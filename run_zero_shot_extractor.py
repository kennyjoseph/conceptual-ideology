"""Ask a gateway model which ideology-codebook elements a text expresses.

The codebook (ideology_nodes_v3.csv, 112 concepts across Values / Attitudes /
Beliefs / Rationalizations) goes into every prompt -- about 3.5K tokens, which
every gateway model handles fine. Commands, roughly in the order you'd use them:

  size       how big the codebook prompt is
  smoke      code one text you type on the command line
  classify   code a few texts sampled from the parquet, printing the full rows
  run-all    code every tweet -> JSONL of rt_id -> [{id, confidence, source, stance}]
  to-matrix  pivot that JSONL into a rt_id x element confidence matrix

Reasoning models (GLM, gpt-oss) spend the whole budget on hidden reasoning and
return nothing, so the default model is a non-reasoning one; raise --max-tokens
to ~16384 if you switch.
"""

import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer

from backends import (
    GATEWAY_BASE_URL,
    GATEWAY_MODEL,
    GATEWAY_MODELS,
    OpenAICompatBackend,
    gateway_key,
)

CODEBOOK = Path("ideology_nodes_v3.csv")
PARQUET = Path("full_data_with_topics_sampleweights.parquet")

# Reasoning models need headroom or they return an empty message.
MAX_TOKENS = 4096

SOURCES = ("author", "other")      # whose thinking is being stated
STANCES = ("support", "reject")    # and with what polarity

# Two questions, not one, and they are independent. Earlier versions asked which elements
# a text "expresses" and let a text that merely "references" an element score the same as
# one asserting it: a tweet reporting that a bomber is a white supremacist scored on
# "white people are superior" exactly like a tweet asserting it. Nothing downstream can
# undo that -- the rate layer sees identical margins either way, and so does the pair
# layer, because describing an ideology reproduces its structure.
#
# The first attempt at a fix used one four-valued field (assert / report / reject /
# accuse). Those four are really the cells of a 2x2, and collapsing them hid the
# structure: there was no way to code "X believes Y, and they are right" (someone else's
# position, endorsed), and opposing an element was ambiguous against the codebook's own
# polarity pairs, which carry both "Support for affirmative action" and "Opposition to
# affirmative action" as separate elements. Asked separately the cells are unambiguous:
#
#                    support                      reject
#   author    the author holds it           the author opposes it
#   other     someone else holds it,        someone else holds it and the
#             the author reporting          author is refuting or accusing
#
# `support` is read widely, and so is `reject`: only the Beliefs layer is asserted in the
# propositional sense. A value is held, an attitude is supported, a rationalization is
# offered -- and each has an opposite that is opposition rather than denial.
#
# One consequence needs saying in the prompt, because getting it backwards silently
# deletes a class of anchors. The codebook already carries both directions of 23 attitude
# pairs as separate elements (A5 "Opposition to affirmative action" against A8 "Support
# for affirmative action", and so on: 26 of the 112 elements have an explicit opposite,
# listed as `direct_negation` in the forbidden-edges file). For those, opposing the one is
# holding the other, and the coding must reach for the matching element with `support`.
# Coding A8 + reject instead would leave A5 -- a C1 anchor for colorblindness -- expressed
# by nobody, since the default filter keeps only author/support. `reject` is for the other
# 86, where the codebook has no element for the position the author holds.
SOURCE_RULES = """\
- author: the author is stating their own thinking.
- other: the author is stating someone else's -- quoting, describing, or summarising
  what another person, group, or movement says, believes, or does."""

STANCE_RULES = """\
- support: the position is asserted or supported -- put forward as true, held as a
  value, endorsed, or backed in what it calls for.
- reject: the position is rejected or refuted -- denied, argued against, mocked, or
  opposed in what it calls for.

Where the codebook carries a separate element for the position the author actually
holds -- it has both "Opposition to affirmative action" and "Support for affirmative
action", and many pairs like them -- code that element with `support` rather than its
opposite with `reject`. Use `reject` only when no element matches the position held."""

EXAMPLES = """\
The two fields are independent, and the four combinations are all different things:

  "The MAGABomber is a hardcore white supremacist who called Black people slurs."
    -> the element about racial superiority: source `other`, stance `support`.
       Someone else's position, reported. The author asserts none of it.

  "Biden is the only candidate with ties to the KKK and white supremacy."
    -> the element about racial superiority: source `other`, stance `reject`.
       Attributed to an opponent as a charge against them.

  "If you can read this then you're racist, because English was invented by white
   devils. Checkmate."
    -> the element about racial superiority: source `other`, stance `reject`.
       The author is mocking a position they attribute to others.

  "Affirmative action is unfair and should be scrapped."
    -> the element about OPPOSITION to affirmative action: source `author`, stance
       `support`. The codebook carries both directions of many attitudes as separate
       elements. Where it has one that matches the author's own position, code that
       one with `support`; do not code its opposite with `reject` instead.

  "People keep saying Jim Crow was ages ago so racism is over. Nonsense."
    -> the element about Jim Crow being too long ago to explain inequality: source
       `author`, stance `reject`. Here the codebook has no separate element for the
       position the author does hold, so `reject` on this one is what records it.
       That is what `reject` is for: use it when, and only when, the codebook has no
       element matching the author's position."""

INSTRUCTIONS = """\
Below is a codebook of ideological elements, grouped into four layers: Values,
Beliefs, Attitudes, and Rationalizations.

{codebook}

Here is a text (a tweet):

\"\"\"{text}\"\"\"

Which codebook elements does this text engage?

Code an element only if the text genuinely engages it. Do NOT code an element merely
because the text is on the same topic, or because a word from the element appears in
it. Most texts engage a handful at most, and an empty list is a fine answer.

For each element you do code, answer two separate questions.

First, WHOSE thinking is being stated?

{source_rules}

Second, with WHAT STANCE is that position put?

{stance_rules}

The two are independent. An author can report someone else's position approvingly
(other + support) or attribute it to an opponent in order to attack them (other +
reject); they can hold a position (author + support) or oppose it (author + reject).
Only `author` + `support` means the author holds the element. The rest record that the
element is present in the text without the author standing behind it, which is
information worth keeping rather than discarding -- so return them, labelled, rather
than omitting them.

{examples}

Return only a JSON object with key `elements`, an array of objects with keys `id` (a
codebook id such as V1 or A12 or B3), `confidence` (0 to 1, how sure you are the text
engages that element at all), `source` (author or other), and `stance` (support or
reject), e.g.
[{{"id": "V1", "confidence": 0.9, "source": "author", "stance": "support"}},
 {{"id": "B11", "confidence": 0.7, "source": "other", "stance": "reject"}}]."""

SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source": {"type": "string", "enum": list(SOURCES)},
                    "stance": {"type": "string", "enum": list(STANCES)},
                },
                "required": ["id", "confidence", "source", "stance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["elements"],
    "additionalProperties": False,
}

app = typer.Typer(help=__doc__, add_completion=False)


def load_codebook(path: Path = CODEBOOK) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def format_codebook(rows: list[dict]) -> str:
    """One line per element, grouped by layer. Drops the CB/WS/IS and links columns --
    the model is being asked what a text engages, not how the elements connect.

    Uses the `coding_statement` column when the node file has one, falling back to
    `element`. The two are kept separate on purpose: `element` is a short label, good for
    reports and for reading a fitted profile, while `coding_statement` spells out what it
    would mean for an author to hold the element ("The author asserts this as true: ...").
    A bare proposition reads as a topic heading, and a codebook of topic headings gets
    coded by topic -- so the stance requirement belongs on the line where the decision is
    made, not only in the instructions above it.
    """
    out = []
    for layer in ["Values", "Beliefs", "Attitudes", "Rationalizations"]:
        out.append(f"## {layer}")
        out += [f'{r["id"]}: {r.get("coding_statement") or r["element"]}'
                for r in rows if r["layer"] == layer]
        out.append("")
    return "\n".join(out)


def build_prompt(text: str, rows: list[dict] | None = None) -> str:
    rows = rows if rows is not None else load_codebook()
    return INSTRUCTIONS.format(codebook=format_codebook(rows), text=text,
                               source_rules=SOURCE_RULES, stance_rules=STANCE_RULES,
                               examples=EXAMPLES)


def resolve(hits: list[dict], rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Join the model's hits back onto the codebook rows, most confident first.

    Each returned row is a copy of the CSV row plus the model's `confidence`.
    Also returns any ids the model made up.
    """
    by_id = {r["id"]: r for r in rows}
    matched = []
    for h in hits:
        if h.get("id") not in by_id:
            continue
        # An unrecognised or missing field is recorded as unknown rather than silently
        # defaulting to author/support. Downstream filters treat unknown as not-held, so
        # a model that ignores a field costs recall, not validity.
        src = str(h.get("source", "")).strip().lower()
        stn = str(h.get("stance", "")).strip().lower()
        matched.append({
            **by_id[h["id"]], "confidence": h.get("confidence"),
            "source": src if src in SOURCES else "unknown",
            "stance": stn if stn in STANCES else "unknown",
        })
    held = lambda r: not (r["source"] == "author" and r["stance"] == "support")
    matched.sort(key=lambda r: (held(r), -(r["confidence"] or 0)))
    bogus = [h.get("id") for h in hits if h.get("id") not in by_id]
    return matched, bogus


def format_rows(matched: list[dict], bogus: list[str], indent: str = "  ") -> str:
    """One line per coded element: confidence, id, layer, and the element text."""
    if not matched and not bogus:
        return f"{indent}(no elements)"
    out = [
        f'{indent}{r["confidence"]:.2f}  {r.get("source", "?"):<7} '
        f'{r.get("stance", "?"):<7} {r["id"]:<4} {r["layer"][:4]:<5} {r["element"]}'
        for r in matched
    ]
    if bogus:
        out.append(f"{indent}!! ids not in the codebook: {', '.join(bogus)}")
    return "\n".join(out)


def load_texts(n: int, seed: int = 0) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(PARQUET, columns=["modeling_text"])
    return df["modeling_text"].dropna().sample(n, random_state=seed).tolist()


@app.command()
def size() -> None:
    """Report how big the codebook prompt is, in characters and tokens."""
    rows = load_codebook()
    prompt = build_prompt("<a tweet goes here>", rows)
    print(f"{len(rows)} elements, {len(prompt):,} chars")
    try:
        import anthropic

        n = anthropic.Anthropic().messages.count_tokens(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": prompt}],
        ).input_tokens
        print(f"{n:,} input tokens (counted by the Anthropic tokenizer)")
    except Exception as e:
        print(f"~{len(prompt) // 4:,} input tokens (rough estimate; {type(e).__name__})")


@app.command()
def smoke(
    text: str = typer.Option(
        "y'all giving white people a lot of your time this week.", help="text to code"
    ),
    model: str = typer.Option(GATEWAY_MODEL, help=f"one of: {', '.join(GATEWAY_MODELS)}"),
    max_tokens: int = typer.Option(MAX_TOKENS),
) -> None:
    """One call, one text -- does the full-codebook prompt actually work?"""
    backend = OpenAICompatBackend(model, GATEWAY_BASE_URL, gateway_key())
    rows = load_codebook()
    start = time.time()
    result = backend.complete(build_prompt(text, rows), SCHEMA, max_tokens)
    matched, bogus = resolve(result.get("elements", []), rows)
    print(f"{text!r}\n")
    print(format_rows(matched, bogus))
    print(f"\n({time.time() - start:.1f}s)")


@app.command()
def classify(
    n: int = typer.Option(5, help="how many texts to sample from the parquet"),
    model: str = typer.Option(GATEWAY_MODEL, help=f"one of: {', '.join(GATEWAY_MODELS)}"),
    max_tokens: int = typer.Option(MAX_TOKENS),
    out: Path = typer.Option(None, help="write results as JSONL here"),
) -> None:
    """Iterate over sampled texts, coding each one against the full codebook."""
    backend = OpenAICompatBackend(model, GATEWAY_BASE_URL, gateway_key())
    rows = load_codebook()
    handle = out.open("w") if out else None

    for i, text in enumerate(load_texts(n), 1):
        start = time.time()
        try:
            result = backend.complete(build_prompt(text, rows), SCHEMA, max_tokens)
        except Exception as e:  # one bad text shouldn't stop the run
            print(f"[{i}] FAILED {type(e).__name__}: {e}")
            continue
        matched, bogus = resolve(result.get("elements", []), rows)
        print(f"[{i}] ({time.time() - start:.1f}s) {text[:100]!r}")
        print(format_rows(matched, bogus, indent="      "))
        if handle:
            handle.write(
                json.dumps({"text": text, "elements": matched, "bogus": bogus}) + "\n"
            )

    if handle:
        handle.close()


@app.command()
def run_all(
    parquet: Path = typer.Option(Path("min_sample.parquet"), help="input tweets"),
    out: Path = typer.Option(Path("codings.jsonl"), help="JSONL, one line per tweet"),
    model: str = typer.Option("google/gemma-3-27b-it", help=f"one of: {', '.join(GATEWAY_MODELS)}"),
    max_tokens: int = typer.Option(MAX_TOKENS),
    workers: int = typer.Option(16, help="concurrent gateway requests"),
    limit: int = typer.Option(0, help="stop after this many tweets (0 = all)"),
    failures: Path = typer.Option(Path("failures.jsonl"),
                                  help="JSONL of tweets that errored, for a later retry run"),
    only: Path = typer.Option(None,
                              help="retry just the rt_ids in this failures JSONL "
                                   "(e.g. a previous run's failures.jsonl, with a higher "
                                   "--max-tokens)"),
) -> None:
    """Code every tweet, writing rt_id -> [{id, confidence, source, stance}].

    Resumable: rt_ids already in `out` are skipped, so re-running picks up where
    a crash or a Ctrl-C left off.
    """
    import pandas as pd

    df = pd.read_parquet(parquet, columns=["rt_id", "modeling_text"]).dropna(
        subset=["modeling_text"]
    )
    done = set()
    if out.exists():
        with out.open() as f:
            done = {json.loads(line)["rt_id"] for line in f if line.strip()}
    todo = [r for r in df.itertuples(index=False) if str(r.rt_id) not in done]
    if only:
        with only.open() as f:
            retry_ids = {json.loads(line)["rt_id"] for line in f if line.strip()}
        todo = [r for r in todo if str(r.rt_id) in retry_ids]
        print(f"--only {only}: {len(retry_ids):,} ids listed, {len(todo):,} still to do")
    if limit:
        todo = todo[:limit]
    print(f"{len(df):,} tweets, {len(done):,} already coded, {len(todo):,} to do")

    backend = OpenAICompatBackend(model, GATEWAY_BASE_URL, gateway_key())
    rows = load_codebook()
    lock = threading.Lock()
    counts = {"ok": 0, "fail": 0}

    def code(rec) -> dict:
        result = backend.complete(build_prompt(rec.modeling_text, rows), SCHEMA, max_tokens)
        matched, bogus = resolve(result.get("elements", []), rows)
        return {
            "rt_id": str(rec.rt_id),
            # `elements` is the canonical field and carries source and stance. `scores`
            # is kept so anything reading the old format still works; it pools all four
            # cells, which is exactly the conflation this version exists to fix, so
            # nothing new should read it.
            "elements": [{"id": r["id"], "confidence": r["confidence"],
                          "source": r["source"], "stance": r["stance"]}
                         for r in matched],
            "scores": {r["id"]: r["confidence"] for r in matched},
            "bogus": bogus,
        }

    start = time.time()
    # Failures are truncated per run: the file always describes the run that just
    # finished, so it can be fed straight back in via --only.
    with out.open("a") as handle, failures.open("w") as fail_handle, \
            ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(code, rec): rec for rec in todo}
        for n, future in enumerate(as_completed(futures), 1):
            rec = futures[future]
            try:
                line = future.result()
            except Exception as e:  # one bad tweet shouldn't stop the run
                counts["fail"] += 1
                print(f"  FAILED {rec.rt_id} {type(e).__name__}: {e}")
                with lock:
                    fail_handle.write(json.dumps({
                        "rt_id": str(rec.rt_id),
                        "error": type(e).__name__,
                        "message": str(e),
                        "max_tokens": max_tokens,
                        "model": model,
                    }) + "\n")
                    fail_handle.flush()
                continue
            counts["ok"] += 1
            with lock:
                handle.write(json.dumps(line) + "\n")
                handle.flush()
            if n % 100 == 0:
                rate = n / (time.time() - start)
                left = (len(todo) - n) / rate / 60
                print(
                    f"  {n:,}/{len(todo):,}  {rate:.1f}/s  ~{left:.0f} min left  "
                    f"({counts['fail']} failed)"
                )

    print(
        f"done: {counts['ok']:,} coded, {counts['fail']:,} failed, "
        f"{time.time() - start:.0f}s -> {out}"
    )
    if counts["fail"]:
        print(f"  {counts['fail']:,} failure(s) -> {failures}\n"
              f"  retry with: --only {failures} --max-tokens {max_tokens * 2}")


@app.command()
def to_matrix(
    codings: Path = typer.Option(Path("codings.jsonl")),
    out: Path = typer.Option(Path("codings.parquet"), help="rt_id x element matrix"),
) -> None:
    """Pivot the JSONL into a rt_id x element confidence matrix (0 where absent)."""
    import pandas as pd

    with codings.open() as f:
        recs = [json.loads(line) for line in f if line.strip()]
    ids = [r["id"] for r in load_codebook()]
    df = pd.DataFrame(
        [{**{"rt_id": r["rt_id"]}, **r["scores"]} for r in recs]
    ).set_index("rt_id")
    df = df.reindex(columns=ids).fillna(0.0)
    df.to_parquet(out)
    print(f"{df.shape[0]:,} tweets x {df.shape[1]} elements -> {out}")
    print(f"mean elements per tweet: {(df > 0).sum(axis=1).mean():.1f}")


if __name__ == "__main__":
    app()
