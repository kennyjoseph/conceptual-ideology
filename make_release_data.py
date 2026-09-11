"""Build the public replication dataset from the private snapshot.

What is published and what is not, and why.

The corpus is 64,459 tweets by 609 accounts, and the tweets themselves are not ours to
redistribute. What the model actually consumes is not text but the extractor's output --
which codebook concepts each unit expresses, with what confidence, in which of the four
source-by-stance cells -- and that derivative carries no tweet content. So the release
ships the codings, the tweet ids that identify the units, and the actor-level covariates,
and withholds the text, the profile bios and the usernames of everyone except the nine
pinned accounts.

  tweet ids       kept. They are the standard unit of Twitter/X data sharing: anyone with
                  API access can rehydrate what still exists, and nothing is republished
                  by listing them.
  tweet text      dropped. `modeling_text` never leaves the private corpus.
  actor ids       pseudonymous. `rt_uid` becomes A0001..A0609, assigned in sorted order of
                  the real id so the mapping is deterministic and carries no signal.
  usernames       dropped, except the nine pinned accounts. Those are public figures the
                  paper names as the anchors of each ideology, and the pinning mechanism
                  looks actors up by handle, so the claim "these accounts anchor this
                  component" stays checkable.
  profile bios    dropped. They are the source of the keyword covariates, and they contain
                  around sixty-five email addresses scraped from the profiles. The derived
                  0/1 columns ship instead, which is what the model reads anyway.
  community       dropped. `grpname` is not used by anything in the reported path.

Actor row order is preserved exactly. `data.load` takes its actor ordering from the C4
file and `cons.actors` follows it, so reordering rows would permute theta and change the
fit; pseudonymizing in place does not.

Run it, then fit against the result to confirm nothing moved:

    python make_release_data.py
    IDEOLOGY_SNAPSHOT=release/snapshot python fit.py final2 --skip-beta
"""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

import actor_keywords as AK
from experiments import PINS

SRC = Path("snapshot")
OUT = Path("release/snapshot")
CODEBOOK = ["ideology_nodes_v3.csv", "ideology_edges_v3.csv",
            "ideology_forbidden_edges_v3.csv"]

# Everything the released actor file keeps. `description`, `grpname` and `m_grpname` are
# absent by design -- see the module docstring.
ACTOR_COLUMNS = ["rt_uid", "username", "kind", "CB", "WS", "IS"] + list(AK.PATTERNS)


def pseudonyms(rt_uids) -> dict:
    """rt_uid -> A0001.., assigned in sorted order of the real id.

    Sorted rather than row order, so the pseudonym leaks nothing about an actor's position
    in the file (which is itself ordered by how the actor table was built).
    """
    return {u: f"A{i:04d}" for i, u in enumerate(sorted(set(rt_uids)), start=1)}


def build(src: Path = SRC, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((src / "actors.csv").open()))
    pins = {u for v in PINS.values() for u in v}
    alias = pseudonyms(r["rt_uid"] for r in rows)

    kept_pins = []
    with (out / "actors.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ACTOR_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:                                   # row order preserved
            real = r["username"]
            if real in pins:
                kept_pins.append(real)
            w.writerow({**r, "rt_uid": alias[r["rt_uid"]],
                        "username": real if real in pins else alias[r["rt_uid"]]})
    missing = pins - set(kept_pins)
    if missing:
        raise SystemExit(f"pinned accounts missing from the actor file: {sorted(missing)}")

    df = pd.read_parquet(src / "corpus.parquet", columns=["rt_id", "rt_uid", "date"])
    df["rt_uid"] = df["rt_uid"].astype(str).map(alias)
    if df["rt_uid"].isna().any():
        raise SystemExit("a corpus actor has no entry in the actor file")
    df.to_parquet(out / "corpus.parquet", index=False)

    shutil.copyfile(src / "coding.jsonl", out / "coding.jsonl")
    for name in CODEBOOK:
        shutil.copyfile(name, out / name)

    sha8 = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:8]
    lines = sum(1 for _ in (out / "coding.jsonl").open())
    fp = json.loads((src / "fingerprint.json").read_text())
    fp = {"frozen": fp.get("frozen"),
          "release": "anonymized: no tweet text, no bios, pseudonymous actor ids",
          "coding": {"path": "coding.jsonl", "sha8": sha8(out / "coding.jsonl"),
                     "lines": lines},
          "actors": {"path": "actors.csv", "sha8": sha8(out / "actors.csv")},
          "parquet": {"path": "corpus.parquet", "sha8": sha8(out / "corpus.parquet")}}
    (out / "fingerprint.json").write_text(json.dumps(fp, indent=2))

    mb = lambda p: p.stat().st_size / 1e6
    print(f"wrote {out}/")
    print(f"  actors.csv      {len(rows)} actors, {len(ACTOR_COLUMNS)} columns, "
          f"{len(kept_pins)} named pins  ({mb(out / 'actors.csv'):.2f} MB)")
    print(f"  corpus.parquet  {len(df):,} units, columns {list(df.columns)}  "
          f"({mb(out / 'corpus.parquet'):.2f} MB)")
    print(f"  coding.jsonl    {lines:,} lines  ({mb(out / 'coding.jsonl'):.2f} MB)")
    print(f"  codebook        {', '.join(CODEBOOK)}")
    print(f"  fingerprint.json  coding {fp['coding']['sha8']}, "
          f"actors {fp['actors']['sha8']}, parquet {fp['parquet']['sha8']}")


if __name__ == "__main__":
    build()
