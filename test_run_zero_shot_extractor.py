"""The extraction side: the prompt, the source/stance fields, and what the corpus does.

The model cannot recover stance that the coding threw away. A tweet reporting that
someone else is a white supremacist and a tweet asserting white supremacy produce
identical margins in the rate layer, and -- because describing an ideology reproduces its
structure -- identical neighbourhoods in the pair layer too. So the distinction has to
survive extraction or it is gone, and these tests guard the places it can be lost.
"""

import json

import numpy as np
import pytest

import data as D
import run_zero_shot_extractor as R


# --------------------------------------------------------------------------- #
# the prompt
# --------------------------------------------------------------------------- #


def test_the_codebook_lines_state_a_position_not_a_topic():
    """A bare proposition reads as a topic heading, and gets coded by topic.

    Gerunds, not "The author ...": whose position it is now has its own field, so the
    statement must not presuppose an answer to it.
    """
    rows = R.load_codebook()
    assert all(r.get("coding_statement") for r in rows), "node file lost coding_statement"
    text = R.format_codebook(rows)
    for frame in ("Holding this as a value:", "Expressing this attitude:",
                  "Asserting this as true:",
                  "Offering this as a reason or justification:"):
        assert frame in text
    assert "The author" not in text


def test_format_codebook_falls_back_to_element():
    rows = [{"id": "B1", "layer": "Beliefs", "element": "short label",
             "coding_statement": "Asserting this as true: short label"}]
    assert "Asserting this as true" in R.format_codebook(rows)
    assert R.format_codebook([{**rows[0], "coding_statement": ""}]).count("short label") == 1


def test_the_prompt_asks_the_two_questions_separately():
    p = R.build_prompt("a tweet")
    assert "WHOSE thinking is being stated" in p and "WHAT STANCE" in p
    for v in R.SOURCES + R.STANCES:
        assert f"- {v}:" in p
    assert "The two are independent." in p
    assert "Only `author` + `support` means the author holds the element" in p


def test_the_prompt_carries_the_observed_failures_as_examples():
    """The examples are the three tweets the previous coding got wrong, plus both sides
    of the polarity-pair case the split makes newly ambiguous."""
    p = R.build_prompt("a tweet")
    # matched on unwrapped fragments: the examples are line-wrapped in the source
    assert "MAGABomber" in p and "ties to the KKK" in p and "Checkmate" in p
    assert "Affirmative action is unfair" in p and "Jim Crow was ages ago" in p


def test_the_prompt_prefers_the_matching_element_over_rejecting_its_opposite():
    """Getting this backwards silently deletes a class of C1 anchors.

    The codebook carries both directions of 23 attitude pairs as separate elements. If a
    text opposing affirmative action were coded as A8 ("Support for affirmative action")
    with stance `reject`, then A5 ("Opposition to affirmative action") -- a colorblindness
    anchor -- would be expressed by nobody, because the default filter keeps only
    author/support. The rule and its exception both have to be in the prompt.
    """
    p = R.build_prompt("a tweet")
    assert ("code that element with `support` rather than its\nopposite with `reject`"
            in p)
    assert "Use `reject` only when no element matches the position held." in p
    assert "stance\n       `support`" in p  # the affirmative-action example resolves that way


def test_the_codebook_really_does_carry_both_directions():
    """The prompt's rule is only right if the node file is really shaped this way."""
    import csv
    rows = {r["id"]: r for r in csv.DictReader(open("ideology_nodes_v3.csv"))}
    neg = [r for r in csv.DictReader(open("ideology_forbidden_edges_v3.csv"))
           if r["type"] == "direct_negation"]
    paired = {i for r in neg for i in (r["id_1"], r["id_2"])}
    assert len(neg) >= 20 and paired >= {"A5", "A8"}
    assert "Opposition to affirmative action" in rows["A5"]["element"]
    assert "Support for affirmative action" in rows["A8"]["element"]
    # ...and that the `reject` example really has no counterpart element
    assert "R2" not in paired


def test_the_schema_requires_both_fields():
    item = R.SCHEMA["properties"]["elements"]["items"]
    assert {"source", "stance"} <= set(item["required"])
    assert set(item["properties"]["source"]["enum"]) == set(R.SOURCES)
    assert set(item["properties"]["stance"]["enum"]) == set(R.STANCES)


# --------------------------------------------------------------------------- #
# resolve
# --------------------------------------------------------------------------- #


def test_resolve_keeps_both_fields_and_sorts_held_first():
    rows = R.load_codebook()
    hits = [{"id": "B11", "confidence": 0.9, "source": "other", "stance": "support"},
            {"id": "V1", "confidence": 0.4, "source": "author", "stance": "support"},
            {"id": "ZZ9", "confidence": 0.5, "source": "author", "stance": "support"}]
    matched, bogus = R.resolve(hits, rows)
    assert bogus == ["ZZ9"]
    assert [m["id"] for m in matched] == ["V1", "B11"]  # held first, then by confidence
    assert (matched[1]["source"], matched[1]["stance"]) == ("other", "support")


@pytest.mark.parametrize("bad,field", [
    ({}, "source"), ({"source": ""}, "source"), ({"source": "AUTHOR!"}, "source"),
    ({"source": 3}, "source"), ({"stance": "endorse"}, "stance"), ({}, "stance"),
])
def test_an_unusable_field_becomes_unknown_not_a_default(bad, field):
    """A model that ignores a field must cost recall, never validity.

    Defaulting to author/support would silently reinstate the conflation the two fields
    exist to remove, and it would do it invisibly.
    """
    rows = R.load_codebook()
    matched, _ = R.resolve([{"id": "B11", "confidence": 0.9, **bad}], rows)
    assert matched[0][field] == "unknown"
    assert (matched[0]["source"], matched[0]["stance"]) not in D.HELD


def test_resolve_accepts_case_and_padding():
    rows = R.load_codebook()
    matched, _ = R.resolve(
        [{"id": "B11", "confidence": 0.9, "source": " Other ", "stance": "REJECT"}], rows)
    assert (matched[0]["source"], matched[0]["stance"]) == ("other", "reject")


# --------------------------------------------------------------------------- #
# reading a coding file
# --------------------------------------------------------------------------- #


def write(tmp_path, lines):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return p


def test_read_codings_reads_all_three_formats(tmp_path):
    new = write(tmp_path, [{"rt_id": "1", "elements": [
        {"id": "B11", "confidence": 0.9, "source": "other", "stance": "support"},
        {"id": "V1", "confidence": 0.5, "source": "author", "stance": "support"}]}])
    coded, has_stance = D.read_codings(new)
    assert has_stance and len(coded["1"]) == 2
    assert {(e["source"], e["stance"]) for e in coded["1"]} == {
        ("other", "support"), ("author", "support")}

    # the short-lived single-field form maps onto the grid it was a collapse of
    mid = write(tmp_path, [{"rt_id": "1", "elements": [
        {"id": "B11", "confidence": 0.9, "voice": "accuse"},
        {"id": "V1", "confidence": 0.5, "voice": "assert"}]}])
    coded, has_stance = D.read_codings(mid)
    assert has_stance
    by = {e["id"]: (e["source"], e["stance"]) for e in coded["1"]}
    assert by == {"B11": ("other", "reject"), "V1": ("author", "support")}

    old = write(tmp_path, [{"rt_id": "1", "scores": {"B11": 0.9}, "bogus": []}])
    coded, has_stance = D.read_codings(old)
    assert not has_stance
    assert coded["1"] == [{"id": "B11", "confidence": 0.9,
                           "source": "unknown", "stance": "unknown"}]


def test_a_file_without_stance_is_not_silently_filtered(tmp_path):
    """The failure mode this guards: filtering a legacy file drops everything to zero."""
    coded, has_stance = D.read_codings(
        write(tmp_path, [{"rt_id": "1", "scores": {"B11": 0.9}}]))
    assert not has_stance
    # `load` needs a parquet, so the contract is checked where it is decidable: nothing in
    # a stance-free file lands in a kept cell, so filtering it would empty the corpus.
    assert all((e["source"], e["stance"]) not in D.HELD
               for v in coded.values() for e in v)


def test_the_default_keeps_only_what_the_author_holds():
    assert D.HELD == (("author", "support"),)
    assert D.CELLS[("author", "support")] == "holds"
    assert set(D.CELLS) == {(s, t) for s in R.SOURCES for t in R.STANCES}


def test_cell_filtering_selects_the_intended_codings():
    """Filtering is per coding, not per unit: a tweet that holds one element and reports
    another keeps the first and loses the second."""
    els = [{"id": "V1", "source": "author", "stance": "support"},
           {"id": "A8", "source": "author", "stance": "reject"},
           {"id": "B11", "source": "other", "stance": "support"},
           {"id": "B9", "source": "other", "stance": "reject"}]
    for cells, want in [
        (D.HELD, {"V1"}),
        ((("other", "support"), ("other", "reject")), {"B11", "B9"}),
        ((("author", "reject"),), {"A8"}),
        (None, {"V1", "A8", "B11", "B9"}),
    ]:
        got = {e["id"] for e in els
               if cells is None or (e["source"], e["stance"]) in cells}
        assert got == want


def test_run_all_record_stays_readable_by_the_old_reader(tmp_path):
    """The written record must satisfy both readers, or a half-migrated directory breaks."""
    rec = {"rt_id": "1",
           "elements": [{"id": "B11", "confidence": 0.9,
                         "source": "other", "stance": "support"}],
           "scores": {"B11": 0.9}, "bogus": []}
    coded, has_stance = D.read_codings(write(tmp_path, [rec]))
    assert has_stance and coded["1"][0]["source"] == "other"
    assert rec["scores"] == {"B11": 0.9}  # the legacy field is still there
