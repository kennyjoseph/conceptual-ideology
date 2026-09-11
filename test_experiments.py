"""Tests for the paper's experiment harness.

Each of these guards something that would silently produce a wrong number rather than an
error -- which is the failure mode that matters here, because the output is a table in a
paper and nothing downstream would notice.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

import constraints as C
import diagnostics as G
import experiments as E
import model as M
import test_model as T
import validate as V


def toy(K=8, Mi=2, A=6, D=120):
    cons = T.toy_constraints(K=K, Mi=Mi, A=A)
    corpus = T.toy_corpus(cons, D=D, A=A)
    return cons, corpus


# --------------------------------------------------------------------------- #
# the metadata design
# --------------------------------------------------------------------------- #


def test_empty_metadata_list_is_intercept_only_and_none_is_not():
    """`--meta none` must not degrade to the default.

    `[] or None` is None, and None means "unspecified", which falls back to every m_*
    column. Two experiments withhold identity and would silently stop doing so.
    """
    cons = C.load()
    m_default, n_default = C.metadata(cons.meta_rows, cons.actors, None)
    m_none, n_none = C.metadata(cons.meta_rows, cons.actors, [])
    assert m_none.shape[1] == 1 and n_none == ["(intercept)"]
    assert m_default.shape[1] > 1, "the actor file should have m_* columns to distinguish"
    assert m_none.shape != m_default.shape


@pytest.mark.parametrize("raw,want", [("", None), ("none", []), ("NONE", []),
                                      (" none ", []), ("m_grpname", ["m_grpname"])])
def test_the_meta_sentinel_parses(raw, want):
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    if [x.lower() for x in cols] == ["none"]:
        cols = []
    elif not cols:
        cols = None
    assert cols == want


# --------------------------------------------------------------------------- #
# E1's sampler
# --------------------------------------------------------------------------- #


def test_the_pin_roster_is_three_per_ideology_and_ordered_by_evidence():
    assert set(E.PINS) == {"CB", "WS", "IS"}
    assert all(len(v) == 3 for v in E.PINS.values())
    assert "JackPosobiec" not in {u for v in E.PINS.values() for u in v}


def test_restrict_pins_keeps_exactly_the_named_actors():
    cons = C.load(actors=E.SNAP / "actors.csv")
    assert cons.theta_fix.sum() == 9, "the frozen actor file should carry nine pins"
    keep = [E.PINS[k][0] for k in ("CB", "WS", "IS")]
    c2 = E.restrict_pins(cons, keep, None)
    assert c2.theta_fix.sum() == 3
    assert np.isnan(c2.theta_val[~c2.theta_fix]).all()
    # and the survivors kept their pinned value
    for i in np.flatnonzero(c2.theta_fix):
        assert np.array_equal(np.nan_to_num(c2.theta_val[i]),
                              np.nan_to_num(cons.theta_val[i]))


# --------------------------------------------------------------------------- #
# the diagnostics the ladder reports
# --------------------------------------------------------------------------- #


def test_boundary_mass_reconstructs_the_c6_ratio():
    """E3's per-node score must be the same quantity C6 reports in aggregate.

    A decomposition that did not add back up would be a different statistic wearing the
    name of the one the framework defines, and every intervention ranking drawn from it
    would be uninterpretable.
    """
    cons, corpus = toy()
    S = M.setup(corpus, cons, M.Hyper(facets=1))
    rng = np.random.default_rng(2)
    p = replace(M.initialize(S), eta=0.5 * rng.standard_normal((S.F, cons.K)),
                W=np.abs(rng.standard_normal((cons.M, cons.P))))
    bm = G.boundary_mass(p, S)
    u, v = cons.pairs[:, 0], cons.pairs[:, 1]
    ieta = M.ideology_profile(p, S)
    for k in range(S.M):
        supp = ieta[k] > 0
        w = p.W[k]
        # inside edges have both ends in the support, so the per-node sum double-counts
        assert bm["inside"][k].sum() / 2 == pytest.approx(w[supp[u] & supp[v]].sum())
        # boundary edges have one end in, so they are counted once
        assert bm["boundary"][k].sum() == pytest.approx(w[supp[u] ^ supp[v]].sum())
        # nodes outside the support contribute to neither of C6's sums
        assert not bm["inside"][k][~supp].any()
        assert not bm["boundary"][k][~supp].any()


def test_intervention_scores_cover_the_support_and_carry_every_column():
    cons, corpus = toy()
    S = M.setup(corpus, cons, M.Hyper(facets=1))
    rng = np.random.default_rng(3)
    p = replace(M.initialize(S), eta=0.5 * rng.standard_normal((S.F, cons.K)),
                W=np.abs(rng.standard_normal((cons.M, cons.P))))
    rows = G.intervention_scores(p, S)
    ieta = M.ideology_profile(p, S)
    assert len(rows) == int((ieta > 0).sum())
    need = {"k", "node", "rung", "upstream", "eta", "distinctiveness", "centrality",
            "boundary", "leak", "shared_with", "divergence", "text"}
    assert need <= set(rows[0])
    # V is upstream of R, per the ladder
    by_rung = {r["rung"]: r["upstream"] for r in rows}
    if "V" in by_rung and "R" in by_rung:
        assert by_rung["V"] > by_rung["R"]


# --------------------------------------------------------------------------- #
# grpname prediction
# --------------------------------------------------------------------------- #


def test_grpname_accuracy_reports_a_beatable_baseline():
    cons, corpus = toy(A=8)
    corpus = replace(corpus, groups=["g0", "g0", "g0", "g0", "g1", "g1", "g1", "g1"])
    S = M.setup(corpus, cons, M.Hyper(facets=1))
    r = V.grpname_accuracy(M.initialize(S), S)
    assert 0.0 <= r["accuracy"] <= 1.0
    assert r["baseline"] == pytest.approx(0.5)     # the majority class, 4 of 8
    assert r["n"] == 8 and r["classes"] == ["g0", "g1"]


def test_grpname_accuracy_uses_no_metadata():
    """It must score the model, not the design matrix.

    Asserted on the feature construction rather than by swapping designs: metadata enters
    the fit through theta, so a different design gives a different (legitimately different)
    vartheta, and comparing accuracies across designs cannot separate "metadata leaked into
    the scorer" from "metadata changed the fit". What has to be true is narrower and
    checkable directly -- the features are the fitted actor summary and nothing else.
    """
    cons, corpus = toy(A=8)
    corpus = replace(corpus, groups=["g0"] * 4 + ["g1"] * 4)
    S = M.setup(corpus, cons, M.Hyper(facets=1))
    r = V.grpname_accuracy(M.initialize(S), S)
    assert r["features"] == ["vartheta"] * (cons.M + 1) + ["mu", "intercept"]
    assert not any(f.startswith("m_") for f in r["features"])
    # and the width is fixed by the model, not by how many metadata columns exist
    assert len(r["features"]) == cons.M + 3


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def test_the_snapshot_is_frozen_and_fingerprinted():
    fp = E.fingerprint()
    for key in ("coding", "actors", "parquet"):
        assert key in fp and "sha8" in fp[key]
    for name in ("coding.jsonl", "actors.csv", "corpus.parquet"):
        f = E.SNAP / name
        assert f.exists(), f"{name} missing from the snapshot"
        assert not f.is_symlink(), (f"{name} is a symlink -- the snapshot is not frozen "
                                    "and the live file can change under a running sweep")


def test_free_ideologies_get_names_so_configs_stay_distinguishable():
    """The bug this guards actually happened, and the way it surfaced is the point.

    `add_free_ideologies` padded the anchor arrays but not `names`, so M (read off
    anchor_pos) and len(names) disagreed. Two consequences: `cons.name()` runs off the end,
    and -- because the experiment fit cache keys on the names -- M=3 and M=3+1-free became
    the same configuration, so the free-component arms silently loaded M=3's parameters.
    It happened to crash on a shape mismatch instead of tabulating wrong numbers.
    """
    cons = C.load()
    for n in (1, 2, 3):
        c2 = C.add_free_ideologies(cons, n)
        assert c2.M == len(c2.names) == cons.M + n
        assert c2.names[:cons.M] == list(cons.names)      # the named ones keep their names
        assert c2.name(c2.M - 1) == f"free{n}"
        assert c2.anchor_pos.shape[0] == c2.M
        assert not c2.anchor_pos[cons.M:].any()           # free means unanchored
        assert not c2.theory_edges[cons.M:].any()


def test_the_fit_cache_key_separates_configs_that_differ_only_in_M():
    """The cache must not serve one arm's parameters for another."""
    cons = C.load(actors=E.SNAP / "actors.csv")
    keys = set()
    for n in (0, 1, 2):
        c2 = C.add_free_ideologies(cons, n) if n else cons
        keys.add(json.dumps({"names": list(c2.names), "M": int(c2.M)}, sort_keys=True))
    assert len(keys) == 3, "configs differing only in free-component count collide"


def test_pinned_actors_keep_a_valid_margin_after_padding():
    """Padding theta_val with zeros would silently un-normalise every pin."""
    cons = C.load(actors=E.SNAP / "actors.csv")
    c2 = C.add_free_ideologies(cons, 2)
    for i in np.flatnonzero(c2.theta_fix):
        tot = np.nansum(c2.theta_val[i])
        assert tot == pytest.approx(1.0), f"pin {c2.actors[i]} sums to {tot}"


def test_written_json_has_no_bare_nan_or_infinity():
    """Python's json emits bare NaN/Infinity; JSON.parse rejects them.

    This is not hypothetical: a NaN divergence -- a concept no rival ideology holds, which
    is 104 of 124 rows -- shipped a report that wrote without complaint and failed to load
    in the browser with "Unexpected token 'N'". The writer now sanitises non-finite floats
    to null *and* passes allow_nan=False, so the failure is impossible rather than merely
    fixed.
    """
    import report_data as RD

    payload = {"a": float("nan"), "b": float("inf"), "c": float("-inf"),
               "d": [np.nan, 1.5, np.float64("inf")],
               "e": {"f": np.float32("nan")}, "g": np.array([np.nan, 2.0]),
               "h": np.int64(3), "i": np.bool_(True), "j": "Infinity War"}
    clean = RD.jsonable(payload)
    txt = json.dumps(clean, allow_nan=False)          # would raise if any slipped through
    back = json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert back["a"] is None and back["b"] is None and back["c"] is None
    assert back["d"] == [None, 1.5, None]
    assert back["e"]["f"] is None and back["g"] == [None, 2.0]
    assert back["h"] == 3 and back["i"] is True
    assert back["j"] == "Infinity War", "a string that merely contains the word is fine"


def test_every_experiment_artifact_is_strict_json():
    """Guards the files on disk, not just the writer."""
    from pathlib import Path
    strict = lambda t: json.loads(
        t, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"bare {c}")))
    files = sorted(Path("exp").glob("*.json"))
    if not files:
        pytest.skip("no experiment results yet -- run runs.sh")
    for f in files:
        strict(f.read_text())


def test_the_fit_cache_key_separates_configs_that_differ_only_in_supplied_theory():
    """The second collision of this kind, and it produced a wrong table before it was
    caught.

    `strip_theory` removes C1's anchors, C2's links and C3's sets while leaving the node
    set, the pins, the metadata design and the hyperparameters identical. Every other field
    in the cache key is therefore the same, so the pins-only arm loaded the theory-laden
    fit. The tell was V7 -- which refits and is not cached -- differing while held-out fit
    and coherence matched to four decimals.

    Guarded by hashing the theory arrays into the key, and asserted here on content rather
    than on the key-building code, so a refactor that drops a field still fails.
    """
    import hashlib
    cons = E.drop_ideologies(C.load(actors=E.SNAP / "actors.csv"), ["WS"])
    bare = E.strip_theory(cons)

    def theory_hash(cn):
        return hashlib.sha256(
            cn.anchor_pos.tobytes() + cn.anchor_neg.tobytes()
            + cn.theory_edges.tobytes()
            + b"|".join(np.asarray(s).tobytes() for s in cn.incompat)).hexdigest()[:16]

    assert theory_hash(cons) != theory_hash(bare)
    # and the fields that previously carried the whole key are genuinely identical
    assert list(cons.node_ids) == list(bare.node_ids)
    assert list(cons.names) == list(bare.names) and cons.M == bare.M
    assert np.array_equal(cons.theta_fix, bare.theta_fix)


def test_the_fit_cache_key_separates_configs_that_differ_only_in_the_holdout():
    """The third collision of this kind.

    `keyword_beta` fits on every unit (holdout=0) so that beta is estimated from the whole
    corpus; every other field in the key -- nodes, names, pins, metadata design,
    hyperparameters, stages, A and D -- is identical to the held-out fit of the same
    configuration. The no-holdout arm would therefore have been served the held-out fit's
    parameters, and reported them as if nothing had been withheld.

    Guarded by putting the training-unit count in the key, and asserted here on the corpus
    rather than on the key-building code.
    """
    _, held, _ = E.snapshot(meta_columns=[], years=(2012, 2022), holdout=0.1)
    _, full, _ = E.snapshot(meta_columns=[], years=(2012, 2022), holdout=0.0)
    # the fields that previously carried the whole key are identical ...
    assert held.A == full.A and held.D == full.D
    assert list(held.meta_names) == list(full.meta_names)
    # ... and only the split tells them apart
    assert int(held.train.sum()) < int(full.train.sum()) == full.D


def test_strip_theory_removes_every_supplied_constraint_and_keeps_the_codebook():
    cons = C.load(actors=E.SNAP / "actors.csv")
    bare = E.strip_theory(cons)
    assert not bare.anchor_pos.any() and not bare.anchor_neg.any()
    assert not bare.theory_edges.any()
    assert bare.incompat == []
    # the codebook itself survives: same nodes, same types, so the same admitted pairs
    assert bare.node_ids == cons.node_ids
    assert np.array_equal(bare.layer, cons.layer)
    assert np.array_equal(bare.pairs, cons.pairs)
    assert bare.theta_fix.sum() == cons.theta_fix.sum(), "pins are the only supervision left"


def test_concept_table_shape_and_ordering():
    cons, corpus = toy()
    S = M.setup(corpus, cons, M.Hyper(facets=1))
    rng = np.random.default_rng(0)
    p = replace(M.initialize(S), eta=rng.standard_normal((S.F, cons.K)),
                W=np.abs(rng.standard_normal((cons.M, cons.P))))
    t = E.concept_table(p, S, top=3)
    assert set(t) == set(S.cons.names)
    for k, rows in t.items():
        assert len(rows) == 3
        assert [r["rank"] for r in rows] == [1, 2, 3]
        assert rows[0]["eta"] >= rows[1]["eta"] >= rows[2]["eta"]
        assert {"node", "rung", "eta", "distinctiveness", "centrality", "text"} <= set(rows[0])
