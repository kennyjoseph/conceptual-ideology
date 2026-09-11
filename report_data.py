"""Turn a saved fit into the JSON the HTML report reads.

Kept separate from the report itself so the numbers can be checked without opening a
browser: every field here is computed by the same functions `fit.py run` prints from.

The corpus is rebuilt and then pinned to the `unit_ids` stored in the pickle. The coding
file is still being written, so re-loading it whole would silently fit the report to a
larger corpus than the one that produced the parameters -- and every rate in the report is
relative to m_u, which is estimated from exactly those units.
"""

import hashlib
import json
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import constraints as C
import data as D
import diagnostics as G
import model as M


def jsonable(obj):
    """Make a payload valid JSON, not merely serialisable.

    Python's json emits bare `NaN`, `Infinity` and `-Infinity`. Those are a non-standard
    extension that `json.dumps` accepts and `JSON.parse` refuses, so a single NaN anywhere
    in the tree -- a divergence for a concept no other ideology holds, a correlation over
    an empty set -- produces a file that writes without complaint and fails to load in the
    browser. They become null, which every reader understands and which the page already
    renders as an em dash.

    Numpy scalars and arrays are converted here too, so callers do not need `default=`.
    """
    import math
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def dump(obj, path: Path, **kw) -> None:
    """Write valid JSON or fail loudly. `allow_nan=False` is the belt to jsonable's braces."""
    Path(path).write_text(json.dumps(jsonable(obj), allow_nan=False, **kw))


def rebuild(pkl: Path, codings: Path, parquet: Path = D.PARQUET):
    saved = pickle.load(open(pkl, "rb"))
    h = saved["hyper"]
    # A fit that dropped an ideology or stripped C1/C2 no longer matches the live theory
    # files, and one fit with an intercept-only design no longer matches the default
    # metadata columns -- both mismatches surface as a shape error deep in setup rather
    # than as anything readable. Fits that carry their own constraints are rebuilt from
    # them; older pickles fall back to the live files, which is what they were fit on.
    cons = saved.get("cons") or C.load()
    # `meta_columns` is reconstructed from the design the fit actually used, so beta and
    # m_a keep the same width. The intercept is added by meta_for and is not a column.
    want_meta = [n for n in saved.get("meta_names", []) if n != "(intercept)"]
    cols = sorted({n.split("=")[0] for n in want_meta}) if want_meta else []
    corpus = D.load(codings=codings, parquet=parquet, cons=cons, holdout=0.0,
                    period="year", meta_columns=cols)
    want = list(saved["unit_ids"])
    pos = {u: i for i, u in enumerate(corpus.unit_ids)}
    keep = np.array([pos[u] for u in want if u in pos])
    if len(keep) != len(want):
        raise SystemExit(f"{len(want) - len(keep)} of the fit's units are no longer in "
                         f"{codings}; the coding file has changed under the fit")
    corpus = replace(corpus, x=corpus.x[keep], actor=corpus.actor[keep],
                     time=corpus.time[keep], stamp=corpus.stamp[keep],
                     unit_ids=[corpus.unit_ids[i] for i in keep],
                     train=np.ones(len(keep), bool))
    # Period labels are built over the whole file, so a fit made on a date window has
    # fewer of them than the reload -- and `time` indexes the label list, so the extra
    # labels shift every index. Remap onto the fit's own periods, which the subset units
    # are guaranteed to lie inside.
    if list(corpus.periods) != list(saved["periods"]):
        at = {lab: i for i, lab in enumerate(saved["periods"])}
        missing = {corpus.periods[t] for t in corpus.time} - set(at)
        if missing:
            raise SystemExit(f"the reload has periods the fit never saw: {sorted(missing)}")
        corpus = replace(corpus, periods=list(saved["periods"]),
                         time=np.array([at[corpus.periods[t]] for t in corpus.time],
                                       dtype=corpus.time.dtype))
    # actors were stored in fit order; drop any the subset lost and reindex
    if corpus.actors != saved["actors"]:
        idx = {a: i for i, a in enumerate(corpus.actors)}
        sel = np.array([idx[a] for a in saved["actors"]])
        remap = -np.ones(corpus.A, int)
        remap[sel] = np.arange(len(sel))
        corpus = replace(corpus, actors=list(saved["actors"]),
                         actor=remap[corpus.actor],
                         meta=corpus.meta[sel],
                         groups=[corpus.groups[i] for i in sel])
        if (corpus.actor < 0).any():
            raise SystemExit("the fit's actor list does not cover the corpus")
    return saved, cons, corpus, M.setup(corpus, cons, h)


CELLS_ORDER = [("author", "support"), ("author", "reject"),
               ("other", "support"), ("other", "reject")]
CELL_NAME = ["holds", "opposes", "reports", "refutes", "unknown"]


def _columns(parquet: Path) -> set:
    import pyarrow.parquet as pq
    return set(pq.read_schema(parquet).names)


def actor_meta(parquet: Path = D.PARQUET) -> dict:
    """rt_uid -> {username, description, group}. The report addresses actors by name.

    Whatever of these the corpus carries. The released corpus carries none of them -- no
    username, no bio, no group -- so the report addresses actors by their pseudonymous id
    and shows no profile text. Reports built against the private corpus are identical to
    before; reports built against the release are safe to publish.
    """
    import pandas as pd
    have = _columns(parquet)
    cols = [c for c in ("rt_uid", "username", "grpname", "description") if c in have]
    df = pd.read_parquet(parquet, columns=cols).drop_duplicates("rt_uid")
    get = lambda r, c: str(getattr(r, c, "") or "")
    return {str(r.rt_uid): {"username": get(r, "username"),
                            "description": get(r, "description").replace("\n", " "),
                            "group": get(r, "grpname")}
            for r in df.itertuples(index=False)}


def unit_text(unit_ids, parquet: Path = D.PARQUET, cap: int = 320) -> dict:
    """unit id -> its text, where the corpus has text.

    The released corpus deliberately has none: tweet ids are published so the units can be
    rehydrated by anyone with API access, but the text itself is not redistributed. An
    empty mapping here is the normal case for a public build, not a failure.
    """
    import pandas as pd
    if "modeling_text" not in _columns(parquet):
        return {}
    want = set(map(str, unit_ids))
    df = pd.read_parquet(parquet, columns=["rt_id", "modeling_text"]).drop_duplicates("rt_id")
    df["rt_id"] = df.rt_id.astype(str)
    df = df[df.rt_id.isin(want)]
    return {r.rt_id: str(r.modeling_text or "").replace("\n", " ")[:cap]
            for r in df.itertuples(index=False)}


def build(pkl="fit.pkl", codings=str(D.CODINGS), out="report_data.json",
          parquet=str(D.PARQUET)) -> dict:
    saved, cons, corpus, S = rebuild(Path(pkl), Path(codings), Path(parquet))
    p = saved["params"]
    th = M.theta_of(p, S)
    vth = M.vartheta_of(th, S)
    ieta = M.ideology_profile(p, S)
    al = G.alignment_summary(p, S)
    names = cons.names
    u, v = cons.pairs[:, 0], cons.pairs[:, 1]
    blocks = C.block_names()
    block_of = cons.block_of()

    # ---- rates: what the extractor said vs what the model predicts ---------
    pred = M.actor_rates(p, S, th)                       # [A, K] per-unit rate
    obs = np.zeros((corpus.A, corpus.K))
    np.add.at(obs, corpus.actor, corpus.x)
    n_units = np.bincount(corpus.actor, minlength=corpus.A).astype(float)
    obs = obs / np.maximum(n_units[:, None], 1.0)

    cells = []
    for a in range(corpus.A):
        if n_units[a] == 0:
            continue
        for k_ in range(corpus.K):
            o, e = float(obs[a, k_]), float(pred[a, k_])
            if o <= 0 and e < 1e-4:
                continue                                  # neither side says anything
            cells.append([a, k_, round(o, 5), round(e, 5),
                          round(float(np.log10(max(o, 1e-6) / max(e, 1e-6))), 3)])

    # ---- per-ideology structure: theory against fit ------------------------
    div = G.theory_divergence(p, S)
    ideologies = []
    for k in range(cons.M):
        anchored = cons.anchor_pos[k]
        theory_e = cons.theory_edges[k]
        w = p.W[k]
        lifted = theory_e & (w > S.h.omega + 1e-6)
        inert = theory_e & ~lifted & (p.W0 <= w + 1e-6)
        shared = theory_e & ~lifted & ~inert
        found = ~theory_e & (w > 0.25)
        nodes = [{"i": int(i), "id": cons.node_ids[i], "rung": int(cons.layer[i]),
                  "anchored": bool(anchored[i]), "eta": round(float(ieta[k, i]), 3),
                  "facets": [round(float(p.eta[f, i]), 3)
                             for f in np.flatnonzero(S.facet_of == k + 1)],
                  "text": cons.element[i]}
                 for i in range(corpus.K)]
        edges = []
        for cat, sel in (("lifted", lifted), ("inert", inert), ("shared", shared),
                         ("found", found)):
            for e in np.flatnonzero(sel):
                edges.append({"u": int(u[e]), "v": int(v[e]), "cat": cat,
                              "w": round(float(w[e]), 3),
                              "w0": round(float(p.W0[e]), 3),
                              "block": blocks[int(block_of[e])]})
        ideologies.append({
            "name": names[k] if k < len(names) else f"free{k}",
            "mass": round(float(al["mass"][k]), 2),
            "effective_actors": round(float(al["effective_actors"][k]), 1),
            "support": int((ieta[k] > 0).sum()),
            "n_anchors": int(anchored.sum()), "n_theory_edges": int(theory_e.sum()),
            "counts": {c: int(s.sum()) for c, s in
                       (("lifted", lifted), ("inert", inert), ("shared", shared),
                        ("found", found))},
            "nodes": nodes, "edges": edges,
        })

    # ---- every live edge, for the block view ------------------------------
    # ideology[k].edges only carries pairs theory named plus strong finds; a block view
    # has to see the whole matrix, including edges that live only in the shared layer.
    live = (p.W0 > 1e-8) | (p.W > 1e-8).any(0)
    theory_any = cons.theory_edges.any(0)
    all_edges = [[int(u[e]), int(v[e]), int(block_of[e]),
                  round(float(p.W0[e]), 3),
                  [round(float(p.W[k, e]), 3) for k in range(cons.M)],
                  bool(theory_any[e])]
                 for e in np.flatnonzero(live | theory_any)]

    coh = G.coherence_ratio(p, S, n_perm=200)
    em = G.edge_mass(p, S)
    typ = G.contestation(p, S)
    from collections import Counter

    # ---- the units behind each coding ------------------------------------
    # Every voice, not only the ones the fit saw. A cell where the model predicts more
    # than the extractor is very often a node the extractor *did* code and the voice
    # filter then dropped, and that is only visible if the dropped codings travel with
    # the text.
    # Both must read the corpus the fit was built against, not the module default: a
    # report built from the anonymized release would otherwise silently reach for the
    # private parquet and re-attach the usernames and tweet text it exists to omit.
    who = actor_meta(Path(parquet))
    texts = unit_text(corpus.unit_ids, Path(parquet))
    raw, _ = D.read_codings(Path(codings))
    cellcode = {c: i for i, c in enumerate(CELLS_ORDER)}
    actor_of = {uid: int(corpus.actor[i]) for i, uid in enumerate(corpus.unit_ids)}
    time_of = {uid: int(corpus.time[i]) for i, uid in enumerate(corpus.unit_ids)}
    # The model's expectation for a coding is P(node appears in a unit by this actor at
    # this time) = 1 - exp(-eta_{a,u,t}). It carries no per-tweet information -- every
    # unit by one actor in one period gets the same number -- so it is attached per
    # coding for convenience, not because the model read the text.
    rt = np.exp(M.period_log_rates(p, S, th))            # [A, T, K]
    p_expr = -np.expm1(-rt)
    units = []
    for uid in corpus.unit_ids:
        els = raw.get(uid) or []
        if not els:
            continue
        a_i, t_i = actor_of[uid], time_of[uid]
        codes = [[cons.index[e["id"]],
                  round(float(e.get("confidence") or 0), 2),
                  cellcode.get((e.get("source"), e.get("stance")), 4),
                  round(float(p_expr[a_i, t_i, cons.index[e["id"]]]), 4)]
                 for e in els if e["id"] in cons.index]
        if not codes:
            continue
        units.append([a_i, t_i, texts.get(uid, ""), codes])

    # The fit pins the unit *ids* but nothing pins the codings on them: the extraction
    # can re-code a unit after the fit was taken, and then the extractor rates shown here
    # and the parameters they are compared against come from different snapshots. Record
    # enough to notice, since the pickle stores no fingerprint of its own input.
    import hashlib
    raw_bytes = Path(codings).read_bytes()
    now = {"sha8": hashlib.sha256(raw_bytes).hexdigest()[:8],
           "lines": raw_bytes.count(b"\n")}
    was = saved.get("codings")          # absent in fits taken before this was recorded
    fingerprint = {
        "now": now, "was": was,
        "state": "unknown" if not was else
                 ("match" if was.get("sha8") == now["sha8"] else "changed"),
        "codings_mtime": time.strftime("%Y-%m-%d %H:%M",
                                       time.localtime(Path(codings).stat().st_mtime)),
    }

    # The paper's analyses, if they have been run. Optional so the report still builds
    # from a bare fit; a stale mixture is refused rather than shown, since `collect`
    # already checks the arms share one snapshot and this only has to notice the file is
    # from a different one than the fit.
    experiments = None
    coll = Path("exp/collected.json")
    if coll.exists():
        experiments = json.loads(coll.read_text())

    payload = {
        "provenance": fingerprint,
        "experiments": experiments,
        "meta": {
            "facets": int(S.h.facets), "kernel": S.ps.label, "M": cons.M,
            "K": corpus.K, "A": corpus.A, "D": corpus.D, "P": cons.P,
            "pi_bar": S.h.pi_bar, "omega": S.h.omega, "alpha": S.h.alpha,
            "codings": codings, "parquet": parquet, "periods": [corpus.periods[0], corpus.periods[-1]],
            "period_labels": list(corpus.periods),
            "stages": [{"name": i["stage"], "train": round(i["history"][-1], 1),
                        "rate": round(i["train_rate"], 1),
                        "pair": round(i["train_pair"], 1),
                        "held_out": round(i["held_out"], 1)} for i in saved["infos"]],
            "ideologies": names[:cons.M],
            "rungs": C.LAYER_CODE, "blocks": blocks,
        },
        "nodes": [{"id": cons.node_ids[i], "rung": int(cons.layer[i]),
                   "text": cons.element[i],
                   "m": round(float(S.m[i]), 3),
                   "anchored": [names[k] for k in range(cons.M)
                                if cons.anchor_pos[k, i]]}
                  for i in range(corpus.K)],
        "cell_names": CELL_NAME,
        "units": units,
        "actors": [{
            "id": corpus.actors[a],
            "name": who.get(corpus.actors[a], {}).get("username") or corpus.actors[a],
            "desc": who.get(corpus.actors[a], {}).get("description", ""),
            "group": corpus.groups[a],
            "units": int(n_units[a]),
            "pinned": bool(S.fixed[a]),
            "vartheta": [round(float(x), 4) for x in vth[a, 1:]],
            "background": round(float(vth[a, 0]), 4),
            "facets": [round(float(x), 4) for x in th[a, 1:]],
            "entropy": round(float(al["entropy"][a]), 3),
            "mu": round(float(p.mu[a]), 3),
        } for a in range(corpus.A)],
        "facet_of": [int(x) for x in S.facet_of],
        "cells": cells,
        "ideology": ideologies,
        "edges": all_edges,
        "skipping": [[C.LAYER_CODE[a] + C.LAYER_CODE[b] for b in range(4)]
                     for a in range(4)],
        "forbidden_blocks": sorted(
            {C.LAYER_CODE[min(a, b)] + C.LAYER_CODE[max(a, b)]
             for fs in C.FORBIDDEN_RUNGS for a in fs for b in fs if a != b}),
        "coherence": [{"k": names[r["k"]], "C": None if not np.isfinite(r["C_k"])
                       else round(r["C_k"], 3), "support": r["support"],
                       "null": None if not np.isfinite(r["null_mean"])
                       else round(r["null_mean"], 3),
                       "p": None if not np.isfinite(r["p"]) else round(r["p"], 4)}
                      for r in coh],
        "edge_mass": {"shared": round(em["shared"], 2),
                      "ideology": round(em["ideology"], 2),
                      "share": round(em["ideology_share"], 4),
                      "per_block": {b: {"pairs": r["pairs"],
                                        "shared": round(r["shared"], 2),
                                        "ideology": round(r["ideology"], 2)}
                                    for b, r in em["per_block"].items()}},
        "typology": dict(Counter(r["type"] for r in typ)),
        "divergence": {"anchors_met": round(div["anchors_met"], 4),
                       "edges_lifted": round(div["edges_lifted"], 4),
                       "n_anchors": div["n_anchors"],
                       "n_theory_edges": div["n_theory_edges"],
                       "poached": div["poached"][:40], "unmet": div["unmet"][:40]},
        "incoherence": [{"set": r["set"], "V_S": round(r["V_S"], 5),
                         "n_units": r["n_units"], "mass": round(r["max_profile"], 4)}
                        for r in G.incoherence(p, S)[:20]],
        "logical": [{"k": r["k"], "nodes": r["nodes"],
                     "eta": [round(e, 2) for e in r["eta"]],
                     "mass": round(r["mass"], 4), "elements": r["elements"]}
                    for r in G.logical_incoherence(p, S)],
        "ladder": [{"k": names[r["k"]], "pair": list(r["pair"]),
                    "layers": list(r["layers"]), "R": round(r["R"], 2),
                    "expected": round(r["expected"], 2)}
                   for r in G.ladder_violations(p, S)[:25]],
        "split": [{"k": names[r["k"]], "n_pairs": r["n_pairs"],
                   "frac_under": None if not np.isfinite(r["frac_under"])
                   else round(r["frac_under"], 3),
                   "median_R": None if not np.isfinite(r["median_R"])
                   else round(r["median_R"], 3)} for r in G.split_signal(p, S)],
    }
    dump(payload, Path(out), separators=(",", ":"))
    print(f"wrote {out}: {Path(out).stat().st_size / 1e6:.2f} MB, "
          f"{len(cells):,} actor-node cells, {len(units):,} coded units, "
          f"{sum(len(i['edges']) for i in ideologies)} edges")
    return payload


if __name__ == "__main__":
    build()
