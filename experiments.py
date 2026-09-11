"""The paper's ladder, and the harness it shares.

`final2_sweep` fits the arms the paper reports -- the facet count crossed with whether a
free component is admitted and whether white supremacy is modelled -- and `keyword_beta`
fits the one arm that puts actor covariates into the alignment regression. `fit.py final2`
drives both; `fit.py final2-pick` writes a chosen arm out.

Two properties of the harness matter more than anything in the individual arms.

The data is frozen. `snapshot/` holds the coding and the actor file as they stood when the
design was fixed, and every output carries their fingerprint. The extraction was still being
written while this was designed -- the corpus went 35,977 -> 53,921 -> 64,459 units over
three days -- so arms fit on different snapshots would be compared as though they differed
by configuration when they differed by corpus.

Fits are cached by configuration. `cached_fit` hashes everything that defines a fit and
reuses `exp/fits/<hash>.pkl` if it is already there, so the sweep is resumable: a crashed
arm does not redo the others, and adding an arm refits only that arm. That is what makes a
twelve-fit ladder practical at ~7 minutes a fit.

The earlier E1-E5 suite and the superseded first ladder (`pins_only_sweep`, `final_sweep`)
produced no results in the paper and now live verbatim in `old/superseded/`.
"""

import hashlib
import json
import os
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import actor_keywords as AK
import constraints as C
import data as D
import diagnostics as G
import model as M
import report_data as RD
import validate as V

# The frozen data directory. Overridable so the anonymized release can be fitted in place
# without copying code: IDEOLOGY_SNAPSHOT=release/snapshot python fit.py final2
SNAP = Path(os.environ.get("IDEOLOGY_SNAPSHOT", "snapshot"))
EXP = Path("exp")
FITS = EXP / "fits"

# Three pins per ideology, in the order the pin ladder adds them: most units first, so the
# one-pin arm is the best-evidenced single actor rather than an average one.
PINS = {
    "CB": ["PrisonPlanet", "RealCandaceO", "bluelivesmtr"],       # 1084, 459, 175
    "WS": ["winstonCovfefe", "PatVPeters", "BlissTabitha"],       #   35,  33,   5
    "IS": ["deray", "BreeNewsome", "jemelehill"],                 # 3902, 2473, 278
}


# --------------------------------------------------------------------------- #
# the frozen corpus
# --------------------------------------------------------------------------- #


def fingerprint() -> dict:
    return json.loads((SNAP / "fingerprint.json").read_text())


def snapshot(period: str = "year", meta_columns=None, years=None, holdout: float = 0.1):
    """The frozen corpus, plus its fingerprint. Nothing else reads the live files."""
    cons = C.load(actors=SNAP / "actors.csv")
    corpus = D.load(codings=SNAP / "coding.jsonl", parquet=SNAP / "corpus.parquet",
                    cons=cons, holdout=holdout, period=period, meta_columns=meta_columns,
                    years=years)
    fp = fingerprint()
    fp["units"], fp["actors_fitted"] = corpus.D, corpus.A
    return cons, corpus, fp


def restrict_pins(cons: C.Constraints, keep_usernames, corpus) -> C.Constraints:
    """Keep only the named actors pinned; release everyone else."""
    import csv
    uid = {r["username"]: r["rt_uid"]
           for r in csv.DictReader((SNAP / "actors.csv").open())}
    want = {uid[n] for n in keep_usernames if n in uid}
    fix = np.array([a in want and cons.theta_fix[i]
                    for i, a in enumerate(cons.actors)])
    val = np.where(fix[:, None], cons.theta_val, np.nan)
    return replace(cons, theta_fix=fix, theta_val=val)


def drop_ideologies(cons: C.Constraints, names) -> C.Constraints:
    for nm in names:
        if nm in cons.names:
            cons = V.drop_ideology(cons, cons.names.index(nm))
    return cons


# --------------------------------------------------------------------------- #
# fitting, cached
# --------------------------------------------------------------------------- #


def cached_fit(tag: str, cons, corpus, h: M.Hyper, stages=("1", "3a", "3b"),
               force: bool = False):
    """Fit, or reload an identical earlier fit.

    The key covers everything that changes the answer: the corpus fingerprint, the node and
    actor sets, the ideology names, which actors are pinned, the metadata design, the
    hyperparameters and the stage list. A collision would silently serve one arm's
    parameters for another, so the key errs toward being too specific -- a spurious refit
    costs seven minutes, a spurious hit costs a wrong table.
    """
    FITS.mkdir(parents=True, exist_ok=True)
    # The theory arrays have to be in the key by content, not by shape. They are what
    # `strip_theory` removes, and two configurations differing only in whether C1/C2/C3
    # were supplied are otherwise identical in every field above -- which served the
    # theory-laden fit for the pins-only arm and made the two look indistinguishable.
    # Hashed rather than inlined: anchor_pos is M x 112 and theory_edges M x 4285.
    theory = hashlib.sha256(
        cons.anchor_pos.tobytes() + cons.anchor_neg.tobytes()
        + cons.theory_edges.tobytes()
        + b"|".join(np.asarray(S_).tobytes() for S_ in cons.incompat)
    ).hexdigest()[:16]
    key = json.dumps({
        "fp": fingerprint()["coding"]["sha8"],
        "nodes": list(cons.node_ids), "names": list(cons.names), "M": int(cons.M),
        "theory": theory,
        "pins": sorted(np.array(cons.actors)[cons.theta_fix].tolist()),
        "pin_values": np.nan_to_num(cons.theta_val[cons.theta_fix]).round(4).tolist(),
        "meta": list(corpus.meta_names), "A": corpus.A, "D": corpus.D,
        # Two fits differing only in the holdout split are otherwise identical in every
        # field here, so a no-holdout fit would be served the held-out fit's parameters.
        "train": int(corpus.train.sum()),
        "h": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
              for k, v in h.__dict__.items()},
        "stages": list(stages),
    }, sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = FITS / f"{digest}.pkl"
    if path.exists() and not force:
        saved = pickle.load(path.open("rb"))
        return saved["params"], M.setup(corpus, cons, h), saved["infos"], True
    t0, c0 = time.time(), time.process_time()
    S = M.setup(corpus, cons, h)
    p, infos = M.fit_stages(S, stages, verbose=False)
    with path.open("wb") as f:
        # Both clocks: wall time is what you waited, CPU time is what it cost. They
        # diverge badly if the machine sleeps mid-sweep -- a fit that reads as 214 minutes
        # and 5 minutes of CPU was asleep, not pathological, and only recording the former
        # sends you hunting a performance bug that is not there.
        pickle.dump({"params": p, "infos": infos, "hyper": h, "tag": tag, "key": key,
                     "seconds": time.time() - t0,
                     "cpu_seconds": time.process_time() - c0}, f)
    return p, S, infos, False


def strip_theory(cons: C.Constraints, parts=("C1", "C2", "C3")) -> C.Constraints:
    """Remove theory-supplied constraints, leaving the concept set and the pins.

    `parts` selects which, because they are not the same kind of claim. C1 (which concepts
    an ideology holds) and C2 (which pairs it links) map *ideologies onto concepts* -- they
    are the content of the theory. C3 is different: incompatible sets are a claim about
    concepts alone, that no coherent position holds both, and it says nothing about which
    ideology holds what. So "no theory linking ideologies to concepts" is parts=("C1","C2"),
    which leaves the coherence cap in force while supplying no affiliations.

    With all three, C1's anchors, C2's links and C3's sets go. What remains is the
    codebook -- which concepts exist, their types, and therefore which pairs the ladder
    admits -- plus whichever actors are pinned.

    This is the configuration that answers "what do a few labelled accounts buy you",
    which E2 does not: E2 withholds identity but still hands the model 40 anchors and 73
    links per ideology, so its strong profile recovery is partly the priors agreeing with
    themselves. Here nothing names a component except its pins, so the components are
    formally exchangeable and the pins are the only thing that breaks it. That is the point
    rather than a hazard, but it does mean the labels have to be checked rather than
    assumed: `concept_table` prints the content so a mislabelled component is visible.

    C5's ladder stays. It is a property of the codebook's type assignments, not a
    theoretical claim about these ideologies, and removing it would change which pairs
    exist rather than which are believed.
    """
    out = cons
    if "C1" in parts:
        out = replace(out, anchor_pos=np.zeros_like(out.anchor_pos),
                      anchor_neg=np.zeros_like(out.anchor_neg))
    if "C2" in parts:
        out = replace(out, theory_edges=np.zeros_like(out.theory_edges))
    if "C3" in parts:
        out = replace(out, incompat=[])
    return out


def concept_table(p, S, top: int = 5) -> dict:
    """The top concepts per ideology, with the columns a paper table needs.

    Was missing: E2 reported how well the profiles agreed with a reference but never wrote
    out what they contained, so there was no way to read the concepts off a fit without an
    ad-hoc script.

    eta is the log-rate deviation, so +1 means adherents express the concept e times more
    than the corpus average once their breadth is removed. distinctiveness subtracts the
    strongest rival, and centrality is edge mass inside the ideology's own support.
    """
    ieta = M.ideology_profile(p, S)
    roles = G.node_roles(p, S)
    bm = G.boundary_mass(p, S)
    out = {}
    for k in range(S.M):
        rows = []
        for rank, u in enumerate(np.argsort(-ieta[k])[:top], start=1):
            rows.append({
                "rank": rank, "node": S.cons.node_ids[u],
                "rung": C.LAYER_CODE[S.cons.layer[u]],
                "eta": round(float(ieta[k, u]), 3),
                "distinctiveness": round(float(roles["distinctiveness"][k, u]), 3),
                "centrality": round(float(bm["inside"][k, u]), 2),
                "text": S.cons.element[u],
            })
        out[S.cons.name(k)] = rows
    return out


def final_csvs(p, S, cons, stem: str = "fit_final", top: int = 40):
    """Two CSVs off a kept arm: the top concepts and the top edges, per ideology.

    Both carry the ideology name rather than an index, and the concept rows carry the
    codebook's own label (`theory`) beside the fitted rank -- the fit never saw it, so
    the column is what a reader checks the naming against.
    """
    import csv as _csv
    ieta = M.ideology_profile(p, S)
    roles = G.node_roles(p, S)
    bm = G.boundary_mass(p, S)
    gold = C.load(actors=SNAP / "actors.csv").anchor_pos
    gold_by_id = {nid: "|".join(nm for j, nm in enumerate(C.IDEOLOGIES) if gold[j, u])
                  for u, nid in enumerate(cons.node_ids)}

    cpath = f"{stem}_concepts.csv"
    with open(cpath, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["ideology", "rank", "node", "rung", "eta", "distinctiveness",
                    "centrality", "theory", "in_support", "text"])
        for k in range(S.M):
            for rank, u in enumerate(np.argsort(-ieta[k])[:top], start=1):
                w.writerow([cons.name(k), rank, cons.node_ids[u],
                            C.LAYER_CODE[cons.layer[u]], round(float(ieta[k, u]), 4),
                            round(float(roles["distinctiveness"][k, u]), 4),
                            round(float(bm["inside"][k, u]), 4),
                            gold_by_id[cons.node_ids[u]], int(ieta[k, u] > 0),
                            cons.element[u]])

    epath = f"{stem}_edges.csv"
    u_i, v_i = cons.pairs[:, 0], cons.pairs[:, 1]
    with open(epath, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["ideology", "rank", "node_u", "node_v", "w", "w_shared",
                    "both_in_support", "rung_u", "rung_v", "text_u", "text_v"])
        for k in range(S.M):
            supp = ieta[k] > 0
            for rank, e in enumerate(np.argsort(-p.W[k])[:top], start=1):
                a, b = int(u_i[e]), int(v_i[e])
                w.writerow([cons.name(k), rank, cons.node_ids[a], cons.node_ids[b],
                            round(float(p.W[k, e]), 4), round(float(p.W0[e]), 4),
                            int(supp[a] and supp[b]),
                            C.LAYER_CODE[cons.layer[a]], C.LAYER_CODE[cons.layer[b]],
                            cons.element[a], cons.element[b]])
    return cpath, epath


# --------------------------------------------------------------------------- #
# the final ladder: WS never modeled, F x one free component
# --------------------------------------------------------------------------- #

# The covariate columns are whatever `actor_keywords` defines patterns for, in that order.
# They were duplicated here as a literal list, which meant adding a keyword to the
# generator silently left the model fitting the old design.
KEYWORD_COLUMNS = list(AK.PATTERNS)


def mapping_vs_theory(p, S, cons) -> dict:
    """Concept mapping, scored only where the codebook has an answer.

    A free component has no row in `anchor_pos` -- that is what makes it free -- so it is
    reported by support size and left out of the macro average rather than scored against
    a gold set it was never given.
    """
    gold = C.load(actors=SNAP / "actors.csv").anchor_pos
    by_name = {nm: gold[i] for i, nm in enumerate(C.IDEOLOGIES)}
    ieta = M.ideology_profile(p, S)
    out, scored = [], []
    for k, nm in enumerate(cons.names):
        supp = ieta[k] > 0
        row = {"ideology": nm, "support": int(supp.sum())}
        if nm in by_name:
            g = by_name[nm].astype(bool)
            tp = int((supp & g).sum())
            row.update(auc=round(float(V.auc(ieta[k], g.astype(float))), 4),
                       precision=round(tp / max(int(supp.sum()), 1), 4),
                       recall=round(tp / max(int(g.sum()), 1), 4), gold=int(g.sum()))
            scored.append(row["auc"])
        else:
            row.update(auc=None, precision=None, recall=None, gold=None)
        out.append(row)
    return {"per_ideology": out,
            "macro_auc": round(float(np.mean(scored)), 4) if scored else None}


def final2_arm(F: int, extra: int, years=(2012, 2022), h=None, meta_columns=(),
               holdout: float = 0.1, tag: str | None = None, ws: bool = False):
    """One arm: all three pins per modeled ideology, C1/C2 stripped, C3 kept.

    `ws` keeps white supremacy as a modeled component. It is off by default because the
    selected ladder does not model it, but the arms that do are the evidence for that
    choice, so they are fit through the same function rather than a parallel one.

    Order matters. WS is dropped first so `restrict_pins` never asks PINS for a name that
    is not being modeled; the free components are added after the pins, so they stay
    unpinned -- a pinned actor's vartheta is zero on them, which is what `free` means here.
    """
    h = h or M.Hyper(max_iter=30)
    hF = replace(h, facets=F)
    cons, corpus, fp = snapshot(meta_columns=list(meta_columns) if meta_columns is not None
                                else None, years=years, holdout=holdout)
    if not ws:
        cons = drop_ideologies(cons, ["WS"])
    cons = restrict_pins(cons, [u for k in cons.names for u in PINS[k]], corpus)
    if extra:
        cons = C.add_free_ideologies(cons, extra)
    cons = strip_theory(cons, ("C1", "C2"))                    # C3 stays
    tag = tag or f"f2-F{F}-x{extra}" + ("-ws" if ws else "")
    p, S, infos, hit = cached_fit(tag, cons, corpus, hF)
    return p, S, cons, corpus, infos, hit, hF, tag


def final2_sweep(ladder=(1, 2, 3, 4), extras=(0, 1), years=(2012, 2022), h=None,
                 out_dir=None, ws: bool = False, out_name="final2.json") -> dict:
    """The final ladder, on the four reported criteria.

    The degeneracy check is `share`: the number of actors whose ideology margin is largest
    on each component. It reads only the model's own assignment, so it needs no community
    metadata -- which is what lets the released corpus carry no group label at all. A
    component at ~100% has absorbed the corpus; one at 0% retains only its pinned actors.
    """
    out_dir = Path(out_dir or EXP)
    rows = []
    for extra in extras:
        for F in ladder:
            p, S, cons, corpus, infos, hit, hF, tag = final2_arm(F, extra, years, h,
                                                                  ws=ws)
            al = G.alignment_summary(p, S)
            assign = M.vartheta_of(M.theta_of(p, S), S)[:, 1:].argmax(1)
            share = [int((assign == j).sum()) for j in range(cons.M)]
            rows.append({
                "arm": tag, "facets": F, "extra": extra, "cached": hit,
                "M": int(cons.M), "names": list(cons.names),
                "n_pins": int(cons.theta_fix.sum()), "units": corpus.D,
                "actors": corpus.A, "held_out": infos[-1]["held_out"],
                "share": share,
                "mapping": mapping_vs_theory(p, S, cons),
                "coherence": [{"k": cons.name(r["k"]), "C": r["C_k"], "null": r["null_mean"],
                               "p": r["p"], "support": r["support"]}
                              for r in G.coherence_ratio(p, S, n_perm=200)],
                "mass": [round(float(x), 1) for x in al["mass"]],
                "min_effective_facet": round(float(min(al["effective_per_facet"])), 1),
                "background": round(float(al["background"].mean()), 4),
            })
            RD.dump({"rows": rows}, out_dir / out_name, indent=1)
    res = {"rows": rows, "years": list(years), "stripped": ["C1", "C2"], "kept": ["C3"],
           "ws": "modeled" if ws else "never modeled", "fingerprint": fingerprint()}
    RD.dump(res, out_dir / out_name, indent=1)
    return res


def keyword_beta(F: int = 1, extra: int = 0, years=(2012, 2022), h=None) -> dict:
    """The keyword-covariate arm: fit on every unit, report beta. No holdout.

    Nothing here is a test -- holdout=0 means there is no held-out anything -- so the
    output is the coefficient table and the two regressions that say how to read it.
    Beta is a *multivariate* coefficient on standardized columns, so a covariate that is
    collinear with a stronger one reads as zero however well it predicts on its own; the
    univariate column is printed beside it for exactly that reason.
    """
    p, S, cons, corpus, infos, hit, hF, tag = final2_arm(
        F, extra, years, h, meta_columns=KEYWORD_COLUMNS, holdout=0.0,
        tag=f"kwbeta-F{F}-x{extra}")
    vth = M.vartheta_of(M.theta_of(p, S), S)
    import csv as _csv
    raw = {r["rt_uid"]: r for r in _csv.DictReader((SNAP / "actors.csv").open())}
    X = {c: np.array([raw.get(a, {}).get(c) == "1" for a in corpus.actors], float)
         for c in KEYWORD_COLUMNS}
    units = np.bincount(corpus.actor, minlength=corpus.A)
    rows = []
    for j, nm in enumerate(corpus.meta_names):
        row = {"covariate": nm, "beta": [round(float(p.beta[j, k]), 4)
                                         for k in range(p.beta.shape[1])]}
        if nm in X:
            y = np.log(np.maximum(vth[:, 1], M.EPS)) - np.log(np.maximum(vth[:, 2], M.EPS))
            Z = np.column_stack([np.ones(corpus.A), X[nm]])
            row["n"] = int(X[nm].sum())
            row["univariate"] = round(float(np.linalg.lstsq(Z, y, rcond=None)[0][1]), 3)
            row["mean_units"] = int(units[X[nm] > 0].mean()) if X[nm].any() else 0
        rows.append(row)
    Xm = np.column_stack([np.ones(corpus.A)] + [X[c] for c in KEYWORD_COLUMNS])
    y = np.log(np.maximum(vth[:, 1], M.EPS)) - np.log(np.maximum(vth[:, 2], M.EPS))
    multi = np.linalg.lstsq(Xm, y, rcond=None)[0][1:]
    for r, mv in zip([r for r in rows if "n" in r], multi):
        r["multivariate"] = round(float(mv), 3)
    res = {"arm": tag, "facets": F, "extra": extra, "cached": hit,
           "columns": ["background"] + list(cons.names), "rows": rows,
           "held_out": None, "note": "holdout=0; no held-out anything is computed",
           "fingerprint": fingerprint()}
    RD.dump(res, EXP / "kwbeta.json", indent=1)
    return res
