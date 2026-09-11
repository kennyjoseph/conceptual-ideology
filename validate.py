"""Predictive tests that hold out things the fit is normally given -- V1 to V8.

The diagnostics in diagnostics.py are residuals: they describe a fit. These are tests:
each one withholds something the corpus or the theory files supply, refits, and asks
whether the model recovers it. That is the difference between reporting that theory was
imposed and showing that the data supports it.

Constraints and priors are inputs; each test withholds one. The ladder (C5) is not among
them -- it is a commitment about the grounding order of ideological content, and its
rung-skipping residuals stay diagnostics of type assignments rather than a hypothesis
this corpus adjudicates. Every test reports a baseline, because most of these numbers
look good in isolation.

  V1  concept-ideology affiliation (C1)      anchor_holdout
  V2  does supplied theory buy prediction?   supply_curve
  V3  incompatibility (C3)                   forbidden_holdout
  V4  actor groups                           group_recovery
  V5  actor stance on a withheld concept     actor_concept_holdout
  V6  the timescale curve                    timescale_curve
  V7  whole-actor holdout                    actor_holdout      <- the primary criterion
  V8  coherence and the culture/ideology partition               coherence_and_partition
"""

from dataclasses import replace

import numpy as np

import constraints as C
import data as D
import diagnostics as G
import model as M

STAGES = ("1", "3a", "3b")


# --------------------------------------------------------------------------- #
# small statistics, kept local so the module has no sklearn dependency
# --------------------------------------------------------------------------- #


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC; 0.5 is chance. Ties are handled by average ranks."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    if labels.all() or not labels.any():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation. The rate target is heavily skewed, so ranks not values."""
    rank = lambda z: np.argsort(np.argsort(z)).astype(float)
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def adjusted_rand(labels: np.ndarray, assign: np.ndarray, n_k: int) -> float:
    names = sorted(set(labels))
    cont = np.array([[int(((labels == g) & (assign == k)).sum()) for k in range(n_k)]
                     for g in names], dtype=float)
    comb = lambda n: n * (n - 1) / 2
    sum_ij, sum_i, sum_j = comb(cont).sum(), comb(cont.sum(1)).sum(), comb(cont.sum(0)).sum()
    n = comb(cont.sum())
    if not n:
        return float("nan")
    exp = sum_i * sum_j / n
    den = 0.5 * (sum_i + sum_j) - exp
    return float((sum_ij - exp) / den) if den else float("nan")


def fit_quiet(S: M.Setup, stages=STAGES):
    return M.fit_stages(S, stages, verbose=False)


# --------------------------------------------------------------------------- #
# V1. concept-ideology affiliation (C1)
# --------------------------------------------------------------------------- #


def anchor_holdout(corpus, cons, h, frac: float = 0.30, seed: int = 0, stages=STAGES) -> dict:
    """Drop a fraction of A^+_k, refit, and rank the held-out affiliations by fitted eta.

    Two comparison sets, because they ask different questions and on this node file only
    one of them exists. `auc` ranks the held-out nodes against everything *this* ideology
    was never anchored to -- the always-available contrast, and the harder one, since most
    of that set is other ideologies' definitional cores. `auc_unaffiliated` uses the
    write-up's literal "never-affiliated" set, nodes anchored to no ideology at all; the
    current node file anchors every node to something, so it is reported as nan rather
    than quietly falling back.
    """
    rng = np.random.default_rng(seed)
    kept = cons.anchor_pos.copy()
    held = np.zeros_like(kept)
    for k in range(cons.M):
        idx = np.flatnonzero(cons.anchor_pos[k])
        n = int(round(frac * len(idx)))
        if n:
            drop = rng.choice(idx, size=n, replace=False)
            kept[k, drop] = False
            held[k, drop] = True
    unaffiliated = ~cons.anchor_pos.any(0)
    S = M.setup(corpus, replace(cons, anchor_pos=kept), h)
    p, infos = fit_quiet(S, stages)

    ieta = M.ideology_profile(p, S)

    def rank(k, negatives):
        if not held[k].any() or not negatives.any():
            return np.nan
        return auc(np.concatenate([ieta[k][held[k]], ieta[k][negatives]]),
                   np.concatenate([np.ones(held[k].sum()), np.zeros(negatives.sum())]))

    rows = []
    for k in range(cons.M):
        other = ~cons.anchor_pos[k]
        rows.append({
            "ideology": cons.name(k),
            "n_held": int(held[k].sum()),
            "n_negative": int(other.sum()),
            "auc": rank(k, other),
            "auc_unaffiliated": rank(k, unaffiliated),
            "mean_eta_held": float(ieta[k][held[k]].mean()) if held[k].any() else np.nan,
            "mean_eta_negative": float(ieta[k][other].mean()) if other.any() else np.nan,
        })
    return {"per_ideology": rows, "held_out": infos[-1]["held_out"],
            "n_unaffiliated": int(unaffiliated.sum())}


# --------------------------------------------------------------------------- #
# V2. does supplied theory buy predictive capacity?
# --------------------------------------------------------------------------- #


def supply_curve(corpus, cons, h, what: str = "anchors", steps=(0.0, 0.25, 0.5, 0.75, 1.0),
                 seed: int = 0, stages=STAGES) -> list[dict]:
    """Supply fractions of A^+_k or E_k and read held-out fit and group recovery.

    Section 10 (V2) makes the prediction sharp: because C1 is now a prior *mean* rather
    than a lower bound, the magnitude claim that damaged fit in earlier versions is gone,
    so the affiliation curve should be flat-to-improving. Monotone degradation would
    indict the prior means themselves rather than the amount of theory supplied.
    """
    rng = np.random.default_rng(seed)
    src = cons.anchor_pos if what == "anchors" else cons.theory_edges
    idx = [np.flatnonzero(src[k]) for k in range(cons.M)]
    perm = [rng.permutation(i) for i in idx]
    rows = []
    for frac in steps:
        keep = np.zeros_like(src)
        for k in range(cons.M):
            n = int(round(frac * len(perm[k])))
            keep[k, perm[k][:n]] = True
        c2 = replace(cons, **({"anchor_pos": keep} if what == "anchors"
                              else {"theory_edges": keep}))
        if what == "anchors" and not keep.any():
            # With no anchors at all every component is exchangeable; the fit is still
            # informative about held-out likelihood, so run it but flag the label switch.
            pass
        S = M.setup(corpus, c2, h)
        p, infos = fit_quiet(S, stages)
        rows.append({
            "supplied": int(keep.sum()),
            "frac": float(frac),
            "held_out": infos[-1]["held_out"],
            "ari": group_recovery(p, S)["ari"],
        })
    return rows


# --------------------------------------------------------------------------- #
# V3. incompatibility (C3)
# --------------------------------------------------------------------------- #


def forbidden_holdout(corpus, cons, h, frac: float = 0.5, seed: int = 0, stages=STAGES
                      ) -> dict:
    """Drop half of X, refit, and report max_k sum_{u in S} e^{m_u + eta_ku} per dropped set.

    The cap is the claim; this asks whether the corpus would have respected it unasked.
    A dropped set whose unconstrained profile mass stays under pi_bar is one the data
    already agreed with.
    """
    rng = np.random.default_rng(seed)
    sets = list(cons.incompat)
    n = int(round(frac * len(sets)))
    drop = set(rng.choice(len(sets), size=n, replace=False).tolist()) if n else set()
    keep = [s for i, s in enumerate(sets) if i not in drop]
    S = M.setup(corpus, replace(cons, incompat=keep), h)
    p, infos = fit_quiet(S, stages)
    rows = []
    ieta = M.ideology_profile(p, S)
    for i in sorted(drop):
        Sx = sets[i]
        mass = [float(np.exp(S.m[Sx] + ieta[k, Sx]).sum()) for k in range(cons.M)]
        rows.append({
            "set": [cons.node_ids[u] for u in Sx],
            "max_mass": max(mass),
            "worst": cons.name(int(np.argmax(mass))),
            "violated": max(mass) > h.pi_bar,
        })
    return {"sets": sorted(rows, key=lambda r: -r["max_mass"]),
            "cap": h.pi_bar, "held_out": infos[-1]["held_out"]}


# --------------------------------------------------------------------------- #
# V4. actor groups
# --------------------------------------------------------------------------- #


def group_recovery(p: M.Params, S: M.Setup) -> dict:
    """grpname against the fitted alignment.

    Reported as adjusted Rand against the hard assignment plus the group-by-ideology
    table, because the interesting question is not the score but the shape: which groups
    the model declines to separate. Run it with the group column dropped from m_a --
    `groups_withheld` below does that -- or the test is circular.
    """
    labels = np.array(S.corpus.groups)
    # The ideology margin, not the facet membership: V4 asks which *ideology* an actor
    # lands on, and a facet split within one ideology is specialisation, not a different
    # answer to that question.
    vth = M.vartheta_of(M.theta_of(p, S), S)[:, 1:]
    assign = vth.argmax(1)
    names = sorted(set(labels))
    table = {g: np.bincount(assign[labels == g], minlength=S.M).tolist() for g in names}
    share = vth / np.maximum(vth.sum(1, keepdims=True), M.EPS)
    return {"ari": adjusted_rand(labels, assign, S.M), "table": table,
            "entropy": float(-(share * np.log(np.maximum(share, M.EPS))).sum(1).mean())}


def group_derived_columns(corpus, cons) -> list[str]:
    """Which `m_*` columns are a relabeling of grpname.

    A column is group-derived when knowing it tells you the group: every one of its
    values maps to a single grpname. `m_grpname` is the obvious case, but so is any
    coarser or finer coding of the same partition, and V4 is circular if either survives
    into m_a.
    """
    if not cons.meta_rows:
        return []
    by_id = {r["rt_uid"]: r for r in cons.meta_rows}
    grp = dict(zip(corpus.actors, corpus.groups))
    out = []
    for col in [c for c in cons.meta_rows[0] if c and c.startswith(C.META_PREFIX)]:
        seen: dict[str, set[str]] = {}
        for a in corpus.actors:
            if a in by_id:
                seen.setdefault(by_id[a].get(col, ""), set()).add(grp[a])
        if seen and all(len(g) == 1 for g in seen.values()):
            out.append(col)
    return out


def groups_withheld(corpus, cons, h, stages=STAGES, **_) -> dict:
    """V4 proper: refit with every group-derived metadata column removed from m_a.

    Alignment then has to come from the remaining metadata, the C4 pins and gamma. If
    actor file's only metadata is the group label this reduces to an intercept-only
    design, and the result reads as "what the pinned anchors alone recover" -- still the
    right baseline, and the printed `meta_used` says which case you are in.
    """
    dropped = group_derived_columns(corpus, cons)
    all_meta = [c for c in (cons.meta_rows[0] if cons.meta_rows else {})
                if c and c.startswith(C.META_PREFIX)]
    keep = [c for c in all_meta if c not in dropped]
    corpus2 = corpus
    if dropped:
        # Rebuild only the design matrix. Reloading the corpus would also discard any
        # subsampling or resplitting the caller did.
        meta, names = D.meta_for(corpus.actors, cons, keep)
        corpus2 = replace(corpus, meta=meta, meta_names=names)
    S = M.setup(corpus2, cons, h)
    p, _ = fit_quiet(S, stages)
    out = group_recovery(p, S)
    out["meta_used"] = corpus2.meta_names
    out["dropped"] = dropped
    return out


# --------------------------------------------------------------------------- #
# V5. actor stance on a withheld concept
# --------------------------------------------------------------------------- #


def actor_concept_holdout(corpus, cons, h, frac: float = 0.15, seed: int = 0,
                          stages=STAGES, baseline_facets: int = 1) -> dict:
    """Mask (actor, node) cells from the rate likelihood and predict the actor's rate.

    Masking is not zeroing. A zero says the actor expresses the node zero times, which is
    a measurement; the mask drops both halves of the Poisson term so the cell contributes
    nothing, and the prediction E[eta_{a,u} | rest] then shrinks toward the aligned
    profile. Baselines, in increasing order of difficulty: the node's corpus rate, breadth
    x node rate, and the peer mean within the actor's group label -- section 10 calls
    beating the third the headline test of the hierarchy.
    """
    rng = np.random.default_rng(seed)
    A, K = corpus.A, corpus.K
    mask = np.ones((A, K))
    n_cells = int(round(frac * A * K))
    flat = rng.choice(A * K, size=n_cells, replace=False)
    mask.reshape(-1)[flat] = 0.0
    S = M.setup(corpus, cons, h, cell_mask=mask)
    p, _ = fit_quiet(S, stages)

    theta = M.theta_of(p, S)
    pred = M.actor_rates(p, S, theta)
    # truth: the actor's observed rate on the held-out cell, over their training units
    obs = np.zeros((A, K))
    np.add.at(obs, corpus.actor[corpus.train], corpus.x[corpus.train])
    n_units = np.bincount(corpus.actor[corpus.train], minlength=A).astype(float)
    truth = obs / np.maximum(n_units[:, None], 1.0)

    a_idx, u_idx = np.unravel_index(flat, (A, K))
    live = n_units[a_idx] > 0
    a_idx, u_idx = a_idx[live], u_idx[live]
    y = truth[a_idx, u_idx]
    ever = y > 0

    node_rate = np.exp(S.m)[u_idx]
    breadth = (obs.sum(1) / np.maximum(n_units, 1.0))
    breadth = breadth / max(breadth.mean(), M.EPS)
    indep = breadth[a_idx] * node_rate
    # peer mean within the group label, excluding the actor
    groups = np.array(corpus.groups)
    peer = np.zeros(len(a_idx))
    for g in set(groups):
        rows = np.flatnonzero(groups == g)
        sel = np.isin(a_idx, rows)
        if not sel.any():
            continue
        tot = truth[rows].sum(0)
        cnt = np.full(K, float(len(rows)))
        peer[sel] = ((tot[u_idx[sel]] - truth[a_idx[sel], u_idx[sel]])
                     / np.maximum(cnt[u_idx[sel]] - 1.0, 1.0))
    # the single-facet model, fitted on the same mask, as the fourth baseline
    auc_f1 = rho_f1 = np.nan
    if h.facets > baseline_facets:
        S1 = M.setup(corpus, cons, replace(h, facets=baseline_facets), cell_mask=mask)
        p1, _ = fit_quiet(S1, stages)
        pred1 = M.actor_rates(p1, S1, M.theta_of(p1, S1))
        auc_f1 = auc(pred1[a_idx, u_idx], ever)
        rho_f1 = spearman(pred1[a_idx, u_idx], y)

    return {
        "n_cells": int(len(a_idx)),
        "positive_rate": float(ever.mean()),
        "facets": h.facets,
        "auc_facets1": auc_f1, "rho_facets1": rho_f1,
        "auc_model": auc(pred[a_idx, u_idx], ever), "rho_model": spearman(pred[a_idx, u_idx], y),
        "auc_node": auc(node_rate, ever), "rho_node": spearman(node_rate, y),
        "auc_independence": auc(indep, ever), "rho_independence": spearman(indep, y),
        "auc_peer": auc(peer, ever), "rho_peer": spearman(peer, y),
    }


# --------------------------------------------------------------------------- #
# V6. the timescale curve
# --------------------------------------------------------------------------- #


def timescale_curve(corpus, cons, h, deltas=(("same", 0.0),), stages=STAGES) -> list[dict]:
    """W_k(delta) across the sweep, with held-out fit, the R diagnostic and coherence.

    Section 10 (V6): which edges live within expressions, which in episodes, which in
    dispositions; whether the theory links E_k concentrate at a timescale. Comparisons
    across delta are made on the excess scale rather than on raw likelihood, because the
    objective is a composite likelihood that double-counts more as delta grows -- so the
    edge mass and R columns are the ones to read across rows, not `held_out`.

    The facet count is whatever `h.facets` carries -- section 10's V6 asks for the
    large-delta end of this curve to be re-run across the F_k ladder, and `facet_contest`
    is what does that, so it is not folded in here.

    Ideology edge mass is broken out per type block, because two of the write-up's
    questions are block-shaped: section 9's "expressed vs. held" reads the
    delta-dependence of the B-R edges as a partial instrument for the expression-belief
    gap, and section 5's lateral-absorption warning is about B-B growing at the others'
    expense. Both are visible only if the blocks are tracked across the sweep.
    """
    rows = []
    for kernel, delta in deltas:
        hh = replace(h, kernel=kernel, delta=delta)
        S = M.setup(corpus, cons, hh)
        p, infos = fit_quiet(S, stages)
        em = G.edge_mass(p, S)
        R, exp = G.residual(p, S)
        blocks = {b: r["ideology"] for b, r in em["per_block"].items()}
        theory = np.array([R[k][cons.theory_edges[k]] for k in range(cons.M)], dtype=object)
        rows.append({
            "kernel": S.ps.label,
            "held_out": infos[-1]["held_out"],
            "shared": em["shared"],
            "ideology": em["ideology"],
            "ideology_share": em["ideology_share"],
            "median_R": float(np.nanmedian(R[exp > 1.0])) if (exp > 1.0).any() else np.nan,
            "median_R_theory": float(np.nanmedian(np.concatenate(
                [np.asarray(t, dtype=float) for t in theory if len(t)]))) if any(
                len(t) for t in theory) else np.nan,
            "per_block": blocks,
            "coherence": [r["C_k"] for r in G.coherence_ratio(p, S, n_perm=50)],
            "W": p.W.copy(),
            "W0": p.W0.copy(),
        })
    return rows


def facet_contest(corpus, cons, h, ladder=(1, 2, 3), kernel: str = "all",
                  delta: float = 0.0, stages=STAGES) -> list[dict]:
    """Section 8's facet--W contest: W_k at large delta across the F_k ladder.

    This is the successor to v3.1's eps--W contest, and the reason the model changed. In
    v3.1 free actor deviations and W were rival explanations of the same between-actor
    covariation, coupled through a continuous shrinkage scale. v3.2 removes the deviations,
    so covariation beyond the facet span must land in W -- but the contest re-enters
    through the facet count, because each added facet is a rank of covariation W no longer
    has to explain, and at F_k large enough the facets *are* free deviations again.

    What changed is that the dial is discrete and selected by held-out fit rather than
    tuned. The required report survives in this form: run the ladder and read the mass and
    the rank agreement. Stable W as facets enter means edges and specialisation carry
    different variance. W collapsing into new facets means "ideological structure" and
    "specialist composition of adherents" are the same variance under two names -- a
    limitation to state, not to tune away.
    """
    rows, base = [], None
    for F in ladder:
        hh = replace(h, kernel=kernel, delta=delta, facets=F)
        S = M.setup(corpus, cons, hh)
        p, infos = fit_quiet(S, stages)
        w = p.W.ravel()
        row = {"facets": F, "kernel": S.ps.label, "mass": float(p.W.sum()),
               "shared": float(p.W0.sum()), "held_out": infos[-1]["held_out"],
               "background": float(M.theta_of(p, S)[:, 0].mean())}
        if base is None:
            base = w
            row["spearman_vs_F1"], row["retained"] = 1.0, 1.0
        else:
            live = (base > 1e-8) | (w > 1e-8)
            row["spearman_vs_F1"] = spearman(base[live], w[live]) if live.any() else np.nan
            row["retained"] = float(w.sum()) / max(float(base.sum()), M.EPS)
        rows.append(row)
    return rows


def facet_curve(corpus, cons, h, ladder=(1, 2, 3, 4), frac: float = 0.2, seed: int = 0,
                stages=STAGES) -> list[dict]:
    """V7 across the facet ladder -- the selector for F_k (section 8, stage 2).

    Whole-actor holdout is the right criterion here for the reason it is the right
    criterion generally: a facet earns its place by predicting an actor the fit never saw,
    and a facet that only fits its own adherents better will not.
    """
    rows = []
    for F in ladder:
        r = actor_holdout(corpus, cons, replace(h, facets=F), frac=frac, seed=seed,
                          stages=stages)
        r["facets"] = F
        rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# V7. whole-actor holdout -- the primary predictive criterion
# --------------------------------------------------------------------------- #


def actor_holdout(corpus, cons, h, frac: float = 0.2, seed: int = 0, stages=STAGES) -> dict:
    """Hold out entire actors and score their units from metadata alone.

    Section 10 calls this the primary predictive criterion and, in v3.2, the selector of
    the facet counts F_k -- see `facet_curve`. The scoring rule is what makes it that: the held-out
    actor's (eps_a, mu_a) are marginalized at the prior -- set to their prior mean of zero
    rather than fitted -- and their alignment comes from beta' m_a with xi_a = 0. Nothing
    about the actor's own stream enters. Baselines: the background rates alone, and the
    background plus the mean fitted profile over training actors.
    """
    rng = np.random.default_rng(seed)
    n_out = max(1, int(round(frac * corpus.A)))
    held_actors = rng.choice(corpus.A, size=n_out, replace=False)
    is_held = np.isin(corpus.actor, held_actors)

    # One setup, whose train is every unit by a non-held actor and whose test is exactly
    # the held-out actors' units. Splitting it in two would fit m_u and the exposure on a
    # different sample than the one being scored.
    S = M.setup(corpus, cons, h, train=~is_held)
    p, _ = fit_quiet(S, stages)

    # Score with the actor layer marginalized at its prior mean and the ideology margin
    # from metadata alone: nothing about the held-out actor's own stream enters. Both
    # actor parameters go to zero -- the facet logits gamma and the breadth mu -- which
    # leaves the actor at the prior's facet split within whatever ideology beta predicts.
    cold = replace(p, gamma=np.zeros_like(p.gamma), mu=np.zeros_like(p.mu))
    theta = M.theta_of(cold, S)
    model = M.rate_loss(cold, S, theta, S.test)
    background = M.rate_loss(replace(cold, eta=np.zeros_like(p.eta)), S, theta, S.test)
    pooled = M.rate_loss(replace(cold, eta=np.tile(p.eta.mean(0), (S.F, 1))), S,
                         theta, S.test)
    n = float(S.test.n.sum())
    return {
        "n_actors": int(n_out),
        "n_units": int(is_held.sum()),
        "loss_model": model / max(n, 1.0),
        "loss_background": background / max(n, 1.0),
        "loss_pooled_profile": pooled / max(n, 1.0),
        "gain_vs_background": (background - model) / max(n, 1.0),
    }


# --------------------------------------------------------------------------- #
# V8. coherence and the culture/ideology partition
# --------------------------------------------------------------------------- #


def coherence_and_partition(p: M.Params, S: M.Setup, n_perm: int = 200) -> dict:
    """The three readings of V8, on an already-fitted model.

    (i) coherence ratios against the permutation null, gate off; (ii) the edge-mass split
    and the typology's concept counts; (iii) the van Dijk check -- predictive gain from
    alignment, per concept. A concept the typology calls cultural should show none, and
    one that does is not culture whatever the typology said.
    """
    typ = G.contestation(p, S)
    from collections import Counter
    return {
        "coherence": G.coherence_ratio(p, S, n_perm=n_perm),
        "edge_mass": G.edge_mass(p, S),
        "typology": dict(Counter(r["type"] for r in typ)),
        "van_dijk": van_dijk_check(p, S, typ),
        "gate": S.h.gate,
    }


def van_dijk_check(p: M.Params, S: M.Setup, typ: list[dict] | None = None) -> list[dict]:
    """Per-concept predictive gain from alignment: the model-external check.

    For each concept, re-score the held-out cells with that concept's ideology term
    switched off (eta_{k,u} -> the alignment-weighted mean over ideologies, which is what
    "alignment carries no information here" means) and read the change in loss. Shared
    meaning should show no gain; a concept the typology called cultural that does show
    one is not culture.
    """
    typ = G.contestation(p, S) if typ is None else typ
    theta = M.theta_of(p, S)
    base = M.rate_loss(p, S, theta, S.test)
    kind = {r["concept"]: r["type"] for r in typ}
    out = []
    for c in range(S.corpus.C):
        nodes = np.flatnonzero(S.cons.concept_of == c)
        eta = p.eta.copy()
        eta[:, nodes] = eta[:, nodes].mean(0, keepdims=True)
        loss = M.rate_loss(replace(p, eta=eta), S, theta, S.test)
        out.append({
            "concept": c,
            "nodes": [S.cons.node_ids[u] for u in nodes],
            "type": kind.get(c, "?"),
            "gain": float(loss - base),  # > 0: alignment was helping
        })
    return sorted(out, key=lambda r: -r["gain"])


# --------------------------------------------------------------------------- #
# the zero-ideology null (section 7's "choosing M")
# --------------------------------------------------------------------------- #


def zero_ideology_null(corpus, cons, h, n_draws: int = 5, seed: int = 0, stages=STAGES
                       ) -> dict:
    """Permute actor metadata *and* shuffle alignments; refit with breadth mu held fixed.

    Section 7 asks for three things and all three are load-bearing.

    Shuffling the alignments has to mean pinning the ideology margin to a permutation of
    its real-fit value, not merely permuting the metadata rows and refitting. Membership
    is softmax(beta'm_a + gamma_a) with a free per-actor gamma, so permuting only m_a
    leaves the association between actor and ideology entirely recoverable through gamma
    -- the "null" then fits about as well as the real model and the test says nothing.
    With the margin pinned to a permutation, the actor-ideology association is genuinely
    destroyed and the facet profiles have to explain the corpus while attached to the
    wrong actors.

    Holding mu_a at its real-fit value is the same argument one layer down: a deviation
    refitted freely re-absorbs the structure the permutation removed.

    The null is *more* flexible than the real fit in one respect -- the real fit's C4 pins
    constrain four actors and the null's pinned-but-shuffled theta constrains all of them
    to a wrong assignment -- so a corpus that does not beat this null decisively contains
    no coherent ideology in the framework's sense.
    """
    S = M.setup(corpus, cons, h)
    real, infos = fit_quiet(S, stages)
    real_vartheta = M.vartheta_of(M.theta_of(real, S), S)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        perm = rng.permutation(corpus.A)
        shuffled = replace(corpus, meta=corpus.meta[perm])
        Sn = M.setup(shuffled, cons, h)
        # Pin the *ideology margin* to a permutation of its fitted value and leave the
        # facets free. Pinning the full facet membership would also freeze the
        # specialisation, which is not what the null is destroying: the claim under test
        # is that actor identity predicts ideology, not that it predicts a facet.
        Sn = replace(Sn, fixed=np.ones(corpus.A, dtype=bool),
                     fixed_vartheta=real_vartheta[perm])
        p0 = replace(M.initialize(Sn), mu=real.mu.copy())
        _, ninfos = M.fit_stages(Sn, stages, p=p0, verbose=False)
        draws.append(ninfos[-1]["held_out"])
    draws = np.array(draws)
    return {"real": infos[-1]["held_out"], "null": draws.tolist(),
            "null_mean": float(draws.mean()),
            "beats": int((draws > infos[-1]["held_out"]).sum()), "n_draws": n_draws}


# --------------------------------------------------------------------------- #
# is a named ideology worth its parameters? the ablation
# --------------------------------------------------------------------------- #


def drop_ideology(cons: C.Constraints, k: int) -> C.Constraints:
    """Remove ideology k from the constraint set, renormalising the C4 pins.

    An actor pinned partly to k keeps the rest of their alignment, renormalised. An actor
    pinned *entirely* to k has nothing left to be pinned to and becomes free rather than
    being given an arbitrary assignment -- which is the honest treatment, since the whole
    question the ablation asks is where that actor goes when k is unavailable.
    """
    keep = [j for j in range(cons.M) if j != k]
    val = cons.theta_val[:, keep] if cons.theta_val.size else cons.theta_val
    fix = cons.theta_fix.copy()
    if val.size:
        tot = np.nansum(val, axis=1)
        orphan = fix & (tot <= 1e-9)
        fix = fix & ~orphan
        with np.errstate(invalid="ignore", divide="ignore"):
            val = np.where(tot[:, None] > 0, val / np.maximum(tot[:, None], M.EPS), val)
    return replace(cons, names=[cons.names[j] for j in keep],
                   anchor_pos=cons.anchor_pos[keep], anchor_neg=cons.anchor_neg[keep],
                   theory_edges=cons.theory_edges[keep], theta_val=val, theta_fix=fix)


def ablation(corpus, cons, h, stages=STAGES) -> list[dict]:
    """Drop each ideology in turn, refit, and read what its absence costs.

    Section 7 chooses M by held-out fit plus the split diagnostic, and `sweep-m` answers
    the upward half of that question -- whether the corpus contains more than the
    anchored ideologies. This is the downward half, which nothing else asks: whether each
    *named* ideology is carrying variance the others cannot.

    Three columns matter, and they answer different questions.

      held-out    the predictive cost of the ablation. A near-zero cost says the corpus
                  does not need this component.
      reassigned  where the dropped ideology's adherents go. If they distribute evenly,
                  the component was a genuine third position; if they all land on one
                  survivor, it was a sub-tendency of that one.
      edge mass   whether the association the component carried survives elsewhere, in
                  the remaining W_k or in the shared layer.

    A component can be predictively worth keeping and still not be an ideology in the
    framework's sense -- it may be a topic every camp discusses, which lands in the rate
    layer as real variance while belonging in W^(0) and the background. The ablation
    measures the variance, not the interpretation.
    """
    S = M.setup(corpus, cons, h)
    full, infos = fit_quiet(S, stages)
    base = infos[-1]["held_out"]
    theta_full = M.theta_of(full, S)
    rows = [{"dropped": "(none)", "M": cons.M, "held_out": base, "cost": 0.0,
             "reassigned": {}, "shared": float(full.W0.sum()),
             "ideology": float(full.W.sum())}]

    for k in range(cons.M):
        nm = cons.name(k)
        c2 = drop_ideology(cons, k)
        S2 = M.setup(corpus, c2, h)
        p2, i2 = fit_quiet(S2, stages)
        th2 = M.theta_of(p2, S2)
        # where the dropped ideology's adherents went, weighted by their old membership
        w = theta_full[:, k]
        survivors = [cons.name(j) for j in range(cons.M) if j != k]
        share = (w[:, None] * th2).sum(0) / max(w.sum(), M.EPS)
        rows.append({
            "dropped": nm, "M": c2.M, "held_out": i2[-1]["held_out"],
            "cost": i2[-1]["held_out"] - base,
            "reassigned": dict(zip(survivors, share.round(3).tolist())),
            "shared": float(p2.W0.sum()), "ideology": float(p2.W.sum()),
        })
    return rows


def grpname_accuracy(p: M.Params, S: M.Setup, seed: int = 0) -> dict:
    """Leave-one-actor-out accuracy predicting an actor's community from the fit.

    The adjusted Rand index in `group_recovery` answers "did the components line up with
    the communities", which conflates recovery with how many components there are: a fit
    with three components and six groups is penalised for a mismatch that is not an error.
    This answers the question the experiment actually asks -- given what the model inferred
    about an actor, can we say which community they belong to -- and it is comparable across
    configurations with different M, which is the whole point of the ladder.

    The features are the fitted actor summary only: the ideology margin, the background
    share, and breadth. No metadata, and the fit itself must have been made with identity
    withheld, or this measures the design matrix rather than the model.

    Multinomial logistic regression by gradient descent, leave-one-actor-out. Written out
    rather than imported because the project has no sklearn and one small solver is cheaper
    than the dependency.
    """
    vth = M.vartheta_of(M.theta_of(p, S), S)
    X = np.column_stack([vth, p.mu[:, None], np.ones(len(vth))])
    labels = np.array(S.corpus.groups)
    classes = sorted(set(labels))
    y = np.array([classes.index(g) for g in labels])
    n, d, C_ = len(y), X.shape[1], len(classes)
    if C_ < 2 or n < 3:
        return {"accuracy": float("nan"), "baseline": float("nan"), "n": n,
                "classes": classes}

    def train(mask, iters=300, lr=0.5, l2=1e-3):
        Wt = np.zeros((d, C_))
        Xm, ym = X[mask], y[mask]
        for _ in range(iters):
            z = Xm @ Wt
            z -= z.max(1, keepdims=True)
            e = np.exp(z)
            pr = e / e.sum(1, keepdims=True)
            oh = np.eye(C_)[ym]
            Wt -= lr * (Xm.T @ (pr - oh) / len(ym) + l2 * Wt)
        return Wt

    ok = 0
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        Wt = train(mask)
        ok += int((X[i] @ Wt).argmax() == y[i])
    # the honest floor: always guess the most common community
    base = np.bincount(y).max() / n
    return {"accuracy": ok / n, "baseline": float(base), "n": n, "classes": classes,
            "features": ["vartheta"] * vth.shape[1] + ["mu", "intercept"]}
