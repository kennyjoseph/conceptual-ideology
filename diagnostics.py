"""Residuals and statistics, per sections 5 and 7 of the write-up.

Every hard constraint on ideologies has a soft residual on data, and the point of this
module is to report those residuals rather than let the fit absorb them:

  R on allowed pairs     R > 1 is what W >= 0 captures; structured R < 1 among pairs
                         both of whose profile rates are elevated is a negative edge the
                         model refused to fit -- the signature of two sub-tendencies
                         sharing a component, and the signal to split k.
  R on skipping pairs    structured R > 1 off P is evidence against the ladder (C5) or,
                         far more often, against a concept's type assignment.
  coherence ratio C_k    C6's falsifiable form: does an ideology's structure stay inside
                         its own support? Reported against a permutation null, fitted
                         with the gate off.
  edge mass              W^(0) against sum_k W_k -- how much of the association in this
                         discourse is culture and how much is contested -- and mass per
                         type block, which is section 5's lateral-absorption watch item.
  contestation typology  per concept: cultural, consensually ideological, contested by
                         presence/exclusion, contested relationally.
  V_S, V^actor_S         C3 binds ideologies but not units or actors, and the gap between
                         the layers is data, not error.
  node roles             sharedness, distinctiveness, structural centrality -- the three
                         numbers that separate a shared hub from an idiosyncratic marker,
                         and the disagreement with BENDING.

All of it is a function of delta. An edge present at small delta and absent at large (or
the reverse) is a finding about where structure lives, not an inconsistency, so every
statistic carries the kernel label it was computed at.
"""

import numpy as np

import constraints as C
import kernels as Kr
import model as M


def dense(w: np.ndarray, pairs: np.ndarray, K: int) -> np.ndarray:
    """[..., K, K] symmetric adjacency from a [..., P] edge vector.

    K is 112 here, so the dense form costs 12,544 floats per layer and makes every
    neighbourhood question -- degree, a node's row, the divergence between two ideologies
    at one node -- a slice instead of a scan over the pair list.
    """
    out = np.zeros(w.shape[:-1] + (K, K))
    out[..., pairs[:, 0], pairs[:, 1]] = w
    out[..., pairs[:, 1], pairs[:, 0]] = w
    return out


# --------------------------------------------------------------------------- #
# the R diagnostic
# --------------------------------------------------------------------------- #


def residual(
    p: M.Params, S: M.Setup, pairs: np.ndarray | None = None,
    ps: Kr.PairStats | None = None, theta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Section 7's R^(delta)_{k,uu'}, and its denominator.

        R = sum_a theta_ak A^(delta)_{a,uu'} / sum_a theta_ak N^(delta)_a eta_au eta_au'

    The eta_a terms in the denominator are what stop actor specialization from
    masquerading as fragmentation, so the denominator is returned alongside: an R
    computed on a near-zero denominator is noise, not a finding.

    `pairs` defaults to the allowed set; pass `skipping_pairs` (with the matching `ps`)
    for the C5 test.
    """
    theta = M.theta_of(p, S) if theta is None else theta
    pairs = S.cons.pairs if pairs is None else pairs
    ps = S.ps if ps is None else ps
    r = M.actor_rates(p, S, theta)
    u, v = pairs[:, 0], pairs[:, 1]
    base = S.ps.N[:, None] * r[:, u] * r[:, v]
    obs = theta.T @ ps.A  # [M, P]
    exp = theta.T @ base  # [M, P]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(exp > 0, obs / np.maximum(exp, M.EPS), np.nan)
    return R, exp


def split_signal(
    p: M.Params, S: M.Setup, eta_min: float = 0.0, exp_min: float = 1.0
) -> list[dict]:
    """Structured R < 1 among pairs an ideology's profile elevates: the split signal.

    W >= 0 cannot represent under-occurrence, so a component that systematically
    under-produces pairs of nodes it claims to hold is one that a split would supply
    with the opposite-level nodes it needs.
    """
    R, exp = residual(p, S)
    ieta = M.ideology_profile(p, S)
    u, v = S.cons.pairs[:, 0], S.cons.pairs[:, 1]
    out = []
    for k in range(S.M):
        live = (ieta[k, u] > eta_min) & (ieta[k, v] > eta_min) & (exp[k] > exp_min)
        if not live.any():
            out.append({"k": k, "n_pairs": 0, "frac_under": np.nan, "median_R": np.nan,
                        "worst": []})
            continue
        r = R[k][live]
        idx = np.flatnonzero(live)[np.argsort(r)]
        out.append({
            "k": k,
            "n_pairs": int(live.sum()),
            "frac_under": float((r < 1).mean()),
            "median_R": float(np.median(r)),
            "worst": [(S.cons.node_ids[u[i]], S.cons.node_ids[v[i]], float(R[k, i]))
                      for i in idx[:5]],
        })
    return out


def ladder_violations(p: M.Params, S: M.Setup, exp_min: float = 1.0, top: int = 15
                      ) -> list[dict]:
    """Structured R > 1 on rung-skipping pairs: evidence against C5 or against T(c).

    Reported per pair rather than absorbed. A concept mistyped one rung off shows up
    here with its true neighbours, at whatever delta its links live, so read a hit as a
    prompt to revisit that node's layer before revisiting the ladder itself.
    """
    off = C.skipping_pairs(S.cons.layer, S.cons.concept_of)
    if not len(off):
        return []
    ps = Kr.pair_counts(
        S.corpus.x, S.corpus.actor, S.corpus.stamp, off, S.h.kernel, S.h.delta,
        n_actors=S.corpus.A, mask=S.corpus.train,
    )
    R, exp = residual(p, S, off, ps)
    out = []
    for k in range(S.M):
        live = np.flatnonzero(exp[k] > exp_min)
        for i in live[np.argsort(-R[k][live])][:top]:
            if not R[k, i] > 1:
                break
            a, b = off[i]
            out.append({
                "k": k,
                "pair": (S.cons.node_ids[a], S.cons.node_ids[b]),
                "layers": (C.LAYER_CODE[S.cons.layer[a]], C.LAYER_CODE[S.cons.layer[b]]),
                "R": float(R[k, i]),
                "expected": float(exp[k, i]),
            })
    return out


# --------------------------------------------------------------------------- #
# C6: structural coherence
# --------------------------------------------------------------------------- #


def coherence_ratio(
    p: M.Params, S: M.Setup, eta_min: float = 0.0, n_perm: int = 200, seed: int = 0
) -> list[dict]:
    """C6's statistic, against a permutation null. Section 3 asks for it in this order.

        C_k = sum_{u,u' in S_k} W_k / sum_{u in S_k, u' not in S_k} W_k,   S_k = {u : eta_ku > 0}

    A component whose structure leaks across its own boundary is either mis-supported or
    not an ideology by this framework's definition -- a reportable failure, not a fitting
    error, which is why the statistic is computed on a fit with the C6 gate off and the
    gate is only ever switched on afterwards.

    The null draws random supports of matched size *and* matched structure: nodes are
    sampled stratified by type (so a support that is all beliefs is compared against
    all-belief supports) and within type by weighted-degree decile in W_k (so a support
    built out of the network's hubs is compared against other hub sets). Matching only on
    size would make almost any support look coherent, since W's mass is concentrated on a
    few high-degree nodes.
    """
    rng = np.random.default_rng(seed)
    ieta = M.ideology_profile(p, S)
    u, v = S.cons.pairs[:, 0], S.cons.pairs[:, 1]
    out = []
    Wd = dense(p.W, S.cons.pairs, S.corpus.K)
    for k in range(S.M):
        w = p.W[k]
        deg = Wd[k].sum(1)
        supp = ieta[k] > eta_min
        stat = _ratio(w, supp, u, v)
        # strata: (type, degree quartile within type)
        strata = np.empty(S.corpus.K, dtype=object)
        for ly in np.unique(S.cons.layer):
            idx = np.flatnonzero(S.cons.layer == ly)
            if len(idx) < 4:
                q = np.zeros(len(idx), dtype=int)
            else:
                edges = np.quantile(deg[idx], [0.25, 0.5, 0.75])
                q = np.searchsorted(edges, deg[idx], side="right")
            for i, qq in zip(idx, q):
                strata[i] = (int(ly), int(qq))
        buckets: dict = {}
        for i, s in enumerate(strata):
            buckets.setdefault(s, []).append(i)
        want: dict = {}
        for i in np.flatnonzero(supp):
            want[strata[i]] = want.get(strata[i], 0) + 1
        null = []
        for _ in range(n_perm):
            draw = np.zeros(S.corpus.K, dtype=bool)
            for s, n in want.items():
                pool = buckets[s]
                draw[rng.choice(pool, size=min(n, len(pool)), replace=False)] = True
            null.append(_ratio(w, draw, u, v))
        null = np.array([x for x in null if np.isfinite(x)])
        out.append({
            "k": k,
            "support": int(supp.sum()),
            "C_k": stat,
            "null_mean": float(null.mean()) if len(null) else np.nan,
            "null_sd": float(null.std()) if len(null) else np.nan,
            "p": float((null >= stat).mean()) if len(null) and np.isfinite(stat) else np.nan,
        })
    return out


def _ratio(w: np.ndarray, supp: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    inside = supp[u] & supp[v]
    boundary = supp[u] ^ supp[v]
    num, den = float(w[inside].sum()), float(w[boundary].sum())
    return num / den if den > 0 else (np.inf if num > 0 else np.nan)


# --------------------------------------------------------------------------- #
# where the association mass sits
# --------------------------------------------------------------------------- #


def edge_mass(p: M.Params, S: M.Setup) -> dict:
    """The culture/ideology split, and mass per type block.

    V8(ii) asks for the share of total edge mass in W^(0) versus {W_k}: how much of the
    discourse is shared meaning that allows interaction across groups, and how much
    covaries with alignment. Section 5's lateral-absorption warning asks for the same
    mass broken out by type block, because the intra-rung blocks -- B-B especially -- can
    absorb co-occurrence that previously had to load on cross-rung edges, and the fitted
    cross-rung structure will thin without anything announcing it.
    """
    block, names = S.cons.block_of(), C.block_names()
    shared, ideo = float(p.W0.sum()), float(p.W.sum())
    per_block = {}
    for i, n in enumerate(names):
        sel = block == i
        per_block[n] = {
            "pairs": int(sel.sum()),
            "shared": float(p.W0[sel].sum()),
            "ideology": float(p.W[:, sel].sum()),
            "nonzero_shared": int((p.W0[sel] > 1e-8).sum()),
            "nonzero_ideology": int((p.W[:, sel] > 1e-8).sum()),
        }
    return {
        "shared": shared,
        "ideology": ideo,
        "ideology_share": ideo / max(shared + ideo, M.EPS),
        "per_ideology": p.W.sum(1).tolist(),
        "per_block": per_block,
        "kernel": S.ps.label,
    }


# --------------------------------------------------------------------------- #
# the contestation typology
# --------------------------------------------------------------------------- #


def contestation(
    p: M.Params, S: M.Setup, eta_hi: float = 0.25, div_hi: float = 0.5
) -> list[dict]:
    """Section 7's typology, per concept, from the fitted facets.

      cultural                  m high, eta_{k,.} ~ 0 for all k; association only in W^(0)
      consensually ideological  one level elevated near-identically across ideologies
      contested by exclusion    ideologies split on which *level* of the concept they hold
      contested relationally    a shared elevated node with diverging W_k rows
      contested by presence     some ideologies elevate the concept and others do not

    The last is not in the write-up's list because it is the single-level degeneration of
    exclusion contrast: with the node file's `concept` column blank every concept has one
    level, so "which level is elevated" collapses to "whether it is elevated", and the
    exclusion category is unreachable until levels are filled in. It is reported under its
    own name rather than folded into exclusion, so the distinction survives the day the
    column is populated.
    """
    Wd = dense(p.W, S.cons.pairs, S.corpus.K)
    W0d = dense(p.W0, S.cons.pairs, S.corpus.K)
    ieta = M.ideology_profile(p, S)
    out = []
    for c in range(S.corpus.C):
        nodes = np.flatnonzero(S.cons.concept_of == c)
        hot = ieta[:, nodes] > eta_hi  # [M, levels]
        holders = np.flatnonzero(hot.any(1))
        row = {
            "concept": c,
            "nodes": [S.cons.node_ids[i] for i in nodes],
            "m": float(S.m[nodes].max()),
            "max_eta": float(ieta[:, nodes].max()),
            "dispersion": float(ieta[:, nodes].max(1).std()),
            "shared_edges": float(W0d[nodes].sum()),
            "divergence": np.nan,
        }
        if len(holders) == 0:
            row["type"] = "cultural"
        elif len(holders) == 1:
            row["type"] = "contested by presence" if S.M > 1 else "ideological"
        elif len(nodes) > 1 and len({int(hot[k].argmax()) for k in holders}) > 1:
            row["type"] = "contested by exclusion"
        else:
            div = _row_divergence(Wd, holders, nodes)
            row["divergence"] = div
            row["type"] = ("contested relationally" if div > div_hi
                           else "consensually ideological")
        out.append(row)
    return out


def _row_divergence(Wd: np.ndarray, holders: np.ndarray, nodes: np.ndarray) -> float:
    """Mean pairwise cosine distance between ideologies' edge rows at a shared concept.

    Relational contrast is exactly this: the same node, different neighbourhoods. One
    pooled graph cannot represent it (section 6), so a high value here is the empirical
    form of the argument for one network per ideology.
    """
    rows = [Wd[k][nodes].sum(0) for k in holders]
    d, n = 0.0, 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < M.EPS or nb < M.EPS:
                continue
            d += 1.0 - float(a @ b) / (na * nb)
            n += 1
    return d / n if n else np.nan


# --------------------------------------------------------------------------- #
# incoherence, roles, identifiability
# --------------------------------------------------------------------------- #


def incoherence(p: M.Params, S: M.Setup) -> list[dict]:
    """V_S and V^actor_S per theoretical set, decomposed by alignment (section 7).

    C3 binds ideologies, not units and not actors: a unit may express two incompatible
    nodes in the same breath and an actor's deviations may elevate both. What this
    measures is how often that happens and where it concentrates. Violations concentrated
    in an actor measure that actor's inconsistency; concentrated in a component, they mean
    the component is mis-fit or X is wrong.
    """
    theta = M.theta_of(p, S)
    r = M.actor_rates(p, S, theta)
    ieta = M.ideology_profile(p, S)
    out = []
    for Sx in S.cons.incompat:
        viol = S.corpus.x[:, Sx].sum(1) > 1.0
        actor_viol = r[:, Sx].sum(1) > S.h.pi_bar
        by_actor = np.zeros(S.corpus.A)
        np.add.at(by_actor, S.corpus.actor, viol.astype(float))
        n_actor = np.bincount(S.corpus.actor, minlength=S.corpus.A).astype(float)
        share = by_actor / np.maximum(n_actor, 1.0)
        out.append({
            "set": [S.cons.node_ids[i] for i in Sx],
            "V_S": float(viol.mean()),
            "n_units": int(viol.sum()),
            "V_actor": float(actor_viol.mean()),
            "by_alignment": (theta[S.corpus.actor[viol]].sum(0)
                             / max(theta.sum(0).sum(), M.EPS)).tolist() if viol.any()
                            else [0.0] * S.M,
            "worst_actor": S.corpus.actors[int(np.argmax(share))],
            "max_profile": float(max(
                np.exp(S.m[Sx] + ieta[k, Sx]).sum() for k in range(S.M))),
        })
    return sorted(out, key=lambda r_: -r_["V_S"])


def node_roles(p: M.Params, S: M.Setup) -> dict:
    """Sharedness, distinctiveness and structural centrality (section 9.3).

    Relational contrast is high on sharedness and centrality and low on distinctiveness:
    a node shared with a rival that is nonetheless the hub of this ideology's structure.
    BENDING's idiosyncrasy criterion would reject exactly those nodes, which is the
    disagreement section 9.3 is about -- so centrality is computed *within type*, as that
    section specifies, rather than over the whole neighbourhood.
    """
    eta = M.ideology_profile(p, S)
    other = np.array([np.delete(eta, k, axis=0).max(0) for k in range(S.M)]) if S.M > 1 \
        else np.zeros_like(eta)
    u, v = S.cons.pairs[:, 0], S.cons.pairs[:, 1]
    same_type = S.cons.layer[u] == S.cons.layer[v]
    centrality = dense(np.where(same_type[None, :], p.W, 0.0),
                       S.cons.pairs, S.corpus.K).sum(2)
    return {"sharedness": eta, "distinctiveness": eta - other, "centrality": centrality,
            "expressed": S.cells.X.sum(0) > 0}


def alignment_summary(p: M.Params, S: M.Setup) -> dict:
    """Effective actors, entropy and the background share (sections 7 and 9).

    Three readings, and the level each is taken at is the point.

    Effective actors is reported **per facet**, because a facet is what can fail to be
    separated from an idiolect: section 9.1 says to read a facet dominated by one actor as
    that actor's version of the ideology. That is a reading, not an error -- a single-actor
    facet is a legitimate outcome and nothing here suppresses or flags it.

    Entropy is taken over the **ideology margin only**. Facet spread within an ideology is
    specialisation, not mixing (section 9.2), so folding it in would make every specialist
    look like a straddler. What the instrument is for is finding actors between ideologies.

    The background share theta[:, 0] is the fraction of an actor's speech the model calls
    unideological, and it is section 9.4's stand-in for NB's personal temperature: how far
    the actor sits from every attractor.
    """
    th = M.theta_of(p, S)
    vth = M.vartheta_of(th, S)
    col = th / np.maximum(th.sum(0, keepdims=True), M.EPS)
    eff_facet = 1.0 / np.maximum((col ** 2).sum(0), M.EPS)          # [1+F]
    vcol = vth / np.maximum(vth.sum(0, keepdims=True), M.EPS)
    eff_ideology = 1.0 / np.maximum((vcol ** 2).sum(0), M.EPS)      # [M+1]
    ideo = vth[:, 1:]
    share = ideo / np.maximum(ideo.sum(1, keepdims=True), M.EPS)
    ent = -(share * np.log(np.maximum(share, M.EPS))).sum(1)
    return {"theta": th, "vartheta": vth, "background": th[:, 0],
            "effective_actors": eff_ideology[1:], "effective_per_facet": eff_facet[1:],
            "entropy": ent, "mass": vth[:, 1:].sum(0), "facet_mass": th[:, 1:].sum(0)}


def facet_release(p: M.Params, S: M.Setup, exp_min: float = 1.0) -> list[dict]:
    """Section 7's test of the facet/ideology boundary: were the facets ideologies?

    Facets share their ideology's network by construction. Release the tie -- fit one W
    per facet instead of one per ideology -- and if the freed networks diverge on
    well-estimated pairs, the facets were ideologies all along and the tied fit was
    undercounting M.

    Implemented as a re-solve of the W block alone, at the fitted rates and memberships,
    with the design widened from the ideology margins to the full facet memberships. That
    is exactly the same convex problem the W-step already solves, so nothing new has to be
    trusted; what changes is one design matrix.
    """
    th = M.theta_of(p, S)
    Z = np.column_stack([np.ones(S.corpus.A), th[:, 1:]])     # facets, not margins
    E = M.exposure(p, S, th)
    Aobs = S.ps.A * S.pmask
    lam = np.vstack([np.full((1, S.P), S.h.lam_sp), np.full((S.F, S.P), S.h.lam_sp)])
    lo = np.zeros((1 + S.F, S.P))
    b = np.zeros((1 + S.F, S.P))
    for _ in range(40):                                        # the w_step sweep, inlined
        for j in range(1 + S.F):
            Om = np.clip(Z @ b, -M.Z_MAX, M.Z_MAX)
            pred = E * np.exp(Om)
            grad = S.h.pair_weight * (Z[:, j] @ (pred - Aobs)) + lam[j]
            hess = S.h.pair_weight * ((Z[:, j] ** 2) @ pred)
            step = np.where(hess > M.EPS, grad / np.maximum(hess, M.EPS), 0.0)
            b[j] = np.clip(b[j] - 0.5 * step, lo[j], np.inf)
    free = b[1:]                                               # [F, P]
    live = (E.sum(0) > exp_min)
    out = []
    for k in range(1, S.cons.M + 1):
        sel = np.flatnonzero(S.facet_of == k)
        nm = S.cons.name(k - 1)
        if len(sel) < 2:
            out.append({"k": nm, "n_facets": len(sel), "divergence": np.nan,
                        "n_pairs": int(live.sum())})
            continue
        rows = free[sel][:, live]
        d_, n = 0.0, 0
        for i in range(len(rows)):
            for j2 in range(i + 1, len(rows)):
                a_, b_ = rows[i], rows[j2]
                na, nb = np.linalg.norm(a_), np.linalg.norm(b_)
                if na < M.EPS or nb < M.EPS:
                    continue
                d_ += 1.0 - float(a_ @ b_) / (na * nb)
                n += 1
        out.append({"k": nm, "n_facets": len(sel), "n_pairs": int(live.sum()),
                    "divergence": d_ / n if n else np.nan,
                    "tied_mass": float(p.W[k - 1][live].sum()),
                    "free_mass": float(rows.sum())})
    return out


# --------------------------------------------------------------------------- #
# is an ideology expressed, or only discussed? the grounding test
# --------------------------------------------------------------------------- #


def cross_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """All [P, 2] pairs with one endpoint in `a` and the other in `b`, u < u'."""
    out = {(min(int(x), int(y)), max(int(x), int(y))) for x in a for y in b if x != y}
    return np.array(sorted(out), dtype=np.int32) if out else np.zeros((0, 2), np.int32)


def actor_pair_totals(
    p: M.Params, S: M.Setup, pairs: np.ndarray, theta: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Observed and expected pair counts per *actor* rather than per ideology: [A, P'].

    Returned unreduced so the caller can pool them -- section 7's R is a ratio of sums,
    and over a set of rare pairs that is the only stable form.

    Section 7's R weights actors by their membership, which answers "does this ideology
    show this excess". The unweighted per-actor form answers "does any actor show it",
    which is the question when the ideology's own membership is what is in doubt.

    `pairs` is arbitrary -- the statistic is defined on any pair, and the sets this
    module needs are mostly rung-skipping (V-B, V-R, A-R), so the counts are built for
    the pair set given rather than read off S.ps.
    """
    theta = M.theta_of(p, S) if theta is None else theta
    if not len(pairs):
        return np.zeros((S.corpus.A, 0)), np.zeros((S.corpus.A, 0))
    ps = Kr.pair_counts(S.corpus.x, S.corpus.actor, S.corpus.stamp, pairs,
                        S.h.kernel, S.h.delta, n_actors=S.corpus.A, mask=S.corpus.train)
    r = M.actor_rates(p, S, theta)
    u, v = pairs[:, 0], pairs[:, 1]
    exp = S.ps.N[:, None] * r[:, u] * r[:, v]
    return ps.A, exp


def ideology_node_sets(cons: C.Constraints, k: int) -> dict:
    """The claim / grounding / rival split of an ideology's anchored nodes.

    An ideology's *claims* are its beliefs and rationalizations: the propositions someone
    who merely reports on the ideology would quote, describe or attack. Its *grounding*
    is its values and attitudes: the commitments the ladder says those claims rest on.
    *Rival* is everything anchored to another ideology -- the vocabulary a critic embeds
    the claims in.

    Only nodes anchored to k and to nothing else count as k's. A node two ideologies both
    claim cannot distinguish them.
    """
    own = cons.anchor_pos[k] & ~np.delete(cons.anchor_pos, k, axis=0).any(0)
    rival = np.delete(cons.anchor_pos, k, axis=0).any(0) & ~cons.anchor_pos[k]
    rivals = {}
    for j in range(cons.M):
        if j == k:
            continue
        nm = cons.name(j)
        rivals[nm] = np.flatnonzero(cons.anchor_pos[j] & ~cons.anchor_pos[k])
    return {"claim": np.flatnonzero(own & (cons.layer >= 2)),
            "ground": np.flatnonzero(own & (cons.layer <= 1)),
            "rival": np.flatnonzero(rival), "rivals": rivals}


def grounding_test(
    p: M.Params, S: M.Setup, k: int, exp_min: float = 0.25, min_units: int = 200
) -> dict:
    """Is ideology k held by anyone, or only talked about? Four statistics, and a warning.

    The rate channel cannot answer this. Someone who endorses "white people are
    biologically superior" and someone who reports that a bomber said so produce the same
    marginal on that node, so the profiles, memberships and held-out fit all measure
    *topic*. The intended instrument was the pair channel and the ladder: an adherent
    should ground k's claims in k's own values and attitudes, a critic should ground them
    in the critic's vocabulary. Identical margins, different rows of W -- section 6's
    relational-contrast argument used to identify stance rather than to describe it.

    THAT INSTRUMENT DOES NOT WORK, and the failure is the finding. A description of an
    ideology reproduces the ideology's internal structure: to report that someone is a
    white supremacist is to say, in one breath, both what they believe and what they
    value. On this corpus 55.9% of units carrying a WS claim also carry a WS grounding
    node, and the actor with the highest such rate is a critic (67.7%) while a
    sympathetic account sits lower (43.4%). `R_own` and `co_ground` therefore rank
    describers *above* adherents, and must not be read as stance. They remain worth
    computing as what they honestly are -- whether k's content is used coherently by
    anyone at all -- but the verdict field says so on every call.

    What retains some discriminating power is the *other* direction: whether the claims
    arrive embedded in a rival's vocabulary (`R_frame`, `co_frame`). A critic argues
    against WS in the language of structural racism; an adherent has no reason to. This
    is weaker than it looks, because it cannot see the third stance the codebook has no
    words for -- accusing an opponent of racism uses neither vocabulary -- so a low
    `co_frame` means "not arguing against it in these terms", not "endorses it".

    `unframed` is the closest available adherent signature: the share of an actor's
    claim-carrying units that reach for no other ideology's vocabulary. A describer
    embeds the claims in the critic's frame, an accuser in their own; only someone
    asserting them needs neither. It is still not a stance measure -- a terse assertion
    and a terse quotation are both unframed -- but it is the column to read.

    The honest conclusion this function is built to support: stance is not recoverable
    from stance-blind extraction by rearranging co-occurrence counts. Settling the
    question needs hand-labelled units or a stance-aware extractor.
    """
    sets = ideology_node_sets(S.cons, k)
    own_p = cross_pairs(sets["claim"], sets["ground"])
    A_own, e_own = actor_pair_totals(p, S, own_p)
    # Per rival, not pooled: for a CB actor, CB vocabulary is their own voice and not a
    # critical frame at all, so pooling the rivals makes every conservative look like
    # they are arguing against WS. Which rival the claims travel with is the whole point.
    riv = {nm: actor_pair_totals(p, S, cross_pairs(sets["claim"], nodes))
           for nm, nodes in sets["rivals"].items() if len(nodes)}

    tr = S.corpus.train
    x = S.corpus.x
    claim_u = x[:, sets["claim"]].sum(1)
    has_claim = (claim_u > 0) & tr
    has_ground = x[:, sets["ground"]].sum(1) > 0
    has_rival = {nm: x[:, nodes].sum(1) > 0 for nm, nodes in sets["rivals"].items()
                 if len(nodes)}
    any_rival = (np.zeros(len(x), bool) if not has_rival
                 else np.logical_or.reduce(list(has_rival.values())))
    units = np.bincount(S.corpus.actor[tr], minlength=S.corpus.A)
    mass = np.zeros(S.corpus.A)
    np.add.at(mass, S.corpus.actor[tr], claim_u[tr])

    def pooled(A, e, a):
        # Ratio of sums, the form section 7's R already takes. A median of per-pair
        # ratios degenerates here: at small delta most of these pairs never co-occur, so
        # the median is zero however strong the signal on the pairs that do.
        live = e[a] > exp_min
        if not live.any():
            return np.nan, 0
        num, den = float(A[a][live].sum()), float(e[a][live].sum())
        return (num / den if den > 0 else np.nan), int(live.sum())

    rows = []
    for a in range(S.corpus.A):
        sel = has_claim & (S.corpus.actor == a)
        if units[a] < min_units or not sel.any():
            continue
        r_own, n_own = pooled(A_own, e_own, a)
        rows.append({
            "actor": S.corpus.actors[a], "group": S.corpus.groups[a],
            "units": int(units[a]), "claim_units": int(sel.sum()),
            "claim_mass": float(mass[a]),
            "R_own": r_own, "n_own": n_own,
            "R_frame": {nm: pooled(A, e, a)[0] for nm, (A, e) in riv.items()},
            "co_ground": float((sel & has_ground).sum() / sel.sum()),
            "co_frame": {nm: float((sel & h).sum() / sel.sum())
                         for nm, h in has_rival.items()},
            # The closest thing to an adherent signature available here: claims carried
            # in no other ideology's vocabulary at all. A describer embeds them in the
            # critic's frame, an accuser in their own; only someone asserting them has
            # no reason to reach for either.
            "unframed": float((sel & ~any_rival).sum() / sel.sum()),
        })
    rows.sort(key=lambda r: -r["unframed"])

    overall = float((has_claim & has_ground).sum() / max(has_claim.sum(), 1))
    nm_k = S.cons.name(k)
    return {
        "ideology": nm_k,
        "claim_nodes": [S.cons.node_ids[u] for u in sets["claim"]],
        "ground_nodes": [S.cons.node_ids[u] for u in sets["ground"]],
        "n_own_pairs": len(own_p), "rivals": list(riv),
        "claim_units": int(has_claim.sum()), "co_ground_overall": overall,
        "actors": rows, "kernel": S.ps.label,
        "verdict": (
            f"{overall:.0%} of units carrying a {nm_k} claim also carry a "
            f"{nm_k} grounding node. Describing an ideology reproduces its "
            "structure, so R_own and co_ground rank reporters above adherents and are "
            "NOT stance measures. Read co_frame (low = not argued against in a rival's "
            "vocabulary) as weak evidence only; it cannot see accusation, which uses "
            "neither vocabulary."),
    }


# --------------------------------------------------------------------------- #
# auditing the extraction against the fitted model
# --------------------------------------------------------------------------- #


def extraction_surprise(p: M.Params, S: M.Setup, coded: dict, top: int = 200) -> dict:
    """Which codings the fitted model thinks should not be there.

    Section 9 lists extraction noise as a thing to bound by calibration against
    hand-coded units, and leaves open which units to hand-code. This ranks them: the
    model is a prior over what each actor expresses, so a coding it assigns a low
    probability is either a real surprise about the actor or a bad extraction, and those
    are the units worth a human's time.

    The probability is the rate layer's own. For a coding of node u in a unit by actor a
    at time t the model predicts a rate

        eta_{a,u,t} = exp( m_u + m_{t,c(u)} + sum_k theta_ak eta_ku + eps_{a,c(u)} )

    and hence a chance 1 - e^{-eta} that the node appears in that unit at all. `surprise`
    is -log10 of that: 3.0 means the model gave it a one-in-a-thousand chance.

    Raw surprise is dominated by how rare the node is corpus-wide, which is not what the
    question is about, so the log rate is reported decomposed into the three facets that
    compose it -- background (m_u + m_tc), ideology (sum_k theta_ak eta_ku) and actor
    (eps) -- and two theory-driven columns single out the part the user cares about:

      mismatch    max_k eta_ku - sum_k theta_ak eta_ku. Some ideology claims this node
                  strongly and this actor is not aligned with it. This is "expressing
                  content from a contrasting ideology" stated on the log-rate scale, and
                  it is the column to sort by when auditing for that specifically.
      off_anchor  the same thing said from theory alone, independent of the fit: node u
                  is a C1 anchor for some ideology, and this actor's alignment with every
                  ideology that anchors it is below `anchor_min`. Reported separately
                  because it survives a bad fit, whereas `mismatch` does not.

    `coded` is the raw {rt_id: [{id, confidence, source, stance}]} from data.read_codings,
    so every cell is scored -- including the ones the fit never saw. That is the point of
    scoring out of sample: if the voice coding is doing its job, `author/support` codings
    should be markedly less surprising than the rest, because the rest are the actor
    talking about positions that are not theirs. `by_cell` is that check.
    """
    corpus, cons = S.corpus, S.cons
    theta = M.theta_of(p, S)
    vth = M.vartheta_of(theta, S)
    unit_of = {u: d for d, u in enumerate(corpus.unit_ids)}
    ideo_term = theta[:, 1:] @ p.eta                            # [A, K]
    ieta = M.ideology_profile(p, S)
    best = ieta.max(0)                                          # [K]
    best_k = ieta.argmax(0)
    anchored = cons.anchor_pos.any(0)

    rows = []
    for rt_id, els in coded.items():
        d = unit_of.get(str(rt_id))
        if d is None:
            continue
        a, t = int(corpus.actor[d]), int(corpus.time[d])
        for e in els:
            u = cons.index.get(e["id"])
            if u is None:
                continue
            c = int(S.cidx[u])
            bg = float(S.m[u] + p.mtc[t, c])
            ide = float(ideo_term[a, u])
            act = float(p.mu[a])
            rate = float(np.exp(np.clip(bg + ide + act, -M.Z_MAX, M.Z_MAX)))
            pr = -np.expm1(-rate)                               # P(at least one)
            owners = [k for k in range(S.M) if cons.anchor_pos[k, u]]
            rows.append({
                "rt_id": str(rt_id), "actor": corpus.actors[a],
                "group": corpus.groups[a], "period": corpus.periods[t],
                "node": e["id"], "confidence": e.get("confidence"),
                "source": e.get("source"), "stance": e.get("stance"),
                "cell": _cell(e), "p_express": pr,
                "surprise": float(-np.log10(max(pr, 1e-300))),
                "background": bg, "ideology": ide, "actor_dev": act,
                "mismatch": float(best[u] - ide),
                "claimed_by": cons.name(int(best_k[u])),
                "anchored_to": "/".join(cons.name(k) for k in owners),
                "theta_on_anchors": (float(M.vartheta_of(theta, S)[a, [o + 1 for o in owners]].sum())
                                     if owners else np.nan),
                "off_anchor": bool(anchored[u] and owners
                                   and vth[a, [o + 1 for o in owners]].sum() < 0.2),
            })
    rows.sort(key=lambda r: -r["surprise"])

    def agg(key):
        out = {}
        for r in rows:
            out.setdefault(r[key], []).append(r)
        return {k: {"n": len(v),
                    "surprise": float(np.mean([x["surprise"] for x in v])),
                    "mismatch": float(np.mean([x["mismatch"] for x in v])),
                    "off_anchor": float(np.mean([x["off_anchor"] for x in v]))}
                for k, v in sorted(out.items(), key=lambda kv: -len(kv[1]))}

    return {"rows": rows, "top": rows[:top], "n": len(rows),
            "by_cell": agg("cell"), "by_node": agg("node"), "by_actor": agg("actor"),
            "worst_mismatch": sorted(rows, key=lambda r: -r["mismatch"])[:top],
            "off_anchor": [r for r in rows if r["off_anchor"]][:top]}


def _cell(e: dict) -> str:
    return {("author", "support"): "holds", ("author", "reject"): "opposes",
            ("other", "support"): "reports", ("other", "reject"): "refutes"
            }.get((e.get("source"), e.get("stance")), "unknown")


# --------------------------------------------------------------------------- #
# where the fit disagrees with theory
# --------------------------------------------------------------------------- #


def theory_divergence(p: M.Params, S: M.Setup, eta_hi: float = 0.25,
                      tol: float = 1e-6) -> dict:
    """Nodes and edges where the fitted model departs furthest from C1 and C2.

    Both constraints are priors in v3.1 rather than bounds -- C1 shifts the mean of the
    eta draw by +/-alpha and C2 shifts the support of W_k up to omega -- so the data is
    free to disagree with either, and what it does with that freedom is a result rather
    than a failure. This is the report of it.

    Six categories, three per constraint.

      C1  unmet     anchored to k, fitted well below the +alpha the prior asked for.
                    Theory named a definitional core the corpus does not support.
          poached   fitted high on k, but anchored to some *other* ideology. The most
                    diagnostic of the six: it says the component has drifted onto
                    content that belongs to a rival, and it is what a mislabeled or
                    unpinned component looks like from the inside.
          adopted   fitted high on k, anchored to no ideology at all. An affiliation the
                    corpus proposes and theory has not named.

      C2  inert     a link in E_k still sitting at omega. C2 is a lower bound, so the
                    fit cannot push such an edge below the floor -- resting on it exactly
                    is the strongest statement the model can make that the corpus offers
                    no support for the link.
          found     a large W_k on a pair E_k does not contain. Structure the corpus has
                    and theory did not name.
          shared    a link in E_k whose mass ended up mostly in W^(0). Theory called the
                    pairing ideological; the fit says every actor makes it, so it is
                    culture. Section 2's whole reason for the shared layer, seen per edge.

    Returns the categories plus headline rates -- the share of anchors met and of theory
    links lifted -- which are the two numbers V2's supply curve is a function of.
    """
    K, u, v = S.corpus.K, S.cons.pairs[:, 0], S.cons.pairs[:, 1]
    anchored_any = S.cons.anchor_pos.any(0)
    ieta = M.ideology_profile(p, S)
    prior = S.cons.prior_mean(S.h.alpha)
    out: dict = {c: [] for c in
                 ("unmet", "poached", "adopted", "inert", "found", "shared")}

    for k in range(S.M):
        nm = S.cons.name(k)
        eta, mu = ieta[k], prior[k]

        for i in np.flatnonzero(S.cons.anchor_pos[k] & (eta < 0.5 * S.h.alpha)):
            out["unmet"].append({"k": nm, "node": S.cons.node_ids[i],
                                 "eta": float(eta[i]), "prior": float(mu[i]),
                                 "gap": float(eta[i] - mu[i]),
                                 "element": S.cons.element[i]})
        for i in np.flatnonzero((eta > eta_hi) & ~S.cons.anchor_pos[k] & anchored_any):
            owners = [S.cons.name(j) for j in range(S.M)
                      if S.cons.anchor_pos[j, i]]
            out["poached"].append({"k": nm, "node": S.cons.node_ids[i],
                                   "eta": float(eta[i]), "owner": "/".join(owners),
                                   "element": S.cons.element[i]})
        for i in np.flatnonzero((eta > eta_hi) & ~anchored_any):
            out["adopted"].append({"k": nm, "node": S.cons.node_ids[i],
                                   "eta": float(eta[i]), "element": S.cons.element[i]})

        w, theory = p.W[k], S.cons.theory_edges[k]
        for e in np.flatnonzero(theory & (w <= S.h.omega + tol) & (p.W0 <= w + tol)):
            out["inert"].append({"k": nm, "pair": (S.cons.node_ids[u[e]],
                                                   S.cons.node_ids[v[e]]),
                                 "W": float(w[e]), "W0": float(p.W0[e]),
                                 "block": C.block_names()[int(S.cons.block_of()[e])]})
        for e in np.flatnonzero(~theory & (w > eta_hi)):
            out["found"].append({"k": nm, "pair": (S.cons.node_ids[u[e]],
                                                   S.cons.node_ids[v[e]]),
                                 "W": float(w[e]), "W0": float(p.W0[e]),
                                 "block": C.block_names()[int(S.cons.block_of()[e])]})
        for e in np.flatnonzero(theory & (p.W0 > np.maximum(w, tol))):
            out["shared"].append({"k": nm, "pair": (S.cons.node_ids[u[e]],
                                                    S.cons.node_ids[v[e]]),
                                  "W": float(w[e]), "W0": float(p.W0[e]),
                                  "block": C.block_names()[int(S.cons.block_of()[e])]})

    out["unmet"].sort(key=lambda r: r["gap"])
    out["poached"].sort(key=lambda r: -r["eta"])
    out["adopted"].sort(key=lambda r: -r["eta"])
    out["inert"].sort(key=lambda r: (r["k"], r["pair"]))
    out["found"].sort(key=lambda r: -r["W"])
    out["shared"].sort(key=lambda r: -(r["W0"] - r["W"]))
    n_anchor = int(S.cons.anchor_pos.sum())
    n_edge = int(S.cons.theory_edges.sum())
    out["anchors_met"] = 1.0 - len(out["unmet"]) / max(n_anchor, 1)
    out["edges_lifted"] = 1.0 - (len(out["inert"]) + len(out["shared"])) / max(n_edge, 1)
    out["n_anchors"], out["n_theory_edges"] = n_anchor, n_edge
    return out


def logical_incoherence(p: M.Params, S: M.Setup, eta_hi: float = 0.25) -> list[dict]:
    """Ideologies holding more than one member of an incompatible set (C3).

    C3 asserts this is impossible, and the cap is supposed to enforce it. It reports
    separately from `incoherence` because the cap is on the *rate* scale -- the mass
    sum_{u in S} e^{m_u + eta_ku} against pi_bar -- and a set of rare nodes can carry two
    fully elevated deviations while sitting an order of magnitude under the cap. The mass
    column then reads as compliant and the contradiction is invisible. This lists the
    contradictions directly, on the deviation scale, so the two readings can be compared:
    a row whose mass is far below pi_bar is one C3 as written cannot reach.
    """
    out = []
    ieta = M.ideology_profile(p, S)
    for Sx in S.cons.incompat:
        for k in range(S.M):
            hot = Sx[ieta[k, Sx] > eta_hi]
            if len(hot) > 1:
                out.append({
                    "k": S.cons.name(k),
                    "nodes": [S.cons.node_ids[i] for i in hot],
                    "eta": [float(ieta[k, i]) for i in hot],
                    "mass": float(np.exp(S.m[Sx] + ieta[k, Sx]).sum()),
                    "elements": [S.cons.element[i] for i in hot],
                })
    return sorted(out, key=lambda r: -min(r["eta"]))


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def head(title: str, *gloss: str) -> None:
    """A section header and a short statement of what its numbers mean.

    Every section of this report answers a question from the write-up, and the numbers
    are only legible if the question is on the page next to them -- so the gloss is part
    of the output rather than something to look up.
    """
    print(f"\n=== {title} ===")
    for line in gloss:
        print(f"    {line}")


def report(p: M.Params, S: M.Setup, top: int = 12) -> None:
    names = S.cons.names
    name = S.cons.name
    roles = node_roles(p, S)
    al = alignment_summary(p, S)
    K = S.corpus.K

    ieta = M.ideology_profile(p, S)
    head(f"components (kernel {S.ps.label}, F = {S.h.facets} facet(s) per ideology)",
         f"mass: expected adherents, out of {S.corpus.A} actors, summed over the "
         "ideology's facets.",
         "support: nodes with max_f eta > 0 -- how many positions the ideology takes, "
         "which section 2",
         "wants small since eta = 0 is 'no position'. effective actors: "
         "(sum t)^2 / sum t^2, given",
         "per ideology and then per facet. A facet with few effective actors is that "
         "actor's version",
         "of the ideology (section 9.1) -- a reading, not a fault.")
    for k in range(S.M):
        supp = int((ieta[k] > 0).sum())
        fac = np.flatnonzero(S.facet_of == k + 1)
        print(f"  {name(k):>6}  mass {al['mass'][k]:7.2f}  support {supp:3d}/{K}"
              f"  edges > 0 {int((p.W[k] > 1e-8).sum()):5d}/{S.P}"
              f"  effective actors {al['effective_actors'][k]:6.1f}")
        if len(fac) > 1:
            for i in fac:
                print(f"          facet {i - fac[0]}  mass {al['facet_mass'][i]:7.2f}"
                      f"  support {int((p.eta[i] > 0).sum()):3d}/{K}"
                      f"  effective actors {al['effective_per_facet'][i]:6.1f}")
    print(f"\n  background share theta[a,0]: mean {al['background'].mean():.3f}, "
          f"max {al['background'].max():.3f} ({S.corpus.actors[int(al['background'].argmax())]})")
    print("  That is the fraction of speech the model calls unideological -- where "
          "off-profile chatter\n  lands instead of distorting a profile. It should fall "
          "as F rises, if the facets are earning it.")
    print(f"  mean entropy over the ideology margin {al['entropy'].mean():.3f} of "
          f"{np.log(S.M):.3f} (uniform).")
    print("  High entropy is straddling -- section 9.2's mixing instrument. Facet spread "
          "within an\n  ideology is deliberately excluded: that is specialisation, not "
          "mixing.")

    head("alignment regression (C4: identity -> ideology)",
         "beta from theta_a = softmax(beta' m_a + xi_a). A positive cell says actors "
         "with that",
         "reference level.",
         "Inverted, a column is that ideology's adherent identity profile -- an "
         "estimand, not a",
         "definition. Column (bg) is the background coordinate. beta is indexed by "
         "ideology only and",
         "shared across its facets: identity informs which ideology, never which facet "
         "(C4), so that",
         "specialisation is never read as identity. Pinned actors contribute nothing "
         "to this fit.")
    print(f"  {'':<32} {'(bg)':>8} " + " ".join(f"{name(k):>8}" for k in range(S.M)))
    for j, nm in enumerate(S.corpus.meta_names):
        print(f"  {nm:<32} " + " ".join(f"{p.beta[j, k]:8.3f}"
                                        for k in range(S.cons.M + 1)))

    head("top nodes per ideology",
         "eta per ideology is max over its facets (section 5's own definition of the "
         "support). The",
         "facet lines below each ideology show how its specialists divide the labour.",
         "eta is the log-rate deviation from the background m_u: +1.0 means adherents "
         "express",
         "the node e^1 = 2.7x more than the corpus average, after their own breadth is "
         "removed.",
         "dist(inctiveness) = eta_k - max over rival ideologies; near zero means the "
         "node is shared.",
         "cen(trality) = summed W_k to same-type neighbours. High eta + low dist + high "
         "cen is a",
         "shared hub: the relational-contrast case that one pooled graph cannot "
         "represent.")
    for k in range(S.M):
        print(f"\n  {name(k)}:")
        for u in np.argsort(-ieta[k])[:top]:
            flag = "" if roles["expressed"][u] else "  [never expressed: not a measurement]"
            print(f"    {ieta[k, u]:+.2f}  dist {roles['distinctiveness'][k, u]:+.2f}  "
                  f"cen {roles['centrality'][k, u]:6.2f}  {S.cons.node_ids[u]:<4} "
                  f"{S.cons.element[u][:58]}{flag}")
        fac = np.flatnonzero(S.facet_of == k + 1)
        if len(fac) > 1:
            for i in fac:
                nodes = np.argsort(-p.eta[i])[:5]
                nodes = [n for n in nodes if p.eta[i, n] > 0.1]
                print(f"      facet {i - fac[0]}: "
                      + (", ".join(f"{S.cons.node_ids[n]} {p.eta[i, n]:+.2f}"
                                   for n in nodes) or "(flat -- takes no position)"))

    head("shared hubs",
         "eta > 0.25, distinctiveness < 0.1, ranked by centrality: nodes an ideology "
         "shares with",
         "a rival while building its own structure around them. BENDING's idiosyncrasy "
         "criterion",
         "would reject exactly these, which is the disagreement section 9.3 is about.")
    any_hub = False
    for k in range(S.M):
        sh, di, ce = (roles[key][k] for key in ("sharedness", "distinctiveness", "centrality"))
        cand = np.flatnonzero((sh > 0.25) & (di < 0.1) & roles["expressed"])
        for u in cand[np.argsort(-ce[cand])][:5]:
            any_hub = True
            print(f"  {name(k):>6}  cen {ce[u]:6.2f}  eta {sh[u]:+.2f}  "
                  f"{S.cons.node_ids[u]:<4} {S.cons.element[u][:58]}")
    if not any_hub:
        print("  none: no node is both elevated and undistinctive. Every ideology's "
              "structure sits\n  on content it does not share, so this corpus shows "
              "exclusion contrast without relational.")

    em = edge_mass(p, S)
    head("where the association mass sits (V8 ii)",
         "W^(0) is culture: pairing excess every actor shows, whatever their alignment. "
         "W_k is",
         "ideology: excess that covaries with theta. The split is an estimation "
         "property, not a",
         "definition -- both come out of one regression whose design is (1, theta_a). "
         "Per block,",
         "this is section 5's lateral-absorption watch: intra-rung blocks (BB "
         "especially) compete",
         "with cross-rung edges for the same co-occurrence, and cross-rung structure "
         "thins quietly.")
    print(f"  shared W(0) {em['shared']:10.2f}   ideology sum_k W_k {em['ideology']:10.2f}"
          f"   ideological share {em['ideology_share']:.1%}")
    print(f"  per ideology: " + "  ".join(
        f"{name(k)} {v:.1f}" for k, v in enumerate(em["per_ideology"])))
    print(f"\n  {'block':>6} {'pairs':>7} {'shared':>10} {'ideology':>10} "
          f"{'nz(0)':>7} {'nz(k)':>7}")
    for b, r in em["per_block"].items():
        print(f"  {b:>6} {r['pairs']:7d} {r['shared']:10.2f} {r['ideology']:10.2f} "
              f"{r['nonzero_shared']:7d} {r['nonzero_ideology']:7d}")

    head(f"structural coherence (C6, gate {'ON' if S.h.gate else 'off'})",
         "C_k = edge mass inside the ideology's own support, over mass crossing its "
         "boundary.",
         "The null draws random supports matched on size, node type and degree "
         "quartile, so a",
         "high C_k is not just 'this support contains the hubs'. C_k below the null is "
         "a reportable",
         "failure, not a fitting error: a component whose structure leaks across its "
         "own boundary",
         "is either mis-supported or not an ideology by this framework's definition.")
    if S.h.gate:
        print("    NOTE: the gate is on, so edges were made cheaper inside the support. "
              "C_k is then\n    partly the prior speaking. Section 3 wants the "
              "statistic to survive with it off first.")
    for row in coherence_ratio(p, S):
        verdict = ("above null" if np.isfinite(row["p"]) and row["p"] < 0.05
                   else "NOT above null")
        print(f"  {name(row['k']):>6}  support {row['support']:3d}  C_k {row['C_k']:8.3f}"
              f"   null {row['null_mean']:8.3f} +/- {row['null_sd']:.3f}   "
              f"p {row['p']:.3f}   {verdict}")

    if S.h.facets > 1:
        head("facet release: were the facets ideologies? (section 7)",
             "Facets share their ideology's network by construction. Release the tie -- "
             "one W per facet",
             "instead of one per ideology -- and refit the W block at the fitted rates. "
             "If the freed",
             "networks diverge on well-estimated pairs, the facets were ideologies all "
             "along and M is",
             "undercounted. Divergence is mean pairwise cosine distance; near zero means "
             "the tie costs",
             "nothing and the facets really are sub-profiles of one ideology.")
        for row in facet_release(p, S):
            if row["n_facets"] < 2:
                print(f"  {row['k']:>6}  one facet -- nothing to release")
                continue
            print(f"  {row['k']:>6}  {row['n_facets']} facets over {row['n_pairs']} live "
                  f"pairs  divergence {row['divergence']:.3f}"
                  f"   tied mass {row['tied_mass']:8.2f}  freed {row['free_mass']:8.2f}")

    head("contestation typology (per concept)",
         "What kind of thing each concept is, from the fitted facets. cultural: no "
         "ideology elevates",
         "it, association only in W^(0). consensually ideological: all holders agree "
         "and connect it",
         "alike. contested relationally: shared elevated node, diverging W_k rows. "
         "contested by",
         "presence: some ideologies elevate it, others do not -- the single-level "
         "degeneration of",
         "exclusion contrast, which needs a filled-in `concept` column to separate.")
    rows = contestation(p, S)
    from collections import Counter
    for t, n in Counter(r["type"] for r in rows).most_common():
        print(f"  {n:4d}  {t}")
    rel = [r for r in rows if r["type"] == "contested relationally"]
    if rel:
        print("  most relationally contested (same node, different neighbourhoods):")
        for r in sorted(rel, key=lambda r_: -r_["divergence"])[:5]:
            print(f"    divergence {r['divergence']:.2f}  {'+'.join(r['nodes']):<6} "
                  f"{S.cons.element[S.cons.index[r['nodes'][0]]][:52]}")

    theory_report(p, S, top)

    head("split diagnostic (R < 1 on pairs the ideology elevates)",
         "R = observed kernel-weighted co-occurrence over what the actors' own margins "
         "predict.",
         "W >= 0 can express R > 1 and cannot express R < 1, so a component that "
         "systematically",
         "under-produces pairs of nodes it claims to hold is one the model refused to "
         "give a negative",
         "edge. That is the signature of two sub-tendencies sharing a component, and "
         "the signal to",
         "split k. A high 'under 1' share is the thing to act on, not the median.")
    for row in split_signal(p, S):
        if not row["n_pairs"]:
            print(f"  {name(row['k']):>6}  no pair has both profiles elevated -- "
                  "nothing to test")
            continue
        print(f"  {name(row['k']):>6}  {row['n_pairs']:5d} pairs, median R "
              f"{row['median_R']:.2f}, {row['frac_under']:.0%} under 1")
        for a, b, r in row["worst"][:3]:
            print(f"          worst: {a}-{b}  R {r:.2f}")

    head("ladder test (R > 1 on rung-skipping pairs; C5)",
         "The same R, computed off the allowed set. C5 pins W to zero on V-B, V-R and "
         "A-R pairs, so",
         "excess co-occurrence there is evidence against the ladder -- or, far more "
         "often, against a",
         "single concept's type assignment. Read a hit as a prompt to revisit T(c) "
         "before the ladder:",
         "a concept mistyped one rung off shows up here paired with its true neighbours.")
    viol = ladder_violations(p, S)
    if not viol:
        print("  no rung-skipping pair shows excess co-occurrence: C5 survives here.")
    for r in viol[:top]:
        print(f"  {name(r['k']):>6}  R {r['R']:6.2f}  {r['pair'][0]}-{r['pair'][1]} "
              f"({r['layers'][0]}-{r['layers'][1]}, expected {r['expected']:.1f})")

    head("incoherence (C3)",
         "C3 binds ideologies but not units and not actors, and the gap between the "
         "layers is data.",
         "V_S: share of units expressing more than one member of an incompatible set. "
         "V_actor: share",
         "of actors whose rates do. mass: max over ideologies of "
         "sum_S e^(m_u + eta_ku), against the",
         "cap pi_bar -- this is the quantity C3 actually constrains.")
    inc = incoherence(p, S)
    if not inc:
        print("  no theoretical set is violated by any unit")
    for r in inc[:top]:
        print(f"  V_S {r['V_S']:.4f} ({r['n_units']:5d} units)  V_actor {r['V_actor']:.3f}"
              f"  mass {r['max_profile']:.3f}/{S.h.pi_bar}  "
              f"{'+'.join(r['set']):<14} worst {r['worst_actor']}")

    bad = logical_incoherence(p, S)
    print("\n  Ideologies holding both sides of an incompatible set -- what C3 asserts "
          "cannot happen:")
    if not bad:
        print("    none.")
    else:
        print(f"    {len(bad)} case(s). Note the mass column above: the cap is on the "
              "rate scale, so a set")
        print(f"    of rare nodes can carry two fully elevated deviations while sitting "
              "far under pi_bar,")
        print(f"    and C3 as written cannot reach it. Compare mass against "
              f"{S.h.pi_bar} on each row.")
        for r in bad[:top]:
            pieces = " | ".join(f"{n} ({e:+.2f})" for n, e in zip(r["nodes"], r["eta"]))
            print(f"    {r['k']:>4}  mass {r['mass']:.4f}  {pieces}")
            for n, el in zip(r["nodes"], r["elements"]):
                print(f"            {n:<4} {el[:66]}")


def theory_report(p: M.Params, S: M.Setup, top: int = 12) -> None:
    """The C1/C2 divergence section. Separate so it can be printed on its own."""
    t = theory_divergence(p, S)
    head("where the fit disagrees with theory (C1 nodes, C2 edges)",
         "C1 and C2 are priors in v3.1, not bounds: C1 shifts the mean of the eta draw "
         f"by +/-{S.h.alpha:g}",
         f"and C2 shifts the support of W_k up to omega = {S.h.omega:g}. The data can "
         "disagree with either,",
         "and what it does with that freedom is a result. These are the largest "
         "disagreements.")
    print(f"  anchors met  {t['anchors_met']:6.1%} of {t['n_anchors']:4d} C1 affiliations "
          f"reached at least half the prior mean")
    print(f"  edges lifted {t['edges_lifted']:6.1%} of {t['n_theory_edges']:4d} C2 links "
          f"rose above the omega floor as *ideological* mass")
    print(f"               ({len(t['shared'])} more carry their mass in the shared "
          f"layer instead; {len(t['inert'])} are inert in both)")

    print(f"\n  -- C1 POACHED: high on this ideology, anchored to a rival "
          f"({len(t['poached'])} nodes) --")
    print("     The most diagnostic category: content that belongs to another ideology "
          "by theory and\n     to this one by fit. A long list here says a component "
          "has drifted off its label.")
    for r in t["poached"][:top]:
        print(f"     {r['k']:>4} eta {r['eta']:+.2f}  {r['node']:<4} "
              f"[theory: {r['owner']}]  {r['element'][:52]}")
    if not t["poached"]:
        print("     none.")

    print(f"\n  -- C1 UNMET: anchored here, fitted below half the prior mean "
          f"({len(t['unmet'])} nodes) --")
    print("     Theory named a definitional core the corpus does not support. The gap "
          "column is\n     eta - prior mean, so the most negative rows are the "
          "affiliations the data most rejects.")
    for r in t["unmet"][:top]:
        print(f"     {r['k']:>4} eta {r['eta']:+.2f}  gap {r['gap']:+.2f}  "
              f"{r['node']:<4} {r['element'][:56]}")
    if not t["unmet"]:
        print("     none.")

    if t["adopted"]:
        print(f"\n  -- C1 ADOPTED: elevated but anchored to no ideology "
              f"({len(t['adopted'])} nodes) --")
        print("     An affiliation the corpus proposes and theory has not named.")
        for r in t["adopted"][:top]:
            print(f"     {r['k']:>4} eta {r['eta']:+.2f}  {r['node']:<4} "
                  f"{r['element'][:56]}")

    print(f"\n  -- C2 FOUND: large W_k on a pair E_k does not contain "
          f"({len(t['found'])} edges) --")
    print("     Structure the corpus has and theory did not name. W0 is the same pair's "
          "shared mass:\n     a high W_k beside a low W0 is specifically ideological "
          "association.")
    for r in t["found"][:top]:
        print(f"     {r['k']:>4} W {r['W']:5.2f}  (W0 {r['W0']:5.2f})  {r['block']}  "
              f"{r['pair'][0]}-{r['pair'][1]}")
    if not t["found"]:
        print("     none.")

    print(f"\n  -- C2 INERT: a link in E_k on the omega floor, with an empty shared "
          f"layer too ({len(t['inert'])} edges) --")
    print("     C2 is a lower bound, so the fit cannot push these below omega. Resting "
          "on it with W0 = 0\n     as well is the strongest available statement that "
          "the corpus offers no support for the\n     pairing in any layer -- neither "
          "ideological nor cultural. Disjoint from SHARED below.")
    for r in t["inert"][:top]:
        print(f"     {r['k']:>4} W {r['W']:5.2f}  (W0 {r['W0']:5.2f})  {r['block']}  "
              f"{r['pair'][0]}-{r['pair'][1]}")
    if not t["inert"]:
        print("     none.")

    print(f"\n  -- C2 SHARED: a link in E_k whose mass went to W^(0) instead "
          f"({len(t['shared'])} edges) --")
    print("     The pairing is there, but every actor makes it regardless of alignment, "
          "so the fit calls\n     it culture and theory called it ideology. The "
          "per-edge form of the V8(ii) mass split,\n     and the more interesting "
          "failure of the two: theory found a real link in the wrong layer.")
    for r in t["shared"][:top]:
        print(f"     {r['k']:>4} W {r['W']:5.2f}  W0 {r['W0']:5.2f}  {r['block']}  "
              f"{r['pair'][0]}-{r['pair'][1]}")
    if not t["shared"]:
        print("     none.")


# --------------------------------------------------------------------------- #
# intervention targets
# --------------------------------------------------------------------------- #


def boundary_mass(p: M.Params, S: M.Setup, eta_min: float = 0.0) -> dict:
    """Per node, the split of its edge mass into inside-support and across-boundary.

    This is C6's ratio decomposed by node rather than aggregated: summing `inside` over an
    ideology's support gives exactly the numerator `coherence_ratio` reports, and summing
    `boundary` gives exactly its denominator. That identity is what licenses reading a
    per-node score as "this node's contribution to the ideology's incoherence" -- a test
    asserts it, because a decomposition that does not add back up would be a different
    quantity wearing the same name.
    """
    Wd = dense(p.W, S.cons.pairs, S.corpus.K)          # [M, K, K]
    ieta = M.ideology_profile(p, S)
    inside = np.zeros((S.M, S.corpus.K))
    boundary = np.zeros((S.M, S.corpus.K))
    for k in range(S.M):
        supp = ieta[k] > eta_min
        inside[k] = (Wd[k] * supp[None, :]).sum(1)
        boundary[k] = (Wd[k] * ~supp[None, :]).sum(1)
        # a node outside the support contributes to neither of C6's sums
        inside[k, ~supp] = 0.0
        boundary[k, ~supp] = 0.0
    return {"inside": inside, "boundary": boundary, "support": ieta > eta_min}


def intervention_scores(p: M.Params, S: M.Setup, eta_min: float = 0.0) -> list[dict]:
    """Per (ideology, node): the four columns section 13's E3 asks for.

    Read as targets worth examining and nothing stronger. The model is observational
    co-occurrence at a chosen timescale, so a high score says that removing the node would
    restructure the fitted *description* -- a hypothesis about intervention, not a
    prediction of its effect. The route to a causal claim (belief dynamics on a fitted
    landscape) is named in the write-up's extensions and is not implemented.

    The quadrant of interest is high centrality with high boundary mass: the bridges C6
    penalises, where an ideology's own structure leaves its support.
    """
    bm = boundary_mass(p, S, eta_min)
    ieta = M.ideology_profile(p, S)
    Wd = dense(p.W, S.cons.pairs, S.corpus.K)
    other = (np.array([np.delete(ieta, k, axis=0).max(0) for k in range(S.M)])
             if S.M > 1 else np.zeros_like(ieta))
    out = []
    for k in range(S.M):
        supp = bm["support"][k]
        for u in np.flatnonzero(supp):
            # relational contestation: both ideologies elevate it, and their rows differ
            shared = [j for j in range(S.M) if j != k and ieta[j, u] > eta_min]
            div = np.nan
            if shared:
                divs = []
                for j in shared:
                    a, b = Wd[k, u], Wd[j, u]
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    if na > M.EPS and nb > M.EPS:
                        divs.append(1.0 - float(a @ b) / (na * nb))
                div = float(np.mean(divs)) if divs else np.nan
            tot = bm["inside"][k, u] + bm["boundary"][k, u]
            out.append({
                "k": S.cons.name(k), "node": S.cons.node_ids[u],
                "rung": C.LAYER_CODE[S.cons.layer[u]],
                "upstream": int(3 - S.cons.layer[u]),   # V=3 (most upstream) .. R=0
                "eta": float(ieta[k, u]),
                "distinctiveness": float(ieta[k, u] - other[k, u]),
                "centrality": float(bm["inside"][k, u]),
                "boundary": float(bm["boundary"][k, u]),
                "leak": float(bm["boundary"][k, u] / tot) if tot > M.EPS else np.nan,
                "shared_with": [S.cons.name(j) for j in shared],
                "divergence": div,
                "text": S.cons.element[u],
            })
    return sorted(out, key=lambda r: -(r["centrality"] + r["boundary"]))
