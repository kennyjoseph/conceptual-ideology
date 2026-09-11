"""Tests for the v3.1 model: gradients, projections, monotonicity, recovery.

Three kinds of test, in the order they catch things:

  gradients      every analytic gradient against a central difference. The model has
                 six of them and they share the composed rate, so a slip in one is
                 usually a slip in the composition.
  projections    project_expsum against a general constrained solve, and the Dykstra
                 loop over overlapping C3 sets against the same. Alternating projection
                 would pass a feasibility check and fail this one.
  monotonicity   the objective must not rise, at any stage, in any block. This is the
                 cheapest test that a gradient or a projection is wrong, and it is what
                 block-coordinate descent's convergence rests on.

Then recovery: synthetic corpora with planted structure, including the relational
contrast (a shared node with disjoint neighbourhoods) that section 6 exists to handle,
and the shared association layer that must not leak into the ideology layers.
"""

import itertools
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import minimize

import constraints as C
import diagnostics as G
import model as M
from data import Corpus


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def toy_constraints(K=8, Mi=2, A=4, seed=0, layers=None, incompat=None):
    rng = np.random.default_rng(seed)
    layer = np.array(layers if layers is not None else [0, 0, 1, 1, 2, 2, 3, 3][:K],
                     dtype=np.int8)
    concept_of = np.arange(K, dtype=np.int32)
    pairs = C.allowed_pairs(layer, concept_of)
    apos = rng.random((Mi, K)) < 0.3
    return C.Constraints(
        names=[C.IDEOLOGIES[k] if k < len(C.IDEOLOGIES) else f"free{k}"
               for k in range(Mi)],
        node_ids=[f"n{i}" for i in range(K)], index={f"n{i}": i for i in range(K)},
        element=[f"e{i}" for i in range(K)], layer=layer, concept_of=concept_of,
        anchor_pos=apos, anchor_neg=np.zeros_like(apos), pairs=pairs,
        theory_edges=rng.random((Mi, len(pairs))) < 0.05,
        incompat=(incompat if incompat is not None
                  else [np.array([0, 2], np.int32), np.array([2, 4], np.int32)]),
        actors=[str(a) for a in range(A)],
        theta_fix=np.array([True] + [False] * (A - 1)),
        theta_val=np.vstack([np.eye(Mi)[0]] + [np.full(Mi, np.nan)] * (A - 1)),
        meta_rows=[],
    )


def toy_corpus(cons, D=60, A=4, T=3, seed=0):
    rng = np.random.default_rng(seed)
    K = cons.K
    x = rng.random((D, K)) * (rng.random((D, K)) < 0.35)
    actor = rng.integers(0, A, D)
    return Corpus(
        x=x, actor=actor, time=rng.integers(0, T, D),
        actors=[str(a) for a in range(A)], periods=[str(t) for t in range(T)],
        unit_ids=[str(d) for d in range(D)],
        train=np.array([d % 10 != 0 for d in range(D)]),
        stamp=np.sort(rng.random(D) * 30),
        meta=np.column_stack([np.ones(A), rng.standard_normal(A)]),
        meta_names=["(intercept)", "z"], groups=["g0", "g0", "g1", "g1"][:A],
        concept_of=cons.concept_of,
    )


def loaded(cons=None, corpus=None, facets=2, **hyper):
    """A setup with every layer at a non-trivial value, for gradient checking.

    Two facets by default: the shapes that changed in v3.2 are the ones indexed by facet,
    and a single-facet fixture would not exercise them.
    """
    cons = toy_constraints() if cons is None else cons
    corpus = toy_corpus(cons) if corpus is None else corpus
    S = M.setup(corpus, cons,
                M.Hyper(kernel="exp", delta=5.0, facets=facets, **hyper))
    rng = np.random.default_rng(1)
    p = M.initialize(S)
    p.eta[:] = 0.3 * rng.standard_normal(p.eta.shape)
    p.mu[:] = 0.2 * rng.standard_normal(p.mu.shape)
    p.mtc[:] = 0.15 * rng.standard_normal(p.mtc.shape)
    p.beta[:] = 0.3 * rng.standard_normal(p.beta.shape)
    p.gamma[:] = 0.25 * rng.standard_normal(p.gamma.shape)
    p.W0[:] = np.abs(0.2 * rng.standard_normal(p.W0.shape))
    p.W[:] = np.abs(0.2 * rng.standard_normal(p.W.shape))
    return S, M.tau_step(p, S)


@pytest.fixture
def toy():
    return loaded()


def numgrad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        o = x[i]
        x[i] = o + eps
        a = f(x)
        x[i] = o - eps
        b = f(x)
        x[i] = o
        g[i] = (a - b) / (2 * eps)
        it.iternext()
    return g


def rel_err(a, b):
    return float(np.abs(a - b).max() / max(1.0, float(np.abs(a).max())))


# --------------------------------------------------------------------------- #
# gradients
# --------------------------------------------------------------------------- #


STAGE = M.STAGES["4"]  # every facet free, so every gradient carries every term


def _blockwise(S, p, name):
    """(objective, gradient, x0) for one block, exactly as the block step builds them."""
    if name == "eta":
        def f(eta):
            q = replace(p, eta=eta)
            th = M.theta_of(q, S)
            dev = eta - S.mu_C1
            return (M.rate_loss(q, S, th) + M.pair_loss(q, S, th)
                    + 0.5 * float((p.tau_I * dev * dev).sum())
                    + M.facet_smoothing(eta, S))

        def g(eta):
            q = replace(p, eta=eta)
            th = M.theta_of(q, S)
            grad = th[S.cells.actor][:, 1:].T @ M._rate_residual(q, S, th, S.cells)
            grad += th[:, 1:].T @ M._pair_node_pressure(q, S, th)
            return grad + p.tau_I * (eta - S.mu_C1) + M.facet_smoothing_grad(eta, S)

        return f, g, p.eta.copy()

    if name == "mu":
        def f(mu):
            q = replace(p, mu=mu)
            th = M.theta_of(q, S)
            return (M.rate_loss(q, S, th) + M.pair_loss(q, S, th)
                    + 0.5 * float((mu ** 2).sum()) / S.h.sigma_mu ** 2)

        def g(mu):
            q = replace(p, mu=mu)
            th = M.theta_of(q, S)
            grad = np.zeros_like(mu)
            np.add.at(grad, S.cells.actor,
                      M._rate_residual(q, S, th, S.cells).sum(1))
            grad += M._pair_node_pressure(q, S, th).sum(1)
            return grad + mu / S.h.sigma_mu ** 2

        return f, g, p.mu.copy()

    if name == "mtc":
        def f(mtc):
            q = replace(p, mtc=mtc)
            th = M.theta_of(q, S)
            return (M.rate_loss(q, S, th) + M.pair_loss(q, S, th)
                    + 0.5 * float((mtc ** 2).sum()) / S.h.sigma_m ** 2)

        def g(mtc):
            q = replace(p, mtc=mtc)
            th = M.theta_of(q, S)
            grad = np.zeros_like(mtc)
            np.add.at(grad, S.cells.period, M._rate_residual(q, S, th, S.cells) @ S.onehot)
            rt = np.exp(M.period_log_rates(q, S, th))
            r = np.einsum("at,atu->au", S.wts, rt)
            share = S.wts[:, :, None] * rt / np.maximum(r[:, None, :], M.EPS)
            grad += np.einsum("atu,au,uc->tc", share,
                              M._pair_node_pressure(q, S, th, r), S.onehot)
            return grad + mtc / S.h.sigma_m ** 2

        return f, g, p.mtc.copy()

    if name == "align":
        J, Mp1, A, F = p.beta.shape[0], S.cons.M + 1, S.corpus.A, S.F
        unpack = lambda v: (v[:J * Mp1].reshape(J, Mp1),
                            v[J * Mp1:].reshape(A, 1 + F))

        def f(v):
            b, gm = unpack(v)
            q = replace(p, beta=b, gamma=gm)
            th = M.theta_of(q, S)
            return (M.rate_loss(q, S, th) + M.pair_loss(q, S, th)
                    + 0.5 * float((b ** 2).sum()) / S.h.sigma_beta ** 2
                    + 0.5 * float((gm ** 2).sum()) / S.h.sigma_gamma ** 2)

        def g(v):
            b, gm = unpack(v)
            q = replace(p, beta=b, gamma=gm)
            th = M.theta_of(q, S)
            vth = M.vartheta_of(th, S)
            gt = np.zeros((A, 1 + F))
            np.add.at(gt[:, 1:], S.cells.actor,
                      M._rate_residual(q, S, th, S.cells) @ p.eta.T)
            r = M.actor_rates(q, S, th)
            pred = M.exposure(q, S, th, r) * np.exp(
                np.clip(M.omega(p, vth), -M.Z_MAX, M.Z_MAX))
            gv = np.zeros((A, Mp1))
            gv[:, 1:] = S.h.pair_weight * ((pred - S.ps.A * S.pmask) @ p.W.T)
            gt += gv @ S.agg.T
            gt[:, 1:] += M._pair_node_pressure(q, S, th, r) @ p.eta.T
            w = th * gt
            free = th * (gt - w.sum(1, keepdims=True))
            blk = np.divide(w @ S.agg, np.maximum(vth, M.EPS),
                            out=np.zeros_like(vth), where=vth > M.EPS)
            gs = np.where(S.fixed[:, None], th * (gt - blk @ S.agg.T), free)
            gb = np.zeros_like(b)
            gb[:, 0] = S.corpus.meta.T @ gs[:, 0]
            np.add.at(gb.T, S.facet_of, (S.corpus.meta.T @ gs[:, 1:]).T)
            return np.concatenate([(gb + b / S.h.sigma_beta ** 2).ravel(),
                                   (gs + gm / S.h.sigma_gamma ** 2).ravel()])

        return f, g, np.concatenate([p.beta.ravel(), p.gamma.ravel()])

    if name == "W":
        def f(flat):
            q = replace(p, W0=flat[:S.P], W=flat[S.P:].reshape(S.M, S.P))
            return M.pair_loss(q, S, M.theta_of(q, S))

        def g(flat):
            q = replace(p, W0=flat[:S.P], W=flat[S.P:].reshape(S.M, S.P))
            th = M.theta_of(q, S)
            vth = M.vartheta_of(th, S)
            pred = M.exposure(q, S, th) * np.exp(
                np.clip(M.omega(q, vth), -M.Z_MAX, M.Z_MAX))
            d = S.h.pair_weight * (pred - S.ps.A * S.pmask)
            return np.concatenate([d.sum(0), (vth[:, 1:].T @ d).ravel()])

        return f, g, np.concatenate([p.W0, p.W.ravel()])

    raise ValueError(name)


@pytest.mark.parametrize("block", ["eta", "mu", "mtc", "align", "W"])
def test_gradients_match_numerical(toy, block):
    S, p = toy
    f, g, x = _blockwise(S, p, block)
    assert rel_err(numgrad(f, x), g(x)) < 1e-5


def test_a_pin_binds_the_ideology_margin_and_frees_the_facets(toy):
    """C4 changed register in v3.2: it pins *which ideology*, not the whole membership.

    Both halves matter. The margin must be exactly what the constraint file supplies and
    stay there through a fit, or components drift off their names. But the facet split
    within that ideology has to remain free and learned, because identity informs which
    ideology and never which facet -- confusing the two would read specialisation as
    identity, which is the thing C4 exists to prevent.
    """
    S, p = toy
    assert S.fixed.any()
    vth = M.vartheta_of(M.theta_of(p, S), S)
    assert np.allclose(vth[S.fixed], S.fixed_vartheta[S.fixed], atol=1e-12)

    q = M.align_step(p, S, STAGE)
    vth2 = M.vartheta_of(M.theta_of(q, S), S)
    assert np.allclose(vth2[S.fixed], S.fixed_vartheta[S.fixed], atol=1e-12)
    # ...and the facets inside the pinned ideology did move
    a = int(np.flatnonzero(S.fixed)[0])
    k = int(np.argmax(S.fixed_vartheta[a]))
    inside = np.flatnonzero(S.facet_of == k) + 1
    assert len(inside) > 1
    assert not np.allclose(M.theta_of(q, S)[a, inside], M.theta_of(p, S)[a, inside])


def test_a_pinned_actor_contributes_no_beta_gradient(toy):
    """beta shifts every logit in an ideology block equally, and a pinned actor's
    membership is a conditional softmax within the block -- which is invariant to that.
    The zero falls out of the Jacobian rather than being imposed by a mask."""
    S, p = toy
    S1 = replace(S, fixed=np.ones(S.corpus.A, bool),
                 fixed_vartheta=np.tile(np.eye(S.cons.M + 1)[1], (S.corpus.A, 1)))
    _, g, x = _blockwise(S1, p, "align")
    J, Mp1 = p.beta.shape[0], S.cons.M + 1
    gbeta = g(x)[:J * Mp1].reshape(J, Mp1)
    assert np.allclose(gbeta, p.beta / S.h.sigma_beta ** 2, atol=1e-9)


# --------------------------------------------------------------------------- #
# projections
# --------------------------------------------------------------------------- #


def test_project_expsum_is_the_true_projection():
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = 6
        m, v, pi = rng.standard_normal(n) - 1.0, rng.standard_normal(n) * 1.5, 0.5
        z = M.project_expsum(v, m, pi)
        assert np.exp(m + z).sum() <= pi + 1e-8
        ref = minimize(lambda y: 0.5 * ((y - v) ** 2).sum(), v, method="SLSQP",
                       constraints=[{"type": "ineq",
                                     "fun": lambda y: pi - np.exp(m + y).sum()}],
                       options={"ftol": 1e-14, "maxiter": 500})
        assert np.abs(z - ref.x).max() < 1e-6


def test_project_expsum_is_the_identity_when_slack():
    m, v = np.full(4, -3.0), np.zeros(4)
    assert M.project_expsum(v, m, 0.5) is v


def test_project_expsum_survives_a_huge_argument():
    """The line search evaluates infeasible points; exp(m + v) overflows there."""
    m, v = np.full(4, -3.0), np.full(4, 900.0)
    z = M.project_expsum(v, m, 0.5)
    assert np.isfinite(z).all() and np.exp(m + z).sum() <= 0.5 + 1e-8


def test_project_caps_is_the_true_projection_on_overlapping_sets():
    """Dykstra, not alternating projection: the latter lands in the set, not on it."""
    rng = np.random.default_rng(3)
    n = 8
    m, v, pi = rng.standard_normal(n) - 1.0, rng.standard_normal(n) * 2, 0.3
    sets = [np.array([0, 1, 2]), np.array([2, 3, 4]), np.array([4, 5])]
    z = M.project_caps(v, m, pi, sets, n_iter=200)
    ref = minimize(
        lambda y: 0.5 * ((y - v) ** 2).sum(), v, method="SLSQP",
        constraints=[{"type": "ineq",
                      "fun": (lambda Sx: lambda y: pi - np.exp(m[Sx] + y[Sx]).sum())(Sx)}
                     for Sx in sets],
        options={"ftol": 1e-14, "maxiter": 1000})
    assert np.abs(z - ref.x).max() < 1e-6
    for Sx in sets:
        assert np.exp(m[Sx] + z[Sx]).sum() <= pi + 1e-8


# --------------------------------------------------------------------------- #
# block objectives, monotonicity, constraints
# --------------------------------------------------------------------------- #


def test_block_objectives_agree_with_the_full_objective(toy):
    """Each block's f must be the full objective up to a constant in that block.

    Checked by moving the block and comparing the *change*, since each block drops the
    terms it cannot affect.
    """
    S, p = toy
    rng = np.random.default_rng(7)
    for block, attr in (("eta", "eta"), ("mu", "mu"), ("mtc", "mtc")):
        f, _, x = _blockwise(S, p, block)
        y = x + 0.05 * rng.standard_normal(x.shape)
        d_block = f(y) - f(x)
        # the full objective with the same move, using the same fixed tau surrogate
        full = lambda z: (M.rate_loss(replace(p, **{attr: z}), S,
                                      M.theta_of(replace(p, **{attr: z}), S))
                          + M.pair_loss(replace(p, **{attr: z}), S,
                                        M.theta_of(replace(p, **{attr: z}), S))
                          + _surrogate_priors(replace(p, **{attr: z}), S, p))
        assert abs(d_block - (full(y) - full(x))) < 1e-8 * max(1.0, abs(d_block))


def _surrogate_priors(q, S, at):
    h = S.h
    dev_i = q.eta - S.mu_C1
    return (0.5 * float((at.tau_I * dev_i * dev_i).sum())
            + M.facet_smoothing(q.eta, S)
            + 0.5 * float((q.mtc ** 2).sum()) / h.sigma_m ** 2
            + 0.5 * float((q.beta ** 2).sum()) / h.sigma_beta ** 2
            + 0.5 * float((q.gamma ** 2).sum()) / h.sigma_gamma ** 2
            + 0.5 * float((q.mu ** 2).sum()) / h.sigma_mu ** 2)


@pytest.mark.parametrize("stage", ["1", "3a", "3b", "4"])
@pytest.mark.parametrize("coupled", [False, True])
def test_objective_is_monotone_across_blocks(stage, coupled):
    """The central test. If a stage rises, a gradient or a projection is wrong.

    Run in both readings of section 4 (see model.couples): each descends a different
    quantity, and `objective` reports whichever one the configuration is minimizing.
    """
    cons = toy_constraints()
    corpus = toy_corpus(cons)
    S = M.setup(corpus, cons, M.Hyper(kernel="exp", delta=5.0, max_iter=15, facets=2,
                                      couple_rates=coupled, pair_weight=0.05))
    st = M.STAGES[stage]
    p = M.initialize(S)
    prev = None
    for _ in range(6):
        p = M.tau_step(p, S)
        for step in (M.align_step, M.eta_step):
            p = step(p, S, st)
            val = M.objective(p, S, st)
            assert prev is None or val <= prev + 1e-7 * max(1.0, abs(prev))
            prev = val
        p = M.mu_step(p, S, st)
        val = M.objective(p, S, st)
        assert val <= prev + 1e-7 * max(1.0, abs(prev))
        prev = val
        if st.mtc:
            p = M.mtc_step(p, S, st)
            val = M.objective(p, S, st)
            assert val <= prev + 1e-7 * max(1.0, abs(prev))
            prev = val
        if st.W0 or st.Wk:
            p = M.w_step(p, S, st)
            val = M.objective(p, S, st)
            assert val <= prev + 1e-7 * max(1.0, abs(prev))
            prev = val


@pytest.mark.parametrize("coupled", [False, True])
def test_fit_stages_never_rises_within_a_stage(coupled):
    cons = toy_constraints()
    S = M.setup(toy_corpus(cons), cons, M.Hyper(kernel="exp", delta=5.0, max_iter=25,
                                                facets=2, couple_rates=coupled,
                                                pair_weight=0.05))
    _, infos = M.fit_stages(S, ("1", "3a", "3b", "4"), verbose=False)
    for info in infos:
        h = info["history"]
        assert all(h[i] <= h[i - 1] + 1e-8 * max(1.0, abs(h[i - 1]))
                   for i in range(1, len(h))), info["stage"]


def test_constraints_hold_after_a_fit():
    cons = toy_constraints()
    S = M.setup(toy_corpus(cons), cons, M.Hyper(kernel="exp", delta=5.0, max_iter=20))
    p, _ = M.fit_stages(S, ("1", "3a", "3b"), verbose=False)
    for f in range(S.F):  # C3 binds per facet: every sub-profile must be coherent
        for Sx in cons.incompat:
            assert np.exp(S.m[Sx] + p.eta[f, Sx]).sum() <= S.h.pi_bar + 1e-6
    assert (p.W >= S.lower_W - 1e-9).all()  # C2 and C7
    assert (p.W0 >= -1e-9).all()
    th = M.theta_of(p, S)
    assert np.allclose(th.sum(1), 1.0)
    assert np.allclose(M.vartheta_of(th, S)[S.fixed], S.fixed_vartheta[S.fixed])


def test_stage_gating_pins_the_facets_it_excludes():
    cons = toy_constraints()
    S = M.setup(toy_corpus(cons), cons, M.Hyper(kernel="exp", delta=5.0, max_iter=5))
    p1, _ = M.fit(S, M.STAGES["1"], verbose=False)
    assert not p1.mtc.any()
    assert not p1.W0.any() and not p1.W.any()
    p3, _ = M.fit(S, M.STAGES["3a"], p=p1, verbose=False)
    assert not p3.mtc.any()
    assert not p3.W.any()  # W_k is exactly zero while only the shared layer is free


# --------------------------------------------------------------------------- #
# the W-step
# --------------------------------------------------------------------------- #


def test_w_step_matches_a_general_solve(toy):
    """Coordinate descent over M+1 coordinates, vectorised across pairs, against SLSQP."""
    S, p = toy
    st = M.STAGES["3b"]
    got = M.w_step(p, S, st, n_sweeps=200)
    th = M.theta_of(p, S)
    Z = np.column_stack([np.ones(S.corpus.A), M.vartheta_of(th, S)[:, 1:]])
    E = M.exposure(p, S, th)
    Aobs = S.ps.A * S.pmask
    lam = np.vstack([np.full((1, S.P), S.h.lam_sp), M.lam_W(p, S)])
    lo = np.vstack([np.zeros((1, S.P)), S.lower_W])
    for pi in (0, S.P // 3, S.P - 1):
        obj = lambda b: float((E[:, pi] * np.exp(Z @ b) - Aobs[:, pi] * (Z @ b)).sum()
                              + lam[:, pi] @ b)
        ref = minimize(obj, lo[:, pi] + 0.1, method="SLSQP",
                       bounds=[(float(l), None) for l in lo[:, pi]],
                       options={"ftol": 1e-14, "maxiter": 800})
        mine = np.concatenate([[got.W0[pi]], got.W[:, pi]])
        assert obj(mine) <= ref.fun + 1e-6


def test_w_step_respects_c2_and_c7(toy):
    S, p = toy
    got = M.w_step(p, S, M.STAGES["3b"])
    assert (got.W0 >= 0).all()
    assert (got.W[S.cons.theory_edges] >= S.h.omega - 1e-12).all()


def test_the_l1_penalty_drives_edges_to_exactly_zero(toy):
    """W is sparse because Exp(lam_sp) is an l1, not because anything is thresholded."""
    S, p = toy
    S = replace(S, h=replace(S.h, lam_sp=50.0))
    got = M.w_step(replace(p, W0=np.zeros_like(p.W0), W=S.lower_W.copy()),
                   S, M.STAGES["3b"])
    assert (got.W0 == 0).mean() > 0.5


# --------------------------------------------------------------------------- #
# recovery on synthetic data
# --------------------------------------------------------------------------- #


def synthetic(cons, theta, eta, W0, W, D=1500, T=2, seed=0, delta_kernel="same"):
    """Draw a corpus from the model, then draw pair counts from the pair channel.

    The two draws are separate on purpose, and that is the model rather than a shortcut:
    section 2 says expressions are conditionally independent Poisson draws with *no*
    within-expression co-occurrence model, and the pair counts are a second observation
    channel over the same units. So planting an edge means drawing A^(delta) from
    Poisson(N eta_u eta_u' e^Omega) directly -- generating x alone could never exhibit
    the excess W is defined to measure.
    """
    rng = np.random.default_rng(seed)
    A, K = len(theta), cons.K
    m = np.full(K, np.log(0.06))
    log_rate = m[None, :] + theta @ eta   # theta here is the [A, M] ideology margin
    per_actor = D // A
    x, actor = [], []
    for a in range(A):
        x.append(rng.poisson(np.exp(log_rate[a]), size=(per_actor, K)).astype(float))
        actor += [a] * per_actor
    x = np.vstack(x)
    actor = np.array(actor)
    corpus = Corpus(
        x=x, actor=actor, time=rng.integers(0, T, len(x)),
        actors=[str(a) for a in range(A)], periods=[str(t) for t in range(T)],
        unit_ids=[str(d) for d in range(len(x))],
        train=np.ones(len(x), dtype=bool), stamp=np.arange(len(x), dtype=float),
        meta=np.column_stack([np.ones(A), theta[:, 0]]),
        meta_names=["(intercept)", "z"], groups=["g"] * A, concept_of=cons.concept_of,
    )
    S = M.setup(corpus, cons, M.Hyper(kernel=delta_kernel, max_iter=40, lam_sp=0.5,
                                      facets=1))
    # the pair channel, drawn from the planted Omega at the true rates
    r = np.exp(log_rate)
    u, v = cons.pairs[:, 0], cons.pairs[:, 1]
    Om = W0[None, :] + theta @ W
    E = S.ps.N[:, None] * r[:, u] * r[:, v]
    S = replace(S, ps=replace(S.ps, A=rng.poisson(E * np.exp(Om)).astype(float)))
    return S


def _relational_setup():
    """Two ideologies sharing a hub node, with disjoint neighbourhoods (section 6).

    A pooled graph reads this as a path h -- a1 and h -- b1 that no actor walks. One
    edge weight per pair *per ideology* is the minimal model that does not, and this is
    the test of that claim.
    """
    K, Mi = 6, 2
    layers = [1, 2, 2, 2, 2, 2]  # h is an attitude, the rest beliefs: all pairs allowed
    cons = toy_constraints(K=K, Mi=Mi, A=8, layers=layers, incompat=[])
    cons = replace(cons, anchor_pos=np.zeros((Mi, K), bool),
                   theory_edges=np.zeros((Mi, cons.P), bool),
                   actors=[str(a) for a in range(8)],
                   theta_fix=np.ones(8, bool),
                   theta_val=np.vstack([np.eye(Mi)[a % 2] for a in range(8)]))
    h_, a1, a2, b1, b2 = 0, 1, 2, 3, 4
    pair_of = {(int(x), int(y)): i for i, (x, y) in enumerate(cons.pairs)}
    eta = np.zeros((Mi, K))
    eta[:, h_] = 1.2  # both ideologies hold the hub
    eta[0, [a1, a2]] = 1.0
    eta[1, [b1, b2]] = 1.0
    W = np.zeros((Mi, cons.P))
    W[0, [pair_of[(h_, a1)], pair_of[(h_, a2)]]] = 1.5
    W[1, [pair_of[(h_, b1)], pair_of[(h_, b2)]]] = 1.5
    theta = np.vstack([np.eye(Mi)[a % 2] for a in range(8)])
    return cons, theta, eta, W, pair_of, (h_, a1, a2, b1, b2)


def test_recovers_the_shared_node_with_distinct_edges():
    cons, theta, eta, W, pair_of, (h_, a1, a2, b1, b2) = _relational_setup()
    S = synthetic(cons, theta, eta, np.zeros(cons.P), W, D=4000, seed=2)
    p, _ = M.fit_stages(S, ("1", "3a", "3b"), verbose=False)
    own = [p.W[0, pair_of[(h_, a1)]], p.W[0, pair_of[(h_, a2)]],
           p.W[1, pair_of[(h_, b1)]], p.W[1, pair_of[(h_, b2)]]]
    cross = [p.W[1, pair_of[(h_, a1)]], p.W[1, pair_of[(h_, a2)]],
             p.W[0, pair_of[(h_, b1)]], p.W[0, pair_of[(h_, b2)]]]
    assert min(own) > 0.8, own
    assert max(cross) < 0.5 * min(own), cross


def test_the_shared_layer_does_not_leak_into_the_ideology_layers():
    """Culture is what every actor does; W_k must carry only what covaries with theta.

    Without W^(0) the ideology networks silently absorb cultural association and overstate
    how much structure is ideological -- section 2's reason for the shared layer -- so a
    planted edge common to all actors has to land in W^(0) and nowhere else.
    """
    cons, theta, eta, _, pair_of, (h_, a1, a2, b1, b2) = _relational_setup()
    W0 = np.zeros(cons.P)
    planted = [pair_of[(h_, a1)], pair_of[(h_, b1)], pair_of[(a1, b1)]]
    W0[planted] = 1.5
    S = synthetic(cons, theta, eta, W0, np.zeros((cons.M, cons.P)), D=4000, seed=3)
    p, _ = M.fit_stages(S, ("1", "3a", "3b"), verbose=False)
    assert min(p.W0[planted]) > 0.8, p.W0[planted]
    assert p.W[:, planted].max() < 0.4, p.W[:, planted]
    # The planted association is attributed to culture rather than to either ideology.
    # Stated on the planted pairs rather than on total mass: sum_k theta_ak = 1, so the
    # design [1, theta] is rank-deficient by one and noise on other pairs can settle in
    # either layer. What must not happen is a shared edge being read as ideological.
    assert p.W0[planted].sum() > 3.0 * p.W[:, planted].sum()


def test_recovers_alignment_from_metadata():
    """C4: beta generalises the pinned anchors to unlabeled actors."""
    cons, theta, eta, W, _, _ = _relational_setup()
    # unpin half the actors: their alignment has to come from beta' m_a
    fix = np.array([a < 4 for a in range(8)])
    cons = replace(cons, theta_fix=fix)
    S = synthetic(cons, theta, eta, np.zeros(cons.P), W, D=4000, seed=4)
    p, _ = M.fit_stages(S, ("1",), verbose=False)
    # the *ideology* margin, not the facet coordinate: C4 is a claim about which ideology
    got = M.vartheta_of(M.theta_of(p, S), S)[:, 1:].argmax(1)
    assert (got[~fix] == theta[~fix].argmax(1)).all(), got


# --------------------------------------------------------------------------- #
# masking
# --------------------------------------------------------------------------- #


def test_masking_a_cell_is_not_the_same_as_zeroing_it():
    """A mask withholds the cell; a zero asserts the actor never expresses the node.

    The distinction is the whole point of V5, and it has to show up in the objective:
    the masked fit drops both halves of the Poisson term, the zeroed one keeps the
    exposure and so is told the actor stayed silent.
    """
    cons = toy_constraints()
    corpus = toy_corpus(cons)
    mask = np.ones((corpus.A, corpus.K))
    mask[0, :3] = 0.0
    Sm = M.setup(corpus, cons, M.Hyper(), cell_mask=mask)
    zeroed = replace(corpus, x=corpus.x.copy())
    zeroed.x[corpus.actor == 0, :3] = 0.0
    Sz = M.setup(zeroed, cons, M.Hyper())
    p = M.initialize(Sm)
    th = M.theta_of(p, Sm)
    assert M.rate_loss(p, Sm, th) < M.rate_loss(p, Sz, th)
    # and the masked cells contribute exactly nothing
    assert Sm.cells.obs[Sm.cells.actor == 0][:, :3].sum() == 0.0


def test_a_masked_pair_drops_from_the_pair_channel():
    cons = toy_constraints()
    corpus = toy_corpus(cons)
    mask = np.ones((corpus.A, corpus.K))
    mask[1, 0] = 0.0
    S = M.setup(corpus, cons, M.Hyper(kernel="exp", delta=5.0), cell_mask=mask)
    touching = (cons.pairs[:, 0] == 0) | (cons.pairs[:, 1] == 0)
    assert S.pmask[1, touching].sum() == 0.0
    assert S.pmask[0, touching].sum() == touching.sum()


# --------------------------------------------------------------------------- #
# diagnostics that are claims, not formatting
# --------------------------------------------------------------------------- #


def test_the_coherence_ratio_separates_a_coherent_from_a_leaking_component():
    """C6's statistic must be able to fail, or it is not falsifiable (section 3)."""
    cons, theta, eta, W, pair_of, (h_, a1, a2, b1, b2) = _relational_setup()
    S = synthetic(cons, theta, eta, np.zeros(cons.P), W, D=1200, seed=5)
    p = M.initialize(S)
    p = replace(p, eta=eta.copy(), W=W.copy())
    inside = G.coherence_ratio(p, S, n_perm=50)[0]["C_k"]
    # move ideology 0's edges onto the other ideology's support: pure boundary leakage
    leak = np.zeros_like(W)
    leak[0, [pair_of[(h_, b1)], pair_of[(h_, b2)]]] = 1.5
    leak[1] = W[1]
    outside = G.coherence_ratio(replace(p, W=leak), S, n_perm=50)[0]["C_k"]
    assert inside > outside


def test_the_ladder_test_reads_off_the_allowed_set():
    """C5 is testable: the R diagnostic runs off P, computed exactly as it is on P."""
    cons = toy_constraints()
    S = M.setup(toy_corpus(cons), cons, M.Hyper(kernel="exp", delta=5.0, max_iter=5))
    p, _ = M.fit_stages(S, ("1",), verbose=False)
    off = C.skipping_pairs(cons.layer, cons.concept_of)
    assert len(off)
    allowed = {tuple(x) for x in cons.pairs.tolist()}
    assert not (allowed & {tuple(x) for x in off.tolist()})
    rows = G.ladder_violations(p, S, exp_min=0.0)
    assert all(r["R"] > 1 for r in rows)


def test_edge_mass_partitions_every_pair_exactly_once():
    cons = toy_constraints()
    S = M.setup(toy_corpus(cons), cons, M.Hyper())
    p = M.initialize(S)
    p = replace(p, W0=np.ones(cons.P), W=np.ones((cons.M, cons.P)))
    em = G.edge_mass(p, S)
    assert sum(r["pairs"] for r in em["per_block"].values()) == cons.P
    assert em["shared"] == pytest.approx(cons.P)
    assert em["ideology"] == pytest.approx(cons.M * cons.P)


def test_allowed_pairs_admit_lateral_blocks_and_forbid_rung_skipping():
    """The C5 change from the previous version, pinned down."""
    layer = np.array([0, 1, 2, 3], dtype=np.int8)
    concept_of = np.arange(4, dtype=np.int32)
    got = {tuple(x) for x in C.allowed_pairs(layer, concept_of).tolist()}
    assert got == {(0, 1), (1, 2), (2, 3)}  # VA, AB, BR; VB, VR, AR excluded
    layer = np.array([2, 2, 2], dtype=np.int8)  # three beliefs: all lateral pairs allowed
    assert len(C.allowed_pairs(layer, np.arange(3, dtype=np.int32))) == 3
    assert "BB" in C.block_names() and "VB" not in C.block_names()


def test_dense_is_a_faithful_adjacency():
    cons = toy_constraints()
    rng = np.random.default_rng(0)
    w = rng.random((cons.M, cons.P))
    A = G.dense(w, cons.pairs, cons.K)
    assert np.allclose(A, np.swapaxes(A, -1, -2))
    for i, (u, v) in enumerate(cons.pairs):
        assert A[:, u, v] == pytest.approx(w[:, i])
    off = C.skipping_pairs(cons.layer, cons.concept_of)
    assert not A[:, off[:, 0], off[:, 1]].any()  # C5's zeros survive the round trip
    assert not np.diagonal(A, axis1=-2, axis2=-1).any()


def test_contestation_names_the_relational_case():
    """A shared node with disjoint neighbourhoods must not be filed as consensual."""
    cons, theta, eta, W, pair_of, (h_, a1, a2, b1, b2) = _relational_setup()
    S = synthetic(cons, theta, eta, np.zeros(cons.P), W, D=400, seed=6)
    p = replace(M.initialize(S), eta=eta.copy(), W=W.copy())
    rows = {r["concept"]: r for r in G.contestation(p, S)}
    assert rows[h_]["type"] == "contested relationally", rows[h_]
    assert rows[a1]["type"] == "contested by presence"  # only ideology 0 holds it
    assert rows[5]["type"] == "cultural"  # elevated by nobody


def test_report_runs_end_to_end():
    """The report is the product; a shape error in it is as bad as a wrong gradient."""
    cons = toy_constraints()
    S = M.setup(toy_corpus(cons), cons, M.Hyper(kernel="exp", delta=5.0, max_iter=5))
    p, _ = M.fit_stages(S, ("1", "3a", "3b"), verbose=False)
    G.report(p, S, top=3)


def test_theory_divergence_sorts_the_six_categories():
    """The C1/C2 disagreement report is a claim about the fit, so it gets a test."""
    cons = toy_constraints()
    S = M.setup(toy_corpus(cons), cons, M.Hyper(kernel="exp", delta=5.0, max_iter=5))
    p = M.initialize(S)
    k, other = 0, 1
    anchored = np.flatnonzero(cons.anchor_pos[k])
    rival = np.flatnonzero(cons.anchor_pos[other] & ~cons.anchor_pos[k])
    assert len(anchored) and len(rival)
    eta = np.zeros_like(p.eta)
    eta[k, anchored[0]] = 0.0        # anchored here, unmet
    eta[k, rival[0]] = 1.5           # anchored to a rival, poached
    W = S.lower_W.copy()
    theory = np.flatnonzero(cons.theory_edges[k])
    free = np.flatnonzero(~cons.theory_edges[k])
    W[k, free[0]] = 2.0              # found
    W0 = np.zeros(cons.P)
    if len(theory):
        W0[theory[0]] = 3.0          # a theory link the shared layer took
    p = replace(p, eta=eta, W=W, W0=W0)

    t = G.theory_divergence(p, S)
    ideol = C.IDEOLOGIES[k]
    assert cons.node_ids[anchored[0]] in [r["node"] for r in t["unmet"] if r["k"] == ideol]
    poached = [r for r in t["poached"] if r["k"] == ideol]
    assert poached and poached[0]["node"] == cons.node_ids[rival[0]]
    assert C.IDEOLOGIES[other] in poached[0]["owner"]
    assert [r["pair"] for r in t["found"] if r["k"] == ideol][0] == (
        cons.node_ids[cons.pairs[free[0], 0]], cons.node_ids[cons.pairs[free[0], 1]])
    if len(theory):
        assert any(r["W0"] == 3.0 for r in t["shared"] if r["k"] == ideol)
    # inert and shared are disjoint: the one theory edge given W0 mass is shared, the
    # rest are on the floor with nothing in either layer
    inert = [r for r in t["inert"] if r["k"] == ideol]
    shared = [r for r in t["shared"] if r["k"] == ideol]
    assert len(inert) + len(shared) == len(theory)
    assert not ({r["pair"] for r in inert} & {r["pair"] for r in shared})
    assert 0.0 <= t["anchors_met"] <= 1.0 and 0.0 <= t["edges_lifted"] <= 1.0


def test_logical_incoherence_catches_what_the_c3_cap_misses():
    """Both sides of an incompatible set elevated, while the rate-scale mass stays legal.

    The point of the statistic: on rare nodes the cap sum_S e^(m_u + eta_ku) <= pi_bar has
    orders of magnitude of slack, so a direct contradiction can sit fully inside C3.
    """
    cons = toy_constraints()
    S = M.setup(toy_corpus(cons), cons, M.Hyper())
    S = replace(S, m=np.full(S.corpus.K, -6.0))  # rare nodes: the cap cannot reach them
    p = M.initialize(S)
    eta = np.zeros_like(p.eta)
    Sx = cons.incompat[0]
    eta[0, Sx] = 1.5                                    # both members held at once
    p = replace(p, eta=eta)
    bad = G.logical_incoherence(p, S)
    assert bad and set(bad[0]["nodes"]) == {cons.node_ids[u] for u in Sx}
    assert bad[0]["mass"] < S.h.pi_bar                  # ...and C3 reports it as legal
    assert not G.logical_incoherence(replace(p, eta=np.zeros_like(eta)), S)


def test_drop_ideology_renormalises_the_pins():
    """Ablating k must leave the surviving C4 pins a valid alignment."""
    import validate as V
    cons = toy_constraints(K=8, Mi=3, A=4)
    val = np.full((4, 3), np.nan)
    val[0] = [0.5, 0.5, 0.0]   # split pin: survives, renormalised
    val[1] = [0.0, 1.0, 0.0]   # pinned entirely to the dropped one: becomes free
    fix = np.array([True, True, False, False])
    cons = replace(cons, theta_val=val, theta_fix=fix)
    out = V.drop_ideology(cons, 1)
    assert out.M == 2
    assert out.anchor_pos.shape[0] == 2 and out.theory_edges.shape[0] == 2
    assert out.theta_fix[0] and not out.theta_fix[1]      # orphaned pin released
    assert np.nansum(out.theta_val[0]) == pytest.approx(1.0)


def test_grounding_test_reports_that_it_does_not_separate_stance():
    """The statistic is real; the claim it cannot support is stated on every call.

    Planting a corpus where every claim-carrying unit also carries the grounding nodes --
    which is what describing an ideology looks like -- must produce a high co_ground and
    a verdict saying so, not a confident stance reading.
    """
    cons = toy_constraints(K=8, Mi=2, A=4)
    # node 0 is a Value, 2 an Attitude, 4 a Belief, 6 a Rationalization (see toy layers)
    apos = np.zeros((2, 8), bool)
    apos[0, [0, 2, 4, 6]] = True      # ideology 0 owns a full ladder
    apos[1, [1, 3]] = True            # ideology 1 owns rival grounding
    cons = replace(cons, anchor_pos=apos, anchor_neg=np.zeros_like(apos))
    corpus = toy_corpus(cons, D=400)
    x = np.zeros_like(corpus.x)
    x[:, [4, 6]] = 1.0                # every unit quotes the claims...
    x[:, [0, 2]] = 1.0                # ...alongside the grounding, as a report does
    corpus = replace(corpus, x=x)
    S = M.setup(corpus, cons, M.Hyper())
    r = G.grounding_test(M.initialize(S), S, 0, min_units=1)
    assert r["co_ground_overall"] == pytest.approx(1.0)
    assert "NOT stance measures" in r["verdict"]
    assert all(a["co_ground"] == pytest.approx(1.0) for a in r["actors"])
    assert set(r["claim_nodes"]) == {"n4", "n6"} and set(r["ground_nodes"]) == {"n0", "n2"}


def test_extraction_surprise_decomposes_the_log_rate():
    """The three facets must add back up to the rate the surprise is computed from."""
    cons = toy_constraints()
    corpus = toy_corpus(cons)
    S = M.setup(corpus, cons, M.Hyper())
    rng = np.random.default_rng(0)
    p = M.initialize(S)
    p = replace(p, eta=0.4 * rng.standard_normal(p.eta.shape),
                mu=0.3 * rng.standard_normal(p.mu.shape))
    coded = {corpus.unit_ids[d]: [{"id": cons.node_ids[u], "confidence": 0.8,
                                   "source": "author", "stance": "support"}]
             for d, u in enumerate(rng.integers(0, cons.K, corpus.D))}
    r = G.extraction_surprise(p, S, coded)
    assert r["n"] == corpus.D
    for x in r["rows"][:20]:
        rate = np.exp(x["background"] + x["ideology"] + x["actor_dev"])
        assert x["p_express"] == pytest.approx(-np.expm1(-rate))
        assert x["surprise"] == pytest.approx(-np.log10(x["p_express"]))
    # sorted most improbable first
    assert all(a["surprise"] >= b["surprise"]
               for a, b in zip(r["rows"], r["rows"][1:]))


def test_a_contrasting_coding_scores_a_high_mismatch():
    """The column the audit is for: an actor pinned to one ideology coded on another's node."""
    cons = toy_constraints(K=8, Mi=2, A=4)
    apos = np.zeros((2, 8), bool)
    apos[0, [0, 2]] = True
    apos[1, [4, 6]] = True
    cons = replace(cons, anchor_pos=apos, anchor_neg=np.zeros_like(apos),
                   theta_fix=np.ones(4, bool),
                   theta_val=np.vstack([np.eye(2)[a % 2] for a in range(4)]))
    corpus = toy_corpus(cons)
    S = M.setup(corpus, cons, M.Hyper())
    eta = np.zeros((2, 8))
    eta[0, [0, 2]] = 2.0        # ideology 0 owns nodes 0 and 2
    eta[1, [4, 6]] = 2.0
    p = replace(M.initialize(S), eta=eta)

    a0 = next(d for d in range(corpus.D) if corpus.actor[d] == 0)  # aligned to ideology 0
    coded = {corpus.unit_ids[a0]: [
        {"id": "n0", "confidence": 1.0, "source": "author", "stance": "support"},
        {"id": "n4", "confidence": 1.0, "source": "author", "stance": "support"}]}
    by = {r["node"]: r for r in G.extraction_surprise(p, S, coded)["rows"]}
    assert by["n0"]["mismatch"] == pytest.approx(0.0, abs=1e-9)  # own ideology's node
    assert by["n4"]["mismatch"] == pytest.approx(2.0, abs=1e-9)  # the rival's
    assert by["n4"]["surprise"] > by["n0"]["surprise"]
    assert by["n4"]["off_anchor"] and not by["n0"]["off_anchor"]
    assert by["n4"]["claimed_by"] == C.IDEOLOGIES[1]


def test_the_audit_scores_cells_the_fit_never_saw():
    """Out-of-sample scoring is the point: the by-cell table is the validation."""
    cons = toy_constraints()
    corpus = toy_corpus(cons)
    S = M.setup(corpus, cons, M.Hyper())
    p = M.initialize(S)
    coded = {corpus.unit_ids[0]: [
        {"id": "n0", "confidence": 1.0, "source": "author", "stance": "support"},
        {"id": "n1", "confidence": 1.0, "source": "other", "stance": "reject"}]}
    r = G.extraction_surprise(p, S, coded)
    assert {x["cell"] for x in r["rows"]} == {"holds", "refutes"}
    assert set(r["by_cell"]) == {"holds", "refutes"}
    assert all(v["n"] == 1 for v in r["by_cell"].values())


# --------------------------------------------------------------------------- #
# v3.2: the faceted actor
# --------------------------------------------------------------------------- #


def test_membership_is_a_simplex_over_background_and_facets():
    cons = toy_constraints(K=8, Mi=2, A=6)
    corpus = toy_corpus(cons, A=6)
    S = M.setup(corpus, cons, M.Hyper(facets=3))
    assert S.F == 6 and list(S.facet_of) == [1, 1, 1, 2, 2, 2]
    p = M.initialize(S)
    th = M.theta_of(p, S)
    assert th.shape == (6, 1 + 6) and np.allclose(th.sum(1), 1.0) and (th >= 0).all()
    vth = M.vartheta_of(th, S)
    assert vth.shape == (6, 3) and np.allclose(vth.sum(1), 1.0)
    # the margin really is the facet sum, and column 0 really is the background
    assert np.allclose(vth[:, 1], th[:, 1:4].sum(1))
    assert np.allclose(vth[:, 2], th[:, 4:7].sum(1))
    assert np.allclose(vth[:, 0], th[:, 0])


def test_the_background_coordinate_has_no_profile():
    """theta[:,0] must dilute toward m_u + mu_a and nothing else.

    That is what makes it readable as the unideological share: mass there costs no
    parameters and pulls the actor toward the corpus background, so off-profile chatter
    has a home that is not a junk component.
    """
    cons = toy_constraints()
    corpus = toy_corpus(cons)
    S = M.setup(corpus, cons, M.Hyper(facets=2))
    p = M.initialize(S)
    p = replace(p, eta=np.ones_like(p.eta), mu=np.zeros_like(p.mu))
    all_bg = np.zeros((S.corpus.A, 1 + S.F))
    all_bg[:, 0] = 1.0
    z = M.cell_log_rates(p, S, all_bg, S.cells)
    assert np.allclose(z, S.m[None, :] + p.mtc[S.cells.period][:, S.cidx])


def test_facets_recover_planted_sub_profiles():
    """The point of the layer: two specialisms inside one ideology, and F=1 cannot see it.

    Actors alternate between two disjoint focal sets within a single ideology. At F=1 the
    profile has to average them and collapses; at F=2 both are recovered and the actors
    land on the right coordinates; at F=3 the spare facet stays flat rather than splitting
    something that is not there.
    """
    K, A, D = 8, 12, 2400
    layer = np.array([1, 1, 2, 2, 2, 2, 3, 3], dtype=np.int8)
    cof = np.arange(K, dtype=np.int32)
    cons = replace(toy_constraints(K=K, Mi=1, A=A, layers=list(layer), incompat=[]),
                   anchor_pos=np.zeros((1, K), bool), anchor_neg=np.zeros((1, K), bool),
                   theory_edges=np.zeros((1, len(C.allowed_pairs(layer, cof))), bool),
                   actors=[str(a) for a in range(A)], theta_fix=np.ones(A, bool),
                   theta_val=np.ones((A, 1)))
    true_eta = np.zeros((2, K))
    true_eta[0, [0, 2, 4]] = 1.6
    true_eta[1, [1, 3, 5]] = 1.6
    rng = np.random.default_rng(0)
    rate = np.exp(np.log(0.08) + np.stack([true_eta[a % 2] for a in range(A)]))
    per = D // A
    x = np.vstack([rng.poisson(rate[a], size=(per, K)).astype(float) for a in range(A)])
    corpus = replace(toy_corpus(cons, D=len(x), A=A),
                     x=x, actor=np.repeat(np.arange(A), per),
                     time=np.zeros(len(x), int), periods=["all"],
                     unit_ids=[str(d) for d in range(len(x))],
                     train=np.ones(len(x), bool),
                     stamp=np.arange(len(x), dtype=float),
                     meta=np.ones((A, 1)), meta_names=["(intercept)"], groups=["g"] * A)

    def fit(F):
        S = M.setup(corpus, cons, M.Hyper(facets=F, max_iter=60))
        p, _ = M.fit_stages(S, ("1",), verbose=False)
        return S, p

    S1, p1 = fit(1)
    assert np.abs(p1.eta).max() < 0.35, "F=1 should average the two specialisms away"

    S2, p2 = fit(2)
    got = [set(np.flatnonzero(p2.eta[f] > 0.4).tolist()) for f in range(2)]
    assert sorted(got, key=min) == [{0, 2, 4}, {1, 3, 5}], got
    th = M.theta_of(p2, S2)
    assert len({th[a].argmax() for a in range(0, A, 2)}) == 1     # even actors agree
    assert ({th[a].argmax() for a in range(0, A, 2)}
            != {th[a].argmax() for a in range(1, A, 2)})          # ...and differ from odd

    S3, p3 = fit(3)
    flat = [f for f in range(3) if np.abs(p3.eta[f]).max() < 0.35]
    assert len(flat) >= 1, "the spare facet should stay flat, not split a real one"


def test_facets_of_one_ideology_diverge_from_the_symmetric_start():
    """C1 is shared across an ideology's facets, so nothing in the prior separates them.

    Guards the initialisation: jittering gamma alone is not enough, because the membership
    block runs first and, with eta at zero, drives gamma straight back to its prior --
    welding the facets together for the rest of the fit.
    """
    cons = toy_constraints()
    S = M.setup(toy_corpus(cons), cons, M.Hyper(facets=2, max_iter=20))
    assert np.abs(np.diff(M.initialize(S).eta[S.facet_of == 1], axis=0)).max() > 0
    p, _ = M.fit_stages(S, ("1", "3a", "3b"), verbose=False)
    spread = max(np.abs(p.eta[S.facet_of == k] - p.eta[S.facet_of == k].mean(0)).max()
                 for k in range(1, S.cons.M + 1))
    assert spread > 1e-6


def test_c3_binds_every_facet_not_just_the_ideology():
    cons = toy_constraints()
    S = M.setup(toy_corpus(cons), cons, M.Hyper(facets=3, max_iter=20))
    p, _ = M.fit_stages(S, ("1", "3a", "3b"), verbose=False)
    for f in range(S.F):
        for Sx in cons.incompat:
            assert np.exp(S.m[Sx] + p.eta[f, Sx]).sum() <= S.h.pi_bar + 1e-6


def test_the_ideology_profile_is_the_max_over_facets():
    cons = toy_constraints(K=6, Mi=2, A=4)
    S = M.setup(toy_corpus(cons), cons, M.Hyper(facets=2))
    p = M.initialize(S)
    eta = np.zeros_like(p.eta)
    eta[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    eta[1] = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    ieta = M.ideology_profile(replace(p, eta=eta), S)
    assert ieta.shape == (2, 6)
    assert np.allclose(ieta[0], [1.0, 2.0, 0, 0, 0, 0])   # union of its facets' supports


def test_facet_smoothing_is_off_by_default_and_pulls_facets_together_when_on():
    cons = toy_constraints()
    S = M.setup(toy_corpus(cons), cons, M.Hyper(facets=2))
    p = M.initialize(S)
    assert M.facet_smoothing(p.eta, S) == 0.0
    assert not M.facet_smoothing_grad(p.eta, S).any()
    S2 = replace(S, h=replace(S.h, lam_facet=1.0))
    eta = np.zeros_like(p.eta)
    eta[S2.facet_of == 1] = np.array([[1.0] * S.corpus.K, [-1.0] * S.corpus.K])
    assert M.facet_smoothing(eta, S2) > 0
    g = M.facet_smoothing_grad(eta, S2)
    assert (g[S2.facet_of == 1][0] > 0).all() and (g[S2.facet_of == 1][1] < 0).all()


def test_w_routes_by_the_ideology_margin_so_facets_share_a_network():
    """Two actors with the same ideology margin but different facet splits must get the
    same Omega -- that is what 'facets share their ideology's network' means."""
    cons = toy_constraints(K=8, Mi=2, A=4)
    S = M.setup(toy_corpus(cons), cons, M.Hyper(facets=2))
    p = M.initialize(S)
    p = replace(p, W=np.abs(np.random.default_rng(0).standard_normal(p.W.shape)),
                W0=np.zeros(S.P))
    th = np.zeros((S.corpus.A, 1 + S.F))
    th[0, 1], th[0, 2] = 0.6, 0.2          # ideology 1 split 0.6/0.2
    th[1, 1], th[1, 2] = 0.2, 0.6          # same margin, opposite split
    th[0, 0] = th[1, 0] = 0.2
    Om = M.omega(p, M.vartheta_of(th, S))
    assert np.allclose(Om[0], Om[1])
