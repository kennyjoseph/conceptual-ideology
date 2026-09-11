"""The v3.2 model: faceted ideologies, composed rates, one pair channel at timescale delta.

M-estimation over (eta, W, beta, gamma, mu) with a variational treatment of the SAGE
variances tau, following sections 4 and 6 of the write-up. Every block is a sparse Poisson
GLM, and block-coordinate descent applies because each is convex with the others fixed.

The parameterisation, in one place (M ideologies each with F_k facets, F = sum_k F_k
in total, K nodes, C concepts, A actors, T periods, P allowed pairs, J metadata columns):

    m_u        [K]      fixed   log corpus rate of node u
    m_tc       [T, C]           temporal offsets -- the news cycle
    eta        [F, K]           facet profiles: sub-profiles of an ideology, the
                                granularity at which specialists engage
    mu         [A]              actor breadth -- verbosity, not content
    beta       [J, M+1]         identity -> ideology regression; column 0 is background
    gamma      [A, 1+F]         facet logits, an actor-level draw
    theta      [A, 1+F] derived softmax over (background, facets)
    vartheta   [A, M+1] derived theta summed within each ideology
    W0         [P]              the shared association layer -- culture
    W          [M, P]           per-ideology networks -- ideology

What changed from v3.1, and why, since it is the whole point of this version. Actors used
to carry a free deviation eps[a, c] over every concept. That layer and W are rival
explanations of the same between-actor covariation at large delta: joint elevation of two
concepts among an ideology's adherents can be absorbed by inflating each actor's eps
(penalised once per actor) or by one shared edge (penalised once). v3.1 could only report
the contest; v3.2 removes it by construction. Actors are now mixed memberships over
(ideology, facet) pairs plus a background coordinate and a breadth scalar, so actor
variation is rank-restricted to the span of the facet profiles and covariation beyond them
*must* land in W. The contest re-enters through F_k -- each added facet is a rank of
covariation W no longer has to explain -- but that dial is discrete and selected by
whole-actor holdout rather than tuned; see validate.facet_contest.

The background is a coordinate of the membership, not a component: theta[:, 0] has no
profile row, so mass there pulls an actor toward m_u + mu_a and nothing else. It reads
directly as the fraction of an actor's speech the model calls unideological, and it is
where off-profile chatter lands instead of distorting a profile or minting a junk
component.

Composed log rate, section 2 -- note there is no actor-by-concept term:

    log eta_{a,u,t} = m_u + m_{t,u} + sum_{k,f} theta_{a,kf} eta_{k,f,u} + mu_a

Pair channel at timescale delta, section 2:

    A^(delta)_{a,uu'} ~ Poisson(N^(delta)_a eta_{a,u} eta_{a,u'} e^{Omega_{a,uu'}}),
    Omega_{a,uu'} = W^(0)_{uu'} + sum_k vartheta_{a,k} W_{k,uu'}

routed by the ideology *margin*: facets share their ideology's network.

Two structural facts make this cheap.

1. The rate depends on a unit only through its (actor, period), so the rate likelihood
   sees the corpus as at most A*T cells -- 739 of them here against 34,670 units. See
   data.rate_cells.
2. The pair likelihood is a Poisson GLM whose design matrix Z = [1, vartheta] is the same
   for every pair. So the W-step is P independent (M+1)-dimensional problems sharing a
   design, and one vectorised coordinate sweep updates all 4285 pairs at once. There is
   no inner solver and no pair loop.

One decision the write-up leaves open is settled here and documented at `couples`:
whether the rate blocks see the pair term. They do not, by default, because section 4's
joint objective is unbounded below along the direction (log rate down, Omega up) once
the pair counts outnumber the expression counts -- which they do on any corpus with more
than a couple of nodes per unit. Section 6's block descriptions, where the pair channel's
N^(delta)_a eta_au eta_au' is an exposure held fixed, are what is implemented.

The SAGE priors are handled by majorization-minimization. Given tau, the prior is
Gaussian and every block objective is the quadratic surrogate; the tau-step re-tightens
the surrogate at the current point. `objective` reports the *true* penalty (a log for
the Jeffreys prior, an l1-like term for the proper Exponential one), and MM guarantees
that is what descends, which is why the tau-step runs first in each iteration.
"""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import wrightomega

import constraints as C
import kernels as Kr
from data import (Cells, Corpus, actor_period_weights, background_rates,
                  rate_cells)

EPS = 1e-12
Z_MAX = 20.0  # log-rate clip; e^20 already dwarfs any cell count on this corpus


# --------------------------------------------------------------------------- #
# hyperparameters and parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Hyper:
    """Section 2's hyperparameter list, plus the numerics."""

    alpha: float = 1.0  # C1 prior mean shift on anchored nodes
    omega: float = 0.10  # C2 lower bound on W_k over the theory links E_k
    pi_bar: float = 0.50  # C3 cap: sum_{u in S} e^{m_u + eta_ku} <= pi_bar
    lam_sp: float = 1.0  # Exp(lam_sp) prior on W -- an l1 penalty
    lam_in: float = 0.5  # C6 gate: penalty inside an ideology's support
    lam_out: float = 2.0  # C6 gate: penalty across its boundary
    gate: bool = False  # C6's prior. Off by default: section 3 wants the statistic
    #                     to survive unpenalized before the prior is switched on.
    tau_prior: str = "jeffreys"  # or "exp" for a proper scale prior
    tau_rate: float = 1.0  # rate of the Exp prior on tau, when proper
    tau_floor: float = 1e-4  # keeps E[1/tau] finite at a zero deviation
    sigma_m: float = 1.0  # N(0, sigma^2) on the temporal offsets
    sigma_beta: float = 1.0
    sigma_gamma: float = 1.0  # N(0, sigma^2) on the facet logits
    sigma_mu: float = 1.0
    facets: int = 1  # F_k, the same for every ideology. Section 8 stage 2 raises it and
    #                  selects by whole-actor holdout (V7); see fit.py sweep-facets.
    lam_facet: float = 0.0  # section 6.3's optional Laplacian smoothing of an ideology's
    #                         facet profiles: lam_facet * sum_k sum_{f<f'} ||eta_f - eta_f'||^2.
    #                         Off by default -- the facets are supposed to differ.
    pair_weight: float = 1.0  # composite-likelihood weight on the pair block
    couple_rates: bool = False  # do the rate blocks see the pair term? see `couples`
    kernel: str = "same"  # the co-occurrence kernel; see kernels.py
    delta: float = 0.0  # its timescale, in days
    max_iter: int = 100
    inner_iter: int = 10
    cap_iter: int = 30  # Dykstra sweeps for the C3 caps
    tol: float = 1e-7
    seed: int = 0


@dataclass
class Params:
    eta: np.ndarray  # [F, K] facet profiles; F = sum_k F_k, background is not stored
    mu: np.ndarray  # [A] breadth, now a rate parameter rather than eps's prior mean
    mtc: np.ndarray  # [T, C]
    beta: np.ndarray  # [J, M+1] ideology logits; column 0 is the background coordinate
    gamma: np.ndarray  # [A, 1+F] facet logits
    W0: np.ndarray  # [P]
    W: np.ndarray  # [M, P]
    tau_I: np.ndarray  # [F, K] variational variances for eta

    def copy(self) -> "Params":
        return replace(self, **{k: v.copy() for k, v in self.__dict__.items()})


@dataclass(frozen=True)
class Stage:
    """Which layers are free. Section 8's build order, as data rather than an integer.

    Dropping a layer is a reattribution, not a simplification, so the stages are named by
    what they add and the fit is reported at each one.

    Section 8's stage 2 is missing from the numbering on purpose. In v3.1 it was "free
    actor deviations"; v3.2 deletes those, and what replaces it -- raising the facet count
    F_k -- is not a block that can be gated inside one fit. F_k changes the shape of eta,
    so the ladder is walked across fits and selected by whole-actor holdout (V7). See
    fit.py sweep-facets. The remaining numbers are kept as section 8 writes them rather
    than renumbered, so a stage label means the same thing in the code and in the paper.
    """

    name: str
    W0: bool = False  # the shared association layer
    Wk: bool = False  # the per-ideology networks
    mtc: bool = False  # temporal offsets


STAGES: dict[str, Stage] = {
    "1": Stage("1 rates only"),
    "3a": Stage("3a + shared association", W0=True),
    "3b": Stage("3b + ideology networks", W0=True, Wk=True),
    "4": Stage("4 + temporal background", W0=True, Wk=True, mtc=True),
}


# --------------------------------------------------------------------------- #
# the fixed context a fit runs against
# --------------------------------------------------------------------------- #


@dataclass
class Setup:
    """Everything derived once from the corpus, the theory and the kernel choice."""

    corpus: Corpus
    cons: C.Constraints
    h: Hyper
    m: np.ndarray  # [K] fixed background log-rates
    cells: Cells  # the training (actor, period) collapse
    test: Cells  # the held-out one
    ps: Kr.PairStats  # A^(delta), N^(delta)
    wts: np.ndarray  # [A, T] actor-period weights for the time-averaged rate
    mu_C1: np.ndarray  # [F, K] C1's prior mean, broadcast from [M, K] over facets
    lower_W: np.ndarray  # [M, P] C2's lower bound, omega on E_k
    fixed: np.ndarray  # [A] bool, C4's pinned actors
    fixed_vartheta: np.ndarray  # [A, M+1] pinned ideology margin, valid where `fixed`
    pmask: np.ndarray  # [A, P] pair observation mask, from the cell mask
    facet_of: np.ndarray  # [F] int, which ideology (1..M) each facet belongs to
    cidx: np.ndarray = field(init=False)  # [K] concept of each node
    onehot: np.ndarray = field(init=False)  # [K, C] node -> concept, for sums
    agg: np.ndarray = field(init=False)  # [1+F, M+1] facet -> ideology margin, incl. bg

    def __post_init__(self):
        self.cidx = self.corpus.concept_of
        self.onehot = np.zeros((self.corpus.K, self.corpus.C))
        self.onehot[np.arange(self.corpus.K), self.cidx] = 1.0
        # Coordinate 0 of theta is the background; coordinate 1+i is facet i, whose
        # ideology is facet_of[i]. `agg` sums a membership vector into its ideology
        # margin, so vartheta = theta @ agg with column 0 the unideological share.
        self.agg = np.zeros((1 + self.F, self.cons.M + 1))
        self.agg[0, 0] = 1.0
        self.agg[np.arange(1, 1 + self.F), self.facet_of] = 1.0

    @property
    def F(self) -> int:
        """Number of ideology facets, excluding the background coordinate."""
        return len(self.facet_of)

    @property
    def M(self) -> int:
        return self.cons.M

    @property
    def P(self) -> int:
        return self.cons.P


def setup(
    corpus: Corpus,
    cons: C.Constraints,
    h: Hyper | None = None,
    cell_mask: np.ndarray | None = None,
    train: np.ndarray | None = None,
) -> Setup:
    """Build the fixed context. `train` overrides the corpus's own split (V7 uses it)."""
    h = h or Hyper()
    train = corpus.train if train is None else train
    cells = rate_cells(corpus, train, cell_mask)
    test = rate_cells(corpus, ~train, cell_mask)
    ps = Kr.pair_counts(
        corpus.x, corpus.actor, corpus.stamp, cons.pairs, h.kernel, h.delta,
        n_actors=corpus.A, mask=train,
    )
    lower = np.zeros((cons.M, cons.P))
    lower[cons.theory_edges] = h.omega
    # F_k facets for every ideology, laid out contiguously: facet_of[i] is the ideology
    # (1..M, since 0 is the background coordinate of theta) that facet i belongs to.
    facet_of = np.repeat(np.arange(1, cons.M + 1), max(int(h.facets), 1)).astype(np.int32)
    # C4 pins the *ideology* margin and leaves the facets within it free (section 3, C4),
    # so the pinned value is stored on the margin's scale, with the background at 0.
    fixed = np.zeros(corpus.A, dtype=bool)
    fixed_vartheta = np.full((corpus.A, cons.M + 1), np.nan)
    by_id = {a: i for i, a in enumerate(cons.actors)}
    for n, a in enumerate(corpus.actors):
        i = by_id.get(a)
        if i is not None and cons.theta_fix[i]:
            fixed[n] = True
            fixed_vartheta[n] = np.concatenate([[0.0], np.nan_to_num(cons.theta_val[i])])
    cm = cells.mask
    return Setup(
        corpus=corpus, cons=cons, h=h,
        m=background_rates(corpus, train),
        cells=cells, test=test, ps=ps,
        wts=actor_period_weights(corpus, train),
        mu_C1=cons.prior_mean(h.alpha)[facet_of - 1],
        lower_W=lower,
        fixed=fixed, fixed_vartheta=fixed_vartheta,
        pmask=cm[:, cons.pairs[:, 0]] * cm[:, cons.pairs[:, 1]],
        facet_of=facet_of,
    )


# --------------------------------------------------------------------------- #
# composed quantities
# --------------------------------------------------------------------------- #


def theta_of(p: Params, S: Setup) -> np.ndarray:
    """[A, 1+F] membership over (background, facets). Section 2's actor block.

    One flat softmax over the (ideology, facet) coordinates plus a background coordinate,
    with the metadata logit shared by all facets of an ideology: identity informs *which
    ideology*, and the facet an actor works within it is an actor-level draw (C4). So

        logit[a, (k,f)] = (m_a' beta)[k] + gamma[a, (k,f)]

    and beta is indexed by ideology only, broadcast across that ideology's facets.

    C4's pins bind the ideology margin and leave the facets free, which is the flat
    softmax conditioned on the pinned ideologies: renormalise each ideology's facet block
    to the margin the constraint file supplies. Pure pins (1, 0, 0) are the common case and
    fall out of the general form, so a half-and-half pin works without a special case.

    theta[:, 0] is the background share -- the fraction of the actor's speech that is
    unideological. Its profile is pinned to zero (there is no eta row for it), so mass
    there pulls the actor's rate toward m_u + mu_a and nothing else. That is what makes it
    a home for off-profile chatter rather than a component that has to be estimated.
    """
    logits = np.zeros((S.corpus.A, 1 + S.F))
    logits[:, 0] = (S.corpus.meta @ p.beta[:, 0])
    logits[:, 1:] = (S.corpus.meta @ p.beta)[:, S.facet_of]
    logits = logits + p.gamma
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    th = e / e.sum(1, keepdims=True)
    if S.fixed.any():
        # Conditional softmax within each pinned ideology, scaled to its pinned margin.
        want = S.fixed_vartheta[S.fixed]                     # [n_fixed, M+1]
        blk = e[S.fixed] @ S.agg                             # [n_fixed, M+1] block sums
        share = np.divide(want, np.maximum(blk, EPS), out=np.zeros_like(want),
                          where=blk > EPS)
        th = th.copy()
        th[S.fixed] = e[S.fixed] * (share @ S.agg.T)
        tot = th[S.fixed].sum(1, keepdims=True)
        th[S.fixed] = np.divide(th[S.fixed], np.maximum(tot, EPS),
                                out=np.zeros_like(th[S.fixed]), where=tot > EPS)
    return th


def ideology_profile(p: "Params", S: Setup) -> np.ndarray:
    """[M, K] an ideology's profile as the pointwise max over its facets.

    Section 5 defines the support this way -- S_k = {u : max_f eta_{k,f,u} > 0}, "the
    union of its facets' supports" -- and the same reduction is what every per-ideology
    read wants: whether the ideology takes a position on a node, not how its specialists
    divide the labour. Diagnostics that care about the division read p.eta directly.
    """
    return np.stack([p.eta[S.facet_of == k].max(0) if (S.facet_of == k).any()
                     else np.zeros(S.corpus.K) for k in range(1, S.cons.M + 1)])


def vartheta_of(theta: np.ndarray, S: Setup) -> np.ndarray:
    """[A, M+1] the ideology margin of a membership, column 0 the background share.

    The pair channel routes by this rather than by theta: facets of one ideology share
    that ideology's network (section 2), so what a facet contributes to Omega is its
    ideology's edge weight, and only the margin matters.
    """
    return theta @ S.agg


def cell_log_rates(p: Params, S: Setup, theta: np.ndarray, cells: Cells) -> np.ndarray:
    """[G, K] the composed log rate at each (actor, period) cell.

    Section 2, with no actor-by-concept term: the actor enters through their facet
    memberships and a breadth scalar, and nothing else. theta[:, 0] is the background
    coordinate, whose profile is zero, so it is simply absent from the matrix product.
    """
    z = S.m[None, :] + (theta[:, 1:] @ p.eta)[cells.actor] + p.mu[cells.actor][:, None]
    z = z + p.mtc[cells.period][:, S.cidx]
    return np.clip(z, -Z_MAX, Z_MAX)


def period_log_rates(p: Params, S: Setup, theta: np.ndarray) -> np.ndarray:
    """[A, T, K] the composed log rate for every actor in every period.

    The pair channel's exposure needs a time-averaged rate, and averaging has to happen
    on the rate scale rather than the log scale, so the periods cannot be collapsed
    before exponentiating.
    """
    z = S.m[None, None, :] + (theta[:, 1:] @ p.eta)[:, None, :] + p.mu[:, None, None]
    z = z + p.mtc[None, :, :][:, :, S.cidx]
    return np.clip(z, -Z_MAX, Z_MAX)


def actor_rates(p: Params, S: Setup, theta: np.ndarray) -> np.ndarray:
    """[A, K] eta_{a,u}: the time-averaged rate section 2 puts in the pair exposure."""
    return np.einsum("at,atu->au", S.wts, np.exp(period_log_rates(p, S, theta)))


def omega(p: Params, vartheta: np.ndarray) -> np.ndarray:
    """[A, P] Omega_{a,uu'} = W^(0) + sum_k vartheta_ak W_k.

    Takes the *ideology margin*, not the full membership: facets share their ideology's
    network. Column 0 of vartheta is the background share and is dropped -- the shared
    layer W^(0) is the intercept, and including a coordinate that completes the simplex
    would make the design collinear with it.
    """
    return p.W0[None, :] + vartheta[:, 1:] @ p.W


def exposure(p: Params, S: Setup, theta: np.ndarray, r: np.ndarray | None = None
             ) -> np.ndarray:
    """[A, P] N^(delta)_a eta_{a,u} eta_{a,u'} -- the pair channel's offset."""
    r = actor_rates(p, S, theta) if r is None else r
    u, v = S.cons.pairs[:, 0], S.cons.pairs[:, 1]
    return S.ps.N[:, None] * r[:, u] * r[:, v] * S.pmask


# --------------------------------------------------------------------------- #
# SAGE priors: the true penalty, and the quadratic surrogate's weight
# --------------------------------------------------------------------------- #


def sage_weight(dev: np.ndarray, h: Hyper) -> np.ndarray:
    """E[1/tau] under the optimal variational q(tau) -- the surrogate's precision.

    Jeffreys (p(tau) ~ 1/tau) gives 1/dev^2, so the effective penalty is log|dev|: the
    scale mixture makes silence the default rather than a budgeted resource, which is
    section 2's whole reason for using it. The proper Exponential prior gives
    sqrt(2*rate)/|dev|, an l1-like penalty, for when an improper prior is unwanted.
    """
    d2 = dev * dev + h.tau_floor
    if h.tau_prior == "jeffreys":
        return 1.0 / d2
    if h.tau_prior == "exp":
        return np.sqrt(2.0 * h.tau_rate) / np.sqrt(d2)
    raise ValueError(f"unknown tau_prior {h.tau_prior!r}; expected 'jeffreys' or 'exp'")


def sage_penalty(dev: np.ndarray, h: Hyper) -> float:
    """The true penalty the surrogate majorizes, up to a constant."""
    d2 = dev * dev + h.tau_floor
    if h.tau_prior == "jeffreys":
        return float(0.5 * np.log(d2).sum())
    return float(np.sqrt(2.0 * h.tau_rate) * np.sqrt(d2).sum())


def tau_step(p: Params, S: Setup) -> Params:
    """Re-tighten the variational surrogate at the current point (the MM majorize half)."""
    return replace(p, tau_I=sage_weight(p.eta - S.mu_C1, S.h))


def lam_W(p: Params, S: Setup) -> np.ndarray:
    """[M, P] the l1 weight on each ideology edge -- C6's support gate, or flat.

    With the gate on, boundary pairs pay more, so edges are cheaper where profiles are
    elevated and W exerts upward pressure on eta. Section 7 says to report fits with the
    gate on and off and read divergence between them as the gate speaking rather than
    the data; W^(0) always keeps the flat penalty, because culture owes no coherence to
    any support.
    """
    h = S.h
    if not h.gate:
        return np.full((S.M, S.P), h.lam_sp)
    g = 1.0 / (1.0 + np.exp(-np.clip(p.eta, -Z_MAX, Z_MAX)))  # [M, K]
    u, v = S.cons.pairs[:, 0], S.cons.pairs[:, 1]
    return h.lam_out - (h.lam_out - h.lam_in) * g[:, u] * g[:, v]


# --------------------------------------------------------------------------- #
# the objective
# --------------------------------------------------------------------------- #


def couples(S: Setup, stage: Stage) -> bool:
    """Do the rate blocks carry the pair term?

    The write-up admits two readings and they are not equivalent.

    Section 4 writes one objective -- rate terms plus pair terms plus priors -- which
    reads as: every block sees all of it. Section 6, listing the blocks, describes them
    the other way: the eta/mu/m_tc steps are "sparse log-link Poisson regressions" (the
    rate likelihood) and the W-step takes N^(delta)_a eta_au eta_au' as an *exposure*,
    i.e. an offset held fixed. `couple_rates` chooses, and the default is section 6's
    reading because section 4's is degenerate. The argument, since it is the reason for
    a non-obvious default:

    The pair term is  sum_{a,p} [ N_a eta_au eta_au' e^Omega - A_ap Omega ].  Scale every
    log rate down by c and every Omega up by 2c and the first half is unchanged, while
    the second falls by 2c * sum_a A_ap. So along that direction the pair term decreases
    without bound, at rate 2 * sum_{p containing u} sum_a A^(delta)_{a,p} per node. What
    pushes back is the rate likelihood, at rate sum_g X_{g,u}, and the l1 penalty, at
    rate 2 * lam_sp * |{p containing u}|. The pair counts are quadratic in the expression
    counts -- roughly n_a * eta * eta per pair, over P pairs -- so on any corpus with more
    than a handful of nodes per unit the pair side wins and the joint objective is
    unbounded below: eta runs to -infinity and W^(0) absorbs the whole rate layer.
    Observed here on both the synthetic and the real corpus, and it is not a numerical
    artifact -- it is what the composite likelihood of section 4 actually says, and it is
    the point estimation counterpart of the standard-error problem that section 4 does
    flag. `pair_weight` below 1 is the graded version of the same correction.

    Under the default the rate blocks minimize the rate likelihood plus the priors, and
    the W-step minimizes the pair likelihood plus the l1 with the exposure held fixed.
    Each is convex and each descends monotonically in its own objective; the two do not
    share one, which is why `objective` reports the one being descended and the pair term
    is reported alongside rather than folded in. Setting `couple_rates=True` recovers
    section 4 literally, with `pair_weight` small enough to keep it bounded.
    """
    return S.h.couple_rates and (stage.W0 or stage.Wk)


def rate_loss(p: Params, S: Setup, theta: np.ndarray, cells: Cells | None = None) -> float:
    """sum_{g,u} obs * (n_g e^z - X_gu z). Section 4's rate block."""
    cells = S.cells if cells is None else cells
    z = cell_log_rates(p, S, theta, cells)
    return float((cells.obs * (cells.n[:, None] * np.exp(z) - cells.X * z)).sum())


def pair_loss(p: Params, S: Setup, theta: np.ndarray) -> float:
    """sum_{a,p} (E e^Omega - A Omega). Section 4's pair block at timescale delta.

    Scaled by `pair_weight`, which is 1.0 -- the objective exactly as section 4 writes
    it -- unless the analyst says otherwise. The knob exists because section 4 also names
    the reason one might want it: the pair counts are built from the same expressions the
    rate likelihood scores, so the joint objective is a composite likelihood, and it
    carries A*P = 133 x 4285 cells against the rate channel's G*K = 739 x 112. Point
    estimation is unproblematic at weight 1, which is why that is the default; a weight
    below 1 is the crude form of the composite-likelihood correction section 4 requires
    for inference, and it should be reported with any number it changes.
    """
    if not (p.W0.any() or p.W.any()):
        E = exposure(p, S, theta)
        return S.h.pair_weight * float(E.sum())  # Omega = 0: e^Omega = 1, A*Omega = 0
    Om = omega(p, vartheta_of(theta, S))
    E = exposure(p, S, theta)
    return S.h.pair_weight * float(
        (E * np.exp(np.clip(Om, -Z_MAX, Z_MAX)) - S.ps.A * S.pmask * Om).sum())


def facet_smoothing(eta: np.ndarray, S: Setup) -> float:
    """Section 6.3's optional Laplacian tie between the facets of one ideology.

    lam_facet * sum_k sum_{f<f'} ||eta_kf - eta_kf'||^2, which for a block of n facets is
    n * ||eta - mean||^2 summed over the block. Off by default: the facets exist in order
    to differ, and tying them is a way to stop a facet running off after a single actor,
    not a default belief about how ideologies are shaped.
    """
    if not S.h.lam_facet:
        return 0.0
    val = 0.0
    for k in range(1, S.cons.M + 1):
        blk = eta[S.facet_of == k]
        if len(blk) > 1:
            val += len(blk) * float(((blk - blk.mean(0)) ** 2).sum())
    return S.h.lam_facet * val


def facet_smoothing_grad(eta: np.ndarray, S: Setup) -> np.ndarray:
    if not S.h.lam_facet:
        return np.zeros_like(eta)
    g = np.zeros_like(eta)
    for k in range(1, S.cons.M + 1):
        sel = S.facet_of == k
        blk = eta[sel]
        if len(blk) > 1:
            g[sel] = 2.0 * S.h.lam_facet * len(blk) * (blk - blk.mean(0))
    return g


def rate_priors(p: Params, S: Setup, stage: Stage) -> float:
    """The priors on the rate-layer parameters: SAGE on the facet profiles, Gaussian else."""
    h = S.h
    val = sage_penalty(p.eta - S.mu_C1, h)
    val += facet_smoothing(p.eta, S)
    val += 0.5 * float((p.mu ** 2).sum()) / h.sigma_mu ** 2
    val += 0.5 * float((p.gamma ** 2).sum()) / h.sigma_gamma ** 2
    val += 0.5 * float((p.beta ** 2).sum()) / h.sigma_beta ** 2
    if stage.mtc:
        val += 0.5 * float((p.mtc ** 2).sum()) / h.sigma_m ** 2
    return val


def w_priors(p: Params, S: Setup, stage: Stage) -> float:
    """The l1 terms: Exp(lam_sp) on W^(0), and C6's gate or a flat penalty on W_k."""
    val = S.h.lam_sp * float(p.W0.sum()) if stage.W0 else 0.0
    if stage.Wk:
        val += float((lam_W(p, S) * p.W).sum())
    return val


def prior_loss(p: Params, S: Setup, stage: Stage) -> float:
    return rate_priors(p, S, stage) + w_priors(p, S, stage)


def objective(p: Params, S: Setup, stage: Stage, theta: np.ndarray | None = None) -> float:
    """The objective the fit is actually descending, with the true SAGE penalty.

    Under section 4's coupled reading that is the whole composite objective. Under the
    default uncoupled one it is the rate half -- rate likelihood plus rate priors -- which
    is the part every block shares: the rate blocks lower it and the W-step leaves it
    untouched, so it is monotone across the iteration. The pair half descends separately
    inside the W-step and is reported next to it rather than added, because summing two
    quantities that no single procedure minimizes would produce a number that rises for
    reasons that are not errors. See `couples`.
    """
    theta = theta_of(p, S) if theta is None else theta
    val = rate_loss(p, S, theta) + rate_priors(p, S, stage)
    if couples(S, stage):
        val += pair_loss(p, S, theta) + w_priors(p, S, stage)
    return val


def held_out(p: Params, S: Setup, theta: np.ndarray | None = None) -> float:
    """Poisson loss on the held-out (actor, period) cells.

    No marginalisation and no correction: v3.1 has no free per-unit parameter, so a
    held-out unit is scored under rates fitted entirely from other units. That is a
    simplification the previous version could not have -- its per-unit responsibilities
    were fitted to held-out rows too, and had to be integrated out to stop the leak.
    """
    theta = theta_of(p, S) if theta is None else theta
    return rate_loss(p, S, theta, S.test)


# --------------------------------------------------------------------------- #
# projections
# --------------------------------------------------------------------------- #


def project_expsum(v: np.ndarray, m: np.ndarray, pi_bar: float, tol: float = 1e-12
                   ) -> np.ndarray:
    """Euclidean projection onto {z : sum_u e^{m_u + z_u} <= pi_bar}. C3's cap.

    The KKT conditions give z = v - lam e^{m + z} for a single multiplier lam >= 0.
    Writing y = m + z and b = m + v this is y + lam e^y = b, whose solution is
    y = b - W(lam e^b) with W the Lambert function -- and scipy's `wrightomega` solves
    exactly w + log w = x, i.e. w = W(e^x), so with t = log lam,

        w = wrightomega(t + b),   y = b - w,   g(t) = sum_u e^{y_u} - pi_bar.

    That leaves a scalar root-find in t, and g is smooth and strictly decreasing:
    w + log w = t + b gives dw/dt = w/(1 + w), so g'(t) = -sum_u e^{y_u} w_u/(1 + w_u).
    So it is solved by Newton inside a bracket rather than by bisection -- worth the
    derivative, because this is the innermost loop of the whole fit (the eta-step calls
    it once per incompatible set per backtracking trial, hundreds of thousands of times
    per stage) and Newton gets there in a handful of steps where bisection needed eighty.
    The bracket is kept and stepped into whenever Newton would leave it, so the solve
    cannot diverge on a badly scaled b.
    """
    b = m + v
    hi_b = b.max()
    lse = hi_b + np.log(np.exp(b - hi_b).sum())
    if lse <= np.log(pi_bar):
        return v  # already feasible; in log space, because b can be large mid-line-search

    def gval(t):
        w = wrightomega(t + b)
        e = np.exp(b - w)
        return float(e.sum()) - pi_bar, w, e

    # bracket: g decreases in t, and g(t) -> sum e^b - pi_bar > 0 as t -> -inf
    lo, hi = 0.0, 0.0
    g_hi, _, _ = gval(hi)
    while g_hi > 0:
        lo, hi = hi, hi + 4.0
        g_hi, _, _ = gval(hi)
        if hi > 400.0:
            break
    g_lo, _, _ = gval(lo)
    while g_lo < 0:
        hi, lo = lo, lo - 4.0
        g_lo, _, _ = gval(lo)
        if lo < -400.0:
            break

    t = 0.5 * (lo + hi)
    for _ in range(60):
        g, w, e = gval(t)
        # `hi` is kept on the feasible side and is what the answer is read off, so a
        # residual within tolerance counts as feasible. The eta-step accepts a trial only
        # if it descends, and a trial outside the feasible set would make that guarantee
        # meaningless -- but insisting on g <= 0 exactly would discard a converged Newton
        # step whose residual is 1e-16 and fall back to the bracket's far end.
        if g <= tol * pi_bar:
            hi = t
        else:
            lo = t
        if abs(g) <= tol * pi_bar:
            break
        deriv = -float((e * w / (1.0 + w)).sum())
        step = t - g / deriv if deriv < 0 else t
        t = step if lo < step < hi else 0.5 * (lo + hi)
    return b - wrightomega(hi + b) - m


def caps_slack(v: np.ndarray, m: np.ndarray, pi_bar: float, sets: list[np.ndarray]) -> bool:
    """True when every C3 cap already holds, so the projection is the identity.

    Worth its own pass: the caps bind rarely once eta has settled, and Dykstra over 51
    overlapping sets is the most expensive thing in the eta-step's line search.
    """
    for S in sets:
        b = m[S] + v[S]
        hi = b.max()
        if hi + np.log(np.exp(b - hi).sum()) > np.log(pi_bar):
            return False
    return True


def project_caps(
    v: np.ndarray, m: np.ndarray, pi_bar: float, sets: list[np.ndarray], n_iter: int = 30
) -> np.ndarray:
    """Projection onto the intersection of the C3 caps, by Dykstra.

    A node sits in several maximal cliques at once, so the sets overlap and there is no
    closed form; plain alternating projection would converge to a point in the
    intersection but not to the projection, and the block steps need the projection for
    their descent guarantee.
    """
    if not sets or caps_slack(v, m, pi_bar, sets):
        return v
    x = v.copy()
    # Corrections are held on each set's own indices rather than as full-length vectors:
    # the sets have two or three members out of 112, and this is called once per
    # backtracking trial in the eta-step.
    corr = [np.zeros(len(S)) for S in sets]
    for _ in range(n_iter):
        moved = 0.0
        for i, S in enumerate(sets):
            y = x[S] + corr[i]
            new = project_expsum(y, m[S], pi_bar)
            corr[i] = y - new
            moved = max(moved, float(np.abs(x[S] - new).max()))
            x[S] = new
        if moved < 1e-11:
            break
    return x


def descend(x0, f, g, project, n_iter: int, tol: float, step: float = 1.0):
    """Projected gradient with backtracking, accepting only steps that descend.

    Monotonicity is the point: block-coordinate descent converges only if no block ever
    increases the objective, and it is also the cheapest test that a gradient or a
    projection is wrong. If backtracking cannot find a descent step, stop rather than
    accept an uphill one.
    """
    x, base = x0.copy(), f(x0)
    for _ in range(n_iter):
        grad = g(x)
        for _ in range(40):
            trial = project(x, step, grad)
            val = f(trial)
            if val <= base:
                break
            step *= 0.5
        else:
            break
        improved = base - val
        x, base = trial, val
        step *= 1.5
        if improved <= tol * max(1.0, abs(base)):
            break
    return x


# --------------------------------------------------------------------------- #
# shared gradient machinery
# --------------------------------------------------------------------------- #


def _rate_residual(p, S, theta, cells) -> np.ndarray:
    """[G, K] obs * (n_g e^z - X_gu): the Poisson residual every rate block shares."""
    z = cell_log_rates(p, S, theta, cells)
    return cells.obs * (cells.n[:, None] * np.exp(z) - cells.X)


def _pair_node_pressure(p, S, theta, r=None) -> np.ndarray:
    """[A, K] sum over pairs containing u of the predicted pair count.

    d(pair loss)/d log eta_{a,u} = sum_{p containing u} E_{ap} e^{Omega_ap}, because the
    observed term A*Omega does not involve the rate. Everything the pair channel says
    about a rate parameter flows through this array -- it is the "W exerts upward
    pressure on eta" coupling of section 3, made explicit.
    """
    r = actor_rates(p, S, theta) if r is None else r
    pred = exposure(p, S, theta, r) * np.exp(
        np.clip(omega(p, vartheta_of(theta, S)), -Z_MAX, Z_MAX))
    u, v = S.cons.pairs[:, 0], S.cons.pairs[:, 1]
    out = np.zeros((S.corpus.A, S.corpus.K))
    np.add.at(out.T, u, pred.T)
    np.add.at(out.T, v, pred.T)
    return S.h.pair_weight * out


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #


def eta_step(p: Params, S: Setup, stage: Stage, n_iter: int | None = None) -> Params:
    """Sparse log-link Poisson regression for the ideology profiles, under C3's caps."""
    h = S.h
    sets = S.cons.incompat

    def f(eta):
        q = replace(p, eta=eta)
        th = theta_of(q, S)
        val = rate_loss(q, S, th)
        if couples(S, stage):
            val += pair_loss(q, S, th)
        dev = eta - S.mu_C1
        return (val + 0.5 * float((p.tau_I * dev * dev).sum())
                + facet_smoothing(eta, S))

    def g(eta):
        q = replace(p, eta=eta)
        th = theta_of(q, S)
        resid = _rate_residual(q, S, th, S.cells)
        grad = th[S.cells.actor][:, 1:].T @ resid
        if couples(S, stage):
            grad += th[:, 1:].T @ _pair_node_pressure(q, S, th)
        return grad + p.tau_I * (eta - S.mu_C1) + facet_smoothing_grad(eta, S)

    def project(x, step, grad):
        out = x - step * grad
        if not sets:
            return out
        return np.stack([project_caps(row, S.m, h.pi_bar, sets, h.cap_iter) for row in out])

    return replace(p, eta=descend(p.eta, f, g, project, n_iter or h.inner_iter, h.tol))


def mu_step(p: Params, S: Setup, stage: Stage, n_iter: int | None = None) -> Params:
    """Actor breadth, now a rate parameter rather than a prior mean.

    In v3.1 mu_a was the mean of the actor's concept deviations eps and had a closed-form
    precision-weighted update. v3.2 deletes eps, so mu is the whole actor layer besides
    membership: one scalar per actor, shifting every one of their rates together. It is
    verbosity, not content -- how much this actor says, as against what they say, which is
    entirely carried by the facet memberships.
    """
    h = S.h

    def f(mu):
        q = replace(p, mu=mu)
        th = theta_of(q, S)
        val = rate_loss(q, S, th)
        if couples(S, stage):
            val += pair_loss(q, S, th)
        return val + 0.5 * float((mu ** 2).sum()) / h.sigma_mu ** 2

    def g(mu):
        q = replace(p, mu=mu)
        th = theta_of(q, S)
        grad = np.zeros_like(mu)
        np.add.at(grad, S.cells.actor, _rate_residual(q, S, th, S.cells).sum(1))
        if couples(S, stage):
            grad += _pair_node_pressure(q, S, th).sum(1)
        return grad + mu / h.sigma_mu ** 2

    project = lambda x, step, grad: x - step * grad
    return replace(p, mu=descend(p.mu, f, g, project, n_iter or h.inner_iter, h.tol))


def mtc_step(p: Params, S: Setup, stage: Stage, n_iter: int | None = None) -> Params:
    """Temporal offsets: the news cycle, shared across actors.

    Section 8 puts this in last on purpose. It is the one facet with no theoretical
    content, existing purely to protect the others' interpretation, and what it protects
    is delta > 0: without it a uniform event surge is absorbed by W^(0) (silently
    redefining culture to include the news) and a camp-differential surge is written into
    W_k as an edge that was merely co-topical for a week.
    """
    h = S.h

    def f(mtc):
        q = replace(p, mtc=mtc)
        th = theta_of(q, S)
        val = rate_loss(q, S, th)
        if couples(S, stage):
            val += pair_loss(q, S, th)
        return val + 0.5 * float((mtc ** 2).sum()) / h.sigma_m ** 2

    def g(mtc):
        q = replace(p, mtc=mtc)
        th = theta_of(q, S)
        resid = _rate_residual(q, S, th, S.cells)
        grad = np.zeros_like(mtc)
        np.add.at(grad, S.cells.period, resid @ S.onehot)
        if couples(S, stage):
            # d eta_{a,u} / d m_{t,c} = wts[a,t] e^{z_{a,t,u}}, so the pair pressure on
            # a log rate has to be re-weighted by that period's share of the rate.
            rt = np.exp(period_log_rates(q, S, th))  # [A, T, K]
            r = np.einsum("at,atu->au", S.wts, rt)
            share = S.wts[:, :, None] * rt / np.maximum(r[:, None, :], EPS)
            press = _pair_node_pressure(q, S, th, r)  # [A, K]
            grad += np.einsum("atu,au,uc->tc", share, press, S.onehot)
        return grad + mtc / h.sigma_m ** 2

    project = lambda x, step, grad: x - step * grad
    return replace(p, mtc=descend(p.mtc, f, g, project, n_iter or h.inner_iter, h.tol))


def align_step(p: Params, S: Setup, stage: Stage, n_iter: int | None = None) -> Params:
    """The membership block: beta and gamma jointly, through the (1+F)-way softmax.

    Sections 6.1-6.2 split this into a membership step and a ridge multinomial regression;
    they share one objective and one softmax Jacobian, so they are done in one gradient
    block rather than iterated against each other. beta carries one column per ideology
    (plus the background) and is broadcast across that ideology's facets, which is what
    makes identity inform *which ideology* and leave the facet to the actor-level draw
    gamma -- so that specialization is never read as identity (C4).

    Pinned actors contribute no gradient to the coordinates their pin determines. The pin
    binds the ideology margin only, so their facet split is still free and still learned;
    what is held is how much of them belongs to each ideology.
    """
    h = S.h
    J, Mp1, A, F = p.beta.shape[0], S.cons.M + 1, S.corpus.A, S.F

    def unpack(vec):
        return vec[: J * Mp1].reshape(J, Mp1), vec[J * Mp1:].reshape(A, 1 + F)

    def f(vec):
        beta, gamma = unpack(vec)
        q = replace(p, beta=beta, gamma=gamma)
        th = theta_of(q, S)
        val = rate_loss(q, S, th)
        if couples(S, stage):
            val += pair_loss(q, S, th)
        val += 0.5 * float((beta ** 2).sum()) / h.sigma_beta ** 2
        val += 0.5 * float((gamma ** 2).sum()) / h.sigma_gamma ** 2
        return val

    def g(vec):
        beta, gamma = unpack(vec)
        q = replace(p, beta=beta, gamma=gamma)
        th = theta_of(q, S)
        # d/d theta_{a,(k,f)}: the rate term sees the facet profile, the pair term sees
        # only the ideology margin, so the latter is spread back over the facets by agg.
        resid = _rate_residual(q, S, th, S.cells)
        vth_all = vartheta_of(th, S)
        gt = np.zeros((A, 1 + F))
        np.add.at(gt[:, 1:], S.cells.actor, resid @ p.eta.T)
        if couples(S, stage):
            vth = vth_all
            r = actor_rates(q, S, th)
            pred = exposure(q, S, th, r) * np.exp(np.clip(omega(p, vth), -Z_MAX, Z_MAX))
            gv = np.zeros((A, Mp1))
            gv[:, 1:] = S.h.pair_weight * ((pred - S.ps.A * S.pmask) @ p.W.T)
            gt += gv @ S.agg.T
            gt[:, 1:] += _pair_node_pressure(q, S, th, r) @ p.eta.T
        # The softmax Jacobian, which is block-diagonal for pinned actors. A pin fixes
        # the ideology margin and leaves the facets free, so a pinned actor's membership
        # is a *conditional* softmax within each ideology block, scaled to the pinned
        # margin -- and its Jacobian normalises within the block rather than over all
        # 1+F coordinates. Zeroing the whole row instead would throw away the facet
        # gradient the pin does not constrain.
        #
        # beta needs no separate mask: it shifts every logit in an ideology block by the
        # same amount, which a within-block conditional softmax is invariant to, so the
        # block form already sends a pinned actor's beta gradient to exactly zero.
        w = th * gt
        free = th * (gt - w.sum(1, keepdims=True))
        blk = np.divide(w @ S.agg, np.maximum(vth_all, EPS), out=np.zeros_like(vth_all),
                        where=vth_all > EPS)
        pinned = th * (gt - blk @ S.agg.T)
        gs = np.where(S.fixed[:, None], pinned, free)
        gbeta = np.zeros_like(beta)
        gbeta[:, 0] = S.corpus.meta.T @ gs[:, 0]
        np.add.at(gbeta.T, S.facet_of, (S.corpus.meta.T @ gs[:, 1:]).T)
        gbeta += beta / h.sigma_beta ** 2
        ggamma = gs + gamma / h.sigma_gamma ** 2
        return np.concatenate([gbeta.ravel(), ggamma.ravel()])

    project = lambda x, step, grad: x - step * grad
    vec = np.concatenate([p.beta.ravel(), p.gamma.ravel()])
    out = descend(vec, f, g, project, n_iter or h.inner_iter, h.tol)
    beta, gamma = unpack(out)
    return replace(p, beta=beta, gamma=gamma)


def w_step(p: Params, S: Setup, stage: Stage, n_sweeps: int = 8) -> Params:
    """The pair channel: a nonnegative l1-penalized Poisson GLM over allowed pairs.

    Section 6.4 describes it as one regression jointly over W^(0) and {W_k} with
    kernel-weighted counts as observations, N^(delta)_a eta_au eta_au' as exposure, and
    (1, theta_a) as the design. The design is the same for every pair, so the problem
    is P independent (M+1)-dimensional convex programs sharing a design matrix, and a
    coordinate sweep updates all of them at once: one damped Newton step per coordinate,
    vectorised across pairs, with the nonnegativity and C2's omega imposed by a clip
    (exact, because the feasible set is a box and the coordinate step is separable).

    That is what makes the culture/ideology split an estimation property rather than a
    definition: coordinate 0 is an intercept fitted from all actors, and the rest are
    alignment covariates, so W_k carries only association that covaries with theta.
    """
    h = S.h
    th = theta_of(p, S)
    # The design is an intercept plus the *ideology margins*: facets share their
    # ideology's network, so a facet's contribution to Omega is its ideology's edge.
    Z = np.column_stack([np.ones(S.corpus.A), vartheta_of(th, S)[:, 1:]])  # [A, M+1]
    E = exposure(p, S, th)  # [A, P]
    Aobs = S.ps.A * S.pmask  # [A, P]
    lam = np.vstack([np.full((1, S.P), h.lam_sp), lam_W(p, S)])  # [M+1, P]
    lo = np.vstack([np.zeros((1, S.P)), S.lower_W])  # [M+1, P]
    hi = np.full_like(lo, np.inf)
    free = np.array([stage.W0] + [stage.Wk] * S.M)

    b = np.vstack([p.W0[None, :], p.W])
    # Only the free coordinates are made feasible: a gated-off W_k stays at whatever the
    # stage pinned it to (zero at 3a), rather than being lifted to C2's floor behind the
    # stage's back.
    b = np.where(free[:, None], np.clip(b, lo, hi), b)

    wt = h.pair_weight

    def obj(bb):
        Om = np.clip(Z @ bb, -Z_MAX, Z_MAX)
        return wt * (E * np.exp(Om) - Aobs * Om).sum(0) + (lam * bb).sum(0)  # [P]

    val = obj(b)
    for _ in range(n_sweeps):
        moved = 0.0
        for j in range(S.M + 1):
            if not free[j]:
                continue
            Om = np.clip(Z @ b, -Z_MAX, Z_MAX)
            pred = E * np.exp(Om)
            grad = wt * (Z[:, j] @ (pred - Aobs)) + lam[j]
            hess = wt * ((Z[:, j] ** 2) @ pred)
            step = np.where(hess > EPS, grad / np.maximum(hess, EPS), 0.0)
            scale = np.ones(S.P)
            for _ in range(30):
                trial = b.copy()
                trial[j] = np.clip(b[j] - scale * step, lo[j], hi[j])
                tv = obj(trial)
                bad = tv > val + 1e-12
                if not bad.any():
                    break
                scale = np.where(bad, scale * 0.5, scale)
            moved = max(moved, float(np.abs(trial[j] - b[j]).max()))
            b, val = trial, np.minimum(tv, val)
        if moved < 1e-10:
            break
    return replace(p, W0=b[0], W=b[1:])


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #


def initialize(S: Setup) -> Params:
    """Start at the background: every profile zero, membership near-uniform.

    Section 2 makes this the honest start rather than a convenience -- the all-zeros
    deviation *is* the background, so a fit that never moves off it is the zero-ideology
    null rather than a failed optimisation. The C1 prior means break the symmetry between
    *ideologies*, entering through mu_C1 in the very first eta-step.

    They cannot break the symmetry between the *facets* of one ideology: C1 is a property
    of the ideology and every facet of k shares mu_C1[k], so facets are exchangeable and a
    perfectly symmetric start would leave them identical forever. Both jitters below exist
    to break it, and the one on eta is the one that does the work.

    Jittering gamma alone is not enough, which is worth recording because it looks as
    though it should be. With eta at zero the rate does not depend on membership at all,
    so the only force on gamma in the first membership step is its own N(0, sigma^2)
    prior -- which drives it to exactly zero, making theta exactly uniform across the
    facets of an ideology, at which point their eta gradients are identical and they are
    welded together for the rest of the fit. The membership block runs before the profile
    block, so it erases the tie-break before anything can use it. Jittering eta puts the
    asymmetry where the exchangeability actually is.
    """
    rng = np.random.default_rng(S.h.seed)
    A, K, C_, T, F = S.corpus.A, S.corpus.K, S.corpus.C, S.corpus.T, S.F
    J = S.corpus.meta.shape[1]
    p = Params(
        eta=0.01 * rng.standard_normal((F, K)),
        mu=np.zeros(A),
        mtc=np.zeros((T, C_)),
        beta=np.zeros((J, S.cons.M + 1)),
        gamma=0.01 * rng.standard_normal((A, 1 + F)),
        W0=np.zeros(S.P),
        W=S.lower_W.copy(),
        tau_I=np.ones((F, K)),
    )
    return tau_step(p, S)


def fit(
    S: Setup,
    stage: Stage,
    p: Params | None = None,
    verbose: bool = True,
) -> tuple[Params, dict]:
    """Block-coordinate descent at one stage of the build order.

    Order within an iteration: the tau-step first (it majorizes the SAGE penalty at the
    current point, so every block below is minimizing a valid surrogate), then membership,
    then the rate layers, then W. The reported objective is the true penalized one, and
    it must never rise -- that is the check that a gradient or a projection is right.
    """
    h = S.h
    p = initialize(S) if p is None else p.copy()
    if not stage.mtc:
        p = replace(p, mtc=np.zeros_like(p.mtc))
    if not stage.W0:
        p = replace(p, W0=np.zeros_like(p.W0))
    if not stage.Wk:
        # Exactly zero, not C2's floor: section 8 reads the movement between stage 3a and
        # stage 3b as the association mass that is genuinely ideological, and a W_k pinned
        # at omega would put theory's own links into the "before" fit and hide them.
        p = replace(p, W=np.zeros_like(p.W))

    history, prev = [], None
    for it in range(h.max_iter):
        p = tau_step(p, S)
        p = align_step(p, S, stage)
        p = eta_step(p, S, stage)
        p = mu_step(p, S, stage)
        if stage.mtc:
            p = mtc_step(p, S, stage)
        if stage.W0 or stage.Wk:
            p = w_step(p, S, stage)

        val = objective(p, S, stage)
        history.append(val)
        if verbose and (it % 10 == 0 or it == h.max_iter - 1):
            print(f"  iter {it:3d}  objective {val:,.2f}")
        if prev is not None and abs(prev - val) <= h.tol * max(1.0, abs(prev)):
            break
        prev = val

    th = theta_of(p, S)
    ho = held_out(p, S, th)
    if verbose:
        print(f"  converged in {len(history)} iters; train {history[-1]:,.2f}  "
              f"held-out {ho:,.2f}")
    return p, {
        "stage": stage.name, "history": history, "held_out": ho, "kernel": S.ps.label,
        "train_rate": rate_loss(p, S, th),
        "train_pair": (pair_loss(p, S, th) + w_priors(p, S, stage)
                       if (stage.W0 or stage.Wk) else 0.0),
        "priors": prior_loss(p, S, stage),
        "coupled": couples(S, stage),
        "held_out_units": float(S.test.n.sum()),
    }


def fit_stages(
    S: Setup,
    stages: tuple[str, ...] = ("1", "3a", "3b"),
    p: Params | None = None,
    verbose: bool = True,
) -> tuple[Params, list[dict]]:
    """Section 8's build order, each stage warm-started from the last.

    Warm-starting is not a convenience: the pair channel is far more flexible than C1's
    prior means are strong, so from a cold start stage 3 can settle into a relabeled or
    flattened solution. Entering it from a converged stage-2 fit gives eta and theta the
    chance to be right before W is allowed to explain anything.

    Stages 3a and 3b are kept separate because the movement between them *is* an
    estimand: 3a fits the shared association layer alone, 3b adds the per-ideology
    networks, and the difference is the association mass that is genuinely ideological.
    """
    infos = []
    for name in stages:
        if name not in STAGES:
            raise ValueError(f"unknown stage {name!r}; expected {list(STAGES)}")
        st = STAGES[name]
        if verbose:
            print(f"stage {st.name}:")
        p, info = fit(S, st, p=p, verbose=verbose)
        infos.append(info)
    if verbose:
        print("\nheld-out by stage: " + "  ".join(
            f"{i['stage'].split()[0]}: {i['held_out']:,.1f}" for i in infos))
    return p, infos
