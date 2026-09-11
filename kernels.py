"""The co-occurrence channel of section 2: kernel-weighted pair counts at timescale delta.

For each actor a and pair (u, u') the model observes

    A^(delta)_{a,uu'} = sum_{d,d'} K_delta(t(d) - t(d')) x_{d,u} x_{d',u'}
    N^(delta)_a       = sum_{d,d'} K_delta(t(d) - t(d'))

where d, d' both range over that actor's units. The timescale is the analyst's
variable and the endpoints are the two familiar readings of "co-occurrence":

    same   K = 1[d = d']      co-occurrence within one expression
    exp    K = e^{-|dt|/delta}   decay: episodes at scale delta
    box    K = 1[|dt| <= delta]  a hard window
    all    K = 1              co-occurrence within an actor, ever (the ResIN /
                              BENDING endpoint, where W is identified across actors
                              rather than within them)

The point of this module is that none of these needs the O(n^2) double sum. Write
X_a for the actor's [n_a, K] block of expressions, sorted by time. Then for any kernel
that factorises over a causal pass,

    A_a = X_a' S_a + S_a' X_a + X_a' X_a

where S_a[d] = sum_{d' < d} K(t_d - t_d') x_{d'} is a running backward sum: the three
terms are d' < d, d' > d, and d' = d. S is built by one pass over the actor's units --
for `exp` a two-term recursion, for `box` a two-pointer prefix sum -- and the pair
matrix then falls out of two [n_a, K] matmuls. Cost is O(D*K + A*K^2) rather than
O(sum_a n_a^2).

`same` and `all` are the degenerate cases of the same formula (S = 0 and S = the full
prefix sum), and they are written out directly because they are exact and cheap:
A = X'X and A = n n' respectively.

A note on the diagonal. The definition above is a double sum over all ordered pairs
(d, d'), so it includes d = d', and N is n_a^2 at the K = 1 endpoint. Section 2's
large-delta display writes the identity with the diagonal removed,
A^(inf) = n_u n_u' - sum_d x_du x_du'; `exclude_self=True` reproduces that convention.
The default follows the definition, since that is what the likelihood's exposure
N^(delta)_a eta_au eta_au' is normalised against.
"""

from dataclasses import dataclass

import numpy as np

KERNELS = ("same", "exp", "box", "all")


@dataclass(frozen=True)
class PairStats:
    """Kernel-weighted pair counts, and the kernel that produced them.

    C7 requires the kernel and timescale to be stated with every reported network, so
    they travel with the counts rather than being remembered by the caller.
    """

    A: np.ndarray  # [A, P] kernel-weighted pair counts on the allowed pairs
    N: np.ndarray  # [A] kernel mass, the exposure's leading factor
    kernel: str
    delta: float
    exclude_self: bool = False

    @property
    def label(self) -> str:
        if self.kernel in ("same", "all"):
            return self.kernel
        return f"{self.kernel}(delta={self.delta:g}d)"


def _causal_sums(x: np.ndarray, t: np.ndarray, kernel: str, delta: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """S[d] = sum_{d' < d} K(t_d - t_d') x_{d'}, and its scalar counterpart s[d].

    Units must arrive sorted by t. Returns ([n, K], [n]).
    """
    n, K = x.shape
    S = np.zeros((n, K))
    s = np.zeros(n)
    if kernel == "exp":
        # S[d] = e^{-(t_d - t_{d-1})/delta} (S[d-1] + x[d-1]); same recursion for s
        # with x = 1. One pass, exact, and numerically harmless because the decay
        # factor is in (0, 1].
        acc = np.zeros(K)
        acc_s = 0.0
        for d in range(1, n):
            decay = np.exp(-(t[d] - t[d - 1]) / delta)
            acc = decay * (acc + x[d - 1])
            acc_s = decay * (acc_s + 1.0)
            S[d] = acc
            s[d] = acc_s
    elif kernel == "box":
        # Prefix sums plus a left pointer at the first d' with t_d - t_d' <= delta.
        cx = np.vstack([np.zeros((1, K)), np.cumsum(x, axis=0)])
        lo = 0
        for d in range(n):
            while t[d] - t[lo] > delta:
                lo += 1
            S[d] = cx[d] - cx[lo]
            s[d] = d - lo
    else:
        raise ValueError(f"_causal_sums does not handle kernel {kernel!r}")
    return S, s


def pair_counts(
    x: np.ndarray,
    actor: np.ndarray,
    stamp: np.ndarray,
    pairs: np.ndarray,
    kernel: str = "same",
    delta: float = 0.0,
    n_actors: int | None = None,
    mask: np.ndarray | None = None,
    exclude_self: bool = False,
) -> PairStats:
    """A^(delta) and N^(delta) per actor, on the allowed pairs.

    `stamp` is the clock the kernel reads, in days. `mask` selects units, so the pair
    channel is built from the training units only, exactly like the rate channel.
    """
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")
    if kernel in ("exp", "box") and not delta > 0:
        raise ValueError(f"kernel {kernel!r} needs delta > 0; use 'same' for delta = 0")
    if mask is not None:
        x, actor, stamp = x[mask], actor[mask], stamp[mask]
    A = int(actor.max()) + 1 if n_actors is None else n_actors
    K = x.shape[1]
    u, v = pairs[:, 0], pairs[:, 1]

    out = np.zeros((A, len(pairs)))
    N = np.zeros(A)
    for a in range(A):
        rows = np.flatnonzero(actor == a)
        if not len(rows):
            continue
        order = rows[np.argsort(stamp[rows], kind="stable")]
        Xa, ta = x[order], stamp[order]
        na = len(order)
        self_term = Xa.T @ Xa  # d = d'
        if kernel == "same":
            full, mass = self_term, float(na)
        elif kernel == "all":
            n_u = Xa.sum(0)
            full, mass = np.outer(n_u, n_u), float(na) ** 2
        else:
            S, s = _causal_sums(Xa, ta, kernel, delta)
            cross = Xa.T @ S
            full = cross + cross.T + self_term
            mass = 2.0 * float(s.sum()) + float(na)
        if exclude_self:
            full = full - self_term
            mass = mass - float(na)
        out[a] = full[u, v]
        N[a] = mass
    return PairStats(A=out, N=N, kernel=kernel, delta=float(delta),
                     exclude_self=exclude_self)


def brute_force(
    x: np.ndarray,
    actor: np.ndarray,
    stamp: np.ndarray,
    pairs: np.ndarray,
    kernel: str = "same",
    delta: float = 0.0,
    n_actors: int | None = None,
    exclude_self: bool = False,
) -> PairStats:
    """The O(n^2) definition, for tests. Never call this on the real corpus."""
    A = int(actor.max()) + 1 if n_actors is None else n_actors
    u, v = pairs[:, 0], pairs[:, 1]
    out = np.zeros((A, len(pairs)))
    N = np.zeros(A)
    for a in range(A):
        rows = np.flatnonzero(actor == a)
        full = np.zeros((x.shape[1], x.shape[1]))
        mass = 0.0
        for d in rows:
            for e in rows:
                if exclude_self and d == e:
                    continue
                dt = abs(stamp[d] - stamp[e])
                if kernel == "same":
                    w = 1.0 if d == e else 0.0
                elif kernel == "all":
                    w = 1.0
                elif kernel == "exp":
                    w = float(np.exp(-dt / delta))
                else:
                    w = 1.0 if dt <= delta else 0.0
                full += w * np.outer(x[d], x[e])
                mass += w
        out[a] = full[u, v]
        N[a] = mass
    return PairStats(A=out, N=N, kernel=kernel, delta=float(delta),
                     exclude_self=exclude_self)
