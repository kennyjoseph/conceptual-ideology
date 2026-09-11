"""The co-occurrence channel, against the O(n^2) definition it is a shortcut for.

kernels.py exists so that A^(delta) never costs a double sum over an actor's units.
Everything else in the model consumes its output as data, so an error here is silent
downstream -- which is why every kernel is checked against `brute_force`, the literal
transcription of section 2's definition, rather than against itself.
"""

import itertools

import numpy as np
import pytest

import kernels as Kr

CASES = [("same", 0.0), ("all", 0.0), ("exp", 3.0), ("exp", 0.05), ("exp", 5e3),
         ("box", 2.0), ("box", 1e4)]


def toy(D=45, K=6, A=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((D, K)) * (rng.random((D, K)) < 0.4)
    actor = rng.integers(0, A, D)
    # deliberately unsorted, and with ties: the causal recursions sort internally and
    # must handle equal timestamps, which a real corpus has plenty of.
    stamp = np.round(rng.random(D) * 25, 1)
    pairs = np.array(list(itertools.combinations(range(K), 2)), dtype=np.int32)
    return x, actor, stamp, pairs, A


@pytest.mark.parametrize("kernel,delta", CASES)
@pytest.mark.parametrize("exclude_self", [False, True])
def test_matches_the_definition(kernel, delta, exclude_self):
    x, actor, stamp, pairs, A = toy()
    fast = Kr.pair_counts(x, actor, stamp, pairs, kernel, delta, A,
                          exclude_self=exclude_self)
    slow = Kr.brute_force(x, actor, stamp, pairs, kernel, delta, A,
                          exclude_self=exclude_self)
    assert np.allclose(fast.A, slow.A)
    assert np.allclose(fast.N, slow.N)


def test_endpoints_are_the_limits_of_the_decay_kernel():
    """Section 2's two endpoints are the limits of the same channel, not separate models.

    Distinct timestamps, deliberately: with ties the delta -> 0 limit is *not* `same`,
    because two units posted at the same instant have K = 1 at every delta. See
    test_simultaneous_units_are_not_the_same_unit.
    """
    x, actor, stamp, pairs, A = toy()
    stamp = np.sort(np.unique(np.linspace(0, 25, len(x))))
    same = Kr.pair_counts(x, actor, stamp, pairs, "same", 0.0, A)
    tiny = Kr.pair_counts(x, actor, stamp, pairs, "exp", 1e-6, A)
    assert np.allclose(same.A, tiny.A) and np.allclose(same.N, tiny.N)

    ever = Kr.pair_counts(x, actor, stamp, pairs, "all", 0.0, A)
    huge = Kr.pair_counts(x, actor, stamp, pairs, "exp", 1e9, A)
    assert np.allclose(ever.A, huge.A) and np.allclose(ever.N, huge.N)


def test_simultaneous_units_are_not_the_same_unit():
    """A tie in time is co-occurrence at any delta; being one expression is not.

    Worth pinning down because the two collapse in the delta -> 0 limit only when the
    clock separates every unit. On a real corpus timestamps repeat, so `same` and a very
    small delta are different measurements and the difference is exactly the tied pairs.
    """
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    actor = np.zeros(2, dtype=int)
    pairs = np.array([[0, 1]], dtype=np.int32)
    tied = np.zeros(2)
    assert Kr.pair_counts(x, actor, tied, pairs, "same", 0.0, 1).A[0, 0] == 0.0
    assert Kr.pair_counts(x, actor, tied, pairs, "exp", 1e-6, 1).A[0, 0] == pytest.approx(1.0)


def test_the_within_actor_identity():
    """A^(inf) = n_u n_u' - sum_d x_du x_du' once the diagonal is removed (section 2)."""
    x, actor, stamp, pairs, A = toy()
    ps = Kr.pair_counts(x, actor, stamp, pairs, "all", 0.0, A, exclude_self=True)
    u, v = pairs[:, 0], pairs[:, 1]
    for a in range(A):
        rows = np.flatnonzero(actor == a)
        n = x[rows].sum(0)
        want = n[u] * n[v] - (x[rows][:, u] * x[rows][:, v]).sum(0)
        assert np.allclose(ps.A[a], want)


def test_counts_are_masked_and_labelled():
    x, actor, stamp, pairs, A = toy()
    mask = np.arange(len(x)) % 3 != 0
    got = Kr.pair_counts(x, actor, stamp, pairs, "exp", 4.0, A, mask=mask)
    want = Kr.brute_force(x[mask], actor[mask], stamp[mask], pairs, "exp", 4.0, A)
    assert np.allclose(got.A, want.A) and np.allclose(got.N, want.N)
    # C7 requires the kernel and timescale to travel with the network
    assert got.label == "exp(delta=4d)"


def test_an_actor_with_no_units_contributes_nothing():
    x, actor, stamp, pairs, A = toy()
    actor = np.where(actor == 2, 1, actor)  # actor 2 now has no units
    ps = Kr.pair_counts(x, actor, stamp, pairs, "exp", 3.0, 3)
    assert not ps.A[2].any() and ps.N[2] == 0.0


@pytest.mark.parametrize("kernel", ["exp", "box"])
def test_a_zero_timescale_is_refused(kernel):
    """delta = 0 has a name -- `same` -- and silently returning it would hide a typo."""
    x, actor, stamp, pairs, A = toy()
    with pytest.raises(ValueError):
        Kr.pair_counts(x, actor, stamp, pairs, kernel, 0.0, A)
