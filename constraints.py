"""The constraint model: theory files in, index arrays out.

Everything the objective needs that comes from theory rather than from data lives
here, as the frozen `Constraints` dataclass built by `load()`. The names follow
the write-up (ideology_operationalization_v3_1.tex, section 3):

  C1  concept-ideology affiliation   -- prior mean +/-alpha on eta_k; node file columns
  C2  within-ideology links          -- W_k >= omega on E_k; the edge file
  C3  incompatibility                -- cap sum_S e^{m_u + eta_ku} <= pi_bar; forbidden file
  C4  actor alignment                -- theta_a pinned for T_fix, and the metadata design
  C5  the type ladder                -- W = 0 off P (rung-skipping pairs); the `layer` column
  C6  structural coherence           -- a statistic first (diagnostics), a gate second (model)
  C7  domain                         -- W >= 0, rates positive; no input

C5 changed in v3.1 and the change is not cosmetic. The ladder forbids *rung-skipping*
only, so the allowed set is

    P = {(u,u') : c(u) != c(u'), {T(u),T(u')} not in {{V,B},{V,R},{A,R}}}

which admits the intra-type blocks V-V, A-A, B-B, R-R that the previous version
excluded. On this node file that takes P from 2604 pairs to 4285. Section 3's warning
travels with it: the lateral blocks, B-B especially, can absorb co-occurrence that
previously had to load on cross-rung edges, so `block_of` exists to report per-block
edge mass rather than let the shift happen silently.

C1 also changed register. It is no longer a box (rho >= delta) but a prior *mean*:
theory asserts direction, the data sets magnitude. `prior_mean` builds it.

Two index maps are kept distinct, because the write-up keeps them distinct:

  layer / type   T(c) in {V, A, B, R}. Supplied by the `layer` column. This is what
                 the ladder runs on.
  concept        c in the node u = (c, l). Supplied by the `concept` column, which is
                 currently blank for every row, meaning each node is its own concept
                 with a single level. Fill the column in to group rows into a
                 multi-level concept and `concept_of` picks it up with no other change;
                 the structural half of C3 (the levels of a concept are incompatible)
                 and the same-concept zeros in W start binding at that point.

Actor metadata (C4's regression, section 2's `m_a`)
--------------------------------------------------
theta_a = softmax(beta' m_a + xi_a), so the alignment regression needs a design matrix
over actors. It is built from the actor CSV, and which columns enter is chosen like
this:

  default    every column whose name begins with `m_`
  override   `--meta col1,col2,...` on any fit.py command names columns explicitly
  none       `--meta none`, or `columns=[]`, gives the intercept alone

A numeric column enters standardised; a non-numeric column enters one-hot with its
alphabetically-first level dropped as the reference. An intercept is always prepended.
`fit.py check` prints the resulting columns and their ranges, so a mis-specified
column is visible before a fit rather than after one.
"""

import csv
import itertools
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

NODES = Path("ideology_nodes_v3.csv")
EDGES = Path("ideology_edges_v3.csv")
FORBIDDEN = Path("ideology_forbidden_edges_v3.csv")
ACTORS = Path("actor_ideology_affiliation.csv")

# Ideology names, in component order. v3.1 has no background component: the background
# ideology is the all-zeros deviation, so ideology k is column IDEOLOGIES[k] throughout.
IDEOLOGIES = ["CB", "WS", "IS"]

# The type ladder (C5). Relations run within a type and between adjacent types; what
# the ladder forbids is skipping a rung.
LAYERS = ["Values", "Attitudes", "Beliefs", "Rationalizations"]
LAYER_CODE = ["V", "A", "B", "R"]
FORBIDDEN_RUNGS = {frozenset({0, 2}), frozenset({0, 3}), frozenset({1, 3})}  # VB, VR, AR

# Prefix that marks a column of the actor file as a regressor in the C4 alignment model.
META_PREFIX = "m_"


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


@dataclass(frozen=True)
class Constraints:
    """Theory as arrays. K nodes, M ideologies, P allowed pairs, C concepts."""

    node_ids: list[str]  # [K] the id strings, in file order
    index: dict[str, int]  # id -> position, for joining codings and edges
    element: list[str]  # [K] the human-readable text, for reporting
    layer: np.ndarray  # [K] int, index into LAYERS -- the type map T
    concept_of: np.ndarray  # [K] int, which concept each node is a level of
    anchor_pos: np.ndarray  # [M, K] bool, C1 prior mean +alpha
    anchor_neg: np.ndarray  # [M, K] bool, C1 prior mean -alpha
    pairs: np.ndarray  # [P, 2] int, the allowed pairs (C5); every other W is 0
    theory_edges: np.ndarray  # [M, P] bool, C2 W >= omega
    incompat: list[np.ndarray]  # C3, each a node-index array S
    names: list[str]  # [M] display names of the ideologies, in component order
    actors: list[str]  # [A] actor ids, in the order theta rows use
    theta_fix: np.ndarray  # [A] bool, actors whose theta is pinned (breaks label switching)
    theta_val: np.ndarray  # [A, M] float, the pinned value where theta_fix, else nan
    meta_rows: list[dict]  # [A] the raw actor CSV rows, for building the design matrix

    @property
    def K(self) -> int:
        return len(self.node_ids)

    @property
    def M(self) -> int:
        return self.anchor_pos.shape[0]

    @property
    def P(self) -> int:
        return len(self.pairs)

    @property
    def C(self) -> int:
        return int(self.concept_of.max()) + 1 if len(self.concept_of) else 0

    def name(self, k: int) -> str:
        """Display name of ideology k.

        On the Constraints rather than read from the module-level IDEOLOGIES, because
        the two can differ: IDEOLOGIES names the columns of the node file and never
        changes, while `names` follows the ideologies a given fit actually has. Dropping
        one (fit.py run --drop) shortens this list, and anything still indexing the
        global would quietly relabel the survivors -- WS's numbers printed under IS's
        name, which is the worst kind of bug because it looks like a result.
        """
        return self.names[k] if 0 <= k < len(self.names) else f"free{k}"

    def prior_mean(self, alpha: float) -> np.ndarray:
        """C1 as the [M, K] mean of the eta_k draw: +alpha, -alpha or 0."""
        return alpha * (self.anchor_pos.astype(float) - self.anchor_neg.astype(float))

    def block_of(self) -> np.ndarray:
        """[P] int, the type block of each pair, indexing `block_names()`."""
        names = block_names()
        lo = np.minimum(self.layer[self.pairs[:, 0]], self.layer[self.pairs[:, 1]])
        hi = np.maximum(self.layer[self.pairs[:, 0]], self.layer[self.pairs[:, 1]])
        lut = {n: i for i, n in enumerate(names)}
        return np.array(
            [lut[LAYER_CODE[a] + LAYER_CODE[b]] for a, b in zip(lo, hi)], dtype=np.int32
        )


def block_names() -> list[str]:
    """The type blocks C5 admits, in (lo, hi) order: VV VA AA AB BB BR RR."""
    return [
        LAYER_CODE[a] + LAYER_CODE[b]
        for a, b in itertools.combinations_with_replacement(range(4), 2)
        if frozenset({a, b}) not in FORBIDDEN_RUNGS
    ]


def allowed_pairs(layer: np.ndarray, concept_of: np.ndarray) -> np.ndarray:
    """C5's P: same-type and adjacent-type pairs, excluding rung-skipping ones.

    Same-concept pairs are excluded too -- that is C3's structural half, and it is a
    zero in W rather than a penalty. Returns [P, 2] with u < u'.
    """
    out = [
        (u, v)
        for u, v in itertools.combinations(range(len(layer)), 2)
        if concept_of[u] != concept_of[v]
        and frozenset({int(layer[u]), int(layer[v])}) not in FORBIDDEN_RUNGS
    ]
    return np.array(out, dtype=np.int32)


def skipping_pairs(layer: np.ndarray, concept_of: np.ndarray) -> np.ndarray:
    """The complement of P: rung-skipping pairs, where W is pinned to zero.

    Section 5 makes these a test rather than an assumption -- excess pairing here,
    computed exactly as the R diagnostic but off P, is evidence against a type
    assignment -- so the diagnostics need the list.
    """
    out = [
        (u, v)
        for u, v in itertools.combinations(range(len(layer)), 2)
        if concept_of[u] != concept_of[v]
        and frozenset({int(layer[u]), int(layer[v])}) in FORBIDDEN_RUNGS
    ]
    return np.array(out, dtype=np.int32)


def maximal_cliques(adj: dict[int, set[int]]) -> list[list[int]]:
    """Bron-Kerbosch with pivoting.

    C3 wants sets of mutually incompatible nodes, and the forbidden file supplies
    pairs. Connected components would be far too aggressive -- the largest component
    of the current file has 21 nodes but only 23 of its 210 possible edges, so
    treating it as one set would assert a 21-way incompatibility nobody wrote down.
    Maximal cliques give exactly the sets theory actually names: mostly pairs, plus
    a triangle wherever three nodes are pairwise incompatible.
    """
    out: list[list[int]] = []

    def expand(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            out.append(sorted(r))
            return
        pivot = max(p | x, key=lambda v: len(adj[v]))
        for v in list(p - adj[pivot]):
            expand(r | {v}, p & adj[v], x & adj[v])
            p.remove(v)
            x.add(v)

    expand(set(), set(adj), set())
    return out


def load_nodes(path: Path = NODES) -> tuple:
    rows = list(csv.DictReader(path.open()))
    ids = [r["id"] for r in rows]
    index = {i: n for n, i in enumerate(ids)}

    unknown = {r["layer"] for r in rows} - set(LAYERS)
    if unknown:
        raise ValueError(f"{path}: unknown layer(s) {sorted(unknown)}; expected {LAYERS}")
    layer = np.array([LAYERS.index(r["layer"]) for r in rows], dtype=np.int8)

    # A blank `concept` means the node is its own concept, with a single level.
    named: dict[str, int] = {}
    concept_of = np.empty(len(rows), dtype=np.int32)
    for n, r in enumerate(rows):
        c = r.get("concept", "").strip()
        concept_of[n] = n if not c else named.setdefault(c, len(rows) + len(named))
    # Renumber to 0..C-1 so the array can index a per-concept quantity directly.
    _, concept_of = np.unique(concept_of, return_inverse=True)
    concept_of = concept_of.astype(np.int32)

    anchor_pos = np.array([[r[k] == "1" for r in rows] for k in IDEOLOGIES], dtype=bool)
    # The node file records affiliation only positively; A^-_k stays empty unless a
    # future column supplies it. C1's negative half is wired up and unused.
    anchor_neg = np.zeros_like(anchor_pos)
    return ids, index, [r["element"] for r in rows], layer, concept_of, anchor_pos, anchor_neg


def load_edges(path: Path, index: dict[str, int], pairs: np.ndarray) -> np.ndarray:
    """C2. Returns [M, P] bool over allowed pairs; rung-skipping edges are dropped."""
    pair_index = {(int(u), int(v)): p for p, (u, v) in enumerate(pairs)}
    out = np.zeros((len(IDEOLOGIES), len(pairs)), dtype=bool)
    missing, skipping = set(), []
    for r in csv.DictReader(path.open()):
        s, t, k = r["source"], r["target"], r["ideology"]
        if s not in index or t not in index:
            missing |= {i for i in (s, t) if i not in index}
            continue
        if k not in IDEOLOGIES:
            warn(f"{path}: unknown ideology {k!r}; ignoring edge {s}-{t}")
            continue
        p = pair_index.get(tuple(sorted((index[s], index[t]))))
        if p is None:
            skipping.append((s, t, k))
            continue
        out[IDEOLOGIES.index(k), p] = True
    if missing:
        warn(f"{path}: {len(missing)} id(s) not in the node file, ignored: {sorted(missing)}")
    if skipping:
        warn(
            f"{path}: {len(skipping)} edge(s) skip a rung of the ladder (C5), dropped: "
            f"{skipping[:5]}{' ...' if len(skipping) > 5 else ''}"
        )
    return out


def load_forbidden(path: Path, index: dict[str, int]) -> list[np.ndarray]:
    """C3 theoretical sets X. Pairwise input, maximal cliques out."""
    adj: dict[int, set[int]] = {}
    missing = set()
    for r in csv.DictReader(path.open()):
        a, b = r["id_1"], r["id_2"]
        if a not in index or b not in index:
            missing |= {i for i in (a, b) if i not in index}
            continue
        u, v = index[a], index[b]
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    if missing:
        warn(f"{path}: {len(missing)} id(s) not in the node file, ignored: {sorted(missing)}")
    return [np.array(c, dtype=np.int32) for c in maximal_cliques(adj) if len(c) > 1]


def structural_sets(concept_of: np.ndarray) -> list[np.ndarray]:
    """C3's structural half: the levels of one concept are mutually incompatible.

    Empty while the `concept` column is blank (one level per concept), which is why
    it is separate from X -- section 3 has the structural sets binding at every layer
    and unrevisable, and X binding only the ideology layer and revisable.
    """
    out = []
    for c in range(int(concept_of.max()) + 1 if len(concept_of) else 0):
        S = np.flatnonzero(concept_of == c)
        if len(S) > 1:
            out.append(S.astype(np.int32))
    return out


def add_free_ideologies(cons: Constraints, n: int) -> Constraints:
    """Extra ideologies with no C1 anchors and no C2 links.

    Free in exactly the sense section 7 wants: the anchored ideologies hold their labels
    through C4's pinned actors, and anything the corpus contains beyond them has
    somewhere to go instead of distorting a named component.
    """
    pad_k = lambda a: np.concatenate([a, np.zeros((n,) + a.shape[1:], dtype=a.dtype)])
    pad_a = lambda a: np.concatenate([a, np.full((a.shape[0], n), np.nan)], axis=1)
    # `names` must grow with them. Without it cons.M (read off anchor_pos) and
    # len(cons.names) disagree, cons.name() falls off the end, and -- the way this actually
    # surfaced -- two configurations differing only in free-component count become
    # indistinguishable to anything keyed on the names, including the experiment fit cache.
    return replace(cons,
                   names=list(cons.names) + [f"free{i + 1}" for i in range(n)],
                   anchor_pos=pad_k(cons.anchor_pos), anchor_neg=pad_k(cons.anchor_neg),
                   theory_edges=pad_k(cons.theory_edges), theta_val=pad_a(cons.theta_val))


def load_actors(path: Path, n_ideologies: int) -> tuple:
    """C4. Blank cells mean 'learn it'; see write_actor_template for the format.

    Returns the raw rows too: the alignment regression's design matrix is built from
    them by `metadata`, and which columns it uses is a run-time choice.
    """
    if not path.exists():
        warn(f"{path} not found; theta is unpinned (C4 empty) and label switching is unbroken")
        return [], np.zeros(0, dtype=bool), np.zeros((0, n_ideologies)), []

    rows = list(csv.DictReader(path.open()))
    actors = [r["rt_uid"] for r in rows]
    fix = np.zeros(len(rows), dtype=bool)
    val = np.full((len(rows), n_ideologies), np.nan)

    for n, r in enumerate(rows):
        kind = r.get("kind", "").strip().lower()
        cells = [r.get(k, "").strip() for k in IDEOLOGIES]
        if not any(cells):
            continue  # fully free actor, whatever `kind` says
        if kind != "fix":
            warn(
                f"{path}: row {n + 1} ({r.get('username', '?')}) has ideology values but "
                f"kind={kind!r}; only kind=fix is honoured in v3.1 (theta is a softmax, so "
                "there is no box to bound), ignoring"
            )
            continue
        for k, cell in enumerate(cells):
            val[n, k] = float(cell) if cell else 0.0
        fix[n] = True
        total = np.nansum(val[n])
        if abs(total - 1.0) > 1e-6:
            warn(f"{path}: row {n + 1} ({r.get('username', '?')}) is kind=fix but sums to "
                 f"{total:.3f}, not 1")
    return actors, fix, val, rows


def metadata(
    rows: list[dict], actors: list[str], columns: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """The C4 design matrix m_a. See the module docstring for how columns are chosen.

    Returns ([A, J] float, [J] names). Column 0 is always the intercept, so an actor
    file with no metadata at all still gives a well-posed (if uninformative) regression
    -- theta is then carried entirely by xi and the pinned anchors.
    """
    if not rows or columns == []:
        # An explicit empty list is "no metadata", and is NOT the same as None. None means
        # "you did not choose", which falls back to every m_* column. Conflating the two
        # silently reinstates identity in designs that exist to exclude it -- see the
        # experiments that withhold community membership.
        return np.ones((len(actors), 1)), ["(intercept)"]
    available = [c for c in rows[0] if c]
    if columns is None:
        columns = [c for c in available if c.startswith(META_PREFIX)]
    else:
        unknown = [c for c in columns if c not in available]
        if unknown:
            raise ValueError(
                f"metadata column(s) {unknown} not in the actor file; it has {available}"
            )
    cols = [np.ones(len(rows))]
    names = ["(intercept)"]
    for c in columns:
        raw = [r.get(c, "").strip() for r in rows]
        numeric = None
        if all(v != "" for v in raw):
            try:
                numeric = np.array([float(v) for v in raw])
            except ValueError:
                numeric = None
        if numeric is not None:
            sd = numeric.std()
            cols.append((numeric - numeric.mean()) / (sd if sd > 0 else 1.0))
            names.append(c)
            continue
        # Categorical: one-hot, alphabetically-first level dropped as the reference.
        levels = sorted({v for v in raw if v})
        if len(levels) < 2:
            warn(f"metadata column {c!r} has fewer than two levels; dropped")
            continue
        for lv in levels[1:]:
            cols.append(np.array([1.0 if v == lv else 0.0 for v in raw]))
            names.append(f"{c}={lv}")
    return np.column_stack(cols), names


def load(
    nodes: Path = NODES,
    edges: Path = EDGES,
    forbidden: Path = FORBIDDEN,
    actors: Path = ACTORS,
) -> Constraints:
    ids, index, element, layer, concept_of, apos, aneg = load_nodes(nodes)
    pairs = allowed_pairs(layer, concept_of)
    actor_ids, theta_fix, theta_val, rows = load_actors(actors, len(IDEOLOGIES))
    c = Constraints(
        names=list(IDEOLOGIES),
        node_ids=ids,
        index=index,
        element=element,
        layer=layer,
        concept_of=concept_of,
        anchor_pos=apos,
        anchor_neg=aneg,
        pairs=pairs,
        theory_edges=load_edges(edges, index, pairs),
        incompat=load_forbidden(forbidden, index) + structural_sets(concept_of),
        actors=actor_ids,
        theta_fix=theta_fix,
        theta_val=theta_val,
        meta_rows=rows,
    )
    check_feasible(c)
    return c


def check_feasible(c: Constraints, alpha: float = 1.0, pi_bar: float = 0.5) -> None:
    """Catch constraint sets that C1 and C3 cannot both satisfy, before the optimizer.

    C1 shifts the prior mean of eta to +alpha on an anchored node and C3 caps
    sum_{u in S} e^{m_u + eta_ku} <= pi_bar. Anchoring one ideology to several nodes of
    one incompatible set does not make the problem infeasible -- eta is free around the
    shifted mean, so the cap simply pulls it back -- but it does assert that an ideology
    holds things theory says nothing coherent holds together, and it means the prior and
    the cap will fight for the whole fit. Read a hit as a disagreement between two theory
    files rather than as a numerical problem.
    """
    for k, name in enumerate(IDEOLOGIES):
        for S in c.incompat:
            hit = S[c.anchor_pos[k, S]]
            if len(hit) > 1:
                warn(
                    f"C1/C3 tension: ideology {name} anchors {len(hit)} nodes of one "
                    f"incompatible set: {[c.node_ids[u] for u in hit]}"
                )
        if not c.anchor_pos[k].any():
            warn(f"ideology {name} has no C1 anchors; it is unidentified up to relabeling")
    if c.anchor_pos.any(axis=0).sum() == 0:
        raise ValueError("no C1 anchors at all; every component is exchangeable")


def check_coding_ids(seen: set[str], c: Constraints) -> None:
    """Fatal check on the ids in a coding file.

    Node ids were renumbered within layer between codebook versions (v2 had 32
    attitudes, v3 has 28), so a stale coding file resolves A29-A31 to nothing and,
    far worse, resolves A1-A28 to the *wrong* nodes -- silently, since they are all
    valid ids. Refuse to fit rather than let that through.
    """
    unknown = sorted(seen - set(c.index))
    if unknown:
        raise ValueError(
            f"{len(unknown)} id(s) in the coding are not in {NODES}: {unknown[:10]}"
            f"{' ...' if len(unknown) > 10 else ''}.\n"
            "This usually means the coding was produced against an older codebook whose "
            "ids have since been renumbered. Ids that *do* resolve may resolve to the "
            "wrong nodes, so re-run run.py against the current node file."
        )


def write_actor_template(
    parquet: Path = Path("min_sample.parquet"), path: Path = ACTORS, force: bool = False
) -> None:
    """Seed the C4 file with every actor in the corpus, for the user to fill in.

    `kind` is one of:
      fix    the row's CB/WS/IS values are theta_a exactly and the alignment step skips
             this actor. This is what pins ideologies to names and breaks label switching.
      free   (or blank) learn theta_a from metadata and xi.

    Metadata columns: any column whose name starts with `m_` enters the alignment
    regression by default; `--meta` names columns explicitly instead. `m_grpname` is
    written out populated so there is a working default -- delete it, or add your own
    `m_*` columns, to change what identity the regression sees.

    Regenerating is merge-aware. Anything already filled in for an actor -- `kind`, the
    ideology cells, and any `m_*` column the file has grown -- is carried over, because
    the C4 pins are hand-work and the whole point of the metadata columns is that the
    analyst adds their own. Only actors missing from the file are appended.
    """
    import pandas as pd

    if path.exists() and not force:
        warn(f"{path} exists; not overwriting (pass force=True to regenerate and merge)")
        return
    prior = {r["rt_uid"]: r for r in csv.DictReader(path.open())} if path.exists() else {}
    extra = [c for r in prior.values() for c in r
             if c and c.startswith(META_PREFIX) and c != "m_grpname"]
    extra = sorted(set(extra))

    df = (
        pd.read_parquet(parquet, columns=["rt_uid", "username", "grpname", "description"])
        .drop_duplicates("rt_uid")
        .sort_values(["grpname", "username"])
    )
    cols = (["rt_uid", "username", "grpname", "description", "kind"] + IDEOLOGIES
            + ["m_grpname"] + extra)
    kept = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in df.itertuples(index=False):
            was = prior.get(str(r.rt_uid), {})
            kept += bool(was.get("kind", "").strip().lower() == "fix")
            w.writerow(
                [r.rt_uid, r.username, r.grpname, str(r.description or "").replace("\n", " "),
                 was.get("kind", "free") or "free"]
                + [was.get(k, "") for k in IDEOLOGIES]
                + [was.get("m_grpname", "") or r.grpname]
                + [was.get(c, "") for c in extra]
            )
    print(f"wrote {path} with {len(df)} actors ({kept} pinned, carried over from the "
          f"previous file); fill in `kind` and the {'/'.join(IDEOLOGIES)} columns to pin "
          "anchors, and add `m_*` columns to change the alignment regression's design")


if __name__ == "__main__":
    c = load()
    m, names = metadata(c.meta_rows, c.actors)
    print(f"\n{c.K} nodes, {c.M} ideologies, {c.P} allowed pairs, "
          f"{len(c.incompat)} incompatible sets")
    print(f"concepts: {c.C} (blank `concept` column -> one per node)")
    print(f"pairs by block: " + "  ".join(
        f"{n}:{int((c.block_of() == i).sum())}" for i, n in enumerate(block_names())))
    print(f"rung-skipping pairs (W pinned to 0): {len(skipping_pairs(c.layer, c.concept_of))}")
    print(f"C1 anchors per ideology: {dict(zip(IDEOLOGIES, c.anchor_pos.sum(1)))}")
    print(f"C2 theory edges per ideology: {dict(zip(IDEOLOGIES, c.theory_edges.sum(1)))}")
    print(f"C3 set sizes: {np.bincount([len(s) for s in c.incompat])[1:]} (index = size - 1)")
    print(f"C4 actors: {len(c.actors)} ({c.theta_fix.sum()} pinned)")
    print(f"metadata design: {m.shape} -> {names}")
