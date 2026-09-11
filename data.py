"""Corpus in, arrays out: soft node expressions x, actor index a(d), period index t(d).

Three inputs, per section 2 of the write-up:

  units -> actors     min_sample.parquet, rt_id -> rt_uid
  units -> nodes      the JSONL written by run.py, rt_id -> {node id: confidence}
  units -> time       min_sample.parquet's `date`, binned to the period grid

x_{d,u} is the extractor's probability that node u is expressed in unit d, treated as
an expected count (section 2: the extraction layer supplies x in [0,1], and the
expression draw is Poisson). It is mostly zero (~4 of 112 nodes per unit), but K is
small enough that dense float64 beats any sparse representation here.

Units with no coded nodes are kept. An all-zero row is not missing data -- it is a
unit that expressed nothing, which is evidence about the actor's rates.

The structural fact the model leans on
--------------------------------------
In v3.1 the rate depends on a unit only through its actor and its period:

    log eta_{a,u,t} = m_u + m_{t,c(u)} + sum_k theta_ak eta_ku + eps_{a,c(u)}

so the rate likelihood sees the corpus only through the (actor, period) collapse that
`rate_cells` builds: a count of units per cell and a column sum of x per cell. On this
corpus that is at most 133 x 13 = 1729 rows against 32k units, and every rate block
costs G*K regardless of D. `rate_cells` is called once for train and once for held-out,
and the held-out score is then the same Poisson loss on the held-out table -- there is
no per-unit free parameter left to leak through it, which is what the previous
version's marginalisation hack existed to prevent.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import constraints as C

# The coding in use. Must match the current node file (see check_coding_ids) and should
# carry source/stance (see read_codings) -- final_v3_coding.jsonl is the pre-voice file and
# is kept only so old fits stay loadable.
CODINGS = Path("absolute_final_v3_coding.jsonl")
PARQUET = Path("min_sample.parquet")

PERIODS = ("year", "quarter", "month", "all")


@dataclass(frozen=True)
class Corpus:
    """D units over K nodes, produced by A actors across T periods."""

    x: np.ndarray  # [D, K] float64 in [0, 1]
    actor: np.ndarray  # [D] int, index into `actors`
    time: np.ndarray  # [D] int, index into `periods`
    actors: list[str]  # [A] actor ids (rt_uid, as strings)
    periods: list[str]  # [T] period labels, in time order
    unit_ids: list[str]  # [D] rt_id, for joining results back to tweets
    train: np.ndarray  # [D] bool, the held-out split (True = train)
    stamp: np.ndarray  # [D] float, days since the corpus start -- the kernel's clock
    meta: np.ndarray  # [A, J] the C4 alignment design matrix
    meta_names: list[str]  # [J]
    groups: list[str]  # [A] grpname; validation only, never enters a fit
    concept_of: np.ndarray  # [K] int -> concept, carried from the constraints

    @property
    def D(self) -> int:
        return len(self.x)

    @property
    def K(self) -> int:
        return self.x.shape[1]

    @property
    def A(self) -> int:
        return len(self.actors)

    @property
    def T(self) -> int:
        return len(self.periods)

    @property
    def C(self) -> int:
        return int(self.concept_of.max()) + 1


@dataclass(frozen=True)
class Cells:
    """The (actor, period) collapse of the rate likelihood.

    `mask` is an optional [A, K] observation mask over actor-node cells, applied here
    so that every downstream block sees a consistent view. A zero means the corpus is
    not being allowed to speak about whether actor a expresses node u, and *both* halves
    of the Poisson term -- the exposure n_g e^z and the count X log(e^z) -- drop that
    cell. The distinction matters: simply zeroing x would tell the model the actor
    expresses the node zero times, which is a measurement, not a withholding. V5's
    held-out (actor, concept) cells are the use; it is all ones otherwise.
    """

    actor: np.ndarray  # [G] int
    period: np.ndarray  # [G] int
    n: np.ndarray  # [G] float, units in the cell
    X: np.ndarray  # [G, K] float, summed expressions
    mask: np.ndarray  # [A, K] float in {0, 1}

    @property
    def G(self) -> int:
        return len(self.n)

    @property
    def obs(self) -> np.ndarray:
        """[G, K] the per-cell observation mask, expanded from the actor mask."""
        return self.mask[self.actor]


# The one cell of the source x stance grid that means the author holds the element.
# `support` is read widely -- asserting, holding as a value, endorsing, or backing what
# the element calls for -- because only the Beliefs layer is asserted in the propositional
# sense; its mirror `reject` covers opposing as well as denying. See run.SOURCE_RULES and
# run.STANCE_RULES.
HELD = (("author", "support"),)

# The four cells, named. `author/reject` is the one the model has no use for yet: it is
# signed evidence about an actor -- opposing an element is not the same as being silent
# on it -- and the rate layer is Poisson, so it has nowhere to put a negative count. It
# is carried through so that C1's unused negative-anchor half has something to be fitted
# against when someone wants it.
CELLS = {("author", "support"): "holds", ("author", "reject"): "opposes",
         ("other", "support"): "reports", ("other", "reject"): "refutes"}


# The first attempt at this used a single four-valued `voice`. Those four values are the
# cells of the source x stance grid, so a file written with them loses nothing by being
# read into the two fields.
_LEGACY_VOICE = {"assert": ("author", "support"), "reject": ("author", "reject"),
                 "report": ("other", "support"), "accuse": ("other", "reject")}


def read_codings(path: Path = CODINGS) -> tuple[dict[str, list[dict]], bool]:
    """rt_id -> [{id, confidence, source, stance}], and whether the file carries them.

    Three formats, all read. The current one writes `elements` with `source` and `stance`
    -- whose thinking is being stated, and with what polarity. A short-lived intermediate
    wrote a single `voice` whose four values are exactly the cells of that grid, and is
    mapped across. The original wrote `scores`, a bare id -> confidence map with neither,
    and is read as unknown so that older codings keep loading -- but the flag comes back
    False and `load` refuses to filter silently, because on such a file "the author holds
    this" and "the author mentioned this" are the same number and nothing can separate
    them after the fact.
    """
    out, coded_stance = {}, False
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "elements" in r:
                coded_stance = True
                els = []
                for e in r["elements"]:
                    src, stn = e.get("source"), e.get("stance")
                    if src is None and "voice" in e:
                        src, stn = _LEGACY_VOICE.get(e["voice"], ("unknown", "unknown"))
                    els.append({"id": e["id"], "confidence": e.get("confidence", 1.0),
                                "source": src or "unknown", "stance": stn or "unknown"})
                out[str(r["rt_id"])] = els
            else:
                out[str(r["rt_id"])] = [
                    {"id": i, "confidence": v, "source": "unknown", "stance": "unknown"}
                    for i, v in r.get("scores", {}).items()
                ]
    return out, coded_stance


def period_labels(dates, period: str) -> tuple[np.ndarray, list[str]]:
    """Bin timestamps onto the m_{t,c} grid.

    Section 8 puts m_{t,c} in at stage 4 and says its resolution is a reported
    sensitivity, not a setting to get right once: coarser leaves event-driven co-surges
    in W, finer leaves most (period, concept) cells empty and leaning on the N(0,
    sigma_m^2) prior. `all` collapses the grid to one period, which is how stages 1-3
    run with the offsets off.
    """
    import pandas as pd

    dates = pd.to_datetime(dates, utc=True)
    if period == "all":
        key = pd.Series(["all"] * len(dates), index=dates.index if hasattr(dates, "index") else None)
    elif period == "year":
        key = dates.dt.year.astype(str)
    elif period == "quarter":
        key = dates.dt.year.astype(str) + "Q" + dates.dt.quarter.astype(str)
    elif period == "month":
        key = dates.dt.strftime("%Y-%m")
    else:
        raise ValueError(f"unknown period {period!r}; expected one of {PERIODS}")
    labels = sorted(set(key))
    lut = {v: i for i, v in enumerate(labels)}
    return np.array([lut[v] for v in key], dtype=np.int32), labels


def load(
    codings: Path = CODINGS,
    parquet: Path = PARQUET,
    cons: C.Constraints | None = None,
    holdout: float = 0.1,
    seed: int = 0,
    period: str = "year",
    meta_columns: list[str] | None = None,
    cells: tuple[tuple[str, str], ...] | None = HELD,
    years: tuple[int, int] | None = None,
) -> Corpus:
    """Join the coding to the actor map and build x, the actor index and the time index.

    The join is an inner one on rt_id; both sides are reported when they disagree.
    Actor ids are taken from the C4 file when it has any, so that theta and the metadata
    design line up row for row; actors present in the corpus but missing from that file
    are appended with an all-reference metadata row rather than dropped.

    `voices` selects which codings count as expression. The default keeps only `assert` --
    read widely, as asserting or supporting -- because the model is about what actors
    hold: a tweet reporting that someone else is a white supremacist is evidence about the
    topic of the discourse, not about the author's ideology, and counting it as expression
    is what makes an ideology's most vocal critics its highest-scoring adherents. Passing `("report", "accuse")` instead builds the
    complement -- the corpus of talk *about* the ideologies -- which is the cheap form of
    a reported-speech layer and needs no change to the model. `voices=None` keeps
    everything, which is what a coding file with no voice information gets.
    """
    import pandas as pd

    cons = cons or C.load()
    coded, coded_stance = read_codings(codings)
    C.check_coding_ids({e["id"] for v in coded.values() for e in v}, cons)
    if cells is not None and not coded_stance:
        print(f"note: {codings} has no source/stance coding, so every coding is kept. It "
              "cannot\n  distinguish an author holding an element from one reporting, "
              "opposing or refuting it,\n  and the ideologies will absorb their own "
              "critics. Re-run run.py to fix.")
        cells = None

    # `grpname` is optional: the released corpus carries no community label, and nothing
    # in the reported path reads one. When it is absent every actor gets an empty group,
    # which only affects diagnostics that report group membership.
    import pyarrow.parquet as _pq
    have = set(_pq.read_schema(parquet).names)
    cols = [c for c in ("rt_id", "rt_uid", "date", "grpname") if c in have]
    df = pd.read_parquet(parquet, columns=cols)
    if "grpname" not in df.columns:
        df["grpname"] = ""
    df["rt_id"] = df["rt_id"].astype(str)
    df["rt_uid"] = df["rt_uid"].astype(str)
    df = df.drop_duplicates("rt_id")

    in_parquet = set(df["rt_id"])
    coded_ids = set(coded)
    if missing := coded_ids - in_parquet:
        print(f"note: {len(missing):,} coded unit(s) not in {parquet}, dropped")
    if uncoded := in_parquet - coded_ids:
        print(f"note: {len(uncoded):,} unit(s) in {parquet} with no coding, dropped")
    df = df[df["rt_id"].isin(coded_ids)].reset_index(drop=True)

    # `years` is inclusive on both ends, and it is applied here -- after the join, before
    # anything is indexed -- so the actor list, the period labels and the `stamp` origin
    # are all built from the window that was kept. Filtering later would leave actors with
    # no units and a day-zero outside the data.
    if years is not None:
        lo, hi = years
        yr = df["date"].dt.year
        keep = (yr >= lo) & (yr <= hi)
        print(f"years {lo}-{hi}: keeping {int(keep.sum()):,} of {len(df):,} units")
        df = df[keep].reset_index(drop=True)

    # Actor ordering follows the C4 file so theta rows match the constraint rows.
    present = set(df["rt_uid"])
    actors = [a for a in cons.actors if a in present]
    extra = sorted(present - set(actors))
    if extra and cons.actors:
        print(f"note: {len(extra)} actor(s) not in {C.ACTORS}, appended; their metadata row "
              "is the reference level, so their alignment rests on gamma alone")
    actors += extra
    actor_index = {a: n for n, a in enumerate(actors)}

    x = np.zeros((len(df), cons.K), dtype=np.float64)
    kept = dropped = 0
    for d, rt_id in enumerate(df["rt_id"]):
        for e in coded[rt_id]:
            if cells is not None and (e["source"], e["stance"]) not in cells:
                dropped += 1
                continue
            kept += 1
            x[d, cons.index[e["id"]]] = e["confidence"]
    if cells is not None:
        names = ", ".join(CELLS.get(c, "/".join(c)) for c in cells)
        print(f"keeping {names}: {kept:,} codings, dropped {dropped:,} "
              f"({dropped / max(kept + dropped, 1):.1%})")

    time, labels = period_labels(df["date"], period)
    day = (df["date"] - df["date"].min()).dt.total_seconds().to_numpy() / 86400.0

    meta, meta_names = meta_for(actors, cons, meta_columns)

    grp = df.drop_duplicates("rt_uid").set_index("rt_uid")["grpname"].to_dict()
    actor_idx = df["rt_uid"].map(actor_index).to_numpy(dtype=np.int32)

    return Corpus(
        x=x,
        actor=actor_idx,
        time=time,
        actors=actors,
        periods=labels,
        unit_ids=list(df["rt_id"]),
        train=split(actor_idx, holdout, seed),
        stamp=day,
        meta=meta,
        meta_names=meta_names,
        groups=[str(grp.get(a, "")) for a in actors],
        concept_of=cons.concept_of,
    )


def meta_for(
    actors: list[str], cons: C.Constraints, columns: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """[A, J] alignment design matrix, in the corpus's actor order.

    Separate from `load` because the design is the one thing a caller may want to change
    without touching the corpus: V4 refits with the group-derived columns removed, and
    reloading 36k units to drop a column would be absurd.

    An actor present in the corpus but absent from the actor file gets the reference row
    -- intercept only -- rather than being dropped: their alignment then rests on the
    actor-level draw alone, which is the honest representation of "we know nothing about
    this actor".
    """
    all_meta, names = C.metadata(cons.meta_rows, cons.actors, columns)
    out = np.zeros((len(actors), all_meta.shape[1]))
    out[:, 0] = 1.0
    by_id = {r["rt_uid"]: i for i, r in enumerate(cons.meta_rows)}
    for n, a in enumerate(actors):
        if a in by_id:
            out[n] = all_meta[by_id[a]]
    return out, names


def split(actor: np.ndarray, holdout: float, seed: int) -> np.ndarray:
    """Held-out mask, stratified within actor.

    Stratifying matters: eps_a, mu_a and xi_a are per-actor, so a plain random split can
    leave a low-volume actor with nothing in train (its parameters unestimable) or
    nothing in test (its fit unscorable). V7's whole-actor holdout is the deliberately
    unstratified counterpart, and it lives in validate.py rather than here.
    """
    rng = np.random.default_rng(seed)
    train = np.ones(len(actor), dtype=bool)
    for a in np.unique(actor):
        idx = np.flatnonzero(actor == a)
        n = int(round(holdout * len(idx)))
        if n:
            train[rng.choice(idx, size=n, replace=False)] = False
    return train


def rate_cells(
    corpus: Corpus, mask: np.ndarray | None = None, cell_mask: np.ndarray | None = None
) -> Cells:
    """Collapse units onto the (actor, period) grid. See the module docstring.

    `mask` selects units (train or held-out); `cell_mask` is the [A, K] observation mask
    described on `Cells`.
    """
    mask = corpus.train if mask is None else mask
    a, t = corpus.actor[mask], corpus.time[mask]
    key = a.astype(np.int64) * corpus.T + t
    uniq, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv, minlength=len(uniq)).astype(float)
    X = np.zeros((len(uniq), corpus.K))
    np.add.at(X, inv, corpus.x[mask])
    cm = np.ones((corpus.A, corpus.K)) if cell_mask is None else cell_mask.astype(float)
    return Cells(
        actor=(uniq // corpus.T).astype(np.int32),
        period=(uniq % corpus.T).astype(np.int32),
        n=n,
        X=X,
        mask=cm,
    )


def background_rates(corpus: Corpus, mask: np.ndarray | None = None, floor: float = 1e-5
                     ) -> np.ndarray:
    """m_u, fixed: the log corpus rate of each node (section 2's Background).

    Floored rather than dropped. A node the corpus never expresses has no rate to take a
    log of, and pinning it at `floor` keeps it in the index -- the diagnostics report it
    as unmeasured rather than as an ideology's considered rejection of it.
    """
    mask = corpus.train if mask is None else mask
    rate = corpus.x[mask].mean(0)
    return np.log(np.maximum(rate, floor))


def actor_period_weights(corpus: Corpus, mask: np.ndarray | None = None) -> np.ndarray:
    """[A, T] share of each actor's units falling in each period.

    Section 2's pair channel uses "time-averaged rates eta_{a,u}" in the exposure. This
    is the average: an actor's rate weighted by where their output actually sits in
    time, so an actor who only spoke in 2020 is not charged with 2014's news cycle.
    """
    mask = corpus.train if mask is None else mask
    w = np.zeros((corpus.A, corpus.T))
    np.add.at(w, (corpus.actor[mask], corpus.time[mask]), 1.0)
    tot = w.sum(1, keepdims=True)
    return np.divide(w, tot, out=np.full_like(w, 1.0 / corpus.T), where=tot > 0)


def describe(corpus: Corpus, cons: C.Constraints) -> None:
    x, per_unit = corpus.x, (corpus.x > 0).sum(1)
    print(f"{corpus.D:,} units x {corpus.K} nodes, {corpus.A} actors, "
          f"{corpus.T} periods ({corpus.periods[0]}..{corpus.periods[-1]})")
    print(f"  held out: {(~corpus.train).sum():,} ({(~corpus.train).mean():.1%})")
    print(f"  nodes per unit: mean {per_unit.mean():.2f}, {(per_unit == 0).mean():.1%} empty")
    print(f"  mean confidence when expressed: {x[x > 0].mean():.2f}")
    unused = [cons.node_ids[u] for u in np.flatnonzero((x > 0).sum(0) == 0)]
    if unused:
        print(f"  {len(unused)} node(s) never expressed: {unused}")
    n = np.bincount(corpus.actor, minlength=corpus.A)
    print(f"  units per actor: min {n.min():,}, median {int(np.median(n)):,}, max {n.max():,}")
    per_period = np.bincount(corpus.time, minlength=corpus.T)
    print(f"  units per period: min {per_period.min():,}, median "
          f"{int(np.median(per_period)):,}, max {per_period.max():,}")
    cells = rate_cells(corpus)
    print(f"  rate cells (actor x period): {cells.G:,} of {corpus.A * corpus.T:,} possible")
    print(f"  metadata design m_a: {corpus.meta.shape} -> {corpus.meta_names}")
    from collections import Counter
    print(f"  actors per group: {dict(sorted(Counter(corpus.groups).items()))}")


if __name__ == "__main__":
    cons = C.load()
    describe(load(cons=cons), cons)
