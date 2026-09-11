# Measuring racial ideology as rates and associations over a concept codebook

Replication code and data for the paper. The model treats an ideology as a pair
`(η_k, W_k)`: a sparse deviation of concept expression rates from a corpus background, and
a network of excess co-occurrence between concepts read at an analyst-set timescale.
Actors are mixed memberships over ideologies, anchored by a small number of pinned
accounts. Estimation is M-estimation with theory entering as priors-as-penalties and hard
caps; the rate layer is a true likelihood and the co-occurrence layer an estimating
equation, so the objective is a composite likelihood rather than a posterior.

## Data statement

The corpus is 64,459 tweets from 609 accounts, 2009–2023; the paper fits the 2012–2022
window (62,045 units, 607 actors). **The tweets themselves are not redistributed here.**
What the model consumes is the extractor's output — which codebook concepts each unit
expresses, with what confidence, in which source-by-stance cell — and that derivative
carries no tweet content.

`release/snapshot/` accordingly contains:

| | |
|---|---|
| tweet ids | **kept** — the standard unit of X data sharing; anyone with API access can rehydrate what still exists |
| tweet text | **dropped** |
| actor ids | **pseudonymous** — `A0001`…`A0609`, assigned in sorted order of the real id |
| usernames | **dropped**, except the nine pinned accounts, which the paper names and which the pinning mechanism looks up by handle |
| profile bios | **dropped** — they are the source of the keyword covariates and contain scraped email addresses; the derived 0/1 columns ship instead |
| community labels | **dropped** — nothing in the reported analyses uses them |

`make_release_data.py` builds that directory from the private snapshot and documents the
reasoning in full. Actor row order is preserved, because `data.load` takes its actor
ordering from the actor file: pseudonymising in place does not change any fit.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q            # 117 tests
```

## Reproducing the paper

All commands read `release/snapshot/` when `IDEOLOGY_SNAPSHOT` points at it; without the
variable they read the private `snapshot/`.

```bash
export IDEOLOGY_SNAPSHOT=release/snapshot

# the ladder: F_k in 1..4, with and without a free component, with and without WS,
# plus the keyword-covariate arm. ~12 fits, roughly 7 minutes each on 6 cores.
python fit.py final2                     # -> exp/final2.json, exp/final2_ws.json, exp/kwbeta.json

# the selected arm: its fit, concept and edge CSVs, and the HTML report
python fit.py final2-pick --arm f2-F1-x0 # -> fit_final2{.pkl,_concepts.csv,_edges.csv,_report.html}

# the paper's tables and figure
python make_ladder_table.py              # -> table_ladder.tex        (main ladder)
python make_ladder_table.py --appendix    # -> table_ladder_full.tex  (per-component)
python make_tables.py                    # -> tables_final2.tex       (concepts, edges)
python make_layer_inputs.py && python plot_ideology_layers.py   # -> ideology_layers.pdf/.png
```

Fits are cached by configuration under `exp/fits/`, so re-running skips finished arms. The
cache key covers the corpus fingerprint, node and actor sets, ideology names, pins, the
metadata design, the hyperparameters, the stage list and the holdout split — a collision
would serve one arm's parameters for another, so it errs toward being too specific.

Reference values for the selected arm `f2-F1-x0`: held-out 21,680, macro AUC 0.878, actor
share 36/64, coherence CB 1.45 and IS 4.38 (both *p* < .005).

## Layout

```
model.py          the model: parameters, stages, composed rates, the pair channel
constraints.py    the codebook and the theory constraints C1-C6; Constraints surgery
data.py           corpus loading, the (actor, period) collapse, the holdout split
kernels.py        kernel-weighted pair counts at timescale delta
diagnostics.py    post-fit reads: coherence, boundary mass, node roles, edge mass
validate.py       the validation suite, including whole-actor holdout (V7)
experiments.py    the ladder: final2_arm / final2_sweep / keyword_beta, and the fit cache
fit.py            CLI: final2, final2-pick, report, template
report{,_data}.py the HTML report; report_template.html is its shell
actor_keywords.py the keyword covariates, and the regexes that derive them
run.py, backends.py   the extraction step (needs the private corpus and a credential)

make_release_data.py  builds release/snapshot/ from the private snapshot
make_tables.py        LaTeX for the concept and edge tables
make_ladder_table.py  LaTeX for the ladder table (main) and its appendix version
make_layer_inputs.py + plot_ideology_layers.py   the layered concept-network figure

test_*.py         117 tests; the ones covering archived code moved with it
release/snapshot/ the anonymized replication data
paper/            the generated tables and figure
old/              superseded code and data, kept for the record; not part of the release
```

`final2-pick` and the figure scripts write to the working directory; `paper/` holds the
copies the write-up includes.

## The extraction step

`run.py` codes tweets against the codebook with an LLM, writing `coding.jsonl`. It needs
the private corpus, so it cannot be re-run from this release; it is included because the
paper's appendix describes the prompt and the source is the authoritative statement of it.
No credential is stored in the repository — set `GATEWAY_API_KEY` in the environment, or
pass `--api-key`.

## What is not published

Built HTML reports, `report_data.json`, and any fit pickle produced against the private
corpus inline actor usernames, profile text and tweet text. The report *code* ships; built
reports do not. Building a report from `release/snapshot/` is safe: with no text or bios in
the corpus, `unit_text` returns nothing and actors are addressed by pseudonym.
