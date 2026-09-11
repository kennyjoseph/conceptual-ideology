"""Fit the v3.2 ideology model and report its diagnostics.

  fit.py check                 load the theory files and the corpus, report sizes
  fit.py run                   fit the build order and print the diagnostics
  fit.py sweep-m               add free ideologies and pick M by held-out fit
  fit.py sweep-delta           V6: W_k(delta) across the timescale sweep
  fit.py contest               section 8's facet--W contest across the F_k ladder
  fit.py sweep-facets          section 8 stage 2: raise F_k, select it by V7
  fit.py validate              V1-V8: hold out theory, groups, cells or whole actors
  fit.py null                  the zero-ideology test of section 7
  fit.py template              (re)write the actor file with every actor in the corpus
  fit.py paper                 the paper's four analyses (E1-E4); see runs.sh
  fit.py report                serve the fitted model's report on localhost

Everything defaults to the files in this directory; --codings is the JSONL that run.py
writes, and must have been produced against the current node file.

Two options are worth reading before the first run.

  --meta      which actor-file columns drive the C4 alignment regression. The default is
              every column named `m_*`; naming columns here overrides that. `check`
              prints the resulting design matrix.
  --kernel/--delta
              the co-occurrence timescale. `same` is within-expression, `all` is
              within-actor-ever, `exp`/`box` with --delta interpolate. C7 requires the
              kernel to be stated with every reported network, so it is printed with all
              of them.
"""

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import typer

import constraints as C
import data as D
import diagnostics as G
import model as M

app = typer.Typer(help=__doc__, add_completion=False)


@app.command()
def report(
    pkl: Path = typer.Option(Path("fit.pkl"), help="the fit to report on"),
    codings: Path = typer.Option(D.CODINGS),
    parquet: Path = typer.Option(D.PARQUET, help="use snapshot/corpus.parquet for a "
                                                 "fit made against the frozen snapshot"),
    port: int = typer.Option(8000),
    host: str = typer.Option("127.0.0.1", help="loopback by default; 0.0.0.0 to expose "
                                               "it on your network"),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
    standalone: Path = typer.Option(None, help="instead of serving, write one HTML file "
                                               "with the data inlined"),
) -> None:
    """Serve the report for a saved fit, or build a standalone copy of it.

    Serving is the working mode: the data is regenerated whenever the fit is newer than
    the last build, so refitting and reloading the browser is the whole loop. Only the
    page and its data are inside the served directory -- the parquet, the codings and the
    pickles are not reachable over the socket.

    The corpus is pinned to the unit ids stored in the fit, so a coding file that has
    grown since the fit does not silently move the baseline the rates are measured
    against; if any of the fit's units have gone missing, the build refuses.
    """
    import report as R

    if standalone is not None:
        R.build(pkl=pkl, codings=codings, parquet=parquet, out=standalone, refresh=True)
        return
    R.serve(pkl=pkl, codings=codings, parquet=parquet, port=port, host=host,
            open_browser=open_browser, root=Path(".report"))


@app.command("final2")
def final2(
    ladder: str = typer.Option("1,2,3,4", help="the F_k values to try"),
    years: str = typer.Option("2012-2022", help="inclusive window on unit dates"),
    max_iter: int = typer.Option(30),
    skip_ws: bool = typer.Option(False, "--skip-ws",
                                 help="omit the arms that model white supremacy"),
    skip_beta: bool = typer.Option(False, "--skip-beta",
                                   help="omit the keyword-covariate arm"),
) -> None:
    """The paper's ladder: F_k x {no free component, one free component} x {WS, no WS}.

    Three sweeps in one command, because they are one table. The first two vary the facet
    count with white supremacy left unmodelled, with and without a free component; the
    third models white supremacy, which is the evidence for leaving it out. `keyword-beta`
    follows as a separate arm fit on every unit with no holdout, so nothing it reports is
    a test.

    Fits are cached by configuration, so re-running skips whatever finished.
    """
    import experiments as E

    lo, hi = (int(x) for x in years.split("-"))
    L = tuple(int(x) for x in ladder.split(",") if x.strip())
    h = M.Hyper(max_iter=max_iter)
    res = E.final2_sweep(ladder=L, extras=(0, 1), years=(lo, hi), h=h)
    rows = list(res["rows"])
    if not skip_ws:
        rows += E.final2_sweep(ladder=L, extras=(0,), years=(lo, hi), h=h, ws=True,
                               out_name="final2_ws.json")["rows"]
    G.head("the final ladder",
           f"Units {lo}-{hi}. Stripped: C1, C2; kept: C3. Three pins per modelled "
           "ideology; m_a intercept-only.",
           "share is the percentage of actors whose ideology margin is largest on each "
           "component -- the degeneracy check. A component at ~100% has absorbed the "
           "corpus; one at 0% retains only its pins.")
    print(f"  {'arm':<14}{'M':>2}{'held-out':>11}{'macroAUC':>10}  "
          f"{'share':<16}coherence")
    for r in rows:
        coh = "  ".join(_coh_cell(c) for c in r["coherence"])
        n = sum(r["share"])
        share = "/".join(f"{round(100 * x / n)}" for x in r["share"])
        print(f"  {r['arm']:<14}{r['M']:>2}{r['held_out']:>11,.0f}"
              f"{r['mapping']['macro_auc']:>10.3f}  {share:<16}{coh}")
    if not skip_beta:
        kb = E.keyword_beta(F=1, extra=0, years=(lo, hi), h=h)
        G.head("the keyword-covariate arm (no holdout; nothing here is a test)",
               "beta is a multivariate coefficient on standardized columns, so a "
               "covariate collinear with a stronger one reads as zero however well it "
               "predicts alone -- the univariate column is printed beside it.")
        print(f"  {'covariate':<17}" + "".join(f"{c:>9}" for c in kb["columns"])
              + f"{'n':>5}{'univar':>9}{'multivar':>10}")
        for r in kb["rows"]:
            tail = (f"{r['n']:>5}{r['univariate']:>9.2f}{r['multivariate']:>10.2f}"
                    if "n" in r else "")
            print(f"  {r['covariate']:<17}"
                  + "".join(f"{x:>9.3f}" for x in r["beta"]) + tail)
    print("\n  wrote exp/final2.json, exp/final2_ws.json, exp/kwbeta.json")
    print("  pick an arm:  python fit.py final2-pick --arm f2-F1-x0")


def _coh_cell(c: dict) -> str:
    """One coherence cell. A dash is not "not significant": it is a component with no
    association mass at all, for which the ratio is undefined."""
    import math
    v, p = c["C"], c["p"]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{c['k']} --"
    vs = "inf" if math.isinf(v) else f"{v:.2f}"
    ps = "" if p is None or (isinstance(p, float) and math.isnan(p)) else f"(p{p:.3f})"
    return f"{c['k']} {vs}{ps}"


@app.command("final2-pick")
def final2_pick(
    arm: str = typer.Option("f2-F1-x0", help="the arm to keep, e.g. f2-F1-x0"),
    years: str = typer.Option("2012-2022"),
    out: Path = typer.Option(Path("fit_final2.pkl")),
    top: int = typer.Option(40, help="rows per ideology in the CSVs"),
    max_iter: int = typer.Option(30),
    report: bool = typer.Option(True, "--report/--no-report",
                                help="also build the standalone HTML report"),
) -> None:
    """Write one arm out: the fit, its concept and edge CSVs, and the HTML report."""
    import experiments as E
    import report as R

    lo, hi = (int(x) for x in years.split("-"))
    F = int(arm.split("-")[1][1:])
    extra = 1 if "-x1" in arm else 0
    ws = arm.endswith("-ws")
    p, S, cons, corpus, infos, hit, hF, tag = E.final2_arm(
        F, extra, (lo, hi), M.Hyper(max_iter=max_iter), ws=ws)
    if tag != arm:
        raise SystemExit(f"asked for {arm} but the config rebuilds as {tag}")
    print(f"{tag}: cached={hit}, M={cons.M} {list(cons.names)}, "
          f"pins={int(cons.theta_fix.sum())}, units={corpus.D}, actors={corpus.A}")

    stem = str(out).replace(".pkl", "")
    with open(out, "wb") as f:
        pickle.dump({"params": p, "infos": infos, "hyper": hF, "cons": cons,
                     "actors": corpus.actors, "periods": corpus.periods,
                     "meta_names": corpus.meta_names, "unit_ids": corpus.unit_ids,
                     "codings": {"path": str(E.SNAP / "coding.jsonl"),
                                 "sha8": E.fingerprint()["coding"]["sha8"],
                                 "lines": E.fingerprint()["coding"]["lines"]},
                     "provenance": {"kind": "final2", "arm": arm, "facets": F,
                                    "extra": extra, "ws": ws, "years": [lo, hi],
                                    "stripped": ["C1", "C2"], "kept": ["C3"],
                                    "pins": int(cons.theta_fix.sum())}}, f)
    print(f"  wrote {out}")
    c, e = E.final_csvs(p, S, cons, stem=stem, top=top)
    print(f"  wrote {c}, {e}")
    if report:
        html = Path(f"{stem}_report.html")
        R.build(pkl=out, codings=E.SNAP / "coding.jsonl",
                parquet=E.SNAP / "corpus.parquet", out=html, refresh=True)


@app.command()
def template(
    parquet: Path = typer.Option(D.PARQUET),
    force: bool = typer.Option(False, "--force", help="regenerate, merging what is "
                                                      "already filled in"),
) -> None:
    """(Re)write the actor file with every actor in the corpus.

    Merge-aware: pinned rows, ideology cells and any `m_*` columns already in the file
    are carried over, so regenerating after new actors appear costs no hand-work.
    """
    C.write_actor_template(parquet, force=force)


if __name__ == "__main__":
    app()
