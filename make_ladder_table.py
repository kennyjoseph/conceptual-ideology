"""The ladder table, in LaTeX, from the sweep JSONs.

Regenerated from `exp/final2.json` and `exp/final2_ws.json` so the printed numbers cannot
drift from the fits. Conventions the table depends on:

  * Bold is reserved for coherence ratios with p < .01, so that a bold cell means one
    thing everywhere. The selected arm is marked by setting its name in bold instead.
  * With `n_perm = 200` the smallest attainable p is .005, so a stored p of 0.000 means
    no null draw beat the observed statistic, not p = 0.
  * A dash is not "not significant": it marks a component with no association mass at
    all, for which no ratio exists. `_ratio` returns nan there and inf in the different
    case of positive inside mass with zero boundary mass.
  * Shares come from `row["share"]`, the model's own assignment (ideology-margin argmax),
    so no community metadata is involved. A share printed as 0 is three actors -- the
    component's own pins, held at their fixed membership -- rather than an empty component.
"""

import json
import math

FILES = ["exp/final2.json", "exp/final2_ws.json"]
NICE = {"CB": "CB", "IS": "IS", "WS": "WS", "free1": "free"}
SELECTED = "f2-F1-x0"
BLOCKS = [("Fixed facets", "Share = CB / IS", lambda r: r["extra"] == 0 and "WS" not in r["names"]),
          ("+ free component", "Share = CB / IS / free", lambda r: r["extra"] == 1),
          ("+ White Supremacy", "Share = CB / WS / IS", lambda r: "WS" in r["names"])]


def ok(p) -> bool:
    return p is not None and not (isinstance(p, float) and math.isnan(p)) and p < 0.01


def coh(c) -> str:
    """The component's name alone: bold where it clears the null, dash where it has no
    association mass at all. The ratio itself lives in the appendix table -- printed here
    it invited reading a large number as a good one, which it is not: CB's ratio rises
    monotonically with F while its null rises with it."""
    v, nm = c["C"], NICE[c["k"]]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{nm}~--"
    return rf"\textbf{{{nm}}}" if ok(c["p"]) else nm


def main(out="table_ladder.tex"):
    rows = [r for f in FILES for r in json.load(open(f))["rows"]]
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\renewcommand{\arraystretch}{1.08}",
         r"\caption{Facet ladder. All arms use units from 2012--2022, withhold "
         r"$\mathcal{C}1$ and $\mathcal{C}2$ while retaining $\mathcal{C}3$, pin three "
         r"actors per modelled ideology, and use intercept-only $m_a$. $M$ is the number "
         r"of modelled components; \emph{free} components have no pins or codebook "
         r"assignment. Held-out is Poisson loss (lower is better); "
         r"$\overline{\mathrm{AUC}}$ is macro concept-mapping AUC against the withheld "
         r"codebook. Share is the percentage of actors assigned to each component, "
         r"ordered as each block indicates; a share of 0 means the component retains only "
         r"its three pinned actors. With 200 permutations the smallest attainable $p$ "
         r"is .005. A dash marks a component with no association mass, for which $C_k$ "
         r"is undefined. Coherence lists each component, in bold where its ratio "
         r"$C_k$ clears the matched permutation null at $p<.01$; the ratios themselves "
         r"are in Table~\ref{tab:ladder-full}. The selected arm is set in bold.}",
         r"\label{tab:ladder}",
         r"\begin{tabular}{@{}lrrrrl@{}}", r"\toprule",
         r"Arm & $M$ & Held-out & $\overline{\mathrm{AUC}}$ &",
         r"Share (\%) & Coherence $C_k$ \\", r"\midrule", ""]
    for bi, (title, order, pred) in enumerate(BLOCKS):
        if bi:
            L.append(r"\addlinespace[2pt]")
        L.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{title}: {order}}}}}\\[-2pt]")
        for r in [r for r in rows if pred(r)]:
            tot = r["share"]
            n = sum(tot)
            share = "/".join(str(round(100 * x / n)) for x in tot)
            arm = r["arm"].replace("f2-", "").replace("-x0", "").replace("-x1", "+free")
            nm = rf"\texttt{{\textbf{{{arm}}}}}" if r["arm"] == SELECTED else rf"\texttt{{{arm}}}"
            L.append(f"{nm} & {r['M']} & {r['held_out']:,.0f} & "
                     f"{r['mapping']['macro_auc']:.3f}".replace("& 0.", "& .")
                     + f" & {share} &")
            L.append("  " + ", ".join(coh(c) for c in r["coherence"]) + r" \\")
        L.append("")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(L) + "\n"
    open(out, "w").write(tex)
    print(tex)




def appendix(out="table_ladder_full.tex"):
    """Every reported quantity, one row per (arm, component), spanning both columns.

    The main table compresses coherence to a pass mark; everything it drops is here:
    the ratio and its null, the support size, the per-component retrieval scores, and
    the arm-level quantities (units, actors, background share, minimum effective actors
    per facet) that decide whether a row is worth reading at all.
    """
    rows = [r for f in FILES for r in json.load(open(f))["rows"]]
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Full results for every arm of the facet ladder. Arm-level rows give "
         r"the Poisson held-out loss, the macro concept-mapping AUC, the smallest "
         r"effective number of actors on any facet (an inverse-Simpson count: 1 means a "
         r"facet carried by a single actor, 607 the number of actors), and the mean "
         r"background share $\theta_{a,0}$. Component rows give the fitted support "
         r"$|S_k|$ and, against the withheld codebook assignment $G_k$, the ranking AUC "
         r"and the precision and recall of $S_k$; the share of actors whose ideology "
         r"margin is largest on that component; the membership mass; and the coherence "
         r"ratio $C_k$ with the mean of its matched permutation null and the resulting "
         r"$p$. Free components have no $G_k$ and are not scored. A dash for $C_k$ marks "
         r"a component with no association mass; $\infty$ marks positive mass inside the "
         r"support with none crossing its boundary, for which the null admits no "
         r"comparison. With 200 permutations the smallest attainable $p$ is .005.}",
         r"\label{tab:ladder-full}",
         r"\begin{tabular}{@{}llrrrrrrrrrr@{}}", r"\toprule",
         r"& & \multicolumn{4}{c}{Concept mapping} & \multicolumn{2}{c}{Actors} "
         r"& \multicolumn{3}{c}{Coherence} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}",
         r"Arm & Comp. & $|S_k|$ & $|G_k|$ & AUC & Prec./Rec. & Share & Mass "
         r"& $C_k$ & Null & $p$ \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i:
            L.append(r"\addlinespace[1.5pt]")
        tot = r["share"]
        n = sum(tot)
        arm = r["arm"].replace("f2-", "").replace("-x0", "").replace("-x1", "+free")
        L.append(rf"\multicolumn{{11}}{{@{{}}l}}{{\texttt{{{arm}}}\quad "
                 rf"\footnotesize $M{{=}}{r['M']}$, {r['n_pins']} pins, "
                 rf"{r['units']:,} units, {r['actors']} actors; "
                 rf"held-out {r['held_out']:,.0f}, "
                 rf"$\overline{{\mathrm{{AUC}}}}$ {r['mapping']['macro_auc']:.3f}, "
                 rf"min.\ eff.\ actors/facet {r['min_effective_facet']:.1f}, "
                 rf"background {r['background']:.3f}}} \\")
        coh_by = {c["k"]: c for c in r["coherence"]}
        for j, m in enumerate(r["mapping"]["per_ideology"]):
            nm = NICE[m["ideology"]]
            c = coh_by[m["ideology"]]
            v = c["C"]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cs, nulls, ps = "--", "--", "--"
            elif math.isinf(v):
                cs, nulls, ps = r"$\infty$", f"{c['null']:.2f}", "--"
            else:
                cs = rf"\textbf{{{v:.2f}}}" if ok(c["p"]) else f"{v:.2f}"
                nulls = f"{c['null']:.2f}"
                ps = r"$<$.005" if c["p"] < 0.005 else f"{c['p']:.3f}".lstrip("0")
            if m["auc"] is None:
                sc = "-- & -- & --"
            else:
                sc = (f"{m['gold']} & {m['auc']:.3f} & "
                      f"{m['precision']:.2f}/{m['recall']:.2f}")
            L.append(f"& {nm} & {m['support']} & {sc} & "
                     f"{100 * tot[j] / n:.0f}\\% & {r['mass'][j]:.1f} & "
                     f"{cs} & {nulls} & {ps} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    tex = "\n".join(L) + "\n"
    open(out, "w").write(tex)
    print(f"wrote {out}")
    return tex


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--appendix", action="store_true",
                    help="the full per-component table instead of the main one")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.appendix:
        appendix(a.out or "table_ladder_full.tex")
    else:
        main(a.out or "table_ladder.tex")
