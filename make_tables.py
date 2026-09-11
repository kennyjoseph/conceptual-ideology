"""LaTeX tables for the top concepts and top edges of a fitted arm.

Reads the CSVs `experiments.final_csvs` writes, so the numbers in the paper and the
numbers in the supplement come from one place.

Two notes on the concept text. It is reproduced verbatim from `ideology_nodes_v3.csv`,
where 52 of the 112 elements separate clauses with semicolons and exactly one contains a
comma -- the pattern of a file whose commas were replaced during a CSV export. They read
oddly in print ("no bearing on worth; ability; or character"), so `--commas` rewrites
"; " to ", " for display; it is off by default because that edits the codebook's own
wording. The bracketed tags ([Abstract Liberalism], [Cultural Racism], ...) are the
codebook's frame labels and are kept.
"""

import argparse
import csv

TEX = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
       "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s: str, commas: bool = False) -> str:
    if commas:
        s = s.replace("; ", ", ")
    return "".join(TEX.get(c, c) for c in s)


RUNG = {"V": "Val", "A": "Attr", "B": "Bel", "R": "Rat"}
IDEO = {"CB": "Colorblindness", "IS": "Intersectionality"}


def concepts(path: str, top: int, commas: bool) -> str:
    """Concept, rung, the codebook's assumed ideology, and eta -- grouped by ideology.

    The grouping is a row header rather than a column: with three rows per ideology a
    repeated label costs more than it explains, and the reader needs to know which block
    they are in, not which block each row belongs to.
    """
    rows = list(csv.DictReader(open(path)))
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Highest-rate concepts per ideology in the pins-only fit "
           r"($F_k{=}1$, 2012--2022, $\mathcal{C}1$ and $\mathcal{C}2$ withheld). "
           r"$\eta$ is the log-rate deviation from the corpus background. "
           r"A \textcolor{green!55!black}{\checkmark} marks a concept the codebook "
           r"assigns to this ideology -- an assignment the fit never saw, since "
           r"$\mathcal{C}1$ was withheld.}",
           r"\label{tab:top-concepts}",
           r"\begin{tabular}{>{\raggedright\arraybackslash}p{8.6cm}lr}",
           r"\toprule",
           r"Concept & Type & $\eta$ \\",
           r"\midrule"]
    for i, k in enumerate(["CB", "IS"]):
        out.append(r"\multicolumn{3}{@{}l}{\footnotesize\itshape " + IDEO[k] + r"} \\")
        for r in [r for r in rows if r["ideology"] == k][:top]:
            # the tick is earned only where the codebook agrees; a concept the fit ranks
            # highly that theory assigns elsewhere gets nothing, which is the case the
            # column exists to make visible
            tick = (r" \textcolor{green!55!black}{\checkmark}"
                    if k in r["theory"].split("|") else "")
            out.append(f"{esc(r['text'], commas)}{tick} & {RUNG[r['rung']]} & "
                       f"{float(r['eta']):.2f} \\\\")
        if i == 0:
            out.append(r"\addlinespace")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def edges(path: str, top: int, commas: bool) -> str:
    rows = list(csv.DictReader(open(path)))
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Strongest excess-association edges per ideology. $W_k$ is the "
           r"kernel-weighted pair count in excess of what the actors' own margins and "
           r"the shared association layer predict; $W^{(0)}$ is that shared layer's mass "
           r"on the same pair, so a pair with $W^{(0)} > W_k$ is closer to common "
           r"currency than to this ideology's own structure.}",
           r"\label{tab:top-edges}",
           r"\begin{tabular}{llcrr>{\raggedright\arraybackslash}p{7.2cm}}",
           r"\toprule",
           r"Ideology & Pair & Rungs & $W_k$ & $W^{(0)}$ & Concepts \\",
           r"\midrule"]
    for i, k in enumerate(["CB", "IS"]):
        sel = [r for r in rows if r["ideology"] == k][:top]
        for j, r in enumerate(sel):
            name = {"CB": "Colorblindness", "IS": "Intersectionality"}[k] if j == 0 else ""
            pair = f"{r['node_u']}--{r['node_v']}"
            out.append(f"{name} & {pair} & {r['rung_u']}{r['rung_v']} & "
                       f"{float(r['w']):.2f} & {float(r['w_shared']):.2f} & "
                       f"{esc(r['text_u'], commas)} \\newline "
                       f"{esc(r['text_v'], commas)} \\\\")
        out.append(r"\addlinespace" if i == 0 else "")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(x for x in out if x != "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="fit_final2")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--commas", action="store_true",
                    help='render the codebook\'s "; " clause separators as ", "')
    ap.add_argument("--out", default="tables_final2.tex")
    a = ap.parse_args()
    tex = (concepts(f"{a.stem}_concepts.csv", a.top, a.commas) + "\n\n"
           + edges(f"{a.stem}_edges.csv", a.top, a.commas) + "\n")
    open(a.out, "w").write(tex)
    print(f"wrote {a.out}")
    print(tex)
