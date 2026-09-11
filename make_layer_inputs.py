"""Build plot_ideology_layers.py's two inputs from a fitted model.

The plotter reads `ideology_nodes.csv` (id, layer, CB, WS, IS) and
`ideology_edges_labeled.csv` (source, target, ideology). Both are written here from the
fit rather than from the codebook, so the picture shows what the model *recovered* --
membership is the fitted support S_k = {u : max_f eta_ku > 0}, not the theory file's
CB/WS/IS columns.

Every concept is written, including the ones the fit assigns to no ideology (12 of 112
in the F=1 arm): they carry 0 in all three columns and the plotter draws them grey. They
are worth showing rather than dropping -- a concept the codebook lists and the extractor
finds, that neither component elevates, is a result about the codebook.

Edges are kept only where both endpoints are in that ideology's own support, which is the
same set the coherence ratio calls "inside"; an edge to a node outside it would also raise
a KeyError from the plotter's `pos`.

WS is written as an all-zero column: it is not modeled in this arm, and the plotter's
IDEOS list expects the field to exist.
"""

import csv
import pickle
import sys

import numpy as np

import model as M


def build(pkl="fit_final2.pkl", nodes_out="ideology_nodes.csv",
          edges_out="ideology_edges_labeled.csv", codebook="ideology_nodes_v3.csv"):
    import experiments as E
    saved = pickle.load(open(pkl, "rb"))
    arm = saved["provenance"]["arm"]
    F, extra = saved["provenance"]["facets"], saved["provenance"]["extra"]
    p, S, cons, corpus, infos, hit, hF, tag = E.final2_arm(F, extra)
    if tag != arm:
        raise SystemExit(f"{pkl} says arm={arm} but rebuilding gave {tag}")

    ieta = M.ideology_profile(p, S)
    supp = {nm: ieta[k] > 0 for k, nm in enumerate(cons.names)}
    book = {r["id"]: r for r in csv.DictReader(open(codebook))}
    drawn, rows = set(), []
    for u, nid in enumerate(cons.node_ids):
        m = [nm for nm in cons.names if supp[nm][u]]
        drawn.add(nid)
        rows.append({"id": nid, "layer": book[nid]["layer"],
                     "CB": int("CB" in m), "WS": 0, "IS": int("IS" in m),
                     "label": book[nid]["element"]})
    with open(nodes_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "layer", "CB", "WS", "IS", "label"])
        w.writeheader()
        w.writerows(rows)

    ui, vi = cons.pairs[:, 0], cons.pairs[:, 1]
    erows = []
    for k, nm in enumerate(cons.names):
        keep = (p.W[k] > 1e-8) & supp[nm][ui] & supp[nm][vi]
        for e in np.flatnonzero(keep):
            a, b = cons.node_ids[ui[e]], cons.node_ids[vi[e]]
            if a in drawn and b in drawn:
                erows.append({"source": a, "target": b, "ideology": nm,
                              "w": round(float(p.W[k, e]), 4)})
    erows.sort(key=lambda r: -r["w"])
    with open(edges_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target", "ideology", "w"])
        w.writeheader()
        w.writerows(erows)

    n_by = {nm: sum(r[nm] for r in rows) for nm in ("CB", "IS")}
    print(f"{pkl} ({arm}): {len(rows)} of {cons.K} nodes drawn "
          f"(CB {n_by['CB']}, IS {n_by['IS']}, "
          f"shared {sum(1 for r in rows if r['CB'] and r['IS'])}, "
          f"unassigned {sum(1 for r in rows if not (r['CB'] or r['IS']))}), "
          f"{len(erows)} edges -> {nodes_out}, {edges_out}")


if __name__ == "__main__":
    build(*(sys.argv[1:] or []))
