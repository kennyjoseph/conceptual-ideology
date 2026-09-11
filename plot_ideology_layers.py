"""
Layered network plot of ideological elements.

Inputs (same folder):
  ideology_nodes.csv          id, layer, CB, WS, IS, label
  ideology_edges_labeled.csv  source, target, ideology

Output:
  ideology_layers.png / .pdf

Usage:
  python plot_ideology_layers.py
"""
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Wedge
from matplotlib.transforms import Affine2D

# ---------------------------------------------------------------- data
nodes = {r["id"]: r for r in csv.DictReader(open("ideology_nodes.csv"))}
edges = list(csv.DictReader(open("ideology_edges_labeled.csv")))

LAYERS = ["Rationalizations", "Beliefs", "Attitudes", "Values"]   # top -> bottom
IDEOS = ["WS", "CB", "IS"]   # left-to-right order of the ideology columns
COLORS = {"CB": "#2b6a8e", "WS": "#b23a3a", "IS": "#8a2b8f"}
UNASSIGNED = "#9e9e9e"               # in no ideology's fitted support
NAMES = {"CB": "Colorblindness", "WS": "White supremacy", "IS": "Intersectionality"}

def membership(n):
    return tuple(i for i in IDEOS if nodes[n][i] == "1")

# ---------------------------------------------------------------- geometry
# Each layer is a parallelogram. A point at (u, v) in layer coordinates
# (u: 0..1 left->right, v: 0..1 front->back) maps to screen coordinates.
W, D, SHEAR = 22.0, 4.0, 4.0        # width, visual depth, horizontal shear of the back edge
GAP = 5.5                            # vertical spacing between layer front edges
DEPTH_ROWS = 3                       # rows of nodes within a layer (front to back)

def layer_base_y(li):                # li = 0 is the top layer
    return -li * GAP

def to_screen(li, u, v):
    return u * W + v * SHEAR, layer_base_y(li) + v * D

# Order nodes within each layer by ideology group so same-ideology nodes cluster
GROUP_ORDER = {("WS",): 0, ("WS", "IS"): 1, ("WS", "CB"): 2, ("CB",): 3, ("CB", "IS"): 4, ("IS",): 5}
pos = {}
for li, layer in enumerate(LAYERS):
    ids = [n for n in nodes if nodes[n]["layer"] == layer]
    ids.sort(key=lambda n: (GROUP_ORDER.get(membership(n), 9), int(n[1:])))
    k = len(ids)
    for j, n in enumerate(ids):
        u = (j + 0.5) / k
        v = ((j % DEPTH_ROWS) + 0.5) / DEPTH_ROWS      # stagger front/back to reduce overlap
        pos[n] = to_screen(li, u, v)

# ---------------------------------------------------------------- draw
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_aspect("equal")
ax.axis("off")

# planes
for li, layer in enumerate(LAYERS):
    corners = [to_screen(li, 0, 0), to_screen(li, 1, 0), to_screen(li, 1, 1), to_screen(li, 0, 1)]
    ax.add_patch(Polygon(corners, closed=True, facecolor="#f2d6c4", edgecolor="none", alpha=0.85, zorder=1))
    ax.text(-1.0, layer_base_y(li) + D / 2, layer, ha="right", va="center", fontsize=22, zorder=6)

# edges: one line per (source, target, ideology) row; shared edges get a small
# perpendicular offset per ideology so both colors stay visible.
OFFSET = {"WS": -0.12, "CB": 0.0, "IS": 0.12}
for e in edges:
    (x1, y1), (x2, y2) = pos[e["source"]], pos[e["target"]]
    dx = OFFSET[e["ideology"]]
    ax.plot([x1 + dx, x2 + dx], [y1, y2], color=COLORS[e["ideology"]], lw=0.9, alpha=0.55, zorder=2)

# nodes: single-ideology nodes are one color; shared nodes are split into
# equal wedges, one per ideology (drawn as a circle then squashed to an ellipse).
RX, RY = 0.45, 0.275
def draw_node(x, y, n):
    m = membership(n)
    squash = Affine2D().scale(1, RY / RX).translate(0, y - y * RY / RX) + ax.transData
    if not m:
        ax.add_patch(Ellipse((x, y), 2 * RX, 2 * RY, facecolor=UNASSIGNED,
                             edgecolor="black", lw=0.5, zorder=4))
    elif len(m) == 1:
        ax.add_patch(Ellipse((x, y), 2 * RX, 2 * RY, facecolor=COLORS[m[0]], edgecolor="black", lw=0.5, zorder=4))
    else:
        step = 360 / len(m)
        for k, ide in enumerate(m):
            w = Wedge((x, y), RX, 90 + k * step, 90 + (k + 1) * step,
                      facecolor=COLORS[ide], edgecolor="none", zorder=4, transform=squash)
            ax.add_patch(w)
        ax.add_patch(Ellipse((x, y), 2 * RX, 2 * RY, facecolor="none", edgecolor="black", lw=0.5, zorder=4))
    ax.text(x, y, n, ha="center", va="center", fontsize=5.5, color="white", zorder=5)

for n, (x, y) in pos.items():
    draw_node(x, y, n)

# legend
handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=COLORS[i], label=NAMES[i]) for i in IDEOS]
handles += [plt.Line2D([], [], marker="o", ls="", ms=10, color=UNASSIGNED,
                       label="In no ideology's support")]
handles += [plt.Line2D([], [], ls="", label="Split nodes are shared by both ideologies")]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=12, ncol=3)

ax.set_xlim(-8, W + SHEAR + 1)
ax.set_ylim(layer_base_y(len(LAYERS) - 1) - 1, D + 1)
plt.tight_layout()
plt.savefig("ideology_layers.png", dpi=200, bbox_inches="tight")
plt.savefig("ideology_layers.pdf", bbox_inches="tight")
print("saved ideology_layers.png / .pdf")
