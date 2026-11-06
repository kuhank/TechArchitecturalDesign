"""Black-and-white line drawings for the provisional specification."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.size": 9})


def box(ax, x, y, w, h, text, ref=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fill=False, lw=1.0, ec="k"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)
    if ref:
        ax.text(x + w + 0.08, y + h - 0.12, ref, fontsize=8, ha="left", va="top")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=0.9, color="k"))


def finish(fig, ax, name, title):
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    fig.savefig(f"drawings/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"drawings/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# FIG. 1 - system architecture
fig, ax = plt.subplots(figsize=(7.5, 9))
ax.set_xlim(0, 10); ax.set_ylim(0, 13)
box(ax, 3, 12, 4, 0.7, "Input / situation data", "102")
box(ax, 3, 10.8, 4, 0.8, "Perturbation module\n(K meaning-preserving variants)", "104")
box(ax, 3, 9.6, 4, 0.8, "Generator (agent / model / planner)\nproduces K candidate outputs", "106")
box(ax, 3, 8.4, 4, 0.8, "Decomposition module\noutput -> D dimensions", "108")
box(ax, 3, 7.2, 4, 0.8, "Chaos sensor: divergence d_j and\nexponent lambda_j per dimension", "110")
xs = [0.3, 2.7, 5.1, 7.5]
labels = ["Controller\ndim 1\n(precision)", "Controller\ndim 2\n(precision)", "Controller\ndim ...", "Controller\ndim D\n(creative)"]
for i, (x, l) in enumerate(zip(xs, labels)):
    box(ax, x, 5.3, 2.1, 1.2, l)
    arrow(ax, 5, 7.2, x + 1.05, 6.5)
ax.text(9.7, 6.3, "112", fontsize=8)
box(ax, 0.3, 3.9, 9.3, 0.8, "Intervention ladder: none | consensus | grounding | verification | escalation;  per-dimension temperature law", "114")
box(ax, 3, 2.6, 4, 0.8, "Master controller: consistency check,\ncomposition, stability index S", "116")
box(ax, 7.5, 2.6, 2.1, 0.8, "Horizon manager\n(agent Lyapunov time)", "118")
box(ax, 3, 1.2, 4, 0.8, "Final output / action", "120")
for (a, b) in [((5, 12), (5, 11.6)), ((5, 10.8), (5, 10.4)), ((5, 9.6), (5, 9.2)), ((5, 8.4), (5, 8.0)),
               ((5, 5.3), (5, 4.7)), ((5, 3.9), (5, 3.4)), ((5, 2.6), (5, 2.0))]:
    arrow(ax, *a, *b)
arrow(ax, 7.5, 3.0, 7.0, 3.0)
arrow(ax, 8.55, 3.4, 8.55, 11.0); arrow(ax, 8.55, 11.0, 7.0, 11.2)
ax.text(8.65, 7.5, "re-anchor\nschedule", fontsize=7.5)
finish(fig, ax, "fig1_system", "FIG. 1")

# FIG. 2 - method flowchart
fig, ax = plt.subplots(figsize=(6, 10))
ax.set_xlim(0, 8); ax.set_ylim(0, 14)
steps = [("S1  Receive input; generate K perturbed variants", 13),
         ("S2  Obtain K candidate outputs", 11.8),
         ("S3  Decompose each output into D dimensions", 10.6),
         ("S4  Compute divergence d_j and exponent\n      lambda_j = ln(d_j / eps) for each dimension", 9.3),
         ("S5  Creative dimension?  -> update temperature,\n      keep variation within band", 7.9),
         ("S6  Precision dimension inside band?\n      -> accept consensus value", 6.5),
         ("S7  Otherwise apply least-cost intervention\n      meeting error tolerance", 5.1),
         ("S8  Check cross-dimension consistency; repair", 3.8),
         ("S9  Compose output; compute stability index S", 2.6),
         ("S10 Estimate growth rate; schedule next\n      checkpoint at alpha * horizon", 1.3)]
for t, y in steps:
    box(ax, 0.5, y - 0.45, 7, 0.9, t)
for (_, y1), (_, y2) in zip(steps[:-1], steps[1:]):
    arrow(ax, 4, y1 - 0.45, 4, y2 + 0.45)
finish(fig, ax, "fig2_method", "FIG. 2")

# FIG. 3 - temperature law: stable region and period doubling (schematic)
import numpy as np, math
a, c, dstar = 1.0, 0.8, 0.4
ts = -math.log(1 - dstar / c) / a
slope = ts * c * a * math.exp(-a * ts)
ec = 2 / slope
fig, ax = plt.subplots(figsize=(6, 3.5))
X, Y = [], []
for eta in np.linspace(0.05, 2.5 * ec, 600):
    u = math.log(0.9)
    for i in range(800):
        tau = math.exp(min(u, 5)); u = max(min(u - eta * (c * (1 - math.exp(-a * tau)) - dstar), 5), -12)
        if i > 760: X.append(eta / ec); Y.append(math.exp(u))
ax.plot(X, Y, ",", color="k")
ax.axvline(1, color="k", ls="--", lw=0.8)
ax.set_yscale("log"); ax.set_xlabel("normalised controller gain  eta / eta_c"); ax.set_ylabel("temperature tau")
ax.text(0.3, 1.2, "stable", fontsize=8); ax.text(1.3, 1.2, "oscillating (period 2)", fontsize=8)
ax.set_title("FIG. 3", fontsize=10)
fig.savefig("drawings/fig3_gain_limit.pdf", bbox_inches="tight"); fig.savefig("drawings/fig3_gain_limit.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# FIG. 4 - robot embodiment
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
box(ax, 0.3, 6.6, 3, 0.9, "Sensors / situation\n(camera, lidar, map, task)", "402")
box(ax, 3.8, 6.6, 3, 0.9, "Planner generates K\ncandidate actions", "404")
box(ax, 0.3, 4.4, 4.4, 1.6, "Creative dimensions (406)\nroute, strategy, approach\n-> controlled chaos kept\n(exploration, new options)")
box(ax, 5.3, 4.4, 4.4, 1.6, "Safety dimensions (408)\nspeed, force, distance to people,\nrules -> controllers suppress\ndivergence, veto / project action")
box(ax, 2.8, 2.4, 4.4, 1.0, "Master controller (410)\ncompose admissible action", None)
box(ax, 2.8, 0.6, 4.4, 0.9, "Actuators (412)", None)
arrow(ax, 3.3, 7.05, 3.8, 7.05)
arrow(ax, 5.3, 6.6, 2.5, 6.0); arrow(ax, 5.3, 6.6, 7.5, 6.0)
arrow(ax, 2.5, 4.4, 4.2, 3.4); arrow(ax, 7.5, 4.4, 5.8, 3.4)
arrow(ax, 5, 2.4, 5, 1.5)
finish(fig, ax, "fig4_robot", "FIG. 4")

# FIG. 5 - checkpoint timeline
fig, ax = plt.subplots(figsize=(7, 3))
e0, D, lam, alpha = 0.01, 0.3, 0.35, 0.7
Th = math.log(D / e0) / lam; iv = alpha * Th
t = np.linspace(0, 30, 3000); e = e0 * np.exp(lam * (t % iv))
ax.plot(t, e, color="k", lw=1); ax.plot(t, e0 * np.exp(lam * t), "k:", lw=0.8)
ax.axhline(D, color="k", ls="--", lw=0.8); ax.text(0.2, D * 1.15, "tolerance Delta", fontsize=8)
for k in range(1, int(30 / iv) + 1): ax.axvline(k * iv, color="0.6", lw=0.5)
ax.set_yscale("log"); ax.set_ylim(e0 / 2, 2); ax.set_xlabel("agent step"); ax.set_ylabel("deviation e(t)")
ax.text(12, 1.1, "without checkpoints (dotted)", fontsize=7.5)
ax.set_title("FIG. 5", fontsize=10)
fig.savefig("drawings/fig5_checkpoints.pdf", bbox_inches="tight"); fig.savefig("drawings/fig5_checkpoints.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok")
