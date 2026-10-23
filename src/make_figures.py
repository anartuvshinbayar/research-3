"""Figures for the paper. Usage: python make_figures.py [name ...]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import PROC, RES, ROOT
from spectra import GRID, marsify, preprocess

FIG = ROOT / "paper/figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRIDC, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "lines.linewidth": 1.4, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def fig_domain_gap():
    meta = pd.read_csv(PROC / "rruff_meta.csv")
    R = np.load(PROC / "rruff.npz", allow_pickle=True)
    raw = np.load(PROC / "rruff_raw.npz", allow_pickle=True)
    M = np.load(PROC / "mars.npz", allow_pickle=True)
    mars = pd.DataFrame({k: M[k] for k in ["product", "target", "primary", "groups", "conf", "split"]})
    T = np.load(RES / "artifact_template.npy")

    examples = [("olivine", "Forsterite", "Olivine"), ("carbonate", "Magnesite", "Fe/Mg carbonate"),
                ("sulfate", "Anhydrite", "Ca sulfate (anhydrite)")]
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(4, 1, figsize=(6.6, 7.4), sharex=True)
    for ax, (grp, ref, title) in zip(axes, examples):
        cand = mars[(mars.split == "dev") & (mars.primary == grp) & (mars.conf == "high")
                    & (mars.groups == grp)]
        if cand.empty:
            cand = mars[(mars.split == "dev") & (mars.primary == grp)]
        i = cand.index[0]
        j = meta[(meta.name == ref) & (meta.laser == "532")].index[0]
        x, y = marsify(raw["x"][j], raw["y"][j], rng, other=None, parts=["res", "shift", "bg", "noise"])
        sim = preprocess(x, y)
        ax.plot(GRID, R["X"][j], color=ORANGE, lw=1.1, label="Lab spectrum (RRUFF)")
        ax.plot(GRID, sim, color=AQUA, lw=1.1, label="Same lab spectrum, simulated Mars conditions")
        ax.plot(GRID, M["X"][i], color=BLUE, lw=1.4, label="Mars spectrum (SuperCam)")
        tname = str(mars.target[i]).replace("_", " ")
        ax.set_title(f"{title}: RRUFF {ref.lower()} vs. SuperCam target “{tname}”",
                     loc="left", color=INK, fontweight="bold")
        ax.set_ylabel("Norm. intensity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.025))
    ax = axes[3]
    ax.fill_between(GRID, 0, T, color=BLUE, alpha=0.12, lw=0)
    ax.plot(GRID, T, color=BLUE, lw=1.4)
    ax.axvspan(150, 560, color=GRIDC, alpha=0.35, lw=0)
    ax.text(160, T.max() * 1.05, "\u201cFiber bump\u201d (skipped region)", ha="left", va="top",
            color=INK2, fontsize=7.5)
    ax.set_title("Fiber-bump template: median of development-split SuperCam spectra", loc="left",
                 color=INK, fontweight="bold")
    ax.set_ylabel("Norm. intensity")
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_xlim(GRID[0], GRID[-1])
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig1_domain_gap")


LABELS = {
    "cosine": "Library search",
    "cosine-window": "Library search, skip fiber bump",
    "cosine-broadened": "Library search, resolution-matched",
    "cosine-broadened-window": "Library search, res.-matched + skip",
    "cnn-clean": "CNN, lab spectra",
    "cnn-clean-window": "CNN, lab spectra, skip fiber bump",
    "cnn-marsified": "CNN, simulated-Mars training",
    "cnn-marsified-window": "CNN, simulated Mars, skip fiber bump",
    "cnn-marsified+artifact": "CNN, simulated Mars + fiber-bump template",
}
MAIN_ORDER = list(LABELS)
LIGHT_BLUE = "#86b6ef"


def pct_axis(ax, axis="x"):
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ["0%", "25%", "50%", "75%", "100%"]
    if axis == "x":
        ax.set_xlim(0, 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ax.set_ylim(0, 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def hbars(ax, d, highlight):
    y = np.arange(len(d))[::-1]
    colors = [BLUE if m == highlight else LIGHT_BLUE for m in d.index]
    ax.barh(y, d.bal_acc, height=0.55, color=colors, lw=0)
    ax.errorbar(d.bal_acc, y, xerr=[d.bal_acc - d.bal_lo, d.bal_hi - d.bal_acc], fmt="none",
                ecolor=INK2, elinewidth=0.8, capsize=2)
    for yi, v, hi in zip(y, d.bal_acc, d.bal_hi):
        ax.text(hi + 0.015, yi, f"{v:.0%}", va="center", color=INK, fontsize=7)
    pct_axis(ax)
    ax.grid(axis="y", visible=False)
    return y


def fig_mars_bars():
    df = pd.read_csv(RES / "mars_results.csv")
    maj = df[df.method == "always-olivine"].bal_acc.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
    for ax, (setting, title) in zip(axes, [("open", "Open library (all RRUFF species)"),
                                           ("mars", "Mars-candidate library (9 groups)")]):
        d = df[df.setting == setting].set_index("method").reindex(MAIN_ORDER)
        hbars(ax, d, d.bal_acc.idxmax())
        ax.set_ylim(-0.6, len(d) + 0.1)
        ax.axvline(maj, color=MUTED, lw=0.8)
        ax.text(maj + 0.012, len(d) - 0.25, "always \u201colivine\u201d", color=INK2, fontsize=6.5, va="bottom")
        ax.axhline(len(d) - 4.5, color=AXIS, lw=0.6)
        ax.set_title(title, loc="left", color=INK, fontweight="bold")
        ax.set_xlabel("Balanced accuracy, Mars test set")
    axes[0].set_yticks(np.arange(len(MAIN_ORDER))[::-1])
    axes[0].set_yticklabels([LABELS[m] for m in MAIN_ORDER])
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig2_mars_accuracy")


def fig_sim_vs_real():
    lab = pd.read_csv(RES / "lab_results.csv")
    mars = pd.read_csv(RES / "mars_results.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1, 1.1]))
    series = [("cosine", BLUE), ("cosine-window", ORANGE), ("cnn-clean", AQUA), ("cnn-marsified", YELLOW)]
    d0 = lab[lab.setting == "open"]
    for m, c in series:
        d = d0[d0.method == m].sort_values("strength")
        a.plot(d.strength, d.group_acc, color=c, lw=2, marker="o", ms=4.5, mec="white", mew=1.2, label=LABELS[m])
    a.set_xlabel("Simulated-Mars strength (0 = clean)")
    a.set_ylabel("Group accuracy, held-out lab samples")
    a.set_ylim(0.6, 1.0)
    a.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    a.set_yticklabels(["60%", "70%", "80%", "90%", "100%"])
    a.set_xticks(sorted(d0.strength.unique()))
    a.legend(loc="lower left", fontsize=6.5)
    a.set_title("(a) Robustness to simulated Mars", loc="left", color=INK, fontweight="bold")

    sim = d0[d0.strength == 1.0].set_index("method").group_acc
    real = mars[mars.setting == "mars"].set_index("method").bal_acc
    ms = [m for m in MAIN_ORDER if m in sim.index]
    is_cnn = np.array([m.startswith("cnn") for m in ms])
    b.scatter(sim[ms][~is_cnn], real[ms][~is_cnn], s=46, color=BLUE, edgecolor="white", lw=1.5, zorder=3,
              label="Library search")
    b.scatter(sim[ms][is_cnn], real[ms][is_cnn], s=46, color=ORANGE, edgecolor="white", lw=1.5, zorder=3,
              marker="s", label="CNN")
    short = {"cosine-window": "skip bump", "cnn-marsified": "simulated Mars", "cnn-clean": "lab only",
             "cosine": "plain", "cnn-marsified+artifact": "sim. Mars + template"}
    for m in ms:
        if m in short:
            b.annotate(short[m], (sim[m], real[m]), xytext=(5, 4), textcoords="offset points", fontsize=6.5,
                       color=INK2)
    b.set_xlabel("Lab accuracy under simulated Mars (strength 1)")
    b.set_ylabel("Balanced accuracy, real Mars test set")
    b.set_xlim(0.8, 1.0)
    b.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    b.set_xticklabels(["80%", "85%", "90%", "95%", "100%"])
    b.set_ylim(0.3, 0.8)
    b.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    b.set_yticklabels(["30%", "40%", "50%", "60%", "70%", "80%"])
    b.legend(loc="lower left", fontsize=6.5)
    b.set_title("(b) Simulated robustness vs. real Mars", loc="left", color=INK, fontweight="bold")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_sim_vs_real")


def _confusion(ax, p, col, title):
    rows = ["olivine", "carbonate", "sulfate", "pyroxene", "feldspar", "silica"]
    cols = rows + [c for c in ["serpentine", "perchlorate", "phosphate", "other"] if c in set(p[col])]
    C = np.zeros((len(rows), len(cols)), int)
    for prim, groups, pred in zip(p.primary, p.groups, p[col]):
        shown = prim if pred in groups.split("|") else pred
        C[rows.index(prim), cols.index(shown)] += 1
    R = C / C.sum(1, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    ax.imshow(R, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if C[i, j]:
                ax.text(j, i, C[i, j], ha="center", va="center", fontsize=6.5,
                        color="white" if R[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r} ({C[i].sum()})" for i, r in enumerate(rows)])
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_confusion():
    p = pd.read_csv(RES / "mars_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    _confusion(axes[0], p, "cosine-window|mars", "Library search, skip fiber bump")
    _confusion(axes[1], p, "cnn-clean|mars", "CNN, lab spectra")
    axes[0].set_ylabel("SuperCam team identification (n)")
    for ax in axes:
        ax.set_xlabel("Predicted group")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig4_confusion")


def fig_single_effects():
    path = RES / "single_effects.csv"
    if not path.exists():
        print("skip single_effects (no results yet)")
        return
    df = pd.read_csv(path)
    d = df[df.split == "test"].set_index("variant")
    order = ["none (lab only)", "only res", "only shift", "only mix", "only bg", "only noise", "all five"]
    names = {"none (lab only)": "No simulation (lab spectra)", "only res": "Resolution blur only",
             "only shift": "Calibration shift only", "only mix": "Mineral mixing only",
             "only bg": "Broad background only", "only noise": "Noise only", "all five": "All five effects"}
    d = d.reindex(order)
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    y = hbars(ax, d, "none (lab only)")
    ax.set_yticks(y)
    ax.set_yticklabels([names[v] for v in order])
    ax.set_xlabel("Balanced accuracy, Mars test set (2-seed CNN)")
    fig.tight_layout()
    save(fig, "fig5_single_effects")


if __name__ == "__main__":
    which = sys.argv[1:] or ["domain_gap", "mars_bars", "sim_vs_real", "confusion", "single_effects"]
    for w in which:
        globals()[f"fig_{w}"]()
