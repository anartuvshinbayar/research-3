"""Write paper/numbers.tex and the LaTeX tables from results/."""
import json

import numpy as np
import pandas as pd

from common import PROC, RES, ROOT

PAPER = ROOT / "paper"
ORDER = ["cosine", "cosine-window", "cosine-broadened", "cosine-broadened-window", "cnn-clean",
         "cnn-clean-window", "cnn-marsified", "cnn-marsified-window", "cnn-marsified+artifact"]
SHORT = {"cosine": "Library search", "cosine-window": "\\quad + skip fiber bump",
         "cosine-broadened": "\\quad + resolution-matched", "cosine-broadened-window": "\\quad + resolution-matched + skip",
         "cnn-clean": "CNN, lab spectra", "cnn-clean-window": "\\quad + skip fiber bump",
         "cnn-marsified": "CNN, simulated-Mars training", "cnn-marsified-window": "\\quad + skip fiber bump",
         "cnn-marsified+artifact": "\\quad + fiber-bump template"}
FULL = {"cosine": "Library search", "cosine-window": "Library search, skip bump",
        "cosine-broadened": "Library search, res.-matched", "cosine-broadened-window": "Library search, res.-matched + skip",
        "cnn-clean": "CNN, lab spectra", "cnn-clean-window": "CNN, lab spectra, skip bump",
        "cnn-marsified": "CNN, simulated Mars", "cnn-marsified-window": "CNN, sim.\\ Mars, skip bump",
        "cnn-marsified+artifact": "CNN, sim.\\ Mars + template"}
RECALL_GROUPS = ["olivine", "carbonate", "sulfate", "pyroxene", "feldspar", "silica"]
STRENGTH_TAG = {0.0: "Clean", 0.5: "Half", 1.0: "One", 1.5: "OneHalf", 2.0: "Two"}
EFFECT_TAG = {"none (lab only)": "None", "only res": "Res", "only shift": "Shift", "only mix": "Mix",
              "only bg": "Bg", "only noise": "Noise", "all five": "All"}
PAIR_TAG = {("cosine-window|mars", "cosine|mars"): "WinVsCos",
            ("cosine-window|mars", "cnn-clean|mars"): "WinVsCnn",
            ("cnn-clean|mars", "cnn-marsified|mars"): "CnnVsSim",
            ("cnn-marsified+artifact|mars", "cnn-marsified|mars"): "TplVsSim",
            ("cosine|mars", "cosine|open"): "MarsVsOpen",
            ("cosine-window|mars", "cosine-broadened-window|mars"): "WinVsBroadWin"}


def pct(x):
    return f"{100 * x:.0f}\\%"


def key(method, setting):
    return "".join(w.capitalize() for w in method.replace("+", "-").split("-")) + setting.capitalize()


def read_json(name):
    with open(RES / name) as f:
        return json.load(f)


def build_macros(summary, meta, mars, mr, lr):
    m = {
        "nLabSpectra": f"{len(meta):,}",
        "nSpecies": f"{meta.name.nunique():,}",
        "nSamples": f"{meta.rruff_id.nunique():,}",
        "nLabTrain": f"{summary['n_lab_train']:,}",
        "nLabTest": f"{summary['n_lab_test']:,}",
        "nLabTestSpecies": f"{summary['n_lab_test_species']:,}",
        "nMarsRows": "343",
        "nMarsMatched": f"{len(mars)}",
        "nMarsEval": f"{summary['n_mars_eval']}",
        "nMarsTargets": f"{summary['n_mars_targets']}",
        "pctOlivine": pct(summary["mars_eval_counts"]["olivine"] / summary["n_mars_eval"]),
    }
    maj = mr[mr.method == "always-olivine"].iloc[0]
    m["majAcc"], m["majBal"] = pct(maj.acc), pct(maj.bal_acc)

    recall_cols = [c for c in mr.columns if c.startswith("recall_")]
    for _, r in mr[mr.method != "always-olivine"].iterrows():
        k = key(r.method, r.setting)
        m[f"bal{k}"] = pct(r.bal_acc)
        m[f"balCI{k}"] = f"{pct(r.bal_lo)}--{pct(r.bal_hi)}"
        m[f"acc{k}"] = pct(r.acc)
        for c in recall_cols:
            if not np.isnan(r[c]):
                m[f"rec{c[7:].capitalize()}{k}"] = pct(r[c])
    for _, r in lr.iterrows():
        k = key(r.method, r.setting)
        m[f"labGrp{STRENGTH_TAG[r.strength]}{k}"] = pct(r.group_acc)
        m[f"labSp{STRENGTH_TAG[r.strength]}{k}"] = pct(r.species_acc)
    for (method, setting), g in pd.read_csv(RES / "mars_seed_results.csv").groupby(["method", "setting"]):
        m[f"seedSd{key(method, setting)}"] = f"{100 * g.bal_acc.std(ddof=1):.1f}" if len(g) > 1 else "--"

    ex = read_json("extra.json")
    m["fracAmorphOpen"] = pct(ex["open_top1_frac_amorphous"])
    m["fracOtherOpen"] = pct(ex["open_top1_frac_other_group"])
    m["fracAmorphWindow"] = pct(ex["open_window_top1_frac_amorphous"])
    m["energyBumpMars"] = pct(ex["median_energy_below_560_mars"])
    m["energyBumpLab"] = pct(ex["median_energy_below_560_lab"])
    m["simMars"] = f"{ex['median_best_cosine_mars']:.2f}"
    m["simLab"] = f"{ex['median_best_cosine_lab']:.2f}"
    cw = ex["acc_by_conf::cosine-window|mars"]
    m.update(confHighCW=pct(cw["high"]), confMedCW=pct(cw["medium"]), confLowCW=pct(cw["low"]))
    nc = ex["n_by_conf"]
    m.update(nConfHigh=str(nc["high"]), nConfMed=str(nc["medium"]), nConfLow=str(nc["low"]))

    devl = pd.read_csv(RES / "dev_library.csv").set_index("method").bal_acc
    devc = pd.read_csv(RES / "dev_selection.csv").set_index("method").bal_acc
    m.update(devCosine=pct(devl["cosine"]), devCosineWindow=pct(devl["cosine-window"]),
             devCnnClean=pct(devc["cnn-clean"]), devCnnMarsified=pct(devc["cnn-marsified"]),
             devCnnMarsifiedArtifact=pct(devc["cnn-marsified+artifact"]))
    m["nDevEval"] = str(summary["n_dev_eval"])

    for _, r in pd.read_csv(RES / "mars_results_sumrule.csv").iterrows():
        m[f"sumBal{key(r.method, r.setting)}"] = pct(r.bal_acc)
    if (RES / "single_effects.csv").exists():
        for _, r in pd.read_csv(RES / "single_effects.csv").iterrows():
            m[f"se{EFFECT_TAG[r.variant]}{r.split.capitalize()}"] = pct(r.bal_acc)
    for _, r in pd.read_csv(RES / "paired_tests.csv").iterrows():
        k = PAIR_TAG[(r.a, r.b)]
        m[f"d{k}"] = f"{100 * r['diff']:+.0f}"
        m[f"dCI{k}"] = f"{100 * r.lo:+.1f} to {100 * r.hi:+.1f}"
        m[f"p{k}"] = "<0.001" if r.p < 0.001 else f"{r.p:.2f}"
    return m


def table_mars(mars):
    ct = pd.crosstab(mars.primary, mars.split)
    ct = ct.reindex(RECALL_GROUPS + ["perchlorate", "serpentine"]).fillna(0).astype(int)
    lines = ["\\begin{tabular}{lrrr}", "\\toprule", "Primary group & Dev & Test & Total\\\\", "\\midrule"]
    for g, r in ct.iterrows():
        note = " (not scored)" if g == "serpentine" else ""
        lines.append(f"{g.capitalize()}{note} & {r.get('dev', 0)} & {r.get('test', 0)} & {r.sum()}\\\\")
    lines += ["\\midrule", f"Total & {ct['dev'].sum()} & {ct['test'].sum()} & {ct.values.sum()}\\\\",
              f"Targets (rocks) & {mars[mars.split == 'dev'].target.nunique()} & "
              f"{mars[mars.split == 'test'].target.nunique()} & {mars.target.nunique()}\\\\",
              "\\bottomrule", "\\end{tabular}"]
    return lines


def table_main(mr, lr):
    maj = mr[mr.method == "always-olivine"].iloc[0]
    lab = lr[lr.strength == 0.0].set_index(["method", "setting"]).group_acc
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             " & Lab (clean) & \\multicolumn{3}{c}{Mars test} \\\\", "\\cmidrule(lr){2-2}\\cmidrule(lr){3-5}",
             "Method & Group acc. & Acc. & Bal.\\ acc. & 95\\% CI (bal.) \\\\", "\\midrule"]
    for setting, title in [("open", "Open library"), ("mars", "Mars-candidate library")]:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}}\\\\")
        d = mr[mr.setting == setting].set_index("method")
        best = d.loc[ORDER, "bal_acc"].idxmax()
        for method in ORDER:
            r = d.loc[method]
            bal = f"\\textbf{{{pct(r.bal_acc)}}}" if method == best else pct(r.bal_acc)
            lines.append(f"{SHORT[method]} & {pct(lab[(method, 'open')])} & {pct(r.acc)} & {bal} & "
                         f"{pct(r.bal_lo)}--{pct(r.bal_hi)}\\\\")
        lines.append("\\addlinespace")
    lines += [f"Always ``olivine'' & -- & {pct(maj.acc)} & {pct(maj.bal_acc)} & -- \\\\",
              "\\bottomrule", "\\end{tabular}"]
    return lines


def table_recall(mr, counts):
    d = mr[mr.setting == "mars"].set_index("method")
    lines = ["\\begin{tabular}{l" + "c" * len(RECALL_GROUPS) + "}", "\\toprule",
             "Method & " + " & ".join(g.capitalize() for g in RECALL_GROUPS) + "\\\\",
             " & " + " & ".join(f"($n$={counts.get(g, 0)})" for g in RECALL_GROUPS) + "\\\\",
             "\\midrule"]
    for method in ORDER:
        lines.append(FULL[method] + " & " + " & ".join(pct(d.loc[method, f"recall_{g}"]) for g in RECALL_GROUPS) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


def main():
    summary = read_json("summary.json")
    meta = pd.read_csv(PROC / "rruff_meta.csv")
    m = np.load(PROC / "mars.npz", allow_pickle=True)
    mars = pd.DataFrame({k: m[k] for k in ["primary", "split", "target", "conf"]})
    mr = pd.read_csv(RES / "mars_results.csv")
    lr = pd.read_csv(RES / "lab_results.csv")

    macros = build_macros(summary, meta, mars, mr, lr)
    with open(PAPER / "numbers.tex", "w") as f:
        f.write("% Generated by src/make_numbers.py\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for name, lines in [("table_mars", table_mars(mars)), ("table_main", table_main(mr, lr)),
                        ("table_recall", table_recall(mr, summary["mars_eval_counts"]))]:
        (PAPER / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(macros)} macros and 3 tables")


if __name__ == "__main__":
    main()
