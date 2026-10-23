"""Write paper/numbers.tex (LaTeX macros) and small tables straight from results/, so the text
can never drift from the experiments."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PAPER = ROOT / "paper"
PROC = ROOT / "data/processed"


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}\\%"


def main():
    s = json.load(open(RES / "summary.json"))
    meta = pd.read_csv(PROC / "rruff_meta.csv")
    M = np.load(PROC / "mars.npz", allow_pickle=True)
    mars = pd.DataFrame({k: M[k] for k in ["primary", "split", "target", "conf"]})
    mr = pd.read_csv(RES / "mars_results.csv")
    lr = pd.read_csv(RES / "lab_results.csv")
    seeds = pd.read_csv(RES / "mars_seed_results.csv")

    macros = {
        "nLabSpectra": f"{len(meta):,}",
        "nSpecies": f"{meta.name.nunique():,}",
        "nSamples": f"{meta.rruff_id.nunique():,}",
        "nLabTrain": f"{s['n_lab_train']:,}",
        "nLabTest": f"{s['n_lab_test']:,}",
        "nLabTestSpecies": f"{s['n_lab_test_species']:,}",
        "nMarsRows": "343",
        "nMarsMatched": f"{len(mars)}",
        "nMarsEval": f"{s['n_mars_eval']}",
        "nMarsTargets": f"{s['n_mars_targets']}",
        "pctOlivine": pct(s["mars_eval_counts"]["olivine"] / s["n_mars_eval"]),
    }
    maj = mr[mr.method == "always-olivine"].iloc[0]
    macros["majAcc"] = pct(maj.acc)
    macros["majBal"] = pct(maj.bal_acc)

    def key(method, setting):
        return "".join(w.capitalize() for w in method.replace("+", "-").split("-")) + setting.capitalize()

    for _, r in mr[mr.method != "always-olivine"].iterrows():
        k = key(r.method, r.setting)
        macros[f"bal{k}"] = pct(r.bal_acc)
        macros[f"balCI{k}"] = f"{pct(r.bal_lo)}--{pct(r.bal_hi)}"
        macros[f"acc{k}"] = pct(r.acc)
        for c in [c for c in mr.columns if c.startswith("recall_")]:
            if not np.isnan(r[c]):
                macros[f"rec{c[7:].capitalize()}{k}"] = pct(r[c])
    for _, r in lr.iterrows():
        k = key(r.method, r.setting)
        tag = {0.0: "Clean", 0.5: "Half", 1.0: "One", 1.5: "OneHalf", 2.0: "Two"}[r.strength]
        macros[f"labGrp{tag}{k}"] = pct(r.group_acc)
        macros[f"labSp{tag}{k}"] = pct(r.species_acc)
    for (m, st), g in seeds.groupby(["method", "setting"]):
        k = key(m, st)
        macros[f"seedSd{k}"] = f"{100 * g.bal_acc.std(ddof=1):.1f}" if len(g) > 1 else "--"


    # Side analyses (results/extra.json, dev_selection.csv, dev_library.csv, single_effects.csv).
    ex = json.load(open(RES / "extra.json"))
    macros["fracAmorphOpen"] = pct(ex["open_top1_frac_amorphous"])
    macros["fracOtherOpen"] = pct(ex["open_top1_frac_other_group"])
    macros["fracAmorphWindow"] = pct(ex["open_window_top1_frac_amorphous"])
    macros["energyBumpMars"] = pct(ex["median_energy_below_560_mars"])
    macros["energyBumpLab"] = pct(ex["median_energy_below_560_lab"])
    macros["simMars"] = f"{ex['median_best_cosine_mars']:.2f}"
    macros["simLab"] = f"{ex['median_best_cosine_lab']:.2f}"
    cw = ex["acc_by_conf::cosine-window|mars"]
    macros.update(confHighCW=pct(cw["high"]), confMedCW=pct(cw["medium"]), confLowCW=pct(cw["low"]))
    nc = ex["n_by_conf"]
    macros.update(nConfHigh=str(nc["high"]), nConfMed=str(nc["medium"]), nConfLow=str(nc["low"]))
    devl = pd.read_csv(RES / "dev_library.csv").set_index("method").bal_acc
    devc = pd.read_csv(RES / "dev_selection.csv").set_index("method").bal_acc
    macros.update(devCosine=pct(devl["cosine"]), devCosineWindow=pct(devl["cosine-window"]),
                  devCnnClean=pct(devc["cnn-clean"]), devCnnMarsified=pct(devc["cnn-marsified"]),
                  devCnnMarsifiedArtifact=pct(devc["cnn-marsified+artifact"]))
    macros["nDevEval"] = str(s["n_dev_eval"])
    sr = pd.read_csv(RES / "mars_results_sumrule.csv")
    for _, r in sr.iterrows():
        macros[f"sumBal{key(r.method, r.setting)}"] = pct(r.bal_acc)
    se_path = RES / "single_effects.csv"
    if se_path.exists():
        se = pd.read_csv(se_path)
        tag = {"none (lab only)": "None", "only res": "Res", "only shift": "Shift", "only mix": "Mix",
               "only bg": "Bg", "only noise": "Noise", "all five": "All"}
        for _, r in se.iterrows():
            macros[f"se{tag[r.variant]}{r.split.capitalize()}"] = pct(r.bal_acc)

    # Paired cluster-bootstrap differences (results/paired_tests.csv, from paired_tests.py).
    pt = pd.read_csv(RES / "paired_tests.csv")
    names_pt = {("cosine-window|mars", "cosine|mars"): "WinVsCos", ("cosine-window|mars", "cnn-clean|mars"): "WinVsCnn",
                ("cnn-clean|mars", "cnn-marsified|mars"): "CnnVsSim", ("cnn-marsified+artifact|mars", "cnn-marsified|mars"): "TplVsSim",
                ("cosine|mars", "cosine|open"): "MarsVsOpen", ("cosine-window|mars", "cosine-broadened-window|mars"): "WinVsBroadWin"}
    for _, r in pt.iterrows():
        k = names_pt[(r.a, r.b)]
        macros[f"d{k}"] = f"{100 * r['diff']:+.0f}"
        macros[f"dCI{k}"] = f"{100 * r.lo:+.1f} to {100 * r.hi:+.1f}"
        macros[f"p{k}"] = "<0.001" if r.p < 0.001 else f"{r.p:.2f}"

    with open(PAPER / "numbers.tex", "w") as f:
        f.write("% Auto-generated by src/make_numbers.py -- do not edit by hand.\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # Table: Mars data by group and split.
    ct = pd.crosstab(mars.primary, mars.split)
    ct = ct.reindex(["olivine", "carbonate", "sulfate", "pyroxene", "feldspar", "silica", "perchlorate",
                     "serpentine"]).fillna(0).astype(int)
    lines = ["\\begin{tabular}{lrrr}", "\\toprule", "Primary group & Dev & Test & Total\\\\", "\\midrule"]
    for g, r in ct.iterrows():
        note = " (not scored)" if g == "serpentine" else ""
        lines.append(f"{g.capitalize()}{note} & {r.get('dev', 0)} & {r.get('test', 0)} & {r.sum()}\\\\")
    lines += ["\\midrule", f"Total & {ct['dev'].sum()} & {ct['test'].sum()} & {ct.values.sum()}\\\\",
              f"Targets (rocks) & {mars[mars.split == 'dev'].target.nunique()} & "
              f"{mars[mars.split == 'test'].target.nunique()} & {mars.target.nunique()}\\\\",
              "\\bottomrule", "\\end{tabular}"]
    (PAPER / "table_mars.tex").write_text("\n".join(lines) + "\n")

    # Table: main results (both settings).
    order = ["cosine", "cosine-window", "cosine-broadened", "cosine-broadened-window", "cnn-clean",
             "cnn-clean-window", "cnn-marsified", "cnn-marsified-window", "cnn-marsified+artifact"]
    names = {"cosine": "Library search", "cosine-window": "\\quad + skip fiber bump",
             "cosine-broadened": "\\quad + resolution-matched",
             "cosine-broadened-window": "\\quad + resolution-matched + skip",
             "cnn-clean": "CNN, lab spectra", "cnn-clean-window": "\\quad + skip fiber bump",
             "cnn-marsified": "CNN, simulated-Mars training",
             "cnn-marsified-window": "\\quad + skip fiber bump",
             "cnn-marsified+artifact": "\\quad + fiber-bump template"}
    lab_clean = lr[(lr.strength == 0.0)].set_index(["method", "setting"])
    rows = ["\\begin{tabular}{lcccc}", "\\toprule",
            " & Lab (clean) & \\multicolumn{3}{c}{Mars test} \\\\", "\\cmidrule(lr){2-2}\\cmidrule(lr){3-5}",
            "Method & Group acc. & Acc. & Bal.\\ acc. & 95\\% CI (bal.) \\\\", "\\midrule"]
    for setting, title in [("open", "Open library"), ("mars", "Mars-candidate library")]:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}}\\\\")
        d = mr[mr.setting == setting].set_index("method")
        best = d.loc[order, "bal_acc"].idxmax()
        for m in order:
            r = d.loc[m]
            lab = lab_clean.loc[(m, "open"), "group_acc"]
            bal = pct(r.bal_acc)
            if m == best:
                bal = f"\\textbf{{{bal}}}"
            rows.append(f"{names[m]} & {pct(lab)} & {pct(r.acc)} & {bal} & {pct(r.bal_lo)}--{pct(r.bal_hi)}\\\\")
        rows.append("\\addlinespace")
    rows += [f"Always ``olivine'' & -- & {pct(maj.acc)} & {pct(maj.bal_acc)} & -- \\\\",
             "\\bottomrule", "\\end{tabular}"]
    (PAPER / "table_main.tex").write_text("\n".join(rows) + "\n")

    # Table: per-class recall for the best Mars-candidate model and library search.
    full = {"cosine": "Library search", "cosine-window": "Library search, skip bump",
            "cosine-broadened": "Library search, res.-matched",
            "cosine-broadened-window": "Library search, res.-matched + skip",
            "cnn-clean": "CNN, lab spectra", "cnn-clean-window": "CNN, lab spectra, skip bump",
            "cnn-marsified": "CNN, simulated Mars", "cnn-marsified-window": "CNN, sim.\\ Mars, skip bump",
            "cnn-marsified+artifact": "CNN, sim.\\ Mars + template"}
    rec_cols = [f"recall_{c}" for c in ["olivine", "carbonate", "sulfate", "pyroxene", "feldspar", "silica"]]
    d = mr[mr.setting == "mars"].set_index("method")
    rows = ["\\begin{tabular}{l" + "c" * len(rec_cols) + "}", "\\toprule",
            "Method & " + " & ".join(c[7:].capitalize() for c in rec_cols) + "\\\\",
            " & " + " & ".join(f"($n$={s['mars_eval_counts'].get(c[7:], 0)})" for c in rec_cols) + "\\\\",
            "\\midrule"]
    for m in order:
        rows.append(full[m] + " & " + " & ".join(pct(d.loc[m, c]) for c in rec_cols) + "\\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "table_recall.tex").write_text("\n".join(rows) + "\n")
    print(f"wrote {len(macros)} macros + 3 tables")


if __name__ == "__main__":
    main()
