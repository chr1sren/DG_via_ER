"""
Reproduction & ablation for "Domain Generalization via Entropy Regularization" (NeurIPS 2020).
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TARGETS = ["art_painting", "cartoon", "photo", "sketch"]
EXP_BASE = "experiments"
NETWORK = "resnet18"
DATASET = "PACS"
ABLATION_COMBOS = [
    (0.5, 0.0, 0.0, "alpha2=0, alpha3=0\n(only alpha1)"),
    (0.5, 0.05, 0.0, "alpha2 on, alpha3=0"),
    (0.5, 0.0, 0.1, "alpha2=0, alpha3 on"),
    (0.5, 0.05, 0.1, "Full\n(alpha1,alpha2,alpha3)"),
]
ABLATION_EPOCHS = 30
PLOT_YLIM = (60, 100)
OUT_DIR = "figures"
EXP_ABLATION = "ablation_combos"


def run_train(extra_args, exp_folder=EXP_BASE, run_suffix=""):
    cmd = [sys.executable, "train.py", "--exp_folder", exp_folder] + extra_args
    if run_suffix:
        cmd += ["--run_suffix", run_suffix]
    return subprocess.call(cmd)


def main_reproduction(epochs):
    os.chdir(ROOT)
    extra = ["--epochs", str(epochs)] if epochs else []
    for i, target in enumerate(TARGETS):
        print("\n[%d/%d] target = %s" % (i + 1, len(TARGETS), target))
        ret = run_train(["--target", target] + extra)
        if ret != 0:
            sys.exit(ret)
    print("Reproduction done. Run: python run_all.py plot")


def main_ablation_all(epochs):
    os.chdir(ROOT)
    epochs = epochs or ABLATION_EPOCHS
    extra = ["--epochs", str(epochs)]
    total = len(ABLATION_COMBOS) * len(TARGETS)
    idx = 0
    for lbd_d, lbd_c, lbd_cp, label in ABLATION_COMBOS:
        run_suffix = "a1_%s_a2_%s_a3_%s" % (_v(lbd_d), _v(lbd_c), _v(lbd_cp))
        for target in TARGETS:
            idx += 1
            cmd = ["--target", target, "--lbd_d", str(lbd_d), "--lbd_c", str(lbd_c), "--lbd_cp", str(lbd_cp)] + extra
            print("\n[%d/%d] %s | %s" % (idx, total, run_suffix, target))
            ret = run_train(cmd, exp_folder=os.path.join(EXP_BASE, EXP_ABLATION), run_suffix=run_suffix)
            if ret != 0:
                sys.exit(ret)
    print("Ablation_all done. Run: python run_all.py plot")


def _v(x):
    s = str(float(x))
    return s.replace(".", "_").replace("-", "m")


def _load_ours():
    base = os.path.join(EXP_BASE, NETWORK, DATASET)
    ours = {}
    for t in TARGETS:
        f = os.path.join(base, t, "results_summary.json")
        if os.path.isfile(f):
            with open(f) as fp:
                ours[t] = json.load(fp)["test_acc"]
        else:
            log = os.path.join(base, t, "loss_log.txt")
            if os.path.isfile(log):
                with open(log) as fp:
                    for line in fp:
                        if "corresponding test" in line:
                            ours[t] = float(line.strip().split("corresponding test")[1].strip())
                            break
    return ours

def _collect_combo_results():
    path = os.path.join(EXP_BASE, EXP_ABLATION, NETWORK, DATASET)
    if not os.path.isdir(path):
        return []

    combo_accs = {}
    for target_name in os.listdir(path):
        target_dir = os.path.join(path, target_name)
        if not os.path.isdir(target_dir):
            continue
        for run in os.listdir(target_dir):
            f = os.path.join(target_dir, run, "results_summary.json")
            if not os.path.isfile(f):
                continue
            with open(f) as fp:
                d = json.load(fp)
            key = run
            if key not in combo_accs:
                combo_accs[key] = {}
            combo_accs[key][target_name] = d["test_acc"]
    out = []
    for lbd_d, lbd_c, lbd_cp, label in ABLATION_COMBOS:
        key = "a1_%s_a2_%s_a3_%s" % (_v(lbd_d), _v(lbd_c), _v(lbd_cp))
        if key not in combo_accs or len(combo_accs[key]) < 4:
            continue
        per = combo_accs[key]
        avg = sum(per[t] for t in TARGETS if t in per) / 4.0
        out.append((label, avg, per))
    return out


def main_plot():
    os.chdir(ROOT)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("pip install matplotlib numpy", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    ylim = PLOT_YLIM

    # 1) Reproduction
    paper_path = os.path.join(OUT_DIR, "paper_baseline.json")
    if os.path.isfile(paper_path):
        with open(paper_path) as f:
            paper = json.load(f)
    else:
        paper = {"art_painting": 84.6, "cartoon": 83.2, "photo": 95.4, "sketch": 77.0}
    ours = _load_ours()
    paper_vals = [paper.get(t, 0) for t in TARGETS]
    ours_vals = [ours.get(t, 0) for t in TARGETS]

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(TARGETS))
    w = 0.35
    ax1.bar(x - w / 2, paper_vals, w, label="Paper", color="C0", alpha=0.8)
    ax1.bar(x + w / 2, ours_vals, w, label="Ours", color="C1", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(TARGETS)
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_title("PACS: Reproduction vs Paper")
    ax1.legend()
    ax1.set_ylim(ylim)
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUT_DIR, "reproduction_plot.png"), dpi=150)
    plt.close(fig1)
    print("Saved figures/reproduction_plot.png")

    # 2) Ablation
    rows = _collect_combo_results()
    if not rows:
        print("No ablation_combos results. Run: python run_all.py ablation_all")
        return

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    labels = [r[0] for r in rows]
    avgs = [r[1] for r in rows]
    x_pos = np.arange(len(labels))
    colors = ["#e74c3c", "#f39c12", "#3498db", "#27ae60"]
    bars = ax2.bar(x_pos, avgs, color=colors[: len(labels)], edgecolor="black", linewidth=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Average test accuracy (%)", fontsize=11)
    ax2.set_title("Ablation: alpha2 & alpha3 off vs Full (alpha1=0.5, 4 targets)", fontsize=12)
    ax2.set_ylim(ylim)
    ax2.grid(axis="y", alpha=0.3)
    for i, (bar, avg) in enumerate(zip(bars, avgs)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, "%.1f" % avg, ha="center", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "ablation_unified.png"), dpi=150)
    plt.close(fig2)
    print("Saved figures/ablation_unified.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["reproduction", "ablation_all", "plot"])
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    if args.mode == "reproduction":
        main_reproduction(args.epochs)
    elif args.mode == "ablation_all":
        main_ablation_all(args.epochs)
    else:
        main_plot()
