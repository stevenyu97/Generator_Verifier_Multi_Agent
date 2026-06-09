#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "gray": "#666666",
    "black": "#000000",
}

CASE_COLORS = {
    "1": "#0072B2",
    "2": "#E69F00",
    "3": "#009E73",
    "4": "#D55E00",
    "5a": "#56B4E9",
    "5b": "#CC79A7",
    "6a": "#999999",
    "6b": "#000000",
}


def setup_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 2.2,
        }
    )


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return rows


def rolling_mean(xs: List[float], w: int) -> List[float]:
    out: List[float] = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        seg = xs[lo : i + 1]
        out.append(sum(seg) / max(len(seg), 1))
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_line_plot(
    x: List[int],
    ys: Dict[str, List[float]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(9, 5))
    color_cycle = [
        PALETTE["blue"],
        PALETTE["orange"],
        PALETTE["green"],
        PALETTE["red"],
        PALETTE["purple"],
        PALETTE["gray"],
    ]
    for i, (name, vals) in enumerate(ys.items()):
        plt.plot(
            x,
            vals,
            marker="o",
            markersize=3.2,
            color=color_cycle[i % len(color_cycle)],
            label=name,
        )
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--smooth-window", type=int, default=5)
    args = ap.parse_args()

    run_dir = args.run_dir
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    eval_metrics = read_jsonl(run_dir / "eval_metrics.jsonl")
    if not metrics:
        print(f"No metrics found at {run_dir / 'metrics.jsonl'}")
        return 1

    out_dir = run_dir / "plots"
    ensure_dir(out_dir)
    setup_paper_style()

    steps = [int(r["step"]) for r in metrics]
    filer_mean = [float(r.get("filer_reward_mean", 0.0)) for r in metrics]
    verifier_mean = [float(r.get("verifier_reward_mean", 0.0)) for r in metrics]
    final_acc = [float(r.get("final_draft_accuracy_strict", 0.0)) for r in metrics]
    loss_total = [float(r.get("loss_total", 0.0)) for r in metrics]
    loss_filer = [float(r.get("loss_filer_term", 0.0)) for r in metrics]
    loss_ver = [float(r.get("loss_verifier_term", 0.0)) for r in metrics]

    w = max(1, int(args.smooth_window))
    save_line_plot(
        steps,
        {
            "filer_reward_mean": filer_mean,
            f"filer_reward_mean_ma{w}": rolling_mean(filer_mean, w),
            "verifier_reward_mean": verifier_mean,
            f"verifier_reward_mean_ma{w}": rolling_mean(verifier_mean, w),
        },
        "Per-step Rewards",
        "Reward",
        out_dir / "rewards.png",
    )
    save_line_plot(
        steps,
        {
            "final_draft_accuracy_strict": final_acc,
            f"final_draft_accuracy_strict_ma{w}": rolling_mean(final_acc, w),
        },
        "Final Draft Accuracy (Train Steps)",
        "Accuracy",
        out_dir / "final_accuracy.png",
    )
    save_line_plot(
        steps,
        {
            "loss_total": loss_total,
            "loss_filer_term": loss_filer,
            "loss_verifier_term": loss_ver,
        },
        "Training Loss Terms",
        "Loss",
        out_dir / "loss_terms.png",
    )

    # Case histogram stacked composition
    case_keys = ["1", "2", "3", "4", "5a", "5b", "6a", "6b"]
    case_vals = {k: [] for k in case_keys}
    for r in metrics:
        h = r.get("case_histogram", {}) or {}
        total = float(sum(float(h.get(k, 0.0)) for k in case_keys))
        for k in case_keys:
            v = float(h.get(k, 0.0))
            case_vals[k].append(v / total if total > 0 else 0.0)

    plt.figure(figsize=(10, 5))
    bottom = [0.0] * len(steps)
    for k in case_keys:
        vals = case_vals[k]
        plt.bar(
            steps,
            vals,
            bottom=bottom,
            label=k,
            width=0.8,
            color=CASE_COLORS[k],
            edgecolor="white",
            linewidth=0.3,
        )
        bottom = [b + v for b, v in zip(bottom, vals)]
    plt.title("Case Histogram Composition per Step")
    plt.xlabel("Step")
    plt.ylabel("Proportion")
    plt.ylim(0, 1.0)
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(out_dir / "case_mix_stacked.png", dpi=140)
    plt.close()

    if eval_metrics:
        ev_steps = [int(r["step"]) for r in eval_metrics]
        ev_acc = [float(r.get("test_accuracy_strict_mean", 0.0)) for r in eval_metrics]
        ev_line = [float(r.get("test_line_accuracy_mean", 0.0)) for r in eval_metrics]
        save_line_plot(
            ev_steps,
            {
                "test_accuracy_strict_mean": ev_acc,
                "test_line_accuracy_mean": ev_line,
            },
            "Periodic Test Evaluation",
            "Accuracy",
            out_dir / "eval_accuracy.png",
        )

    print(f"Saved plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

