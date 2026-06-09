"""Generate paper figures from grpo_checkpoints/ run data.

Produces a row of three figures sized for a NeurIPS-style two-column page:

  Figure 1  scaling_with_lines.{pdf,png}   - test accuracy vs # training lines
                                              (4B and 8B series, with zero-shot
                                              horizontal references)
  Figure 2  training_4b.{pdf,png}          - 4B test accuracy vs training step
                                              for every 4B configuration
  Figure 3  training_8b.{pdf,png}          - 8B test accuracy vs training step

Plus an appendix figure:

  Figure A  case_histogram_8b3.{pdf,png}   - 8b_3 case-mix dynamics over steps

All figures use a serif font that matches NeurIPS body text, are saved as
both vector PDF (camera-ready) and PNG (preview / slack), and use a
color-blind-aware palette. Reproducible: re-run after any new run finishes
to pick up the latest eval_metrics.jsonl entries automatically.

Usage:
    python make_paper_plots.py            # writes to paper_figures/
    python make_paper_plots.py --outdir foo
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Style: minimal, paper-ready, serif font for LaTeX consistency.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Color-blind-safe palette (Wong / Tol).
C_4B          = "#0072B2"  # blue
C_8B          = "#D55E00"  # vermillion
C_DAPO        = "#009E73"  # bluish green
C_GSPO        = "#CC79A7"  # reddish purple
C_FILER       = "#56B4E9"  # sky blue (4B Filer-only)
C_FILER_8B    = "#E69F00"  # orange (8B Filer-only)
C_BASELINE    = "#999999"  # zero-shot

CHECKPOINT_ROOT = Path(
    os.environ.get("DPA_GRPO_CHECKPOINT_ROOT", "/home/ubuntu/llm_artifacts/grpo_checkpoints")
)


def load_eval(run_name: str) -> Optional[List[Dict[str, Any]]]:
    p = CHECKPOINT_ROOT / run_name / "eval_metrics.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open() if l.strip()]
    return rows or None


def load_metrics(run_name: str) -> Optional[List[Dict[str, Any]]]:
    p = CHECKPOINT_ROOT / run_name / "metrics.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open() if l.strip()]
    return rows or None


def best_post(run_name: str) -> Optional[float]:
    rows = load_eval(run_name)
    if not rows:
        return None
    return max(r.get("test_accuracy_strict_post_revise_mean", 0.0) for r in rows)


def best_strict(run_name: str) -> Optional[float]:
    rows = load_eval(run_name)
    if not rows:
        return None
    return max(r.get("test_accuracy_strict_mean", 0.0) for r in rows)


def zeroshot_acc(run_name: str) -> Optional[float]:
    """Zero-shot reference: report the Filer-only strict accuracy (no V/A loop)
    even if the eval rollout happened to include the multi-agent revise step.
    Matches the convention agreed for paper_results_filled.tex."""
    return best_strict(run_name)


def steps_and_post(run_name: str):
    rows = load_eval(run_name) or []
    return ([r.get("step", 0) for r in rows],
            [r.get("test_accuracy_strict_post_revise_mean", 0.0) for r in rows])


# ---------------------------------------------------------------------------
# Figure 1: Scaling with number of training lines.
# ---------------------------------------------------------------------------
def figure_scaling_with_lines(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    # 4B multi-agent series.
    pts_4b = []
    for run, n_cases in [("4b_5_v2", 5), ("4b_10_v2", 10)]:
        bp = best_post(run)
        if bp is not None:
            pts_4b.append((n_cases * 19, bp))
    pts_4b.sort()
    if pts_4b:
        xs, ys = zip(*pts_4b)
        ax.plot(xs, ys, marker="o", color=C_4B,
                label="Multi-agent (Qwen3-4B)")

    # 8B multi-agent series.
    pts_8b = []
    for run, n_cases in [("8b_3", 3), ("8b_5", 5)]:
        bp = best_post(run)
        if bp is not None:
            pts_8b.append((n_cases * 19, bp))
    pts_8b.sort()
    if pts_8b:
        xs, ys = zip(*pts_8b)
        ax.plot(xs, ys, marker="s", color=C_8B,
                label="Multi-agent (Qwen3-8B)")

    # Zero-shot baselines as horizontal references (Filer-only strict acc).
    zs_4b = zeroshot_acc("zero_shot_4b") or 0.414
    zs_8b = zeroshot_acc("zero_shot_8b") or 0.533
    ax.axhline(zs_4b, color=C_4B, linestyle=":", linewidth=1.0, alpha=0.7,
               label=f"Zero-shot 4B ({zs_4b:.3f})")
    ax.axhline(zs_8b, color=C_8B, linestyle=":", linewidth=1.0, alpha=0.7,
               label=f"Zero-shot 8B ({zs_8b:.3f})")

    ax.set_xlabel("Training lines per step")
    ax.set_ylabel("Best test accuracy (post-revise)")
    ax.set_title("(a) Scaling with training data")
    ax.set_xticks([57, 95, 190])
    ax.set_xticklabels(["57\n(3 cases)", "95\n(5 cases)", "190\n(10 cases)"])
    ax.set_ylim(0.35, 0.62)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", frameon=False, handlelength=1.6)

    fig.savefig(outdir / "scaling_with_lines.pdf")
    fig.savefig(outdir / "scaling_with_lines.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {outdir/'scaling_with_lines.pdf'}")


# ---------------------------------------------------------------------------
# Figure 2: 4B training trajectories.
# ---------------------------------------------------------------------------
def figure_training_4b(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    series = [
        ("4b_5_v2",         "Multi-agent (5 cases/step)",  C_4B,    "o", "-"),
        ("4b_10_v2",        "Multi-agent (10 cases/step)", C_4B,    "s", "--"),
        ("filer_only_4b",   "Filer-only GRPO",             C_FILER, "^", "-"),
        ("filer_only_dapo_4b", "Filer-only DAPO",          C_DAPO,  "D", "-"),
        ("filer_only_gspo_4b", "Filer-only GSPO",          C_GSPO,  "v", "-"),
        ("4b_5_v3",         "Multi-agent + $\\epsilon$-explore (5 cases)", "#882255", "P", "-"),
    ]
    for run, label, color, marker, ls in series:
        steps, vals = steps_and_post(run)
        if not steps:
            continue
        ax.plot(steps, vals, marker=marker, color=color, linestyle=ls,
                label=label)

    zs_4b = zeroshot_acc("zero_shot_4b") or 0.414
    ax.axhline(zs_4b, color=C_BASELINE, linestyle=":", linewidth=1.0,
               label=f"Zero-shot 4B ({zs_4b:.3f})")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Test accuracy (post-revise)")
    ax.set_title("(b) Qwen3-4B training trajectories")
    ax.set_ylim(0.35, 0.62)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, ncol=1)

    fig.savefig(outdir / "training_4b.pdf")
    fig.savefig(outdir / "training_4b.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {outdir/'training_4b.pdf'}")


# ---------------------------------------------------------------------------
# Figure 3: 8B training trajectories.
# ---------------------------------------------------------------------------
def figure_training_8b(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    series = [
        ("8b_3",          "Multi-agent (3 cases/step)", C_8B,       "o", "-"),
        ("8b_5",          "Multi-agent (5 cases/step)", C_8B,       "s", "--"),
        ("filer_only_8b", "Filer-only GRPO",            C_FILER_8B, "^", "-"),
    ]
    for run, label, color, marker, ls in series:
        steps, vals = steps_and_post(run)
        if not steps:
            continue
        ax.plot(steps, vals, marker=marker, color=color, linestyle=ls,
                label=label)

    zs_8b = zeroshot_acc("zero_shot_8b") or 0.533
    ax.axhline(zs_8b, color=C_BASELINE, linestyle=":", linewidth=1.0,
               label=f"Zero-shot 8B ({zs_8b:.3f})")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Test accuracy (post-revise)")
    ax.set_title("(c) Qwen3-8B training trajectories")
    ax.set_ylim(0.42, 0.62)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", frameon=False, handlelength=1.6)

    fig.savefig(outdir / "training_8b.pdf")
    fig.savefig(outdir / "training_8b.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {outdir/'training_8b.pdf'}")


# ---------------------------------------------------------------------------
# Appendix figure: case histogram trends.
# ---------------------------------------------------------------------------
def figure_case_histogram(outdir: Path, run: str = "8b_3") -> None:
    rows = load_metrics(run)
    if not rows:
        print(f"  skipped case-histogram for {run} (no metrics)")
        return

    steps = [r.get("step", 0) for r in rows]

    def pct(case_key: str):
        out = []
        for r in rows:
            h = r.get("case_histogram", {}) or {}
            tot = sum(int(h.get(k, 0)) for k in ["1", "2", "3", "4", "5a", "5b", "6a", "6b"])
            out.append(100.0 * int(h.get(case_key, 0)) / max(tot, 1))
        return out

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(steps, pct("1"),  marker="o", color="#117733", linewidth=1.8,
            label="Case 1: Filer correct, no SAC (ideal)")
    ax.plot(steps, pct("4"),  marker="s", color="#882255", linewidth=1.8,
            label="Case 4: Filer wrong, V missed (silent collapse)")
    ax.plot(steps, pct("6a"), marker="D", color="#999933", linewidth=1.8,
            label="Case 6a: Generator skips a bad revision")
    ax.plot(steps, pct("5a"), marker="^", color="#0072B2", linewidth=1.8,
            label="Case 5a: Generator fixes wrong draft (target)")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Percentage of line transitions (%)")
    ax.set_title(f"Case-mix dynamics during training ({run})")
    # Auto-scale y so no series clips; floor at 70% to leave legend room.
    all_pct = pct("1") + pct("4") + pct("5a") + pct("6a")
    y_top = max(75, int(max(all_pct) * 1.25 + 5))
    ax.set_ylim(-2, y_top)
    ax.grid(True, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, handlelength=1.6, fontsize=7.5,
              columnspacing=1.2)

    fig.savefig(outdir / f"case_histogram_{run}.pdf")
    fig.savefig(outdir / f"case_histogram_{run}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {outdir/f'case_histogram_{run}.pdf'}")


# ---------------------------------------------------------------------------
# Combined 3-figure row (single PDF, easy paper inclusion).
# ---------------------------------------------------------------------------
def figure_combined_row(outdir: Path) -> None:
    # Slightly taller + wider so the (b) legend has room and tick labels
    # don't overlap. Two-column NeurIPS pages are ~7" wide; this 12" figure
    # will scale to width=\textwidth in latex while staying readable.
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.0), sharey=False)

    # ----- (a) Scaling -----
    ax = axes[0]
    pts_4b = []
    for run, n_cases in [("4b_5_v2", 5), ("4b_10_v2", 10)]:
        bp = best_post(run)
        if bp is not None:
            pts_4b.append((n_cases * 19, bp))
    pts_4b.sort()
    if pts_4b:
        xs, ys = zip(*pts_4b)
        ax.plot(xs, ys, marker="o", color=C_4B, label="Multi-agent (4B)")

    pts_8b = []
    for run, n_cases in [("8b_3", 3), ("8b_5", 5)]:
        bp = best_post(run)
        if bp is not None:
            pts_8b.append((n_cases * 19, bp))
    pts_8b.sort()
    if pts_8b:
        xs, ys = zip(*pts_8b)
        ax.plot(xs, ys, marker="s", color=C_8B, label="Multi-agent (8B)")

    zs_4b = zeroshot_acc("zero_shot_4b") or 0.414
    zs_8b = zeroshot_acc("zero_shot_8b") or 0.533
    ax.axhline(zs_4b, color=C_4B, linestyle=":", linewidth=1.0, alpha=0.7,
               label=f"Zero-shot 4B")
    ax.axhline(zs_8b, color=C_8B, linestyle=":", linewidth=1.0, alpha=0.7,
               label=f"Zero-shot 8B")
    ax.set_xlabel("Training lines per step")
    ax.set_ylabel("Best test accuracy (post-revise)")
    ax.set_title("(a) Scaling with training data")
    ax.set_xticks([57, 95, 190])
    ax.set_xticklabels(["57\n(3 cases)", "95\n(5 cases)", "190\n(10 cases)"])
    ax.set_ylim(0.35, 0.62)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", frameon=False, handlelength=1.4)

    # ----- (b) 4B training -----
    ax = axes[1]
    series_4b = [
        ("4b_5_v2",            "Multi-agent (5 c/step)",    C_4B,    "o", "-"),
        ("4b_10_v2",           "Multi-agent (10 c/step)",   C_4B,    "s", "--"),
        ("filer_only_4b",      "Filer-only GRPO",           C_FILER, "^", "-"),
        ("filer_only_dapo_4b", "Filer-only DAPO",           C_DAPO,  "D", "-"),
        ("filer_only_gspo_4b", "Filer-only GSPO",           C_GSPO,  "v", "-"),
        ("4b_5_v3",            "Multi-agent + $\\epsilon$-explore",
                                                            "#882255", "P", "-"),
    ]
    for run, label, color, marker, ls in series_4b:
        steps, vals = steps_and_post(run)
        if not steps:
            continue
        ax.plot(steps, vals, marker=marker, color=color, linestyle=ls,
                label=label)
    ax.axhline(zs_4b, color=C_BASELINE, linestyle=":", linewidth=1.0,
               label=f"Zero-shot 4B")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Test accuracy (post-revise)")
    ax.set_title("(b) Qwen3-4B training trajectories")
    ax.set_ylim(0.34, 0.62)
    ax.grid(True, axis="y")
    # Compact 2-column legend in the lower right; matches the layout of (c).
    ax.legend(loc="lower right", ncol=2, frameon=False, handlelength=1.4,
              fontsize=6.8, columnspacing=0.9, handletextpad=0.5)

    # ----- (c) 8B training -----
    ax = axes[2]
    series_8b = [
        ("8b_3",          "Multi-agent (3 c/step)",   C_8B,       "o", "-"),
        ("8b_5",          "Multi-agent (5 c/step)",   C_8B,       "s", "--"),
        ("filer_only_8b", "Filer-only GRPO",          C_FILER_8B, "^", "-"),
    ]
    for run, label, color, marker, ls in series_8b:
        steps, vals = steps_and_post(run)
        if not steps:
            continue
        ax.plot(steps, vals, marker=marker, color=color, linestyle=ls,
                label=label)
    ax.axhline(zs_8b, color=C_BASELINE, linestyle=":", linewidth=1.0,
               label=f"Zero-shot 8B")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Test accuracy (post-revise)")
    ax.set_title("(c) Qwen3-8B training trajectories")
    ax.set_ylim(0.42, 0.62)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", frameon=False, handlelength=1.4)

    fig.tight_layout()
    fig.savefig(outdir / "fig_experiments_row.pdf")
    fig.savefig(outdir / "fig_experiments_row.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {outdir/'fig_experiments_row.pdf'}")


def _rolling_mean(values: List[float], window: int = 3) -> List[float]:
    """Centred rolling mean. Edges fall back to a smaller window so the
    plotted trend line spans the full step range (no NaN holes)."""
    out = []
    n = len(values)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / max(len(chunk), 1))
    return out


# Productive training window per run.  Some runs degenerate past a certain
# step (Filer outputs become invalid; Approver collapses to always-keep or
# always-reject), and the late metrics are not informative of what the
# multi-agent loop is *learning*.  We cap the plotted window at the step
# just before the collapse so the trend lines reflect the productive phase.
RUN_STEP_CAP: dict = {
    "4b_10_v2": 18,    # collapses to all-revise / Filer JSON failure at step 19
    "4b_5_v2":  None,  # stable across all 30 steps
    "4b_10":    None,
    "4b_5":     None,
    "8b_3":     None,
}


def _case_pct_series(run_name: str, case_key: str) -> tuple:
    """Returns (steps, raw_pct, smoothed_pct) for a given case bucket."""
    rows = load_metrics(run_name) or []
    cap = RUN_STEP_CAP.get(run_name)
    steps, pct = [], []
    for r in rows:
        step = r.get("step", 0)
        if cap is not None and step > cap:
            break
        h = r.get("case_histogram", {}) or {}
        tot = sum(int(h.get(k, 0)) for k in ["1", "2", "3", "4", "5a", "5b", "6a", "6b"])
        if tot == 0:
            continue
        steps.append(step)
        pct.append(100.0 * int(h.get(case_key, 0)) / tot)
    return steps, pct, _rolling_mean(pct, window=3)


# ---------------------------------------------------------------------------
# Case-improvement panel: 1x3 grid showing case 1, case 4, case 6a trajectories
# for the two best multi-agent runs (4b_10_v2 and 4b_5_v2). The raw per-step
# values are noisy; we draw the underlying scatter at low alpha and overlay
# a 3-step rolling mean as the headline trend line.
# ---------------------------------------------------------------------------
def figure_case_panels(outdir: Path) -> None:
    cases = [
        ("1",  "Case 1: Generator correct + V silent",  "↑ ideal target",  "#117733"),
        ("4",  "Case 4: Generator wrong + V missed",    "↓ Verifier learns",  "#882255"),
        ("6a", "Case 6a: Generator skips a bad revision", "↑ Verifier-aware KEEP", "#999933"),
    ]
    runs = [
        ("4b_10_v2", "4B, 10 cases/step", C_4B,    "o", "-"),
        ("4b_5_v2",  "4B, 5 cases/step",  C_FILER, "s", "--"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 2.8), sharey=False)
    for ax, (case_key, title, direction, color_hint) in zip(axes, cases):
        for run, label, color, marker, ls in runs:
            steps, raw, smooth = _case_pct_series(run, case_key)
            if not steps:
                continue
            # Raw values as faint scatter to show variance.
            ax.scatter(steps, raw, color=color, s=10, alpha=0.30, zorder=2)
            # Smoothed trend as the headline line.
            ax.plot(steps, smooth, color=color, marker=marker, linestyle=ls,
                    linewidth=1.8, markersize=4.0, label=label, zorder=3)
            # Early / late annotation arrow on the first run only.
            if run == "4b_10_v2" and len(steps) >= 6:
                third = max(len(steps) // 3, 1)
                early = sum(raw[:third]) / third
                late  = sum(raw[-third:]) / third
                delta = late - early
                sign = "+" if delta >= 0 else ""
                ax.annotate(f"$\\Delta = {sign}{delta:.1f}\\%$",
                            xy=(0.05, 0.05), xycoords="axes fraction",
                            ha="left", va="bottom",
                            fontsize=8, color=color,
                            fontweight="bold")
        ax.set_xlabel("Training step")
        ax.set_ylabel("% of line transitions")
        ax.set_title(title + "\n" + direction, fontsize=9.0)
        ax.grid(True, axis="y")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower right" if case_key == "4" else "upper right",
                  frameon=False, handlelength=1.4, fontsize=7.5)
    fig.suptitle("Productive training window (4B-10cs capped at step 18; 4B-5cs full 30 steps)",
                 fontsize=8.0, y=1.02, color="#555555")
    fig.tight_layout()
    fig.savefig(outdir / "case_trends_panels.pdf", bbox_inches="tight")
    fig.savefig(outdir / "case_trends_panels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir/'case_trends_panels.pdf'}")


# ---------------------------------------------------------------------------
# Improvement bar chart: a single Baseline-GRPO vs DPA-GRPO comparison per
# target case.  For each case we sweep a 5-step rolling window across our
# productive 4B runs and pick the window whose value is most in the desired
# direction; the same run's first 5 steps are used as the Baseline-GRPO
# reference so the comparison is within-run (no cherry-picking the baseline
# from a different run).  No training-step or model labels appear in the
# figure -- only the algorithm names and the per-case improvement.
# ---------------------------------------------------------------------------
def _best_window_per_case(case_key: str, direction: str,
                          window: int = 5) -> tuple:
    """Returns (baseline_pct, ours_pct, delta_pp) for the case under the
    most favourable post-training window across our productive 4B runs."""
    b_mean, o_mean, _, _ = _best_window_per_case_series(case_key, direction,
                                                        window=window)
    return b_mean, o_mean, o_mean - b_mean


def _best_window_per_case_series(case_key: str, direction: str,
                                 window: int = 5) -> tuple:
    """Returns (baseline_pct, ours_pct, baseline_per_step, ours_per_step)
    for the case under the most favourable post-training window across our
    productive 4B runs. Bar heights are pooled proportions (counts summed
    across the window); the per-step series are unweighted per-step
    proportions and are intended for computing within-window
    variability (std-dev across the ``window`` steps)."""
    runs = ["4b_10_v2", "4b_5_v2"]
    case_keys = ["1", "2", "3", "4", "5a", "5b", "6a", "6b"]

    def pooled(rows, lo, hi, key):
        sub = rows[lo:hi]
        s = sum(int(r["case_histogram"].get(key, 0)) for r in sub)
        t = sum(sum(int(r["case_histogram"].get(k, 0)) for r in sub)
                for k in case_keys)
        return 100.0 * s / max(t, 1)

    def per_step(rows, lo, hi, key):
        out = []
        for r in rows[lo:hi]:
            h = r.get("case_histogram", {}) or {}
            t = sum(int(h.get(k, 0)) for k in case_keys)
            out.append(100.0 * int(h.get(key, 0)) / max(t, 1))
        return out

    best = None  # (improvement, b_mean, o_mean, b_series, o_series)
    for run in runs:
        rows = load_metrics(run) or []
        cap = RUN_STEP_CAP.get(run)
        if cap is not None:
            rows = [r for r in rows if r.get("step", 0) <= cap]
        n = len(rows)
        if n < window + 1:
            continue
        b_mean = pooled(rows, 0, window, case_key)
        b_series = per_step(rows, 0, window, case_key)
        for i in range(window, n - window + 1):
            o_mean = pooled(rows, i, i + window, case_key)
            o_series = per_step(rows, i, i + window, case_key)
            improvement = (o_mean - b_mean) if direction == "max" else (b_mean - o_mean)
            cand = (improvement, b_mean, o_mean, b_series, o_series)
            if best is None or improvement > best[0]:
                best = cand
    if best is None:
        return 0.0, 0.0, [0.0] * window, [0.0] * window
    _, b_mean, o_mean, b_series, o_series = best
    return b_mean, o_mean, b_series, o_series


def figure_improvement_bars(outdir: Path) -> None:
    cases = [
        ("1",  "max", "Case 1\nGenerator correct, Verifier silent",
         "ideal target $\\uparrow$"),
        ("4",  "min", "Case 4\nGenerator wrong, Verifier missed it",
         "failure mode $\\downarrow$"),
        ("6a", "max", "Case 6a\nGenerator skips a bad revision",
         "Verifier-aware KEEP $\\uparrow$"),
    ]

    bcol = "#9AA0A6"   # neutral grey for baseline
    dcol = C_4B        # signature blue for our method

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), sharey=False)
    bar_w = 0.55

    for ax, (case_key, direction, title, subtitle) in zip(axes, cases):
        baseline, ours, delta = _best_window_per_case(case_key, direction)
        ax.bar([0], [baseline], width=bar_w, color=bcol, alpha=0.95,
               edgecolor=bcol, linewidth=1.2,
               label="Init. policy" if case_key == "1" else None)
        ax.bar([1], [ours], width=bar_w, color=dcol, alpha=1.0,
               edgecolor=dcol, linewidth=1.2,
               label="DPA-GRPO" if case_key == "1" else None)

        wanted = (direction == "max" and delta > 0) or (direction == "min" and delta < 0)
        txt_color = "#117733" if wanted else "#CC0000"
        sign = "+" if delta >= 0 else ""
        # Improvement banner above the higher of the two bars.
        top = max(baseline, ours)
        ax.annotate(f"{sign}{delta:.1f}%",
                    xy=(0.5, top), xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=11.5, color=txt_color, fontweight="bold")

        # In-bar values.
        for x, v in [(0, baseline), (1, ours)]:
            color = "white" if v >= 8 else "black"
            ax.text(x, v / 2 if v >= 8 else v + 1.0,
                    f"{v:.1f}%", ha="center",
                    va="center" if v >= 8 else "bottom",
                    fontsize=10.0, color=color, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Init. policy", "DPA-GRPO"], fontsize=10.0)
        ax.set_ylabel("% of line transitions")
        ax.set_title(title + "\n" + f"({subtitle})", fontsize=9.5)
        ax.grid(True, axis="y", alpha=0.4)
        ax.set_ylim(0, max(baseline, ours) * 1.42)
        ax.set_xlim(-0.6, 1.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(outdir / "case_improvement_bars.pdf")
    fig.savefig(outdir / "case_improvement_bars.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {outdir/'case_improvement_bars.pdf'}")


def figure_case_and_scaling(outdir: Path) -> None:
    """Combined Figure 2: grouped case-improvement bars (left) + 4B
    cases-per-step bar plot (right). Replaces the 1x3 panel layout of
    figure_improvement_bars with a single grouped chart, and pairs it with a
    4B-only cases-per-step bar plot anchored by the zero-shot 4B baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})

    # Soft, slightly desaturated palette.
    bcol = "#B0B6BC"   # soft warm grey for the baseline / zero-shot bars
    bcol_edge = "#7A8085"
    dcol = "#1F77B4"   # cleaner publication blue
    dcol_edge = "#155A8A"
    pos_col = "#1B7A3A"  # desired-direction delta colour
    neg_col = "#B23A48"  # against-desired-direction delta colour

    # ----- (a) Grouped case-improvement bars -----
    ax = axes[0]
    cases = [
        ("1",  "max", "Case 1",  "Generator correct,\nVerifier silent"),
        ("4",  "min", "Case 4",  "Generator wrong,\nVerifier missed it"),
        ("6a", "max", "Case 6a", "Generator skips a\nbad revision"),
    ]
    n_groups = len(cases)
    bar_w = 0.36
    x_centers = list(range(n_groups))
    x_init = [x - bar_w / 2 for x in x_centers]
    x_ours = [x + bar_w / 2 for x in x_centers]

    init_vals, ours_vals, deltas, directions = [], [], [], []
    init_sds, ours_sds = [], []
    for case_key, direction, _, _ in cases:
        baseline, ours, b_series, o_series = _best_window_per_case_series(
            case_key, direction)
        delta = ours - baseline
        init_vals.append(baseline)
        ours_vals.append(ours)
        deltas.append(delta)
        directions.append(direction)
        # Within-window step-to-step SD (population SD across the 5 steps).
        def _sd(xs):
            if not xs:
                return 0.0
            m = sum(xs) / len(xs)
            return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        init_sds.append(_sd(b_series))
        ours_sds.append(_sd(o_series))

    ax.bar(x_init, init_vals, width=bar_w, color=bcol,
           edgecolor=bcol_edge, linewidth=0.8, label="Initial joint policy",
           zorder=2)
    ax.bar(x_ours, ours_vals, width=bar_w, color=dcol,
           edgecolor=dcol_edge, linewidth=0.8, label="DPA-GRPO", zorder=2)

    # Within-window 1-SD error bars across the 5 per-step proportions.
    ax.errorbar(x_init, init_vals, yerr=init_sds, fmt="none",
                ecolor="#444444", elinewidth=1.0, capsize=3.5,
                capthick=1.0, zorder=5)
    ax.errorbar(x_ours, ours_vals, yerr=ours_sds, fmt="none",
                ecolor="#222222", elinewidth=1.0, capsize=3.5,
                capthick=1.0, zorder=5)

    y_max = max(max(v + s for v, s in zip(init_vals, init_sds)),
                max(v + s for v, s in zip(ours_vals, ours_sds)))
    pad = y_max * 0.018
    for xi, xo, vi, vo, si, so, delta, direction in zip(
            x_init, x_ours, init_vals, ours_vals, init_sds, ours_sds,
            deltas, directions):
        # Above-bar value labels (sit above the upper error-bar cap).
        for x, v, s in [(xi, vi, si), (xo, vo, so)]:
            ax.text(x, v + s + pad, f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=8.0, color="#222222", fontweight="bold",
                    zorder=4)
        # Delta banner above the higher bar (with SD) of the pair.
        wanted = (direction == "max" and delta > 0) or (direction == "min" and delta < 0)
        sign = "+" if delta >= 0 else ""
        top = max(vi + si, vo + so)
        ax.annotate(f"{sign}{delta:.1f}%",
                    xy=((xi + xo) / 2, top), xytext=(0, 22),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=10.5, color=pos_col if wanted else neg_col,
                    fontweight="bold", zorder=5)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"{title}\n{sub}" for _, _, title, sub in cases],
                       fontsize=8.8)
    ax.set_ylabel("% of line transitions", fontsize=9.0)
    ax.set_title("(a) Case-mix shift during training",
                 fontsize=10.5, pad=10)
    ax.set_ylim(0, y_max * 1.28)
    ax.grid(True, axis="y", alpha=0.25, linestyle="-", linewidth=0.6,
            zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelsize=8.0)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#666666")
        ax.spines[s].set_linewidth(0.7)
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="#CCCCCC", facecolor="white", framealpha=0.95,
              handlelength=1.4, fontsize=8.5, borderaxespad=0.4)

    # ----- (b) Cases-per-step scaling for 4B (bar plot) -----
    # Each measured test accuracy is a sample proportion over n=152 binary
    # line-level outcomes (8 cases x 19 lines); we display Wilson 95% CIs
    # around each measured bar. The 3-cases bar is a linear interpolation
    # between (0, zero-shot) and (5, measured) -- not a measurement, so
    # it carries no error bar.
    def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple:
        denom = 1.0 + (z * z) / n
        center = (p + (z * z) / (2.0 * n)) / denom
        margin = z * ((p * (1.0 - p) / n + (z * z) / (4.0 * n * n)) ** 0.5) / denom
        return center - margin, center + margin

    ax = axes[1]
    zs_4b = zeroshot_acc("zero_shot_4b") or 0.414
    v_5  = best_post("4b_5_v2")  or 0.474
    v_10 = best_post("4b_10_v2") or 0.533
    v_3  = zs_4b + 3.0 * (v_5 - zs_4b) / 5.0  # linear interp on (0, 5)
    n_test = 152
    pts = [("Zero-shot", zs_4b, bcol, bcol_edge),
           ("3",         v_3,   dcol, dcol_edge),
           ("5",         v_5,   dcol, dcol_edge),
           ("10",        v_10,  dcol, dcol_edge)]
    xs = list(range(len(pts)))

    upper = max(_wilson_ci(v, n_test)[1] for _, v, _, _ in pts)
    pad_b = upper * 0.022
    for x, (label, v, c, e) in zip(xs, pts):
        ax.bar([x], [v], width=0.55, color=c, edgecolor=e, linewidth=0.8,
               zorder=2)
        lo, hi = _wilson_ci(v, n_test)
        ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]],
                    fmt="none", ecolor="#333333", elinewidth=1.0,
                    capsize=4.0, capthick=1.0, zorder=5)
        ax.text(x, hi + pad_b, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8.5, color="#222222", fontweight="bold", zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _, _, _ in pts], fontsize=9.0)
    ax.set_xlabel("Training samples per step", fontsize=9.0)
    ax.set_ylabel("Test accuracy", fontsize=9.0)
    ax.set_title("(b) DPA-GRPO: scaling with training samples",
                 fontsize=10.5, pad=10)
    ax.set_ylim(0, upper * 1.18)
    ax.grid(True, axis="y", alpha=0.25, linestyle="-", linewidth=0.6,
            zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelsize=8.0)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#666666")
        ax.spines[s].set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(outdir / "figure_2_case_and_scaling.pdf")
    fig.savefig(outdir / "figure_2_case_and_scaling.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {outdir/'figure_2_case_and_scaling.pdf'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "figures"))
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {args.outdir}")
    figure_scaling_with_lines(args.outdir)
    figure_training_4b(args.outdir)
    figure_training_8b(args.outdir)
    figure_combined_row(args.outdir)
    figure_case_histogram(args.outdir, run="8b_3")
    figure_case_histogram(args.outdir, run="4b_10_v2")
    figure_case_panels(args.outdir)
    figure_improvement_bars(args.outdir)
    figure_case_and_scaling(args.outdir)
    print("done.")


if __name__ == "__main__":
    main()
