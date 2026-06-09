#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return rows


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _normalize_hist(h: Dict[str, float]) -> Dict[str, float]:
    z = float(sum(h.values()))
    if z <= 0:
        return {k: 0.0 for k in h}
    return {k: float(v) / z for k, v in h.items()}


def _tv_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys)


def _lift_main_cases(hist: Dict[str, float]) -> Dict[str, float]:
    out = {str(i): 0.0 for i in range(1, 7)}
    # Direct 1..6 bins (or mixed with split bins)
    for k, v in hist.items():
        if k in out:
            out[k] += float(v)
    # Split bins
    out["5"] = float(hist.get("5a", 0.0)) + float(hist.get("5b", 0.0))
    out["6"] = float(hist.get("6a", 0.0)) + float(hist.get("6b", 0.0))
    return out


def _case_preference_score(hist_main: Dict[str, float]) -> float:
    # Positive means mass shifts toward desired cases 1 and 5.
    h = _normalize_hist(hist_main)
    return (h.get("1", 0.0) + h.get("5", 0.0)) - (
        h.get("2", 0.0)
        + h.get("3", 0.0)
        + h.get("4", 0.0)
        + h.get("6", 0.0)
    )


def _window(rows: List[dict], size: int) -> List[List[dict]]:
    if size <= 0:
        return [rows]
    if len(rows) < size:
        return []
    return [rows[i : i + size] for i in range(0, len(rows) - size + 1)]


def _aggregate_hist(rows: List[dict]) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for r in rows:
        h = r.get("case_histogram", {}) or {}
        for k, v in h.items():
            agg[str(k)] = agg.get(str(k), 0.0) + float(v)
    return agg


def analyze(metrics_path: Path, window_size: int) -> Tuple[str, int]:
    rows = _read_jsonl(metrics_path)
    if not rows:
        return f"No rows found in {metrics_path}", 1

    filer_rewards = [float(r.get("filer_reward_mean", 0.0)) for r in rows]
    loss_total = [float(r.get("loss_total", 0.0)) for r in rows]
    loss_filer = [float(r.get("loss_filer_term", 0.0)) for r in rows]
    loss_verifier = [float(r.get("loss_verifier_term", 0.0)) for r in rows]
    replay_used = [int(r.get("replay_updates_used", 0)) for r in rows]

    lines: List[str] = []
    lines.append(f"Run: {metrics_path}")
    lines.append(f"Steps: {len(rows)}")
    lines.append(
        f"Filer reward mean/std: {_mean(filer_rewards):.4f} / {_std(filer_rewards):.4f}"
    )
    lines.append(f"Loss total mean/std: {_mean(loss_total):.4f} / {_std(loss_total):.4f}")
    lines.append(f"Filer loss term mean/std: {_mean(loss_filer):.4f} / {_std(loss_filer):.4f}")
    lines.append(
        f"Verifier loss term mean/std: {_mean(loss_verifier):.4f} / {_std(loss_verifier):.4f}"
    )
    lines.append(
        f"Replay active steps: {sum(1 for x in replay_used if x > 0)} / {len(replay_used)}"
    )

    # Rolling stability on reward
    windows = _window(rows, window_size)
    if windows:
        w_reward_mean = [_mean([float(r.get("filer_reward_mean", 0.0)) for r in w]) for w in windows]
        w_reward_std = [_std([float(r.get("filer_reward_mean", 0.0)) for r in w]) for w in windows]
        w_filer_loss_mean = [_mean([float(r.get("loss_filer_term", 0.0)) for r in w]) for w in windows]
        w_verifier_loss_mean = [_mean([float(r.get("loss_verifier_term", 0.0)) for r in w]) for w in windows]
        w_filer_loss_std = [_std([float(r.get("loss_filer_term", 0.0)) for r in w]) for w in windows]
        w_verifier_loss_std = [_std([float(r.get("loss_verifier_term", 0.0)) for r in w]) for w in windows]
        lines.append(
            f"Rolling({window_size}) reward mean range: {min(w_reward_mean):.4f}..{max(w_reward_mean):.4f}"
        )
        lines.append(
            f"Rolling({window_size}) reward std avg: {_mean(w_reward_std):.4f}"
        )
        lines.append(
            f"Rolling({window_size}) filer-loss mean range: {min(w_filer_loss_mean):.4f}..{max(w_filer_loss_mean):.4f}"
        )
        lines.append(
            f"Rolling({window_size}) verifier-loss mean range: {min(w_verifier_loss_mean):.4f}..{max(w_verifier_loss_mean):.4f}"
        )
        lines.append(
            f"Rolling({window_size}) filer-loss std avg: {_mean(w_filer_loss_std):.4f}"
        )
        lines.append(
            f"Rolling({window_size}) verifier-loss std avg: {_mean(w_verifier_loss_std):.4f}"
        )

        # Histogram stability: TV distance between consecutive windows.
        tvs: List[float] = []
        prefs: List[float] = []
        prev = None
        for w in windows:
            h_main = _lift_main_cases(_aggregate_hist(w))
            hn = _normalize_hist(h_main)
            prefs.append(_case_preference_score(h_main))
            if prev is not None:
                tvs.append(_tv_distance(prev, hn))
            prev = hn
        if tvs:
            lines.append(
                f"Rolling({window_size}) case-hist TV mean/max: {_mean(tvs):.4f} / {max(tvs):.4f}"
            )
        if prefs:
            lines.append(
                f"Rolling({window_size}) case preference mean/range: {_mean(prefs):.4f} / {min(prefs):.4f}..{max(prefs):.4f}"
            )

    # First-half vs second-half summary
    mid = len(rows) // 2
    if mid >= 1 and len(rows) - mid >= 1:
        a = rows[:mid]
        b = rows[mid:]
        a_reward = _mean([float(r.get("filer_reward_mean", 0.0)) for r in a])
        b_reward = _mean([float(r.get("filer_reward_mean", 0.0)) for r in b])
        lines.append(f"Half split reward: first={a_reward:.4f}, second={b_reward:.4f}, delta={b_reward-a_reward:+.4f}")
        a_f = _mean([float(r.get("loss_filer_term", 0.0)) for r in a])
        b_f = _mean([float(r.get("loss_filer_term", 0.0)) for r in b])
        a_v = _mean([float(r.get("loss_verifier_term", 0.0)) for r in a])
        b_v = _mean([float(r.get("loss_verifier_term", 0.0)) for r in b])
        lines.append(
            f"Half split filer-loss: first={a_f:.4f}, second={b_f:.4f}, delta={b_f-a_f:+.4f}"
        )
        lines.append(
            f"Half split verifier-loss: first={a_v:.4f}, second={b_v:.4f}, delta={b_v-a_v:+.4f}"
        )

        ah = _normalize_hist(_lift_main_cases(_aggregate_hist(a)))
        bh = _normalize_hist(_lift_main_cases(_aggregate_hist(b)))
        lines.append(f"Half split case TV distance: {_tv_distance(ah, bh):.4f}")
        lines.append(
            "Half split case mass first: "
            + ", ".join(f"{k}:{ah[k]:.3f}" for k in sorted(ah.keys()))
        )
        lines.append(
            "Half split case mass second: "
            + ", ".join(f"{k}:{bh[k]:.3f}" for k in sorted(bh.keys()))
        )

    # Simple verdict heuristic
    verdict = "inconclusive"
    filer_verdict = "inconclusive"
    verifier_verdict = "inconclusive"
    if len(rows) >= max(window_size * 2, 10):
        # Improvement if reward grows and TV shrinks.
        # Compare early vs late rolling windows.
        ws = _window(rows, window_size)
        early = ws[: max(1, len(ws) // 3)]
        late = ws[-max(1, len(ws) // 3) :]
        early_r = _mean([_mean([float(r.get("filer_reward_mean", 0.0)) for r in w]) for w in early])
        late_r = _mean([_mean([float(r.get("filer_reward_mean", 0.0)) for r in w]) for w in late])
        early_tvs: List[float] = []
        late_tvs: List[float] = []
        early_h = [_normalize_hist(_lift_main_cases(_aggregate_hist(w))) for w in early]
        late_h = [_normalize_hist(_lift_main_cases(_aggregate_hist(w))) for w in late]
        for i in range(1, len(early_h)):
            early_tvs.append(_tv_distance(early_h[i - 1], early_h[i]))
        for i in range(1, len(late_h)):
            late_tvs.append(_tv_distance(late_h[i - 1], late_h[i]))
        early_tv = _mean(early_tvs) if early_tvs else 0.0
        late_tv = _mean(late_tvs) if late_tvs else 0.0
        if late_r > early_r + 0.03 and late_tv <= early_tv:
            verdict = "likely improving toward stable regime"
        elif abs(late_r - early_r) < 0.02 and late_tv < 0.10:
            verdict = "near stationary / possible local equilibrium"
        else:
            verdict = "not yet at equilibrium (or too noisy)"
        # Agent-specific heuristic verdicts from their own optimization terms.
        early_f = _mean([_mean([float(r.get("loss_filer_term", 0.0)) for r in w]) for w in early])
        late_f = _mean([_mean([float(r.get("loss_filer_term", 0.0)) for r in w]) for w in late])
        early_v = _mean([_mean([float(r.get("loss_verifier_term", 0.0)) for r in w]) for w in early])
        late_v = _mean([_mean([float(r.get("loss_verifier_term", 0.0)) for r in w]) for w in late])
        if abs(late_f - early_f) < 10.0:
            filer_verdict = "filer appears near stationary regime"
        else:
            filer_verdict = "filer still drifting (not stationary)"
        if abs(late_v - early_v) < 10.0:
            verifier_verdict = "verifier appears near stationary regime"
        else:
            verifier_verdict = "verifier still drifting (not stationary)"
    lines.append(f"Verdict: {verdict}")
    lines.append(f"Filer verdict: {filer_verdict}")
    lines.append(f"Verifier verdict: {verifier_verdict}")
    return "\n".join(lines), 0


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze GRPO run stability/equilibrium proxies.")
    p.add_argument(
        "--metrics-jsonl",
        type=Path,
        required=True,
        help="Path to run metrics.jsonl",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Rolling window size for stability metrics (default: 5).",
    )
    args = p.parse_args()
    report, code = analyze(args.metrics_jsonl, args.window_size)
    print(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

