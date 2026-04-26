from __future__ import annotations

import json
import math
import os
import textwrap
from collections import defaultdict
from statistics import mean
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

METRICS_PATH = "training_metrics.json"
DOCS_DIR = "docs"
DPI = 200


def _init_matplotlib():
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#1e293b",
            "axes.labelcolor": "#0f172a",
            "text.color": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#cbd5e1",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#e2e8f0",
            "figure.titlesize": 12,
            "lines.linewidth": 1.5,
            "lines.solid_capstyle": "round",
        }
    )


def _ensure_docs_dir():
    os.makedirs(DOCS_DIR, exist_ok=True)


def _load_metrics():
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"{METRICS_PATH} not found. Run training first.")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"{METRICS_PATH} is empty.")
    return data


def _key_episode(m: dict) -> tuple:
    return (m.get("mode", "heuristic"), int(m.get("scenario", 0)))


def collapse_to_one_row_per_episode(raw: list) -> list:
    """
    `train_llm` now writes one row per outer scenario. Older logs may still have
    multiple rows per (mode, scenario); this keeps the last step and success=any(…).
    """
    if not raw:
        return []
    out: list = []
    i = 0
    n = len(raw)
    while i < n:
        k = _key_episode(raw[i])
        j = i
        while j < n and _key_episode(raw[j]) == k:
            j += 1
        group = raw[i:j]
        last = dict(group[-1])
        last["success"] = any(bool(x.get("success")) for x in group)
        out.append(last)
        i = j
    return out


def _has_real_loss(metrics):
    for m in metrics:
        v = m.get("loss")
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        if isinstance(v, (int, float)):
            return True
    return False


def _loss_proxy_from_reward(r):
    return float(min(5.0, max(0.05, abs(float(r)) / 10.0)))


def _style_grid(ax, show_x=True, show_y=True):
    if show_x:
        ax.grid(True, axis="x", which="major")
    if show_y:
        ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)


def _fig_save(fig, path: str, *, rect: Optional[Tuple[float, float, float, float]] = None) -> None:
    fig.tight_layout(pad=1.1, rect=rect if rect is not None else (0, 0, 1, 1))
    p = os.path.join(DOCS_DIR, path)
    fig.savefig(p, dpi=DPI, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _plot_reward_curve(metrics, *, is_episode: bool) -> None:
    rewards = np.array([m["reward"] for m in metrics], dtype=np.float64)
    steps = np.arange(1, len(rewards) + 1, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor="white")
    n = len(rewards)

    # High-density series: filled band + line so it stays visible at any N
    ax.fill_between(
        steps,
        rewards,
        0.0,
        color="#3b82f6",
        alpha=0.12,
        interpolate=True,
        zorder=1,
        linewidth=0,
    )
    (line_raw,) = ax.plot(
        steps,
        rewards,
        color="#1d4ed8",
        linewidth=max(0.4, min(1.0, 8000.0 / max(n, 1))),
        alpha=0.85,
        zorder=2,
        label="Final reward" if is_episode else "Reward (each loop step)",
    )
    if n > 5000:
        line_raw.set_rasterized(True)

    h_idx = [i for i, m in enumerate(metrics) if m.get("mode") != "council"]
    c_idx = [i for i, m in enumerate(metrics) if m.get("mode") == "council"]
    if c_idx:
        ax.scatter(
            steps[c_idx],
            rewards[c_idx],
            c="#c2410c",
            s=max(3, min(20, 12000 // max(n, 1))),
            alpha=0.9,
            zorder=4,
            label="Council (marker)",
            edgecolors="#7c2d12",
            linewidths=0.2,
        )
    if n >= 20:
        w = int(min(80, max(25, n // 20)))
        ma = np.array(
            [float(mean(rewards[max(0, i - w + 1) : i + 1])) for i in range(n)],
        )
        ax.plot(
            steps,
            ma,
            color="#15803d",
            linewidth=2.4,
            zorder=5,
            label=f"Moving avg (w={w})",
        )

    if is_episode:
        ax.set_title(
            f"Final reward per training episode (N={n} points, e.g. 50 heuristic + 10 council by default)",
            pad=8,
            fontsize=10.5,
        )
    else:
        ax.set_title(
            "Reward at every backprop / loop in old multi-row metrics (set PLOTS_USE_PER_LOOP_METRICS=1)",
            pad=8,
            fontsize=10.5,
        )
    ax.set_xlabel("Episode order" if is_episode else "Update index (each loop in the log)", labelpad=6)
    ax.set_ylabel("Reward", labelpad=6)
    _style_grid(ax)
    ax.margins(x=0.01, y=0.08)
    ax.legend(loc="lower right" if n > 5 else "best", fontsize=9, ncol=1, frameon=True)
    ylim = ax.get_ylim()
    if ylim[0] < 0 < ylim[1] or (ylim[0] < 0 and ylim[1] < 0) or (ylim[0] > 0 and ylim[1] > 0):
        ax.axhline(0, color="#64748b", linewidth=0.8, linestyle="--", zorder=0)
    _fig_save(fig, "reward_curve.png")


def _plot_loss_curve(metrics, use_proxy: bool, *, is_episode: bool) -> None:
    if use_proxy:
        losses = [_loss_proxy_from_reward(m["reward"]) for m in metrics]
        sub = "Proxy = clip(|reward|/10, 0.05, 5). For real loss, use metrics from train_llm.py, not a parsed log."
    else:
        losses = []
        for m in metrics:
            v = m.get("loss")
            if v is not None and isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                losses.append(float(v))
            else:
                losses.append(_loss_proxy_from_reward(m["reward"]))
        sub = "From training loop tensor loss." if not any(m.get("loss") is None for m in metrics) else "Real loss where available; |reward| proxy for gaps."

    losses = np.array(losses, dtype=np.float64)
    steps = np.arange(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(12, 5.0), facecolor="white")
    (ln,) = ax.plot(steps, losses, color="#b91c1c", linewidth=1.4, alpha=0.92, zorder=2)
    if len(losses) > 8000:
        ln.set_rasterized(True)
    title = "Loss" + (" (proxy from rewards)" if use_proxy else "")
    ax.set_title(title, pad=6)
    ax.set_xlabel("Training episode" if is_episode else "Backprop / loop step (same order as metrics rows)", labelpad=6)
    if use_proxy:
        _yl = "Proxy loss"
    else:
        _yl = "Loss (last backprop in each episode)" if is_episode else "Loss (each loop)"
    ax.set_ylabel(_yl, labelpad=6)
    foot = "\n".join(textwrap.wrap(sub, 72))
    ax.text(0.5, -0.2, foot, transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color="#64748b", linespacing=1.2)
    _style_grid(ax)
    _fig_save(fig, "loss_curve.png", rect=(0, 0.1, 1, 0.99))


def _plot_baseline_vs_trained(metrics, *, is_episode: bool) -> None:
    rewards = [m["reward"] for m in metrics]
    n = len(rewards)
    if n < 10:
        split = 1
        baseline = mean(rewards)
        trained = mean(rewards)
    else:
        split = max(1, n // 5)
        baseline = mean(rewards[:split])
        trained = mean(rewards[-split:])

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    xlabels = [f"Early\n(first {100 * split // n}%)", f"Late\n(last {100 * split // n}%)"]
    colors = ("#94a3b8", "#16a34a")
    bars = ax.bar(xlabels, [baseline, trained], color=colors, edgecolor="#0f172a", linewidth=0.8, width=0.55, zorder=2)
    ax.set_title("Average final reward: early vs late segment of the run" + ("" if is_episode else " (per loop step)"), pad=8)
    ax.set_ylabel("Average reward", labelpad=6)
    for b, v in zip(bars, (baseline, trained)):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h,
            f"{v:.2f}",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=11,
            fontweight="bold",
            color="#0f172a",
        )
    _style_grid(ax, show_x=False)
    ax.axhline(0, color="#64748b", linewidth=0.9, zorder=1)
    ylim = ax.get_ylim()
    pad = 0.12 * (ylim[1] - ylim[0] + 1e-9)
    ax.set_ylim(min(ylim[0], 0) - pad, ylim[1] + pad)
    _fig_save(fig, "baseline_vs_trained.png")


def _plot_rolling_success(metrics, *, is_episode: bool) -> None:
    succ = np.array([1.0 if m.get("success") else 0.0 for m in metrics], dtype=np.float64)
    n = len(succ)
    w = int(min(250, max(15, n // 20))) if n >= 15 else max(1, n // 2)
    w = min(w, n) if n else 1
    steps = np.arange(1, n + 1, dtype=np.int32)
    rolling = np.empty(n, dtype=np.float64)
    csum = np.cumsum(np.r_[0.0, succ])
    for i in range(n):
        a = max(0, i - w + 1)
        rolling[i] = (csum[i + 1] - csum[a]) / (i - a + 1)

    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor="white")
    ax.fill_between(steps, 0, rolling, color="#7c3aed", alpha=0.2, zorder=1, linewidth=0, step=None)
    ax.plot(steps, rolling, color="#4c1d95", linewidth=2.2, zorder=3, drawstyle="default", solid_capstyle="round")
    if n > 1:
        ax.margins(x=0.01, y=0.03)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if is_episode:
        ax.set_title("Rolling success rate: episode had at least one SUCCESS in any loop", pad=8, fontsize=10.5)
    else:
        ax.set_title("Rolling share of log lines marked SUCCESS (per-loop metrics only)", pad=8, fontsize=10.5)
    ax.set_xlabel("Episode order" if is_episode else "Log line index (each loop in the log)", labelpad=6)
    ax.set_ylabel("Success rate", labelpad=6)
    _style_grid(ax)
    wtxt = f"Window: {w} (adaptive by {('episode' if is_episode else 'step')} count, capped)"
    ax.text(0.5, -0.16, wtxt, transform=ax.transAxes, ha="center", va="top", fontsize=7.8, color="#64748b")
    _fig_save(fig, "success_rate_rolling.png", rect=(0, 0.1, 1, 0.99))


def _plot_reward_histogram(metrics, *, is_episode: bool) -> None:
    rewards = np.array([m["reward"] for m in metrics], dtype=np.float64)
    n = len(rewards)
    n_bins = int(min(80, max(20, 3 * int(round(np.log2(n + 1))))) if n else 20)

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    n_pos, _, _patches = ax.hist(
        rewards,
        bins=n_bins,
        color="#0891b2",
        edgecolor="#0e7490",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.set_title(
        "Distribution of final reward (one value per training episode)" if is_episode
        else "Distribution of reward at every log line (each loop step)",
        pad=8,
    )
    ax.set_xlabel("Reward", labelpad=6)
    ax.set_ylabel("Count", labelpad=6)
    med = float(np.median(n_pos)) if len(n_pos) else 0.0
    mx = float(np.max(n_pos)) if len(n_pos) else 0.0
    if med > 0 and mx / med > 50 and len(n_pos) > 3:
        ax.set_yscale("log")
        ax.set_ylabel("Count (log scale)")
        ax.set_ylim(bottom=0.5)

    _style_grid(ax)
    _fig_save(fig, "reward_distribution.png")


def _plot_per_episode_outcome(metrics, *, is_episode: bool) -> None:
    by = defaultdict(list)
    order = []
    for m in metrics:
        key = (m.get("mode", "heuristic"), m["scenario"])
        if key not in order:
            order.append(key)
        by[key].append(m)
    n_ep = len(order)
    if n_ep == 0:
        return

    last_rewards = []
    won = []
    for key in order:
        rows = by[key]
        last_rewards.append(rows[-1]["reward"])
        won.append(any(r.get("success") for r in rows))
    x = np.arange(1, n_ep + 1, dtype=np.int32)
    last_rewards = np.array(last_rewards, dtype=np.float64)
    s = max(1.0, min(6.0, 2500.0 / max(n_ep, 1)))
    s = s**2

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor="white")
    for w in (True, False):
        mask = np.array(won) == w
        if not np.any(mask):
            continue
        lab = "Episode with ≥1 SUCCESS" if w else "All loops PENALTY in episode"
        c = "#16a34a" if w else "#dc2626"
        ax.scatter(
            x[mask],
            last_rewards[mask],
            c=c,
            s=s,
            alpha=0.85,
            label=lab,
            zorder=3,
            edgecolors="black" if s > 10 else "none",
            linewidths=0.15,
        )
    if n_ep > 2:
        ax.plot(x, last_rewards, color="#94a3b8", linewidth=0.3, zorder=1, alpha=0.5, label="Connect (trend)")

    t = "Final reward and outcome per episode (rebuilt from metrics rows; same as main curve if JSON is one row per loop)"
    if is_episode:
        t = "Per-episode: final reward vs. index (N matches scenario count, e.g. 50 + 10 default)"
    ax.set_title(t, pad=6, fontsize=9.5)
    ax.set_xlabel("Index in this run (heuristic 0..N-1, then council)", labelpad=6, fontsize=9.5)
    ax.set_ylabel("Final step reward in episode", labelpad=6)
    _style_grid(ax)
    if last_rewards.size and np.nanmax(np.abs(last_rewards)) < 1e-6:
        ax.margins(y=0.5)
    if abs(np.nanmax(last_rewards) - np.nanmin(last_rewards)) > 0.1:
        ax.margins(y=0.05)
    h = float(np.max(last_rewards)) if len(last_rewards) else 0.0
    l = float(np.min(last_rewards)) if len(last_rewards) else 0.0
    if l < 0 < h:
        ax.axhline(0, color="#64748b", linewidth=0.8, linestyle="--", zorder=0)
    ax.legend(loc="best", fontsize=8, ncol=1)
    _fig_save(fig, "per_episode_final_reward.png")


def main():
    _init_matplotlib()
    _ensure_docs_dir()
    raw = _load_metrics()
    use_per_loop = os.environ.get("PLOTS_USE_PER_LOOP_METRICS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    is_episode = not use_per_loop
    if use_per_loop:
        plot = raw
        if len(raw) > 0:
            print(
                f"Plotting every backprop/loop: {len(plot)} points. "
                "For one x-step per training scenario, leave PLOTS_USE_PER_LOOP_METRICS unset; JSON is 1 row/scenario."
            )
    else:
        plot = collapse_to_one_row_per_episode(raw)
        n_ep, n_loops = len(plot), len(raw)
        print(
            f"Plotting one point per training episode: {n_ep} points (collapsed from {n_loops} loop-level JSON rows). "
            f"E.g. 50+10 default scenarios => {n_ep} episode rows; raw JSON had {n_loops} row(s) before collapse."
        )

    real_loss = _has_real_loss(plot)

    _plot_reward_curve(plot, is_episode=is_episode)
    _plot_loss_curve(plot, use_proxy=not real_loss, is_episode=is_episode)
    _plot_baseline_vs_trained(plot, is_episode=is_episode)
    _plot_rolling_success(plot, is_episode=is_episode)
    _plot_reward_histogram(plot, is_episode=is_episode)
    # Use full `raw` so episode grouping (multi-loop) is visible; same as collapsed `plot` when one row/episode.
    _plot_per_episode_outcome(raw, is_episode=is_episode)

    out = [
        "docs/reward_curve.png",
        "docs/loss_curve.png",
        "docs/baseline_vs_trained.png",
        "docs/success_rate_rolling.png",
        "docs/reward_distribution.png",
        "docs/per_episode_final_reward.png",
    ]
    print("Saved:")
    for p in out:
        print(" ", p)
    if not real_loss:
        print("Note: loss curve uses a |reward| proxy when loss is not in the JSON (e.g. parsed log).")


if __name__ == "__main__":
    main()
