import json
import os
from statistics import mean

import matplotlib.pyplot as plt


METRICS_PATH = "training_metrics.json"
DOCS_DIR = "docs"


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


def _plot_reward_curve(metrics):
    rewards = [m["reward"] for m in metrics]
    steps = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards, linewidth=1.3, label="Per update reward")
    if len(rewards) >= 20:
        window = 20
        moving_avg = [mean(rewards[max(0, i - window + 1): i + 1]) for i in range(len(rewards))]
        plt.plot(steps, moving_avg, linewidth=2.0, label="Moving average (20)")
    plt.title("Reward Curve")
    plt.xlabel("Training Update")
    plt.ylabel("Reward")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, "reward_curve.png"), dpi=200)
    plt.close()


def _plot_loss_curve(metrics):
    losses = [m["loss"] for m in metrics]
    steps = list(range(1, len(losses) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, linewidth=1.3, color="#d62728")
    plt.title("Loss Curve")
    plt.xlabel("Training Update")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, "loss_curve.png"), dpi=200)
    plt.close()


def _plot_baseline_vs_trained(metrics):
    rewards = [m["reward"] for m in metrics]
    if len(rewards) < 10:
        baseline = mean(rewards)
        trained = mean(rewards)
    else:
        split = max(1, len(rewards) // 5)
        baseline = mean(rewards[:split])
        trained = mean(rewards[-split:])

    plt.figure(figsize=(7, 5))
    plt.bar(["Baseline (early)", "Trained (late)"], [baseline, trained], color=["#7f7f7f", "#2ca02c"])
    plt.title("Baseline vs Trained Reward")
    plt.ylabel("Average Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, "baseline_vs_trained.png"), dpi=200)
    plt.close()


def main():
    _ensure_docs_dir()
    metrics = _load_metrics()
    _plot_reward_curve(metrics)
    _plot_loss_curve(metrics)
    _plot_baseline_vs_trained(metrics)
    print("Saved: docs/reward_curve.png, docs/loss_curve.png, docs/baseline_vs_trained.png")


if __name__ == "__main__":
    main()
