"""Plot autoresearch optimization progress -- Karpathy-style training curve.

Usage:
    uv run python development/autoresearch/plot_progress.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCORE_FILE = Path(__file__).parent / "scores.jsonl"


def load_scores() -> list[dict]:
    entries = []
    with open(SCORE_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    entries = load_scores()

    # Group by git_sha to get committed optimization steps
    # Use median of all runs per sha as the representative score
    sha_order = []
    sha_scores: dict[str, list[float]] = {}
    for e in entries:
        sha = e["git_sha"]
        if sha not in sha_scores:
            sha_order.append(sha)
            sha_scores[sha] = []
        sha_scores[sha].append(e["score"])

    # Labels for each committed optimization (in order)
    opt_labels = {
        "9e8be0c": "Baseline",
        "594f81a": "Workspace caching +\nhistogram subtraction",
        "4e657b0": "In-place gradient +\nCHUNK_SIZE 8192",
        "a867c8c": "CHUNK_SIZE\n16384",
        "6ee0497": "Smaller-child\ntrick",
        "459a8e4": "D2D copies +\nworkspace outputs",
        "8f78243": "Binning\noptimization",
    }

    # Build plot data: one point per sha, median score, with error bars
    shas = []
    medians = []
    mins = []
    maxs = []
    labels = []
    for sha in sha_order:
        scores = sha_scores[sha]
        # Filter obvious outliers (Modal cold-start noise > 20s)
        scores = [s for s in scores if s < 20]
        if not scores:
            continue
        shas.append(sha)
        med = float(np.median(scores))
        medians.append(med)
        mins.append(min(scores))
        maxs.append(max(scores))
        labels.append(opt_labels.get(sha, sha[:7]))

    x = np.arange(len(medians))
    err_low = [med - mn for med, mn in zip(medians, mins)]
    err_high = [mx - med for med, mx in zip(medians, maxs)]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Dark background like Karpathy's plots
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Error bars (Modal variance)
    ax.errorbar(
        x, medians,
        yerr=[err_low, err_high],
        fmt="none",
        ecolor="#4a90d9",
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        alpha=0.5,
        zorder=2,
    )

    # Main line
    ax.plot(
        x, medians,
        color="#e94560",
        linewidth=2.5,
        marker="o",
        markersize=10,
        markerfacecolor="#e94560",
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=3,
    )

    # Scatter individual runs faintly
    for i, sha in enumerate(shas):
        scores = [s for s in sha_scores[sha] if s < 20]
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(scores))
        ax.scatter(
            [i + j for j in jitter], scores,
            color="#4a90d9",
            alpha=0.3,
            s=20,
            zorder=1,
        )

    # Annotations: speedup relative to baseline
    baseline = medians[0]
    for i, (med, label) in enumerate(zip(medians, labels)):
        speedup = baseline / med
        delta_pct = (1 - med / baseline) * 100
        # Score label
        ax.annotate(
            f"{med:.1f}s",
            (i, med),
            textcoords="offset points",
            xytext=(0, -22),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
        # Speedup label (skip baseline)
        if i > 0:
            ax.annotate(
                f"{speedup:.1f}x",
                (i, med),
                textcoords="offset points",
                xytext=(0, 14),
                ha="center",
                fontsize=8,
                color="#0f3460",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="#e94560",
                    edgecolor="none",
                    alpha=0.8,
                ),
            )

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color="#a0a0a0", ha="center")

    # Y-axis
    ax.set_ylabel("Fit Time (seconds)", fontsize=12, color="white", fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0fs"))
    ax.tick_params(axis="y", colors="#a0a0a0")

    # Title
    ax.set_title(
        "OpenBoost GPU Training Optimization\n"
        "1M samples, 100 features, 200 trees, depth 8 -- Modal A100-SXM4-40GB",
        fontsize=14,
        fontweight="bold",
        color="white",
        pad=15,
    )

    # Y-axis starts at 0 (no truncation crimes)
    ax.set_ylim(bottom=0)

    # Grid
    ax.grid(axis="y", color="#2a3a5e", linewidth=0.5, alpha=0.7)
    ax.grid(axis="x", visible=False)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_color("#2a3a5e")

    # Add a "diminishing returns" annotation
    ax.annotate(
        "Modal A100 variance\n~7% run-to-run",
        xy=(len(medians) - 1, medians[-1]),
        xytext=(len(medians) - 2.8, medians[-1] + 3),
        fontsize=8,
        color="#a0a0a0",
        fontstyle="italic",
        arrowprops=dict(
            arrowstyle="->",
            color="#a0a0a0",
            connectionstyle="arc3,rad=-0.2",
        ),
    )

    # Bottom text
    fig.text(
        0.5, 0.01,
        "Each dot = one Modal A100 benchmark run. Line = median. Error bars = min/max.",
        ha="center",
        fontsize=8,
        color="#606060",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out_path = Path(__file__).parent / "optimization_progress.png"
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
