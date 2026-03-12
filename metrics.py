"""
Utility Functions
==================
  wilcoxon_test  : Wilcoxon signed-rank test to validate statistical significance
                   (mirrors the paper's validation methodology, α = 0.05)
  plot_convergence: Plot accuracy-vs-round curves for all compared algorithms
  plot_comm_cost  : Bar chart of normalised communication cost
  save_results    : Persist experiment results to JSON
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------------
# Statistical Validation
# ---------------------------------------------------------------------------

def wilcoxon_test(
    baseline_scores: list[float],
    proposed_scores: list[float],
    alpha: float = 0.05,
    alternative: str = "greater",
) -> dict:
    """
    Wilcoxon signed-rank test comparing proposed vs baseline accuracy runs.

    Parameters
    ----------
    baseline_scores : list of accuracy values from 30 independent runs (baseline)
    proposed_scores : list of accuracy values from 30 independent runs (proposed)
    alpha           : significance level (default 0.05)
    alternative     : 'greater' tests if proposed > baseline

    Returns
    -------
    dict with keys: statistic, p_value, reject_null, interpretation
    """
    stat, p = wilcoxon(proposed_scores, baseline_scores, alternative=alternative)
    reject = bool(p < alpha)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "reject_null": reject,
        "interpretation": (
            f"FedM-ALO2 is significantly better (p={p:.4f} < {alpha})."
            if reject
            else f"No significant difference (p={p:.4f} ≥ {alpha})."
        ),
    }


def run_wilcoxon_table(results: dict, proposed_key: str = "FedM-ALO2") -> None:
    """
    Print a formatted Wilcoxon table for all baseline vs proposed comparisons.

    Parameters
    ----------
    results     : {algorithm_name: {dataset_name: [run_accuracies]}}
    proposed_key: key for the proposed algorithm in `results`
    """
    proposed = results.get(proposed_key, {})
    print(f"\n{'='*70}")
    print(f"  Wilcoxon Signed-Rank Test — {proposed_key} vs. all baselines")
    print(f"{'='*70}")
    header = f"{'Dataset':<20} {'Algorithm':<18} {'p-value':>10} {'Result':>10}"
    print(header)
    print("-" * 70)

    for alg, datasets in results.items():
        if alg == proposed_key:
            continue
        for ds, scores in datasets.items():
            prop_scores = proposed.get(ds, [])
            if len(prop_scores) != len(scores) or len(scores) == 0:
                continue
            res = wilcoxon_test(scores, prop_scores)
            verdict = "Reject H0" if res["reject_null"] else "Fail to Reject"
            print(f"{ds:<20} {alg:<18} {res['p_value']:>10.4f} {verdict:>10}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(
    histories: dict[str, list[float]],
    title: str = "Convergence",
    xlabel: str = "Communication Round",
    ylabel: str = "Test Accuracy",
    save_path: str = None,
):
    """
    Line plot of accuracy vs round for multiple algorithms.

    Parameters
    ----------
    histories : {algorithm_name: [acc_per_round]}
    title     : plot title
    save_path : if provided, saves the figure to this path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed — skipping plot.")
        return

    STYLES = {
        "FedAvg":    ("gray",   "--"),
        "FedPSO":    ("blue",   "-."),
        "FedSCA":    ("green",  ":"),
        "FedGWO":    ("orange", "-."),
        "FedALO":    ("purple", "--"),
        "FedM-ALO1": ("red",    "-"),
        "FedM-ALO2": ("black",  "-"),
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, acc_list in histories.items():
        color, ls = STYLES.get(name, ("steelblue", "-"))
        ax.plot(range(1, len(acc_list) + 1), acc_list,
                label=name, color=color, linestyle=ls, linewidth=2)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[INFO] Convergence plot saved → {save_path}")
    plt.show()


def plot_comm_cost(
    costs: dict[str, float],
    title: str = "Normalized Communication Cost",
    save_path: str = None,
):
    """Bar chart of normalised communication cost per algorithm."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed — skipping plot.")
        return

    algs = list(costs.keys())
    vals = [costs[a] for a in algs]
    colors = ["#4C72B0" if "M-ALO2" not in a else "#C44E52" for a in algs]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(algs, vals, color=colors, edgecolor="black", width=0.55)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Normalised Cost", fontsize=12)
    ax.set_ylim(0, max(vals) * 1.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[INFO] Communication cost plot saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str = "results.json"):
    """Serialise experiment results to JSON (converts numpy types)."""
    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, default=_convert, indent=2)
    print(f"[INFO] Results saved → {out}")
