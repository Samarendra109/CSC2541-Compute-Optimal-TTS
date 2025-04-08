import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
# Set global font sizes
plt.rcParams.update(
    {
        "font.size": 12,  # Default font size
        "axes.titlesize": 14,  # Title font size
        "axes.labelsize": 12,  # Axis label font size
        "xtick.labelsize": 10,  # X-axis tick label font size
        "ytick.labelsize": 10,  # Y-axis tick label font size
        "legend.fontsize": 10,  # Legend font size
        "figure.titlesize": 16,  # Figure title font size
    }
)


def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        dct = json.load(f)
    return dct


DATASET = "math"

RES_DIR = Path(os.environ.get("HOME"), "llm-tts", "openr", "debug")
datasets = [p.name for p in RES_DIR.glob("*")]
results = []
for dataset in datasets:
    if DATASET is not None and dataset.lower() != DATASET:
        logger.info("Skipping results for %s", dataset)
        continue
    logger.info("Processing results for %s", dataset)
    methods = [p.name for p in (RES_DIR / dataset).glob("*")]
    for method in methods:
        for exp_dir in (RES_DIR / dataset / method).glob("*"):
            # logger.info("%s, %s", exp_dir, list(exp_dir.glob("*")))
            if not (exp_dir / "avg_result.json").exists():
                continue
            conf = load_json(exp_dir / "config.json")
            avg_res = load_json(exp_dir / "avg_result.json")[0]
            # import code

            # code.interact(local=locals() | globals())
            beam_key = "tree_max_width" if method == "beam_search" else "num_sequence"
            lm = conf["LM"]
            num_params = float(re.search(r"-(\d+(?:\.\d+)?)B", lm).group(1))
            d = {
                "dataset": dataset,
                "method": method.replace("_", " ").title(),
                "temperature": conf["gen_config"]["temperature"],
                "beam_width": conf["method_config"][beam_key],
                "majority_vote": avg_res["majority_vote"],
                "total_completion_tokens": avg_res["total_completion_tokens"],
                "LM": lm,
                "RM": conf["RM"],
                "num_params": num_params,
                "created_at": os.path.getmtime(exp_dir),
            }
            results.append(d)

df = pd.DataFrame(results)
df["budget"] = df["num_params"] * df["beam_width"]
df = df.sort_values(
    "created_at",
)


def plot_fixed_budget():
    fig = plt.figure(figsize=(6, 6))
    tmp = df[(45 <= df["budget"]) & (df["budget"] <= 65)]
    sns.lineplot(
        tmp,
        x="num_params",
        y="majority_vote",
        marker="o",
        hue="method",
        palette="tab10",
    )
    plt.xlabel("# Params")
    plt.xscale("log", base=2)
    plt.ylabel("Accuracy")
    plt.legend(title="Method")
    plt.title(f"{dataset.upper()} Performance Over Fixed Budget")

    plt.tight_layout()
    plt.savefig(f"{DATASET.lower()}-res-fixed-budget.pdf", dpi=300, bbox_inches="tight")


def plot_params_accuracy():
    fig = plt.figure(figsize=(6, 6))
    sns.lineplot(
        df,
        x="num_params",
        y="majority_vote",
        marker="o",
        hue="method",
        palette="tab10",
    )
    plt.xlabel("# Params")
    plt.xscale("log", base=2)
    plt.ylabel("Accuracy")
    plt.legend(title="Method")
    plt.title(f"{dataset.upper()} Performance Averaged Over Beam Widths")
    plt.tight_layout()
    plt.savefig(f"{DATASET.lower()}-res-avg-acc.pdf", dpi=300, bbox_inches="tight")


def plot_accuracy_beam():
    fig = plt.figure(figsize=(6, 8))
    sns.lineplot(
        df,
        x="beam_width",
        y="majority_vote",
        marker="o",
        style="method",
        hue="num_params",
        palette="tab10",
    )
    plt.xlabel("Beam Width")
    plt.xscale("log", base=2)
    plt.ylabel("Accuracy")
    # Move legend to the bottom, outside the plot
    plt.legend(
        title="# Params",
        loc="upper center",  # Position at upper center
        bbox_to_anchor=(
            0.5,
            -0.15,
        ),  # Place below the plot (x=0.5 centers it, y=-0.15 places it below)
        ncol=4,  # Arrange items in 3 columns for better horizontal spread
        frameon=True,  # Add a frame around the legend
        fontsize=10,
    )

    plt.title(f"{dataset.upper()} Performance vs. Beam Widths")
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(f"{DATASET.lower()}-res-acc-beam.pdf", dpi=300, bbox_inches="tight")


plot_accuracy_beam()
plot_fixed_budget()
plot_params_accuracy()
