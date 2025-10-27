"""Analysis utilities for aggregating and computing statistics from eval logs."""

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dataset.dataloader import Dataloader
from experiments.utils.log_reader import read_eval_log_safe, extract_metadata
from logging_config import get_logger

logger = get_logger(__name__)


def get_difficulty_weights(dataloader: Dataloader, paper_name: str, task_results: Dict[str, float]) -> Tuple[float, float]:
    """Calculate difficulty-weighted score for a paper.

    Returns:
        (difficulty_weighted_score, normal_score)
    """
    if paper_name not in dataloader.papers:
        normal_score = sum(task_results.values()) / len(task_results) if task_results else 0.0
        return 0.0, normal_score

    paper = dataloader.papers[paper_name]
    weighted_score = 0.0
    total_difficulty = 0.0

    for task_id, task in paper.tasks.items():
        task_id_clean = task_id.replace('.json', '')
        difficulty = task.difficulty
        total_difficulty += difficulty

        if task_id_clean in task_results:
            weighted_score += task_results[task_id_clean] * difficulty

    normal_score = sum(task_results.values()) / len(task_results) if task_results else 0.0

    if total_difficulty > 0:
        return weighted_score / total_difficulty, normal_score

    return 0.0, normal_score


def load_eval_logs_to_dataframe(
    log_dirs: List[str],
    dataloader: Optional[Dataloader] = None
) -> pd.DataFrame:
    """Load all eval logs from directories into a MultiIndex DataFrame.

    Returns DataFrame with MultiIndex (model, run, paper, task) and columns for metrics.
    """
    if dataloader is None:
        dataloader = Dataloader(
            task_types=["numeric", "code"],
            load_text=False
        )

    all_data = []

    for log_dir in log_dirs:
        dir_name = os.path.basename(log_dir)

        if "o3" in dir_name:
            model = "o3"
        elif "o4-mini" in dir_name:
            model = "o4-mini"
        elif "sonnet-37" in dir_name or "sonnet-3.7" in dir_name:
            model = "Sonnet 3.7"
        elif "sonnet-4" in dir_name:
            model = "Sonnet 4"
        elif "gemini-25" in dir_name or "gemini-2.5" in dir_name:
            model = "Gemini 2.5"
        elif "kimik2" in dir_name:
            model = "Kimi K2"
        else:
            model = dir_name

        run_parts = dir_name.split('-')
        run_match = run_parts[-1] if run_parts else "1"
        run = int(run_match) if run_match.isdigit() else 1

        eval_files = glob.glob(os.path.join(log_dir, "logs", "*.eval"))
        if not eval_files:
            eval_files = glob.glob(os.path.join(log_dir, "*.eval"))

        for eval_file in eval_files:
            eval_log = read_eval_log_safe(eval_file)
            if not eval_log:
                continue

            metadata = extract_metadata(eval_log)
            paper_name = metadata["paper_name"]

            weighted_score, regular_score = get_difficulty_weights(
                dataloader, paper_name, metadata["task_results"]
            )

            all_data.append({
                "model": model,
                "run": run,
                "paper": paper_name,
                "task": "_summary",
                "accuracy": metadata["accuracy"],
                "difficulty_weighted_accuracy": weighted_score,
                "response_rate": metadata["response_stats"]["response_rate"],
                "answered_tasks": metadata["response_stats"]["answered_tasks"],
                "total_tasks": metadata["response_stats"]["total_tasks"],
                "input_tokens": metadata["token_stats"]["input_tokens"],
                "output_tokens": metadata["token_stats"]["output_tokens"],
                "reasoning_tokens": metadata["token_stats"]["reasoning_tokens"],
                "runtime_minutes": metadata["runtime_minutes"],
                "has_transcript": metadata["has_transcript"],
                "task_score": None,
                "task_difficulty": None
            })

            if paper_name in dataloader.papers:
                paper = dataloader.papers[paper_name]
                for task_id, task in paper.tasks.items():
                    task_id_clean = task_id.replace('.json', '')
                    task_score = metadata["task_results"].get(task_id_clean, None)

                    all_data.append({
                        "model": model,
                        "run": run,
                        "paper": paper_name,
                        "task": task_id_clean,
                        "accuracy": None,
                        "difficulty_weighted_accuracy": None,
                        "response_rate": None,
                        "answered_tasks": None,
                        "total_tasks": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "reasoning_tokens": None,
                        "runtime_minutes": None,
                        "has_transcript": None,
                        "task_score": task_score,
                        "task_difficulty": task.difficulty
                    })

    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.set_index(["model", "run", "paper", "task"])
        df = df.sort_index()

    return df


def aggregate_runs(df: pd.DataFrame, metric: str, agg_func: str = "mean") -> pd.DataFrame:
    """Aggregate metrics across runs for each model.

    Args:
        df: MultiIndex DataFrame from load_eval_logs_to_dataframe
        metric: Column name to aggregate
        agg_func: Aggregation function ('mean', 'max', 'min', 'std')
    """
    summary_df = df[df.index.get_level_values("task") == "_summary"].copy()

    if agg_func == "mean":
        result = summary_df.groupby(level=["model", "paper"])[metric].mean()
    elif agg_func == "max":
        result = summary_df.groupby(level=["model", "paper"])[metric].max()
    elif agg_func == "min":
        result = summary_df.groupby(level=["model", "paper"])[metric].min()
    elif agg_func == "std":
        result = summary_df.groupby(level=["model", "paper"])[metric].std()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")

    return result.unstack(level="paper", fill_value=0)


def get_model_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics for each model across all papers and runs."""
    summary_df = df[df.index.get_level_values("task") == "_summary"].copy()

    stats = []
    for model in summary_df.index.get_level_values("model").unique():
        model_data = summary_df.xs(model, level="model")

        accuracy_by_run = model_data.groupby('run')['accuracy'].mean()
        average_accuracy = accuracy_by_run.mean()
        std_accuracy = accuracy_by_run.std()
        best_accuracy = accuracy_by_run.max()

        stats.append({
            "Model": model,
            "Avg Accuracy": average_accuracy,
            "Std Accuracy": std_accuracy,
            "Best Accuracy": best_accuracy,
            "Avg Difficulty-Weighted": model_data["difficulty_weighted_accuracy"].mean(),
            "Best Difficulty-Weighted": model_data["difficulty_weighted_accuracy"].max(),
            "Avg Response Rate": model_data["response_rate"].mean(),
            "Avg Output Tokens": model_data["output_tokens"].mean(),
            "Avg Reasoning Tokens": model_data["reasoning_tokens"].mean(),
            "Avg Runtime (min)": model_data["runtime_minutes"].mean()
        })

    return pd.DataFrame(stats).set_index("Model")


def get_per_paper_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Get per-paper statistics across all models and runs."""
    summary_df = df[df.index.get_level_values("task") == "_summary"].copy()

    stats = []
    for paper in summary_df.index.get_level_values("paper").unique():
        paper_data = summary_df.xs(paper, level="paper")

        stats.append({
            "Paper": paper,
            "Avg Accuracy": paper_data["accuracy"].mean(),
            "Std Accuracy": paper_data["accuracy"].std(),
            "Best Accuracy": paper_data["accuracy"].max(),
            "Avg Difficulty-Weighted": paper_data["difficulty_weighted_accuracy"].mean(),
            "Avg Response Rate": paper_data["response_rate"].mean(),
            "Num Evaluations": len(paper_data)
        })

    return pd.DataFrame(stats).set_index("Paper")
