"""Core utilities for reading and extracting data from Inspect AI .eval logs."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from inspect_ai.log import read_eval_log

from logging_config import get_logger

logger = get_logger(__name__)


def read_eval_log_safe(log_file: str) -> Optional[Any]:
    """Safely read an eval log file, returning None if it fails."""
    try:
        return read_eval_log(log_file)
    except Exception as e:
        logger.error(f"Error reading {log_file}: {e}")
        return None


def extract_metadata(eval_log) -> Dict[str, Any]:
    """Extract comprehensive metadata from an eval log."""
    metadata = {
        "paper_name": eval_log.eval.task if eval_log.eval else "unknown",
        "model": eval_log.eval.model if eval_log.eval else "unknown",
        "status": eval_log.status if hasattr(eval_log, 'status') else "unknown",
        "started_at": None,
        "completed_at": None,
        "runtime_minutes": None,
        "accuracy": 0.0,
        "task_results": {},
        "response_stats": {
            "total_tasks": 0,
            "answered_tasks": 0,
            "response_rate": 0.0,
            "no_submission": False
        },
        "token_stats": {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0
        },
        "has_transcript": False,
        "num_messages": 0
    }

    if eval_log.stats:
        if hasattr(eval_log.stats, 'started_at'):
            metadata["started_at"] = eval_log.stats.started_at
        if hasattr(eval_log.stats, 'completed_at'):
            metadata["completed_at"] = eval_log.stats.completed_at

        if metadata["started_at"] and metadata["completed_at"]:
            try:
                start = datetime.fromisoformat(metadata["started_at"])
                end = datetime.fromisoformat(metadata["completed_at"])
                metadata["runtime_minutes"] = (end - start).total_seconds() / 60
            except Exception:
                pass

    if eval_log.stats and eval_log.stats.model_usage:
        for model, usage in eval_log.stats.model_usage.items():
            metadata["token_stats"]["input_tokens"] += usage.input_tokens
            metadata["token_stats"]["output_tokens"] += usage.output_tokens
            metadata["token_stats"]["total_tokens"] += usage.total_tokens
            if hasattr(usage, 'reasoning_tokens') and usage.reasoning_tokens is not None:
                metadata["token_stats"]["reasoning_tokens"] += usage.reasoning_tokens

    if eval_log.results and eval_log.results.scores:
        for score in eval_log.results.scores:
            if score.name == "submission_file_scorer":
                for metric_name, metric in score.metrics.items():
                    if metric_name == "accuracy":
                        metadata["accuracy"] = metric.value

    if hasattr(eval_log, 'reductions') and eval_log.reductions:
        for reduction in eval_log.reductions:
            if hasattr(reduction, 'samples'):
                for sample in reduction.samples:
                    if hasattr(sample, 'explanation'):
                        try:
                            explanation = json.loads(sample.explanation)
                            field_scores = explanation.get('field_scores', {})
                            metadata["task_results"] = field_scores
                        except json.JSONDecodeError:
                            pass

    if hasattr(eval_log, 'samples') and eval_log.samples:
        for sample in eval_log.samples:
            if hasattr(sample, 'messages') and sample.messages:
                metadata["has_transcript"] = True
                metadata["num_messages"] = len(sample.messages)

            submission_found = False
            if hasattr(sample, 'scores') and sample.scores:
                for scorer_name, score in sample.scores.items():
                    if scorer_name == "submission_file_scorer":
                        submission_found = True
                        if hasattr(score, 'answer') and score.answer:
                            try:
                                answer_data = json.loads(score.answer)
                                total_tasks = len(answer_data)
                                none_tasks = sum(1 for v in answer_data.values() if v is None)
                                answered_tasks = total_tasks - none_tasks

                                metadata["response_stats"]["total_tasks"] = total_tasks
                                metadata["response_stats"]["answered_tasks"] = answered_tasks
                                metadata["response_stats"]["response_rate"] = answered_tasks / total_tasks if total_tasks > 0 else 0.0
                            except json.JSONDecodeError:
                                metadata["response_stats"]["no_submission"] = True
                        else:
                            metadata["response_stats"]["no_submission"] = True

            if not submission_found and hasattr(sample, 'scores'):
                metadata["response_stats"]["no_submission"] = True

    return metadata


def extract_transcript(eval_log) -> str:
    """Extract transcript from an eval log."""
    if not eval_log.samples or len(eval_log.samples) == 0:
        return ""

    sample = eval_log.samples[0]
    if hasattr(sample, 'transcript') and sample.transcript:
        return str(sample.transcript)
    return ""


def extract_paper_id_from_filename(filename: str) -> str:
    """Extract paper ID from eval log filename."""
    import re
    import os

    basename = os.path.basename(filename)
    match = re.search(r'_([^_]+)_[A-Za-z0-9]+\.eval$', basename)
    return match.group(1) if match else "unknown"
