"""CLI for running single paper evaluation."""

import argparse
import json
import os
import logging
from pathlib import Path
from typing import Optional, List
from inspect_ai import eval
from dataset.dataloader import Dataloader
from evaluation.inspect_task import paper as paper_task

logger = logging.getLogger(__name__)


def run_single_evaluation(
    paper_id: str,
    model: str,
    log_dir: str,
    workspace_base: Optional[str] = None,
    papers_dir: Optional[str] = None,
    tasks_dir: Optional[str] = None,
    manuscripts_dir: Optional[str] = None,
    task_types: Optional[List[str]] = None,
    masked: bool = True,
    message_limit: int = 500,
    token_limit: int = 200000,
    execution_timeout: int = 6000,
    time_limit: int = 12000,
    attempts: int = 1,
    cache: bool = True,
    mode: str = "base",
    include_workspace: bool = True,
    display: str = "full"
) -> None:
    """Run evaluation for a single paper.

    Args:
        paper_id: Paper ID to evaluate
        model: Model name for Inspect AI
        log_dir: Directory for evaluation logs
        workspace_base: Base workspace directory
        papers_dir: Papers directory (optional, uses default if None)
        tasks_dir: Tasks directory (optional, uses default if None)
        manuscripts_dir: Manuscripts directory (optional, uses default if None)
        task_types: Task types to evaluate
        masked: Whether to use masked text
        message_limit: Max messages in conversation
        token_limit: Max tokens in conversation
        execution_timeout: Tool execution timeout
        time_limit: Total task timeout
        attempts: Number of attempts
        cache: Use prompt caching
        mode: Agent mode ('base' or 'react')
        include_workspace: Alert agent about pre-downloaded data
        display: Display mode for Inspect
    """
    logger.info(f"Running evaluation for paper: {paper_id}")
    os.makedirs(log_dir, exist_ok=True)

    loader = Dataloader(
        papers_dir=papers_dir,
        tasks_dir=tasks_dir,
        manuscripts_dir=manuscripts_dir,
        paper_ids=[paper_id],
        task_types=task_types,
        load_text=True,
        masked=masked
    )

    if paper_id not in loader.papers:
        logger.error(f"Paper {paper_id} not found")
        raise ValueError(f"Paper {paper_id} not found")

    paper_obj = loader.papers[paper_id]
    paper_workspace = f"{workspace_base}/{paper_id}" if workspace_base else None

    task = paper_task(
        paper_obj=paper_obj,
        workspace=workspace_base,
        attempts=attempts,
        message_limit=message_limit,
        token_limit=token_limit,
        execution_timeout=execution_timeout,
        time_limit=time_limit,
        cache=cache,
        mode=mode,
        include_workspace=include_workspace
    )

    model_args = {}
    if model.startswith("openai/") or model.startswith("azure/"):
        model_args = {"responses_api": False}

    logger.info(f"Starting evaluation with model: {model}")
    eval(
        task,
        model=model,
        model_args=model_args,
        log_dir=log_dir,
        display=display,
        retry_on_error=3
    )
    logger.info("Evaluation complete")


def main():
    parser = argparse.ArgumentParser(description="Run single paper evaluation")

    parser.add_argument("--paper_id", "-p", required=True, help="Paper ID to evaluate")
    parser.add_argument("--model", "-m", required=True, help="Model name")
    parser.add_argument("--log_dir", "-l", required=True, help="Log directory")

    parser.add_argument("--workspace_base", help="Base workspace directory")
    parser.add_argument("--papers_dir", help="Papers directory")
    parser.add_argument("--tasks_dir", help="Tasks directory")
    parser.add_argument("--manuscripts_dir", help="Manuscripts directory")

    parser.add_argument("--task_types", nargs="+", help="Task types to evaluate")
    parser.add_argument("--no_masking", action="store_true", help="Disable text masking")

    parser.add_argument("--message_limit", type=int, default=500)
    parser.add_argument("--token_limit", type=int, default=200000)
    parser.add_argument("--execution_timeout", type=int, default=6000)
    parser.add_argument("--time_limit", type=int, default=12000)
    parser.add_argument("--attempts", type=int, default=1)

    parser.add_argument("--no_cache", action="store_true", help="Disable prompt caching")
    parser.add_argument("--mode", choices=["base", "react"], default="base")
    parser.add_argument("--no_workspace", action="store_true", help="Don't alert agent about workspace")
    parser.add_argument("--display", default="full")

    parser.add_argument("--config", help="JSON config file (overrides other args)")

    args = parser.parse_args()

    if args.config:
        logger.info(f"Loading config from {args.config}")
        with open(args.config) as f:
            config = json.load(f)

        for key, value in config.items():
            if not hasattr(args, key.lower()):
                setattr(args, key.lower(), value)

    run_single_evaluation(
        paper_id=args.paper_id,
        model=args.model,
        log_dir=args.log_dir,
        workspace_base=args.workspace_base,
        papers_dir=args.papers_dir,
        tasks_dir=args.tasks_dir,
        manuscripts_dir=args.manuscripts_dir,
        task_types=args.task_types,
        masked=not args.no_masking,
        message_limit=args.message_limit,
        token_limit=args.token_limit,
        execution_timeout=args.execution_timeout,
        time_limit=args.time_limit,
        attempts=args.attempts,
        cache=not args.no_cache,
        mode=args.mode,
        include_workspace=not args.no_workspace,
        display=args.display
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
