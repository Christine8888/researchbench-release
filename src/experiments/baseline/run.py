"""CLI for running baseline evaluations."""

import argparse
import asyncio
import json
from pathlib import Path

from dataset.dataloader import Dataloader
from experiments.baseline.baseline import (
    evaluate_single_paper,
    evaluate_multiple_papers,
    load_and_summarize_results,
    print_summary
)
from logging_config import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline evaluation")

    parser.add_argument(
        "--paper_id", "-p",
        type=str,
        default=None,
        help="Specific paper ID to evaluate (default: all papers)"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="openai/gpt-4o",
        help="Model name in Inspect AI format (e.g., openai/gpt-4o)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./baseline_results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--workspace_base",
        type=str,
        default=None,
        help="Base workspace directory"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples per paper"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for model generation"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max tokens for model generation"
    )

    parser.add_argument(
        "--max_connections",
        type=int,
        default=10,
        help="Maximum concurrent API connections"
    )

    parser.add_argument(
        "--papers_dir",
        type=str,
        default=None,
        help="Papers directory"
    )

    parser.add_argument(
        "--tasks_dir",
        type=str,
        default=None,
        help="Tasks directory"
    )

    parser.add_argument(
        "--manuscripts_dir",
        type=str,
        default=None,
        help="Manuscripts directory"
    )

    parser.add_argument(
        "--task_types",
        type=str,
        nargs="+",
        default=["numeric", "code"],
        help="Task types to load"
    )

    parser.add_argument(
        "--filter_on",
        type=str,
        default=None,
        help='Filter papers by attributes (JSON format, e.g., \'{"source": "showyourwork"}\')'
    )

    parser.add_argument(
        "--masked",
        action="store_true",
        default=True,
        help="Use masked manuscripts"
    )

    parser.add_argument(
        "--no_masked",
        action="store_false",
        dest="masked",
        help="Use unmasked manuscripts"
    )

    parser.add_argument(
        "--include_workspace",
        action="store_true",
        default=False,
        help="Include workspace information in prompt"
    )

    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Use thinking mode with <answer> tags"
    )

    parser.add_argument(
        "--json_mode",
        action="store_true",
        default=False,
        help="Enable JSON mode for compatible models"
    )

    parser.add_argument(
        "--zero_tolerance",
        action="store_true",
        default=False,
        help="Override all tolerances to 0"
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip papers with existing results"
    )

    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Re-evaluate papers even if results exist"
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="Load and summarize existing results instead of running evaluation"
    )

    parser.add_argument(
        "--summary_model",
        type=str,
        default=None,
        help="Model name for summary (default: all models)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (overrides all other arguments)"
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(args, key, value)

    log_level = getattr(__import__("logging"), args.log_level)
    setup_logger("src.experiments.baseline", level=log_level)
    logger = setup_logger(__name__, level=log_level)

    if args.summary:
        logger.info("Loading and summarizing existing results")
        results = load_and_summarize_results(
            args.output_dir,
            model_name=args.summary_model,
            paper_id=args.paper_id
        )
        print_summary(results)
        return

    filter_on = json.loads(args.filter_on) if args.filter_on else None

    logger.info("Loading dataloader")
    dataloader = Dataloader(
        papers_dir=args.papers_dir,
        tasks_dir=args.tasks_dir,
        manuscripts_dir=args.manuscripts_dir,
        task_types=args.task_types,
        filters=filter_on,
        masked=args.masked,
        load_text=True
    )

    logger.info(f"Loaded {len(dataloader.papers)} papers")

    paper_ids = [args.paper_id] if args.paper_id else None

    eval_kwargs = {
        "workspace_base": args.workspace_base,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_connections": args.max_connections,
        "thinking": args.thinking,
        "json_mode": args.json_mode,
        "zero_tolerance": args.zero_tolerance,
        "include_workspace": args.include_workspace,
        "skip_existing": args.skip_existing
    }

    asyncio.run(
        evaluate_multiple_papers(
            dataloader=dataloader,
            paper_ids=paper_ids,
            model_name=args.model,
            output_dir=args.output_dir,
            **eval_kwargs
        )
    )


if __name__ == "__main__":
    main()
