"""CLI for running paper masking."""

import argparse
import asyncio
import json
from pathlib import Path

from dataset.dataloader import Dataloader
from experiments.mask.mask import PaperMasker
from logging_config import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Mask numerical results in academic papers")

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="openai/gpt-4o",
        help="Model name in Inspect AI format (e.g., openai/gpt-4o)"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="./text",
        help="Directory containing unmasked paper .txt files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./masked_papers",
        help="Directory to write masked papers"
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
        "--paper_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific paper IDs to process (default: all papers)"
    )

    parser.add_argument(
        "--task_types",
        type=str,
        nargs="+",
        default=["numeric"],
        help="Task types to load"
    )

    parser.add_argument(
        "--filter_on",
        type=str,
        default=None,
        help='Filter papers by attributes (JSON format, e.g., \'{"source": "showyourwork"}\')'
    )

    parser.add_argument(
        "--num_lines",
        type=int,
        default=50,
        help="Number of lines per text chunk"
    )

    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Minimum similarity ratio for masked text validation"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max tokens for model generation"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model generation"
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

    log_level = getattr(__import__("logging"), args.log_level)
    setup_logger("src.experiments.mask", level=log_level)
    logger = setup_logger(__name__, level=log_level)

    filter_on = json.loads(args.filter_on) if args.filter_on else None

    logger.info("Loading dataloader")
    dataloader = Dataloader(
        papers_dir=args.papers_dir,
        tasks_dir=args.tasks_dir,
        paper_ids=args.paper_ids,
        task_types=args.task_types,
        filters=filter_on,
        load_text=False
    )

    logger.info(f"Loaded {len(dataloader.papers)} papers with {sum(len(p.tasks) for p in dataloader.papers.values())} total tasks")

    masker = PaperMasker(
        model_name=args.model,
        num_lines=args.num_lines,
        similarity_threshold=args.similarity_threshold,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    logger.info("Starting paper masking")
    results = asyncio.run(
        masker.process_all_papers(
            dataloader,
            args.input_dir,
            args.output_dir
        )
    )

    summary_path = Path(args.output_dir) / "masking_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved processing summary to: {summary_path}")


if __name__ == "__main__":
    main()
