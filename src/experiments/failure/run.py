"""CLI for running transcript analysis with custom prompts."""

import argparse
import asyncio
from pathlib import Path

from experiments.failure.analyze import analyze_directory
from experiments.failure.prompts import get_prompt_template, AVAILABLE_PROMPTS
from logging_config import setup_logger


PROMPT_TAG_MAP = {
    "environment": ["summary", "bugs"],
    "model-failure": ["failure_summary"],
    "classify": ["classification"]
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Inspect AI evaluation transcripts using custom prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available built-in prompts:
{chr(10).join(f"  {name}" for name in AVAILABLE_PROMPTS.keys())}

Examples:
  # Use built-in environment analysis prompt
  python -m experiments.failure.run --input-dir ./logs --prompt environment

  # Use built-in model failure prompt
  python -m experiments.failure.run --input-dir ./logs --prompt model-failure

  # Use custom prompt from file
  python -m experiments.failure.run --input-dir ./logs --prompt-file my_prompt.txt --expected-tags result,confidence
"""
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing .eval log files"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="analysis_results.jsonl",
        help="Output JSONL file (default: analysis_results.jsonl)"
    )

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        type=str,
        choices=list(AVAILABLE_PROMPTS.keys()),
        help=f"Built-in prompt to use. Available: {', '.join(AVAILABLE_PROMPTS.keys())}"
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=str,
        help="Path to custom prompt template file"
    )

    parser.add_argument(
        "--expected-tags",
        type=str,
        nargs="+",
        default=None,
        help="XML-style tags to extract from responses (e.g., summary bugs). Required if using --prompt-file"
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model name in Inspect AI format (default: openai/gpt-4o-mini)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for model generation (default: 1.0)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens for model generation"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API calls (default: 10)"
    )

    parser.add_argument(
        "--max-tokens-per-segment",
        type=int,
        default=400000,
        help="Maximum tokens per transcript segment (default: 400000)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    log_level = getattr(__import__("logging"), args.log_level)
    setup_logger("experiments.failure", level=log_level)
    logger = setup_logger(__name__, level=log_level)

    if args.prompt:
        prompt_template = get_prompt_template(args.prompt)
        expected_tags = PROMPT_TAG_MAP.get(args.prompt, ["result"])
        logger.info(f"Using built-in prompt: {args.prompt}")
        logger.info(f"Expected tags: {expected_tags}")
    elif args.prompt_file:
        if not args.expected_tags:
            logger.error("--expected-tags is required when using --prompt-file")
            return

        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            logger.error(f"Prompt file not found: {args.prompt_file}")
            return

        with open(prompt_path, 'r') as f:
            prompt_template = f.read()

        expected_tags = args.expected_tags
        logger.info(f"Using custom prompt from: {args.prompt_file}")
        logger.info(f"Expected tags: {expected_tags}")
    else:
        logger.error("Must specify either --prompt or --prompt-file")
        return

    logger.info(f"Analyzing transcripts in: {args.input_dir}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Model: {args.model}")

    asyncio.run(
        analyze_directory(
            log_dir=args.input_dir,
            output_file=args.output,
            prompt_template=prompt_template,
            expected_tags=expected_tags,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_concurrent=args.max_concurrent,
            max_tokens_per_segment=args.max_tokens_per_segment
        )
    )


if __name__ == "__main__":
    main()
