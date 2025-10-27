"""Core transcript analysis functionality using LLM prompts."""

import asyncio
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from inspect_ai.model import get_model, GenerateConfig, Model

from experiments.utils.log_reader import (
    read_eval_log_safe,
    extract_metadata,
    extract_transcript,
    extract_paper_id_from_filename
)
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    eval_id: str
    paper_id: str
    response: str
    parsed_data: Dict[str, Any]
    metadata: Dict[str, Any]
    was_segmented: bool = False
    num_segments: int = 1


def estimate_token_count(text: str, model: str = "gpt-4") -> int:
    """Estimate token count for a given text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def split_transcript_into_segments(
    transcript: str,
    max_tokens_per_segment: int = 400000,
    model: str = "gpt-4"
) -> List[Tuple[str, int, int]]:
    """Split transcript into segments that fit within token limit."""
    total_tokens = estimate_token_count(transcript, model)

    if total_tokens <= max_tokens_per_segment:
        return [(transcript, 0, total_tokens)]

    num_segments = (total_tokens + max_tokens_per_segment - 1) // max_tokens_per_segment

    chars_per_token = 4
    total_chars = len(transcript)
    chars_per_segment = total_chars // num_segments

    segments = []
    overlap_chars = int(chars_per_segment * 0.1)

    for i in range(num_segments):
        start_char = max(0, i * chars_per_segment - (overlap_chars if i > 0 else 0))
        end_char = min(total_chars, (i + 1) * chars_per_segment + overlap_chars)

        segment = transcript[start_char:end_char]
        start_token = start_char // chars_per_token
        end_token = end_char // chars_per_token

        segment_header = f"\n[SEGMENT {i+1}/{num_segments} - Tokens ~{start_token:,} to ~{end_token:,}]\n"
        segment_with_header = segment_header + segment

        segments.append((segment_with_header, start_token, end_token))

    return segments


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with provided arguments."""
    return template.format(**kwargs)


def parse_response(response: str, expected_tags: List[str]) -> Dict[str, str]:
    """Parse LLM response and extract content from XML-style tags."""
    parsed = {}

    for tag in expected_tags:
        pattern = f'<{tag}>\\s*(.*?)\\s*</{tag}>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            parsed[tag] = match.group(1).strip()
        else:
            parsed[tag] = f"Failed to extract {tag}"

    return parsed


async def analyze_with_prompt(
    model: Model,
    prompt: str,
    expected_tags: List[str],
    config: GenerateConfig
) -> Tuple[str, Dict[str, str]]:
    """Send prompt to LLM and parse response."""
    messages = [{"role": "user", "content": prompt}]

    try:
        response = await model.generate(messages, config=config)
        completion = response.completion

        parsed = parse_response(completion, expected_tags)

        return completion, parsed

    except Exception as e:
        logger.error(f"Error in LLM call: {e}")
        return "", {tag: f"Error: {e}" for tag in expected_tags}


async def analyze_transcript(
    model: Model,
    eval_log,
    log_file: str,
    prompt_template: str,
    expected_tags: List[str],
    config: GenerateConfig,
    max_tokens_per_segment: int = 400000
) -> Optional[AnalysisResult]:
    """Analyze a single transcript using the provided prompt template."""
    metadata = extract_metadata(eval_log)

    filename = os.path.basename(log_file)
    paper_id = extract_paper_id_from_filename(filename)

    transcript = extract_transcript(eval_log)

    if not transcript:
        logger.info(f"Skipping {log_file}: No transcript available")
        return None

    token_count = estimate_token_count(transcript)

    if token_count > max_tokens_per_segment:
        logger.info(f"Transcript for {filename} has ~{token_count:,} tokens - splitting into segments")
        segments = split_transcript_into_segments(transcript, max_tokens_per_segment)

        all_responses = []
        all_parsed = {tag: [] for tag in expected_tags}

        for i, (segment_text, start_token, end_token) in enumerate(segments):
            segment_info = f"{filename} segment {i+1}/{len(segments)}"
            logger.info(f"Analyzing {segment_info}")

            prompt = format_prompt(
                prompt_template,
                paper_id=paper_id,
                transcript=segment_text
            )

            completion, parsed = await analyze_with_prompt(
                model, prompt, expected_tags, config
            )

            all_responses.append(completion)
            for tag, value in parsed.items():
                all_parsed[tag].append(value)

        combined_response = "\n\n".join(all_responses)
        combined_parsed = {tag: " ".join(values) for tag, values in all_parsed.items()}

        return AnalysisResult(
            eval_id=filename,
            paper_id=paper_id,
            response=combined_response,
            parsed_data=combined_parsed,
            metadata=metadata,
            was_segmented=True,
            num_segments=len(segments)
        )
    else:
        logger.info(f"Transcript for {filename} has ~{token_count:,} tokens - analyzing as single segment")

        prompt = format_prompt(
            prompt_template,
            paper_id=paper_id,
            transcript=transcript
        )

        completion, parsed = await analyze_with_prompt(
            model, prompt, expected_tags, config
        )

        return AnalysisResult(
            eval_id=filename,
            paper_id=paper_id,
            response=completion,
            parsed_data=parsed,
            metadata=metadata,
            was_segmented=False,
            num_segments=1
        )


async def analyze_directory(
    log_dir: str,
    output_file: str,
    prompt_template: str,
    expected_tags: List[str],
    model_name: str = "openai/gpt-4o-mini",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    max_concurrent: int = 10,
    max_tokens_per_segment: int = 400000
) -> List[AnalysisResult]:
    """Analyze all eval logs in a directory using the provided prompt."""
    model = get_model(
        model_name,
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens
        )
    )

    eval_files = glob.glob(os.path.join(log_dir, "*.eval"))

    if not eval_files:
        logger.warning(f"No .eval files found in {log_dir}")
        return []

    logger.info(f"Found {len(eval_files)} eval files to analyze")

    valid_analyses = []
    for i in range(0, len(eval_files), max_concurrent):
        batch = eval_files[i:i+max_concurrent]
        tasks = []

        for log_file in batch:
            eval_log = read_eval_log_safe(log_file)
            if eval_log:
                tasks.append(
                    analyze_transcript(
                        model,
                        eval_log,
                        log_file,
                        prompt_template,
                        expected_tags,
                        GenerateConfig(temperature=temperature, max_tokens=max_tokens),
                        max_tokens_per_segment
                    )
                )

        results = await asyncio.gather(*tasks)
        valid_analyses.extend([r for r in results if r])
        logger.info(f"Processed {min(i+max_concurrent, len(eval_files))}/{len(eval_files)} files")

    logger.info(f"Successfully analyzed {len(valid_analyses)} out of {len(eval_files)} files")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for analysis in valid_analyses:
            result = {
                "eval_id": analysis.eval_id,
                "paper_id": analysis.paper_id,
                "parsed_data": analysis.parsed_data,
                "metadata": analysis.metadata,
                "was_segmented": analysis.was_segmented,
                "num_segments": analysis.num_segments
            }
            f.write(json.dumps(result) + '\n')

    logger.info(f"Results written to {output_file}")

    print_analysis_summary(valid_analyses, expected_tags)

    return valid_analyses


def print_analysis_summary(analyses: List[AnalysisResult], expected_tags: List[str]):
    """Print summary statistics from analyses."""
    segmented_count = sum(1 for a in analyses if a.was_segmented)
    if segmented_count > 0:
        logger.info(f"{segmented_count} transcripts were segmented due to token limits")

    logger.info("\n" + "="*60)
    logger.info("EVALUATION STATISTICS")
    logger.info("="*60)

    scores = [a.metadata.get('accuracy', 0.0) for a in analyses]
    if scores:
        mean_score = sum(scores) / len(scores)
        logger.info(f"Mean accuracy across all papers: {mean_score:.2%}")

        logger.info("\nPer-paper accuracy scores:")
        paper_scores = {}
        for analysis in analyses:
            paper_id = analysis.paper_id
            accuracy = analysis.metadata.get('accuracy', 0.0)
            if paper_id not in paper_scores:
                paper_scores[paper_id] = []
            paper_scores[paper_id].append(accuracy)

        for paper_id, accuracies in sorted(paper_scores.items()):
            avg_accuracy = sum(accuracies) / len(accuracies)
            logger.info(f"  {paper_id}: {avg_accuracy:.2%}")

    runtimes = [a.metadata.get('runtime_minutes', 0) for a in analyses if a.metadata.get('runtime_minutes')]
    if runtimes:
        logger.info(f"\nAverage runtime: {sum(runtimes) / len(runtimes):.1f} minutes")
        logger.info(f"Total runtime: {sum(runtimes):.1f} minutes")

    logger.info("\n" + "="*60)
    logger.info("SAMPLE ANALYSES")
    logger.info("="*60)
    for i, analysis in enumerate(analyses[:5]):
        logger.info(f"\n{analysis.paper_id} ({analysis.eval_id}):")
        for tag in expected_tags:
            content = analysis.parsed_data.get(tag, "N/A")
            preview = content[:200] + "..." if len(content) > 200 else content
            logger.info(f"  {tag}: {preview}")
        if analysis.was_segmented:
            logger.info(f"  (Combined from {analysis.num_segments} segments)")

    if len(analyses) > 5:
        logger.info(f"\n... and {len(analyses) - 5} more transcripts")
