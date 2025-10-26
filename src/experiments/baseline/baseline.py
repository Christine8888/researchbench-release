"""Core baseline evaluation functionality."""

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from inspect_ai.model import get_model, GenerateConfig, Model

from dataset.dataloader import Dataloader
from dataset.wrappers.paper import Paper
from evaluation.core.prompts import get_paper_prompt, get_task_instructions
from evaluation.core.scorer import parse_json_answer, check_json_equality
from evaluation.core.utils import recursive_zero_out
from logging_config import get_logger

logger = get_logger(__name__)


def create_paper_prompt(
    paper: Paper,
    workspace: Optional[str] = None,
    thinking: bool = False,
    include_workspace: bool = False
) -> str:
    """Create evaluation prompt for a paper."""
    paper_prompt = """TASK INSTRUCTIONS:
    -----------------
    You are an expert astrophysicist tasked with reproducing the key results of a scientific research paper. Your goal is to recall the results of the paper from your memory and by reading the paper manuscript. You will be given a list of tasks, and you will need to answer each task based on your knowledge of the paper.
    """

    paper_prompt += get_paper_prompt(
        paper=paper,
        workspace=workspace,
        include_workspace=include_workspace
    )

    for i, (task_id, task) in enumerate(paper.tasks.items()):
        paper_prompt += f"TASK {i+1}:\n"
        paper_prompt += get_task_instructions(task)

    paper_prompt += """The user requests that you do NOT perform any computations, code usage, or other actions. Do your best to answer the question using only guessing from the paper manuscript given and your training data."""

    if thinking:
        paper_prompt += """First, think carefully about the tasks, paper manuscript, and your understanding of the paper. Submit an educated guess for every single entry. Do not leave any fields as "null". Then submit your final answer as a valid JSON dictionary string with no additional text. You should submit only this JSON dictionary in <answer>....</answer> tags."""
    else:
        paper_prompt += """Submit an educated guess for every single entry. Do not leave any fields as "null". Submit as your final answer as a valid JSON dictionary string with no additional text. Return only this JSON dictionary in <answer>....</answer> tags."""

    return paper_prompt


def extract_answer_from_completion(completion: str, thinking: bool = False) -> str:
    """Extract JSON answer from completion text."""
    if not thinking:
        json_block_match = re.search(r"```json\s*\n?(.*?)\n?```", completion, re.DOTALL)
        if json_block_match:
            return json_block_match.group(1).strip()
        return completion

    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    json_block_match = re.search(r"```json\s*\n?(.*?)\n?```", completion, re.DOTALL)
    if json_block_match:
        logger.info("Extracted JSON from code block")
        return json_block_match.group(1).strip()

    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", completion, re.DOTALL)
    if json_match:
        logger.info("Extracted JSON structure directly")
        return json_match.group(0).strip()

    logger.warning("Could not find answer tags or JSON structure in response")
    return completion


async def generate_completions(
    model: Model,
    prompt: str,
    n_samples: int,
    config: GenerateConfig,
    json_mode: bool = False,
    thinking: bool = False
) -> List[str]:
    """Generate n completions concurrently."""
    messages = [{"role": "user", "content": prompt}]

    async def get_single_completion(sample_id: int) -> Optional[str]:
        try:
            if json_mode and not thinking:
                response = await model.generate(
                    messages,
                    config=config,
                    response_format={"type": "json_object"}
                )
            else:
                response = await model.generate(messages, config=config)

            return response.completion
        except Exception as e:
            logger.error(f"Error on sample {sample_id}: {e}")
            return None

    completions = await asyncio.gather(*[get_single_completion(i) for i in range(n_samples)])
    return [c for c in completions if c is not None]


def evaluate_completions(
    completions: List[str],
    expected_output: Dict[str, Any],
    output_tolerance: Dict[str, Any],
    thinking: bool = False
) -> Tuple[List[float], Dict[str, List[float]], List[Dict[str, Any]]]:
    """Evaluate completions and return scores and detailed results."""
    scores = []
    field_averages = {}
    results = []

    for completion in completions:
        answer = extract_answer_from_completion(completion, thinking)
        success, completion_data, error_msg = parse_json_answer(answer)

        if not success or not completion_data:
            logger.warning(f"Could not parse completion: {error_msg}")
            results.append({
                "completion": completion,
                "parsed": None,
                "score": 0.0,
                "error_msg": error_msg
            })
            continue

        score_result = check_json_equality(completion_data, expected_output, output_tolerance)
        scores.append(score_result["total_score"])

        for field, field_score in score_result["field_scores"].items():
            if field not in field_averages:
                field_averages[field] = []
            field_averages[field].append(field_score)

        results.append({
            "completion": completion,
            "parsed": completion_data,
            "score": score_result["total_score"],
            "field_scores": score_result["field_scores"],
            "field_issues": score_result["field_issues"],
            "error_msg": error_msg
        })

    return scores, field_averages, results


def save_results(
    output_dir: str,
    model_name: str,
    paper_id: str,
    results: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> Path:
    """Save results to hierarchical directory structure."""
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    output_file = model_dir / f"{paper_id}.jsonl"

    if output_file.exists():
        logger.warning(f"Overwriting existing results at {output_file}")

    with open(output_file, "w") as f:
        metadata = {
            "metadata": True,
            "paper_id": paper_id,
            "model": model_name,
            "n_samples": len(results),
            "timestamp": datetime.now().isoformat(),
            "config": config
        }
        f.write(json.dumps(metadata) + "\n")

        for i, result in enumerate(results):
            result_with_id = {"sample_id": i, **result}
            f.write(json.dumps(result_with_id) + "\n")

    logger.info(f"Saved results to {output_file}")
    return output_file


def check_existing_results(
    output_dir: str,
    model_name: str,
    paper_id: str
) -> Optional[float]:
    """Check if results already exist and return average score."""
    model_dir = Path(output_dir) / model_name
    output_file = model_dir / f"{paper_id}.jsonl"

    if not output_file.exists():
        return None

    logger.info(f"Results already exist at {output_file}")

    try:
        with open(output_file, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None

            scores = []
            for line in lines[1:]:
                if line.strip():
                    data = json.loads(line)
                    if "score" in data:
                        scores.append(data["score"])

            if scores:
                return float(np.mean(scores))
    except Exception as e:
        logger.error(f"Error reading existing results: {e}")

    return None


async def evaluate_single_paper(
    paper: Paper,
    model_name: str,
    output_dir: str,
    workspace_base: Optional[str] = None,
    n_samples: int = 10,
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    max_connections: int = 10,
    thinking: bool = False,
    json_mode: bool = False,
    zero_tolerance: bool = False,
    include_workspace: bool = False,
    skip_existing: bool = True
) -> Optional[float]:
    """Evaluate a single paper with baseline prompting."""
    paper_id = paper.paper_id
    logger.info(f"Evaluating paper: {paper_id}")

    if skip_existing:
        existing_score = check_existing_results(output_dir, model_name, paper_id)
        if existing_score is not None:
            logger.info(f"Skipping {paper_id}, average score: {existing_score:.4f}")
            return existing_score

    workspace = f"{workspace_base}/{paper_id}" if workspace_base else None
    prompt = create_paper_prompt(paper, workspace, thinking, include_workspace)

    expected_output = paper.get_output()
    output_tolerance = paper.get_output_tolerance()

    if zero_tolerance:
        output_tolerance = recursive_zero_out(output_tolerance, fill=0)
        logger.info(f"Overriding all tolerances to 0 for paper {paper_id}")

    model = get_model(
        model_name,
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            max_connections=max_connections
        )
    )

    logger.info(f"Generating {n_samples} completions for {paper_id}")
    completions = await generate_completions(
        model, prompt, n_samples,
        GenerateConfig(temperature=temperature, max_tokens=max_tokens),
        json_mode, thinking
    )

    logger.info(f"Evaluating {len(completions)} completions")
    scores, field_averages, results = evaluate_completions(
        completions, expected_output, output_tolerance, thinking
    )

    if not scores:
        logger.warning(f"No valid scores for {paper_id}")
        return None

    config = {
        "model": model_name,
        "n_samples": n_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "json_mode": json_mode,
        "zero_tolerance": zero_tolerance
    }

    save_results(output_dir, model_name, paper_id, results, config)

    avg_score = float(np.mean(scores))
    logger.info(f"Paper {paper_id} - Average score: {avg_score:.4f} ({len(scores)} samples)")

    return avg_score


async def evaluate_multiple_papers(
    dataloader: Dataloader,
    paper_ids: Optional[List[str]],
    model_name: str,
    output_dir: str,
    **kwargs
) -> Dict[str, float]:
    """Evaluate multiple papers."""
    papers_to_eval = paper_ids if paper_ids else list(dataloader.papers.keys())
    results = {}

    for paper_id in papers_to_eval:
        if paper_id not in dataloader.papers:
            logger.warning(f"Paper {paper_id} not found in dataloader")
            continue

        paper = dataloader.papers[paper_id]
        score = await evaluate_single_paper(
            paper, model_name, output_dir, **kwargs
        )

        if score is not None:
            results[paper_id] = score

    if results:
        logger.info("\n" + "="*60)
        logger.info("BASELINE EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Model: {model_name}")
        logger.info(f"Papers evaluated: {len(results)}")
        logger.info(f"Overall average score: {np.mean(list(results.values())):.4f}")
        logger.info("\nPer-paper scores:")
        for paper_id, score in sorted(results.items()):
            logger.info(f"  {paper_id}: {score:.4f}")
        logger.info("="*60)

    return results


def load_and_summarize_results(
    output_dir: str,
    model_name: Optional[str] = None,
    paper_id: Optional[str] = None
) -> Dict[str, Any]:
    """Load results and compute summary statistics."""
    output_path = Path(output_dir)

    if not output_path.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return {}

    all_results = {}

    if model_name and paper_id:
        result_file = output_path / model_name / f"{paper_id}.jsonl"
        if result_file.exists():
            all_results[model_name] = {paper_id: _load_single_result(result_file)}
    elif model_name:
        model_dir = output_path / model_name
        if model_dir.exists():
            all_results[model_name] = {}
            for result_file in model_dir.glob("*.jsonl"):
                paper_id = result_file.stem
                all_results[model_name][paper_id] = _load_single_result(result_file)
    else:
        for model_dir in output_path.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                all_results[model_name] = {}
                for result_file in model_dir.glob("*.jsonl"):
                    paper_id = result_file.stem
                    all_results[model_name][paper_id] = _load_single_result(result_file)

    return all_results


def _load_single_result(result_file: Path) -> Dict[str, Any]:
    """Load a single result file."""
    with open(result_file, "r") as f:
        lines = f.readlines()

        if not lines:
            return {}

        metadata = json.loads(lines[0]) if lines[0].strip() else {}

        scores = []
        field_scores = {}

        for line in lines[1:]:
            if not line.strip():
                continue
            data = json.loads(line)
            if "score" in data:
                scores.append(data["score"])
            if "field_scores" in data:
                for field, score in data["field_scores"].items():
                    if field not in field_scores:
                        field_scores[field] = []
                    field_scores[field].append(score)

        return {
            "metadata": metadata,
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "scores": scores,
            "n_samples": len(scores),
            "field_averages": {field: float(np.mean(scores)) for field, scores in field_scores.items()}
        }


def print_summary(results: Dict[str, Any]) -> None:
    """Print formatted summary of results."""
    logger.info("\n" + "="*60)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("="*60)

    for model_name, papers in results.items():
        logger.info(f"\nModel: {model_name}")
        logger.info(f"Papers: {len(papers)}")

        paper_scores = [paper_data["avg_score"] for paper_data in papers.values()]
        if paper_scores:
            logger.info(f"Overall average: {np.mean(paper_scores):.4f}")

        logger.info("\nPer-paper scores:")
        for paper_id, paper_data in sorted(papers.items()):
            logger.info(f"  {paper_id}: {paper_data['avg_score']:.4f} (n={paper_data['n_samples']})")

    logger.info("="*60)
