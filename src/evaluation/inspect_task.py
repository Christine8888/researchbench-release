"""Inspect AI task definitions for paper evaluation."""

import json
import logging
from inspect_ai import Task, task
from inspect_ai.agent import react, agent
from inspect_ai.tool import bash, python, think, tool_with
from inspect_ai.dataset import Sample
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.agent import AgentPrompt
from evaluation.core.prompts import SYSTEM_PROMPT, get_submission_instructions, get_paper_prompt
from evaluation.core.scorer import submission_file_scorer

logger = logging.getLogger(__name__)


@task
def paper(
    paper_obj,
    workspace: str = None,
    attempts: int = 1,
    message_limit: int = 500,
    token_limit: int = 200000,
    execution_timeout: int = 6000,
    time_limit: int = 12000,
    cache: bool = True,
    mode: str = "base",
    include_workspace: bool = True
):
    """Create Inspect task for a paper evaluation.

    Args:
        paper_obj: Paper object with tasks and metadata
        workspace: Paper-specific workspace directory path (e.g., /workspace/paper_id)
        attempts: Number of attempts to solve the task
        message_limit: Maximum messages in conversation
        token_limit: Maximum tokens in conversation
        execution_timeout: Timeout for agent tools (seconds)
        time_limit: Timeout for full task (seconds)
        cache: Whether to use prompt caching
        mode: Agent mode ('base' or 'react')
        include_workspace: Alert agent that data is pre-downloaded

    Returns:
        Inspect AI Task object
    """
    paper_prompt = get_paper_prompt(
        paper=paper_obj,
        workspace=workspace,
        include_workspace=include_workspace
    )

    system_prompt_with_submission = SYSTEM_PROMPT + "\n\n" + get_submission_instructions()

    if mode == "react":
        solver = react_agent(
            prompt=system_prompt_with_submission,
            attempts=attempts,
            timeout=execution_timeout,
            cache=cache
        )
    else:
        solver = basic_agent(
            init=system_message(system_prompt_with_submission),
            tools=[
                tool_with(
                    tool=python(timeout=execution_timeout),
                    name="execute_python"
                ),
                tool_with(
                    tool=bash(timeout=execution_timeout),
                    name="execute_bash"
                ),
                think()
            ],
            max_attempts=attempts,
            cache=cache
        )

    expected_output = paper_obj.get_output()
    output_tolerance = paper_obj.get_output_tolerance()

    return Task(
        dataset=[Sample(
            input=paper_prompt,
            target=json.dumps(expected_output),
        )],
        solver=solver,
        message_limit=message_limit,
        token_limit=token_limit,
        time_limit=time_limit,
        sandbox="local",
        scorer=submission_file_scorer(output_tolerance),
        attempts=attempts,
        name=paper_obj.paper_id,
        sandbox_env={
            "HOME": "/tmp",
            "GALPYRC": "/tmp/.galpyrc",
            "XDG_CONFIG_HOME": "/tmp/.config",
            "XDG_CACHE_HOME": "/tmp/.cache",
        }
    )


@agent
def react_agent(
    prompt: str = None,
    attempts: int = 1,
    timeout: int = 6000,
    cache: bool = True
):
    """Set up a ReAct agent for reproducing research papers.

    Args:
        prompt: System prompt to use
        attempts: Number of attempts to solve the task
        timeout: Timeout for agent tools (seconds)
        cache: Whether to use prompt caching

    Returns:
        ReAct agent
    """
    if prompt:
        agent_prompt = AgentPrompt(
            instructions=prompt,
            assistant_prompt="Always try again if you are stuck. DO NOT give up early or copy answers. DO NOT submit the task early unless the task is completed, impossible, or running out of tokens/time. NEVER guess values - just submit None for fields you cannot compute."
        )
    else:
        agent_prompt = AgentPrompt(
            instructions="You are an expert at using code and shell execution to reproduce research papers.",
            assistant_prompt="Always try again if you are stuck. DO NOT give up early or copy answers. DO NOT submit the task early unless the task is completed, impossible, or running out of tokens/time. NEVER guess values - just submit None for fields you cannot compute."
        )

    return react(
        name="researcher",
        description="Expert astrophysics researcher",
        prompt=agent_prompt,
        tools=[
            tool_with(
                tool=python(timeout=timeout),
                name="execute_python"
            ),
            tool_with(
                tool=bash(timeout=timeout),
                name="execute_bash"
            ),
            think()
        ],
        attempts=attempts,
    )
