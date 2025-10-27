"""Pre-defined prompt templates for transcript analysis."""

ENVIRONMENT_BUGS_PROMPT = """You are analyzing a transcript from an agentic AI environment where language models attempt to reproduce scientific paper results. The AI agents have access to tools like bash, python execution, file reading/writing, and web browsing. They are given paper text and specific numerical/code tasks to complete.

This is a debugging exercise - we're trying to identify environment issues, tool failures, and other technical problems that prevent the AI from successfully completing tasks.

Paper being reproduced: {paper_id}

Please analyze the following transcript and provide:
1. A concise summary of what the AI attempted to do
2. Any environment bugs, tool failures, or technical issues encountered

Common issues to look for:
- Tool execution errors or timeouts, where it seems that the AI might need more instructions on how to use the tool
- Missing dependencies or packages that the model is unable to install
- File system permission issues or warnings
- Environment configuration problems
- Resource exhaustion (memory, disk space, etc.)

TRANSCRIPT:
---
{transcript}
---

Please provide your analysis in exactly this format:
<summary>
[2-3 sentences summarizing what the AI tried to do]
</summary>

<bugs>
[Bullet points listing any bugs, tool failures, or technical issues found. If none found, write "No significant technical issues identified."]
</bugs>"""


MODEL_FAILURE_PROMPT = """You are analyzing a transcript from an agentic AI environment where language models attempt to reproduce scientific paper results. The AI agents have access to tools like bash, python execution, file reading/writing, and web browsing. They are given paper text and specific numerical/code tasks to complete.

Paper being reproduced: {paper_id}

Please analyze the following transcript and provide a single sentence summary of the agent's main failure.

Focus on:
- What the agent was ultimately unable to accomplish
- The core reason for failure (e.g., couldn't find required data, misunderstood the task, calculation errors, timeout, etc.)
- Be specific and concise

TRANSCRIPT:
---
{transcript}
---

Please provide your analysis in exactly this format:
<failure_summary>
[One sentence describing the agent's main failure]
</failure_summary>

The sentence should be specific and actionable. Examples of good summaries:
- "The agent failed to locate the required dataset file 'gw_data.csv' despite multiple search attempts."
- "The agent misunderstood the task and calculated the wrong metric, computing mean instead of median."
- "The agent's calculation timed out while trying to process a 10GB dataset without proper memory management."
- "The agent successfully completed all tasks without any failures."
"""


CLASSIFY_FAILURES_PROMPT = """You are analyzing failure summaries from AI agents attempting to reproduce scientific paper results.

Paper: {paper_id}
Failure Summary: {failure_summary}

Please classify this failure into one of the following categories:

1. EXECUTION - Problems with tool usage, environment limits, or technical execution
   Examples:
   - Making incorrect assumptions about a dataset's structure
   - Failing to reason about and account for environment limits, particularly the computation time limit
   - Failing to reason about and account for tool settings, particularly the non-persistent Python state
   - File not found errors, import errors, or dependency issues
   - Timeout errors or memory issues

2. CONCEPTUAL - Fundamental misunderstanding of the task or scientific procedure
   Examples:
   - Skipping key parts of the procedure (i.e. failing to use the correct basis transformation)
   - Implementing simplified versions of procedures (i.e. training a logistic regression classifier instead of a neural net)
   - Implementing incorrect procedures (i.e. calculating likelihoods or distance corrections incorrectly)
   - Misunderstanding the research question or objective
   - Using wrong formulas or algorithms

3. PERSISTENCE - Giving up before completing all tasks or not attempting required steps
   Examples:
   - Returning an empty submission with majority or all None values
   - Stopping after partial completion without attempting remaining tasks
   - Not trying alternative approaches after initial failures
   - Declaring inability to complete without exhausting reasonable options
   - Giving up due to perceived complexity rather than actual blockers

4. OTHER - Failures that don't clearly fit the above categories
   Examples:
   - Successfully completing all tasks (no failure)
   - Ambiguous or unclear failure reasons
   - External factors beyond the agent's control

Analyze the failure summary and provide your classification in the following format:

<classification>
CATEGORY: [One of: EXECUTION, CONCEPTUAL, PERSISTENCE, OTHER]
CONFIDENCE: [One of: HIGH, MEDIUM, LOW]
EXPLANATION: [1-2 sentences explaining why you chose this category]
</classification>"""


AVAILABLE_PROMPTS = {
    "environment": ENVIRONMENT_BUGS_PROMPT,
    "model-failure": MODEL_FAILURE_PROMPT,
    "classify": CLASSIFY_FAILURES_PROMPT
}


def get_prompt_template(prompt_name: str) -> str:
    """Get a pre-defined prompt template by name."""
    if prompt_name not in AVAILABLE_PROMPTS:
        raise ValueError(f"Unknown prompt: {prompt_name}. Available: {list(AVAILABLE_PROMPTS.keys())}")
    return AVAILABLE_PROMPTS[prompt_name]
