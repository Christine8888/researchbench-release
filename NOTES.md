# ResearchBench: Detailed Code Functionality Documentation

**Date**: 2025-10-20
**Purpose**: This document provides a comprehensive technical overview of ResearchBench's architecture, data structures, and code functionality to support reproduction and understanding of the benchmark.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Structure and Organization](#data-structure-and-organization)
3. [Core Data Classes (Wrappers)](#core-data-classes-wrappers)
4. [Data Loading System](#data-loading-system)
5. [Prompt Generation System](#prompt-generation-system)
6. [Scoring System](#scoring-system)
7. [Evaluation Framework](#evaluation-framework)
8. [Essential Files for Benchmark Replication](#essential-files-for-benchmark-replication)
9. [Data Flow Diagram](#data-flow-diagram)

---

## Overview

ResearchBench is an AI evaluation benchmark that tests language models' ability to reproduce key numerical results from scientific astrophysics papers. The system:

1. **Loads** scientific papers with their full LaTeX text
2. **Extracts** specific reproducible tasks from each paper
3. **Evaluates** AI models by having them reproduce numerical results
4. **Scores** model outputs against expected results with configurable tolerances

The benchmark is built on the Inspect AI framework and uses a modular, object-oriented design.

---

## Data Structure and Organization

### Directory Layout

```
ResearchBench/
├── papers/                     # Paper metadata and texts
│   ├── {paper_id}.json        # Paper metadata (63 papers)
│   ├── full_texts/            # LaTeX source texts (38 texts)
│   │   └── {paper_id}.txt
│   └── masked_texts/          # Texts with results masked (34 texts)
│       └── {paper_id}.txt
│
├── tasks/                     # Original task definitions
│   └── {paper_id}/           # Tasks for each paper (31 paper dirs)
│       └── {task_id}.json    # Individual task files
│
├── detailed_tasks/            # Expanded task definitions (30 paper dirs)
│   └── {paper_id}/           # Step-by-step task instructions
│       └── {task_id}.json    # Detailed task with procedural steps
│
├── wrappers/                  # Core data structures
│   ├── paper.py              # Paper class and loading functions
│   ├── task.py               # Task class
│   ├── dataset.py            # Dataset classes (HF, wget, local, API)
│   └── executor.py           # Execution environment specs
│
├── dataloader.py             # Main data loading orchestrator
├── prompts.py                # Prompt generation functions
├── json_score.py             # Scoring system with tolerances
├── inspect_setup.py          # Inspect AI task configuration
└── single.py                 # Single paper evaluation script
```

### Data Statistics

- **63** paper JSON files
- **38** full text files
- **34** masked text files
- **31** task directories in `tasks/`
- **30** task directories in `detailed_tasks/`

---

## Core Data Classes (Wrappers)

All wrapper classes are located in `wrappers/` and provide type-safe, validated data structures.

### 1. Paper Class (`wrappers/paper.py`)

**Purpose**: Represents a scientific paper with all metadata and text.

**Core Fields**:
```python
@dataclass
class Paper:
    paper_id: str                           # Unique identifier (e.g., "gw_nsbh")
    paper_link: str                         # ArXiv or publication URL
    code_available: bool                    # Whether code is available
    publication_date: datetime              # Publication date
    title: str                              # Paper title
    abstract: str                           # Abstract text
    full_text: str                          # Full LaTeX source (loaded from file)
    dataset: List[Dataset]                  # Associated datasets
    execution_requirements: Executor        # Code execution specs
    code_link: Optional[str]                # Link to code repository
    other_instructions: Optional[str]       # Additional instructions
    browsing_instructions: Optional[BrowsingInstructions]
    source: str                             # "expert" or "showyourwork"
    blacklist_packages: List[str]           # Forbidden packages

    # Populated by load_paper_and_tasks():
    task_list: List[str]                    # List of task IDs
    task_dir: str                           # Path to task directory
    output: Dict[str, Any]                  # Expected outputs by task_id
    output_tolerance: Dict[str, Any]        # Tolerances by task_id
    blank_output: Dict[str, Any]            # Template with zeros
```

**Key Methods**:

- `__init__(mask: bool = True, **kwargs)`:
  - Initializes paper from kwargs
  - Loads full text from `papers/full_texts/` or `papers/masked_texts/`
  - Handles dataset, executor, and date parsing

- `from_json(json_file: str, mask: bool)`:
  - Class method to load paper from JSON file
  - Returns initialized Paper object

- `get(key, default)` / `set(key, value)`:
  - Dictionary-style access to attributes

**Loading Function**:

```python
def load_paper_and_tasks(
    paper_file: str,
    paper_id: str,
    mask: bool = True,
    task_types: List[str] = ["numeric"],
    task_root_dir: str = "./tasks"
) -> Tuple[Paper, Dict[str, Task]]
```

This function:
1. Loads the paper from JSON
2. Loads all tasks from `{task_root_dir}/{paper_id}/`
3. Filters tasks by type (numeric, code)
4. Populates `paper.output`, `paper.output_tolerance`, `paper.blank_output`
5. Validates tolerance and expected_output types match
6. Returns (Paper object, Dict of Task objects)

**Utility Function**:

```python
def recursive_zero_out(value: Any, fill: Any = 0, deep_copy: bool = True) -> Any
```
- Recursively zeros out nested structures (dicts, lists, tuples)
- Used to create `blank_output` templates for models

### 2. Task Class (`wrappers/task.py`)

**Purpose**: Represents a single reproducible task from a paper.

**Core Fields**:
```python
@dataclass
class Task(Generic[T]):
    task_id: str              # Unique identifier
    paper_id: str             # Parent paper
    kind: str                 # "numeric" or "code"
    description: str          # Short task description
    instructions: str         # Detailed instructions (or list for detailed_tasks)
    expected_output: T        # Expected result (any JSON type)
    difficulty: T             # 1-10 scale (1=15 min, 10=1+ months)
    tolerance: T              # Matching structure to expected_output
    parents: Optional[List[str]]  # Dependent tasks
```

**Key Methods**:
- `from_json(json_file: str)`: Load task from JSON
- `dump_json(json_file: str)`: Save task to JSON
- `get(key, default)` / `set(key, value)`: Attribute access

**Validation**:
- Ensures `tolerance` matches `expected_output` type
- Type-checked with `typeguard`

**Task Type Differences**:

1. **Regular tasks** (`tasks/` directory):
   - `instructions` is a string
   - Brief, high-level description

2. **Detailed tasks** (`detailed_tasks/` directory):
   - `instructions` is a list of strings
   - Step-by-step procedural instructions
   - Each step numbered and explicit

### 3. Dataset Classes (`wrappers/dataset.py`)

**Purpose**: Manages dataset access and downloading for papers.

**Base Class**:
```python
@dataclass
class Dataset:
    paper_id: str
    dataset_name: str
    kind: str  # "wget", "API", "local", "huggingface", "none"
    data_instructions: Dict[str, str]  # access, usage, read instructions
```

**Factory Method**:
```python
@classmethod
def create(cls, **kwargs) -> Dataset:
    # Returns appropriate subclass based on 'kind'
```

**Subclasses**:

1. **NoneDataset**: No dataset required
2. **APIDataset**: API-based access
   - Additional fields: `api_key`, `api_url`

3. **LocalDataset**: Bundled with benchmark
   - Fields: `data_path: List[str]`, `size: List[float]`
   - `download()`: Copies from `data/` to workspace

4. **WgetDataset**: Downloadable via wget
   - Fields: `url: List[str]`, `size: List[float]`
   - `download()`: Downloads and extracts (zip/tar)

5. **HuggingFaceDataset**: HuggingFace Hub datasets
   - Fields: `hf_name: List[str]`, `hf_split: List[str]`, `hf_type: List[str]`
   - `download()`: Uses `snapshot_download()` or `load_dataset()`

**Key Methods**:
- `print_instructions(workspace, include_workspace)`: Generates prompt text
- `download(workspace)`: Downloads/copies data to workspace

### 4. Executor Class (`wrappers/executor.py`)

**Purpose**: Specifies code execution environment requirements.

**Core Fields**:
```python
@dataclass
class Executor:
    code_language: str          # "python", etc.
    dependencies: List[str]     # Required packages
    needs_gpu: bool             # GPU requirement
```

**Methods**:
- `print_instructions()`: Generates execution environment prompt
- `to_dict()`: Serializes to dictionary

**BrowsingInstructions Class**:
```python
@dataclass
class BrowsingInstructions:
    browsing_url: List[str]     # URLs to browse
    browsing_text: str          # Instructions for browsing
```

---

## Data Loading System

### Dataloader Class (`dataloader.py`)

**Purpose**: Central orchestrator for loading papers and tasks with filtering.

**Constructor**:
```python
def __init__(
    self,
    dir: str = "./papers",
    papers: List[str] = [],              # Empty = load all
    mask: bool = True,                   # Use masked texts
    task_types: List[str] = ["numeric"], # Task type filter
    filter_on: Dict[str, Any] = {"source": "showyourwork"},  # Paper filters
    task_root_dir: str = "./tasks"       # Task directory
)
```

**Instance Variables**:
- `self.papers: Dict[str, Paper]` - Loaded papers by paper_id
- `self.tasks: Dict[str, Dict[str, Task]]` - Tasks by paper_id and task_id
- `self.mask: bool` - Masking enabled
- `self.task_types: List[str]` - Task type filter
- `self.filter_on: Dict[str, Any]` - Paper attribute filters
- `self.task_root_dir: str` - Root task directory

**Key Methods**:

1. **`load(dir: str, papers: List[str])`**:
   - Scans directory for `*.json` files
   - Filters by paper list if provided
   - Calls `load_paper_and_tasks()` for each paper
   - Applies `filter_on` criteria (skips papers matching filter)
   - Populates `self.papers` and `self.tasks`
   - Returns `(papers_dict, tasks_dict)`

2. **`paper_prompt(paper_id, workspace, include_workspace, submit_file)`**:
   - Generates complete prompt for a paper
   - Combines: START_PROMPT + paper info + tasks + END_PROMPT
   - Includes dataset instructions
   - Adds workspace path information
   - Appends blank output format template
   - Returns full prompt string

**Filter Logic**:
```python
for filter, value in self.filter_on.items():
    if paper.get(filter) == value:
        skip = True  # Skip papers matching the filter
```

**Example Usage**:
```python
# Load all numeric expert tasks
dl = Dataloader(
    dir="./papers",
    task_types=["numeric"],
    filter_on={"source": "showyourwork"}  # Excludes showyourwork papers
)

# Load specific papers with detailed tasks
dl = Dataloader(
    papers=["gw_nsbh", "bayes_cal"],
    task_root_dir="./detailed_tasks",
    mask=True
)
```

---

## Prompt Generation System

### Module: `prompts.py`

**Purpose**: Generates all prompts used in evaluation.

### 1. System Prompt

**`SYSTEM_PROMPT`**: Core instructions for the AI model.

**Key Components**:
- Role: "Expert astrophysicist tasked with reproducing results"
- Critical restrictions (no hardcoding results, must compute from scratch)
- Failure handling (return None if cannot compute)
- Execution instructions:
  - Ephemeral Python/Bash environment
  - Must write code to files for persistence
  - Path handling requirements (absolute paths, `./` prefix)
  - Sanity check guidelines (physical ranges, etc.)
- Tool usage instructions
- Evaluation criteria

### 2. Submission Instructions

**`get_submission_instructions(submit_file: bool)`**:

Two modes:

**File Mode** (`submit_file=True`):
- Write results to `submission.json` in working directory
- Save progress frequently (partial credit)
- Call `submit()` tool when done

**Direct Mode** (`submit_file=False`):
- Submit results via `submit()` tool with JSON string
- Single submission at end

### 3. Task Prompt Components

**`START_PROMPT`**: Opening instructions
- Sets context about reproducibility
- Emphasizes no guessing/copying

**`END_PROMPT`**: Closing instructions
- Reminds about ephemeral environment
- Provides tool usage tips
- Encourages persistence

**`get_paper_prompt(paper, workspace, browsing, include_workspace)`**:
Generates paper-specific prompt including:
- Title and publication date
- Full paper text (LaTeX)
- Execution environment (from `Executor`)
- Dataset instructions (from each `Dataset`)
- Browsing instructions (if applicable)
- Additional author instructions
- Forbidden packages warning

**`get_task_prompt(task)`**:
Formats individual task:
- Task ID
- Task type (numeric/code)
- Short description
- Detailed instructions
- Expected output format hints (list/dict/float/tuple)

### 4. Prompt Assembly

Full prompt structure:
```
SYSTEM_PROMPT
+ SUBMISSION_INSTRUCTIONS
---
START_PROMPT
+ PAPER_INFORMATION
+ TASK_1
+ TASK_2
+ ...
+ END_PROMPT
+ WORKSPACE_PATH
+ EXPECTED_OUTPUT_FORMAT (blank template)
```

---

## Scoring System

### Module: `json_score.py`

**Purpose**: Flexible JSON comparison with nested tolerance support.

### Core Functions

#### 1. `compare_values(actual, expected, tolerance)`

**Recursive comparison function**.

**Returns**: `(is_correct: bool, issues: List[str])`

**Logic**:
1. **None handling**: Check if both None
2. **Type checking**: Ensure types match (int/float interchangeable)
3. **Recursive comparison by type**:

   **Dictionaries**:
   - Check keys match
   - Report missing keys
   - Recursively compare each key with nested tolerance

   **Lists/Tuples**:
   - Check length match
   - Recursively compare each element with indexed tolerance

   **Numerics (int/float)**:
   - Check `abs(actual - expected) <= tolerance`

   **Other (str, bool)**:
   - Exact equality check

**Tolerance Structure**: Mirrors expected_output structure
```python
expected_output = {
    "mass": 42.5,
    "values": [1.0, 2.0, 3.0],
    "nested": {"a": 10, "b": 20}
}

tolerance = {
    "mass": 0.1,
    "values": [0.01, 0.01, 0.01],
    "nested": {"a": 0.5, "b": 0.5}
}
```

#### 2. `check_json_equality(answer_data, target_data, tolerance_data)`

**Top-level comparison function**.

**Returns**:
```python
{
    "total_score": 0.75,           # Overall score (0.0-1.0)
    "field_scores": {              # Per-field scores
        "task1": 1.0,
        "task2": 0.0,
        "task3": 1.0
    },
    "field_issues": {              # Per-field issue descriptions
        "task1": ["Correct"],
        "task2": ["Value 50 exceeds tolerance 0.1 from expected 42"],
        "task3": ["Correct"]
    },
    "missing_fields": [],          # Fields in target but not answer
    "extra_fields": []             # Fields in answer but not target
}
```

**Scoring Logic**:
- Binary per-field: 1.0 if correct, 0.0 if incorrect
- Overall score: `correct_fields / total_fields`
- Handles dict and list top-level structures

#### 3. `parse_json_answer(answer_text)`

**Extracts JSON from model output**.

**Returns**: `(success: bool, parsed_data: Any, error_msg: str)`

**Parsing Strategies** (in order):
1. Try direct JSON parse
2. Extract JSON from code blocks (```json ... ```)
3. Find dictionary patterns with regex
4. Find list patterns with regex
5. Convert Python dict syntax to JSON (`'` → `"`)

#### 4. `json_scorer(output_tolerance)`

**Inspect AI scorer factory function**.

**Usage**:
```python
scorer = json_scorer(
    output_tolerance={
        "task1": 0.1,
        "task2": [0.05, 0.05],
        "task3": {"a": 0.01, "b": 0.02}
    }
)
```

**Process**:
1. Parse target (expected output)
2. Parse answer (model output)
3. Compare using `check_json_equality()`
4. Return `Score` object with:
   - `value`: total_score
   - `answer`: parsed answer JSON
   - `explanation`: field scores and issues
   - `metadata`: tolerance used

---

## Evaluation Framework

### Module: `inspect_setup.py`

**Purpose**: Configures Inspect AI tasks for evaluation.

### Main Task Function

```python
@task
def paper(
    dataloader: Dataloader,
    paper_id: str,
    workspace: str = None,
    attempts: int = 1,
    message_limit: int = 500,
    token_limit: int = 200000,
    execution_timeout: int = 6000,
    time_limit: int = 12000,
    cache: bool = True,
    mode: str = "base",          # "react", "base", or "plan"
    include_workspace: bool = True,
    submit_file: bool = False
) -> Task
```

**Functionality**:
1. Generates paper prompt using dataloader
2. Builds system prompt with submission instructions
3. Configures solver based on mode:
   - `"react"`: ReAct agent with reasoning
   - `"base"`: Basic agent with tools
   - `"plan"`: Multi-agent with planner handoff
4. Creates Inspect AI `Task` object with:
   - Dataset: Single sample with prompt and expected output
   - Solver: Configured agent
   - Scorer: `json_scorer` or `submission_file_scorer`
   - Limits: message, token, time
   - Sandbox: local execution environment

### Agent Configurations

#### 1. ReAct Agent (`react_agent`)

**Tools**:
- `execute_python(timeout)`: Python code execution
- `execute_bash(timeout)`: Bash command execution
- `think()`: Reasoning/reflection tool

**Prompt**:
- Instructions: System prompt
- Assistant prompt: Encourages persistence, no guessing

#### 2. Basic Agent (default)

**Tools**:
- `execute_python(timeout)`
- `execute_bash(timeout)`
- `think()`

**Features**:
- Built-in `submit()` tool
- Retry logic via `max_attempts`

#### 3. Multi-Agent with Planner

**Structure**:
- **Planner Agent**: Creates detailed plans
  - Only has `execute_bash` tool
  - Writes plans to `[task_id].md` files

- **Main Agent**: Executes plans
  - Has all execution tools
  - Can `handoff(planner)` to request plans

**Workflow**:
1. Main agent calls `transfer_to_planner`
2. Planner analyzes paper and creates plan
3. Control returns to main agent
4. Main agent follows plan

### Execution Environment

**Sandbox Settings**:
```python
sandbox = "local"
sandbox_env = {
    "HOME": "/tmp",
    "GALPYRC": "/tmp/.galpyrc",
    "XDG_CONFIG_HOME": "/tmp/.config",
    "XDG_CACHE_HOME": "/tmp/.cache",
}
```

---

### Module: `single.py`

**Purpose**: Command-line interface for single paper evaluation.

**Main Function**: `eval_single(args)`

**Process**:
1. Create log directory structure
2. Load paper via `Dataloader`
3. Construct paper workspace path
4. Create task using `paper()` function
5. Check GPU requirements
6. Configure model args (e.g., for OpenAI)
7. Run `eval()` with:
   - Task object
   - Model name
   - Log directory
   - Display settings
   - Retry settings

**Arguments** (via `parse_args()`):
```bash
--PAPER_ID / -p: Paper to evaluate
--MODEL / -m: Model name (e.g., "openai/gpt-4")
--RUN_NAME: Name for this run
--WORKSPACE_BASE: Base path for data
--MESSAGE_LIMIT: Max messages (default: 500)
--TOKEN_LIMIT: Max tokens (default: 1M)
--EXECUTION_TIMEOUT: Tool timeout (default: 6000s)
--TIME_LIMIT: Task timeout (default: 6000s)
--MODE: Agent mode (react/base/plan)
--MASKING: Use masked texts
--SUBMIT_FILE: Use file-based submission
--CONFIG: JSON config file (overrides all args)
```

**Config File Support**:
- JSON file with all parameters
- Allows complex settings (TASK_TYPES, FILTER_ON)
- Overrides command-line args

---

## Essential Files for Benchmark Replication

To replicate the ResearchBench benchmark, you need:

### 1. Data Files (REQUIRED)

```
papers/
├── *.json              # All 63 paper metadata files
├── full_texts/         # All paper texts
│   └── *.txt
└── masked_texts/       # Masked versions
    └── *.txt

tasks/                  # OR detailed_tasks/
└── {paper_id}/
    └── *.json         # Task definitions
```

### 2. Core Code (REQUIRED)

```
wrappers/
├── paper.py           # Paper class and loading
├── task.py            # Task class
├── dataset.py         # Dataset classes
└── executor.py        # Execution specs

dataloader.py          # Data loading orchestrator
prompts.py             # Prompt generation
json_score.py          # Scoring system
inspect_setup.py       # Inspect AI integration
single.py              # Evaluation runner
```

### 3. Configuration (REQUIRED)

```
configs/experiments/   # Example configurations
pyproject.toml        # Python dependencies
```

### 4. Optional but Useful

```
analysis/             # Analysis scripts
utils/                # Utility functions
baseline.py           # Baseline evaluation (no tools)
launch_batch.py       # Batch evaluation on SLURM
```

### 5. NOT Required for Basic Replication

```
logs/                 # Evaluation results (generated)
analysis/             # Analysis notebooks (post-hoc)
baseline_test/        # Old baseline results
auto/                 # Automation scripts
cleaning_evaluations/ # Data cleaning tools
failures/             # Failure analysis
experiment_configs/   # Specific experiment configs
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    BENCHMARK INITIALIZATION                  │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ Dataloader  │
                     └─────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │Load Papers│   │Load Tasks │   │Apply      │
    │from JSON  │   │from dirs  │   │Filters    │
    └───────────┘   └───────────┘   └───────────┘
            │               │               │
            └───────────────┴───────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │Papers: Dict[str, Paper]       │
            │Tasks: Dict[str, Dict[str,Task]]│
            └───────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                     PROMPT GENERATION                         │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │paper_prompt() │
                    └───────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐   ┌──────────────┐
│SYSTEM_PROMPT │    │Paper Info    │   │Task Prompts  │
│+ Submission  │    │+ Datasets    │   │x N           │
│Instructions  │    │+ Execution   │   │              │
└──────────────┘    └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Full Prompt   │
                    │ + Blank Output│
                    └───────────────┘

┌──────────────────────────────────────────────────────────────┐
│                        EVALUATION                             │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │inspect_setup  │
                    │  paper()      │
                    └───────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐   ┌──────────────┐
│Create Sample │    │Configure     │   │Setup Scorer  │
│(prompt,      │    │Agent/Solver  │   │(json_scorer) │
│ target)      │    │(react/base)  │   │              │
└──────────────┘    └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Inspect Task  │
                    └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  eval()       │
                    │  Run Model    │
                    └───────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
        ┌──────────────┐        ┌──────────────┐
        │Model Uses    │        │Model Submits │
        │Tools:        │        │JSON Result   │
        │- Python      │        │              │
        │- Bash        │        └──────────────┘
        │- Think       │                │
        └──────────────┘                │
                                        ▼

┌──────────────────────────────────────────────────────────────┐
│                          SCORING                              │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │parse_json_    │
                    │answer()       │
                    └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │check_json_    │
                    │equality()     │
                    └───────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐   ┌──────────────┐
│Compare Each  │    │Apply         │   │Calculate     │
│Field with    │    │Tolerances    │   │Total Score   │
│Tolerance     │    │(nested)      │   │              │
└──────────────┘    └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │Score Object:  │
                    │- value        │
                    │- field_scores │
                    │- issues       │
                    └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │Write to       │
                    │.eval Log      │
                    └───────────────┘
```

---

## Key Design Patterns

### 1. Factory Pattern
- `Dataset.create()` returns appropriate subclass based on `kind`
- Enables polymorphic dataset handling

### 2. Decorator Pattern
- `@task` decorator converts functions to Inspect Task objects
- `@scorer` decorator wraps scoring functions
- `@agent` decorator configures agent solvers

### 3. Composition Pattern
- `Paper` contains `Dataset` and `Executor` objects
- `Dataloader` orchestrates `Paper` and `Task` loading
- Flexible composition without tight coupling

### 4. Validation Pattern
- Type checking with `typeguard`
- Runtime validation in `__init__` methods
- Clear error messages for invalid data

### 5. Template Method Pattern
- `print_instructions()` in base classes
- Subclasses override with specific details
- Consistent interface across dataset types

---

## Important Implementation Details

### 1. Masking System
- Located in `utils/masking.py` (not covered in detail here)
- Replaces numerical values in paper texts with `[MASKED]`
- Prevents models from copying answers
- Applied based on `mask` parameter in Dataloader

### 2. Tolerance System
- Three types supported (though `json_score.py` currently uses absolute):
  - `absolute`: Fixed difference threshold
  - `relative`: Percentage difference
  - `significant_figures`: Match to N sig figs
- Tolerance structure must mirror expected_output structure
- Enables fine-grained control per field

### 3. Task Dependencies
- `parents` field in Task indicates dependencies
- Not currently enforced in evaluation
- Useful for understanding task order

### 4. Workspace Management
- Each paper gets workspace at `{WORKSPACE_BASE}/{paper_id}/`
- Datasets downloaded to workspace before evaluation
- Models read from workspace, write to working directory
- Separation prevents accidental data modification

### 5. File vs. Direct Submission
- **File mode**: Write to `submission.json`, enables partial credit
- **Direct mode**: Single `submit()` call with full JSON
- File mode preferred for long tasks with token limits

---

## Summary: What's Essential?

For someone to **replicate the benchmark**, they need:

1. **All paper JSONs** (`papers/*.json`) - Contains metadata, datasets, execution specs
2. **All paper texts** (`papers/full_texts/` and `papers/masked_texts/`) - The actual papers
3. **All task definitions** (`tasks/` or `detailed_tasks/`) - What to reproduce
4. **Wrapper classes** (`wrappers/*.py`) - Data structure definitions
5. **Dataloader** (`dataloader.py`) - Loads and validates data
6. **Prompt system** (`prompts.py`) - Generates evaluation prompts
7. **Scoring system** (`json_score.py`) - Grades model outputs
8. **Evaluation setup** (`inspect_setup.py`) - Integrates with Inspect AI
9. **Runner script** (`single.py`) - Executes evaluations

Optional but helpful:
- **Analysis tools** (`analysis/`) - For result visualization
- **Batch launcher** (`launch_batch.py`) - For large-scale runs
- **Config examples** (`configs/`) - Template configurations

---

## Version Information

This documentation reflects the state of the codebase as of October 20, 2025.

Key dependencies:
- Inspect AI framework (version printed in `inspect_setup.py`)
- Python 3.x with type hints
- Standard scientific stack (numpy, pandas)
- HuggingFace datasets and hub

---

## Contact and Contribution

For questions about this documentation or the benchmark design, refer to:
- `REPO_GUIDE.md` - Technical repository guide
- `CLAUDE.md` - Project-specific instructions
- `README.md` - User-facing documentation
