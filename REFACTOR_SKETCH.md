# Dataset Refactoring Sketch

## Overview

This sketch shows how to cleanly separate:
1. **Data models** (pure data, no prompts/evaluation logic)
2. **Data loading** (simplified, single responsibility)
3. **Prompt generation** (moved to evaluation/)
4. **Evaluation setup** (scoring metadata separate from data)

---

## 1. Pure Data Models (dataset/wrappers/)

### paper.py - REFACTORED

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import json

@dataclass
class Paper:
    """Pure data model for a research paper.

    Contains only paper metadata and content.
    No evaluation or prompt generation logic.
    """
    paper_id: str
    title: str
    abstract: str
    publication_date: datetime

    # Paper content (loaded separately)
    full_text: str = ""

    # Metadata
    paper_link: str = ""
    code_available: bool = False
    code_link: Optional[str] = None
    source: str = "expert"  # "expert" or "showyourwork"

    # Requirements (reference to other data models)
    datasets: List['Dataset'] = field(default_factory=list)
    execution_requirements: Optional['Executor'] = None
    browsing_instructions: Optional['BrowsingInstructions'] = None

    # Additional instructions
    other_instructions: Optional[str] = None
    blacklist_packages: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, json_path: str) -> 'Paper':
        """Load paper metadata from JSON file.

        Note: Does NOT load text content. Use load_paper_text() separately.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        return cls(
            paper_id=data['paper_id'],
            title=data['title'],
            abstract=data['abstract'],
            publication_date=_parse_date(data['publication_date']),
            paper_link=data.get('paper_link', ''),
            code_available=data.get('code_available', False),
            code_link=data.get('code_link'),
            source=data.get('source', 'expert'),
            datasets=_parse_datasets(data.get('dataset', [])),
            execution_requirements=_parse_executor(data.get('execution_requirements')),
            browsing_instructions=_parse_browsing(data.get('browsing_instructions')),
            other_instructions=data.get('other_instructions'),
            blacklist_packages=data.get('blacklist_packages', [])
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'abstract': self.abstract,
            'publication_date': self.publication_date.strftime('%Y-%m-%d'),
            'paper_link': self.paper_link,
            'code_available': self.code_available,
            'code_link': self.code_link,
            'source': self.source,
            # ... other fields
        }

# Helper functions (extracted from __init__ complexity)
def _parse_date(date_str: str) -> datetime:
    """Parse publication date string."""
    return datetime.strptime(date_str, "%Y-%m-%d")

def _parse_datasets(dataset_data) -> List['Dataset']:
    """Parse dataset configuration into Dataset objects."""
    from .dataset import Dataset

    if isinstance(dataset_data, dict):
        return [Dataset.create(**dataset_data)]
    elif isinstance(dataset_data, list):
        return [Dataset.create(**item) for item in dataset_data]
    else:
        return []

def _parse_executor(exec_data) -> Optional['Executor']:
    """Parse execution requirements."""
    from .executor import Executor

    if exec_data is None:
        return None
    return Executor(**exec_data)

def _parse_browsing(browsing_data) -> Optional['BrowsingInstructions']:
    """Parse browsing instructions."""
    from .executor import BrowsingInstructions

    if browsing_data is None:
        return None
    return BrowsingInstructions(**browsing_data)
```

### task.py - REFACTORED

```python
from dataclasses import dataclass
from typing import Any, List, Optional, Union
import json

@dataclass
class Task:
    """Pure data model for a task.

    Contains only task definition and expected output.
    No prompt generation or scoring logic.
    """
    task_id: str
    paper_id: str
    kind: str  # "numeric" or "code"
    difficulty: int  # 1-10

    # Instructions (normalized to always be a list)
    description: str
    instructions: List[str]  # Always a list of instruction steps

    # Expected output and tolerance
    expected_output: Any
    tolerance: Any

    # Dependencies
    parents: Optional[List[str]] = None

    @classmethod
    def from_json(cls, json_path: str) -> 'Task':
        """Load task from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Normalize instructions to always be a list
        instructions = data['instructions']
        if isinstance(instructions, str):
            instructions = [instructions]

        return cls(
            task_id=data['task_id'],
            paper_id=data['paper_id'],
            kind=data['kind'],
            difficulty=data['difficulty'],
            description=data['description'],
            instructions=instructions,
            expected_output=data['expected_output'],
            tolerance=data['tolerance'],
            parents=data.get('parents')
        )

    def validate_tolerance(self) -> bool:
        """Validate that tolerance matches expected_output structure."""
        return _validate_tolerance_structure(self.expected_output, self.tolerance)

def _validate_tolerance_structure(output: Any, tolerance: Any) -> bool:
    """Recursively validate tolerance matches output structure."""
    if isinstance(output, dict):
        if not isinstance(tolerance, dict):
            return False
        return all(
            key in tolerance and _validate_tolerance_structure(val, tolerance[key])
            for key, val in output.items()
        )
    elif isinstance(output, list):
        if not isinstance(tolerance, list):
            return False
        if len(output) != len(tolerance):
            return False
        return all(
            _validate_tolerance_structure(out_item, tol_item)
            for out_item, tol_item in zip(output, tolerance)
        )
    else:
        # For primitives, tolerance should be numeric or dict with tolerance spec
        return True
```

### dataset.py - REFACTORED

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Dataset:
    """Pure data model for dataset information.

    Contains only dataset metadata and access information.
    No prompt generation logic.
    """
    paper_id: str
    dataset_name: str
    kind: str  # "wget", "API", "local", "huggingface", "none"

    # Access instructions (just data)
    access_instructions: str = ""
    usage_instructions: str = ""
    read_instructions: str = ""

    @classmethod
    def create(cls, **kwargs) -> 'Dataset':
        """Factory method for different dataset types."""
        kind = kwargs.get("kind")

        # Extract data_instructions if present
        data_instructions = kwargs.get("data_instructions", {})
        kwargs['access_instructions'] = data_instructions.get('access_instructions', '')
        kwargs['usage_instructions'] = data_instructions.get('usage_instructions', '')
        kwargs['read_instructions'] = data_instructions.get('read_instructions', '')

        if kind == "API" or kind == "api":
            return APIDataset(**kwargs)
        elif kind == "local":
            return LocalDataset(**kwargs)
        elif kind == "wget":
            return WgetDataset(**kwargs)
        elif kind == "none":
            return NoneDataset(**kwargs)
        elif kind == "huggingface":
            return HuggingFaceDataset(**kwargs)
        else:
            return Dataset(**kwargs)

    def download(self, workspace: str) -> None:
        """Download dataset to workspace. Override in subclasses."""
        pass


@dataclass
class WgetDataset(Dataset):
    """Dataset downloadable via wget."""
    urls: List[str] = field(default_factory=list)
    sizes: List[float] = field(default_factory=list)  # in MB

    def __post_init__(self):
        # Handle old 'url' field name
        if hasattr(self, 'url') and self.urls == []:
            self.urls = self.url if isinstance(self.url, list) else [self.url]
        if hasattr(self, 'size') and self.sizes == []:
            self.sizes = self.size if isinstance(self.size, list) else [self.size]

    def download(self, workspace: str) -> None:
        """Download files using wget."""
        from .download_utils import download_and_extract_file

        for url in self.urls:
            download_and_extract_file(url, workspace)

# Similar for APIDataset, LocalDataset, HuggingFaceDataset...
# All just contain data, no print_instructions()
```

### executor.py - REFACTORED

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Executor:
    """Pure data model for execution environment requirements.

    Contains only requirements specification.
    No prompt generation logic.
    """
    code_language: str
    dependencies: List[str]
    needs_gpu: bool = False

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'code_language': self.code_language,
            'dependencies': self.dependencies,
            'needs_gpu': self.needs_gpu
        }


@dataclass
class BrowsingInstructions:
    """Browsing requirements for a task."""
    browsing_urls: List[str]
    browsing_text: str
```

---

## 2. Simplified Data Loading (dataset/)

### dataloader.py - REFACTORED

```python
from typing import List, Dict, Optional
from pathlib import Path
from .wrappers.paper import Paper
from .wrappers.task import Task
from .text_loader import load_paper_text

class Dataloader:
    """Loads papers and tasks from directory.

    Simplified: Just loads data, no prompt generation or evaluation setup.
    """

    def __init__(
        self,
        papers_dir: str = "./papers",
        tasks_dir: str = "./tasks",
        paper_ids: Optional[List[str]] = None,
        task_types: Optional[List[str]] = None
    ):
        """Initialize dataloader.

        Args:
            papers_dir: Directory containing paper JSON files
            tasks_dir: Directory containing task subdirectories
            paper_ids: Optional list of specific papers to load (None = all)
            task_types: Optional list of task types to load (None = all)
        """
        self.papers_dir = Path(papers_dir)
        self.tasks_dir = Path(tasks_dir)
        self.paper_ids = paper_ids
        self.task_types = task_types

        # Loaded data
        self.papers: Dict[str, Paper] = {}
        self.tasks: Dict[str, Dict[str, Task]] = {}

        # Load on initialization
        self._load()

    def _load(self) -> None:
        """Load papers and tasks from directories."""
        paper_files = self._discover_papers()

        for paper_file in paper_files:
            paper_id = paper_file.stem

            # Skip if not in requested papers
            if self.paper_ids and paper_id not in self.paper_ids:
                continue

            try:
                paper = self._load_paper(paper_file)
                tasks = self._load_tasks(paper_id)

                self.papers[paper_id] = paper
                self.tasks[paper_id] = tasks

            except Exception as e:
                print(f"Warning: Failed to load paper {paper_id}: {e}")

    def _discover_papers(self) -> List[Path]:
        """Discover all paper JSON files."""
        return sorted(self.papers_dir.glob("*.json"))

    def _load_paper(self, paper_file: Path) -> Paper:
        """Load a single paper from JSON file."""
        return Paper.from_json(str(paper_file))

    def _load_tasks(self, paper_id: str) -> Dict[str, Task]:
        """Load all tasks for a paper."""
        task_dir = self.tasks_dir / paper_id

        if not task_dir.exists():
            return {}

        tasks = {}
        for task_file in task_dir.glob("*.json"):
            task = Task.from_json(str(task_file))

            # Filter by task type if specified
            if self.task_types and task.kind not in self.task_types:
                continue

            # Validate tolerance structure
            if not task.validate_tolerance():
                print(f"Warning: Invalid tolerance structure for {task.task_id}")

            tasks[task.task_id] = task

        return tasks

    def load_paper_text(
        self,
        paper_id: str,
        masked: bool = True
    ) -> str:
        """Load text content for a paper.

        Separated from Paper loading to decouple text preprocessing.

        Args:
            paper_id: Paper identifier
            masked: Load masked text (True) or full text (False)

        Returns:
            Paper text content
        """
        from .text_loader import load_paper_text
        return load_paper_text(self.papers_dir, paper_id, masked=masked)

    def get_paper_with_text(
        self,
        paper_id: str,
        masked: bool = True
    ) -> Paper:
        """Get paper with text loaded.

        Args:
            paper_id: Paper identifier
            masked: Load masked text (True) or full text (False)

        Returns:
            Paper object with full_text populated
        """
        paper = self.papers[paper_id]
        paper.full_text = self.load_paper_text(paper_id, masked=masked)
        return paper

    def filter_papers(self, **criteria) -> List[str]:
        """Filter papers by metadata criteria.

        Examples:
            loader.filter_papers(source="expert")
            loader.filter_papers(code_available=True)

        Returns:
            List of paper IDs matching criteria
        """
        matching = []
        for paper_id, paper in self.papers.items():
            if all(getattr(paper, key, None) == value for key, value in criteria.items()):
                matching.append(paper_id)
        return matching
```

### text_loader.py - NEW

```python
"""Utilities for loading paper text content."""

from pathlib import Path

def load_paper_text(
    papers_dir: Path,
    paper_id: str,
    masked: bool = True
) -> str:
    """Load text content for a paper.

    Args:
        papers_dir: Directory containing paper files
        paper_id: Paper identifier
        masked: Load masked text (True) or full text (False)

    Returns:
        Paper text content

    Raises:
        FileNotFoundError: If text file doesn't exist
    """
    text_subdir = "masked_texts" if masked else "full_texts"
    text_path = Path(papers_dir) / text_subdir / f"{paper_id}.txt"

    if not text_path.exists():
        raise FileNotFoundError(
            f"Text file not found: {text_path}\n"
            f"(Looking for {'masked' if masked else 'full'} text)"
        )

    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read()


def has_masked_text(papers_dir: Path, paper_id: str) -> bool:
    """Check if masked text exists for a paper."""
    text_path = Path(papers_dir) / "masked_texts" / f"{paper_id}.txt"
    return text_path.exists()


def has_full_text(papers_dir: Path, paper_id: str) -> bool:
    """Check if full text exists for a paper."""
    text_path = Path(papers_dir) / "full_texts" / f"{paper_id}.txt"
    return text_path.exists()
```

### download_utils.py - NEW

```python
"""Utilities for downloading datasets."""

import wget
import zipfile
import tarfile
from pathlib import Path

def download_and_extract_file(url: str, dest_dir: str) -> None:
    """Download file from URL and extract if archive.

    Args:
        url: URL to download from
        dest_dir: Destination directory
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Detect filename
    filename = wget.detect_filename(url)
    filepath = dest_path / filename

    # Skip if already exists
    if filepath.exists():
        print(f"File {filename} already exists, skipping")
        return

    # Download
    print(f"Downloading {filename}...")
    wget.download(url, str(dest_dir))

    # Extract if archive
    if filename.endswith('.zip'):
        _extract_zip(filepath, dest_path)
    elif filename.endswith(('.tar.gz', '.tar')):
        _extract_tar(filepath, dest_path)


def _extract_zip(filepath: Path, dest_dir: Path) -> None:
    """Extract zip archive."""
    print(f"Extracting {filepath.name}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)


def _extract_tar(filepath: Path, dest_dir: Path) -> None:
    """Extract tar archive."""
    print(f"Extracting {filepath.name}...")
    mode = 'r:gz' if filepath.name.endswith('.gz') else 'r:'
    with tarfile.open(filepath, mode) as tar:
        tar.extractall(dest_dir)
```

---

## 3. Evaluation Setup (evaluation/)

### evaluation_config.py - NEW

```python
"""Configuration for evaluation, separate from data loading."""

from dataclasses import dataclass, field
from typing import Dict, Any
from dataset.wrappers.paper import Paper
from dataset.wrappers.task import Task

@dataclass
class EvaluationConfig:
    """Configuration for evaluating a paper.

    Contains evaluation-specific metadata (expected outputs, tolerances).
    Separate from paper data model.
    """
    paper_id: str
    task_ids: list[str]

    # Scoring metadata
    expected_outputs: Dict[str, Any] = field(default_factory=dict)
    tolerances: Dict[str, Any] = field(default_factory=dict)
    blank_outputs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tasks(
        cls,
        paper_id: str,
        tasks: Dict[str, Task]
    ) -> 'EvaluationConfig':
        """Create evaluation config from loaded tasks.

        Args:
            paper_id: Paper identifier
            tasks: Dictionary of Task objects

        Returns:
            EvaluationConfig with scoring metadata
        """
        expected_outputs = {}
        tolerances = {}

        for task_id, task in tasks.items():
            expected_outputs[task_id] = task.expected_output
            tolerances[task_id] = task.tolerance

        # Create blank output template
        blank_outputs = _create_blank_outputs(expected_outputs)

        return cls(
            paper_id=paper_id,
            task_ids=list(tasks.keys()),
            expected_outputs=expected_outputs,
            tolerances=tolerances,
            blank_outputs=blank_outputs
        )


def _create_blank_outputs(expected_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Create blank output template (all zeros/None).

    Helper function extracted for clarity.
    """
    from .output_utils import zero_out_structure
    return {
        task_id: zero_out_structure(output)
        for task_id, output in expected_outputs.items()
    }
```

### output_utils.py - NEW

```python
"""Utilities for working with output structures."""

from typing import Any

def zero_out_structure(value: Any, fill: Any = 0) -> Any:
    """Recursively zero out a nested structure.

    Creates a template with the same shape but filled with zeros/None.

    Args:
        value: Structure to zero out
        fill: Fill value (default: 0)

    Returns:
        Zeroed structure with same shape
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return fill

    if isinstance(value, str):
        return str(fill) if fill is not None else ""

    if isinstance(value, list):
        return [zero_out_structure(item, fill) for item in value]

    if isinstance(value, tuple):
        return tuple(zero_out_structure(item, fill) for item in value)

    if isinstance(value, dict):
        return {
            key: zero_out_structure(val, fill)
            for key, val in value.items()
        }

    # Fallback for other types
    return fill
```

### prompt_builder.py - NEW (moved from dataset)

```python
"""Build prompts from data models.

All prompt generation moved here from data models.
"""

from typing import List, Optional
from dataset.wrappers.paper import Paper
from dataset.wrappers.task import Task
from dataset.wrappers.dataset import Dataset
from dataset.wrappers.executor import Executor, BrowsingInstructions

class PromptBuilder:
    """Builds prompts from data models.

    Separates prompt generation from data loading.
    """

    def build_paper_info(
        self,
        paper: Paper,
        workspace: Optional[str] = None
    ) -> str:
        """Build paper information section of prompt."""
        sections = []

        sections.append(self._build_title_section(paper))
        sections.append(self._build_text_section(paper))
        sections.append(self._build_executor_section(paper.execution_requirements))

        for dataset in paper.datasets:
            sections.append(self._build_dataset_section(dataset, workspace))

        if paper.browsing_instructions:
            sections.append(self._build_browsing_section(paper.browsing_instructions))

        if paper.other_instructions:
            sections.append(self._build_additional_instructions(paper))

        if paper.blacklist_packages:
            sections.append(self._build_blacklist_section(paper.blacklist_packages))

        return "\n\n".join(sections)

    def _build_title_section(self, paper: Paper) -> str:
        """Build title section."""
        return (
            f"PAPER INFORMATION:\n"
            f"------------------------------\n"
            f"TITLE: {paper.title}\n"
            f"PUBLICATION DATE: {paper.publication_date}"
        )

    def _build_text_section(self, paper: Paper) -> str:
        """Build full text section."""
        return f"FULL PAPER TEXT:\n{paper.full_text}"

    def _build_executor_section(self, executor: Optional[Executor]) -> str:
        """Build execution requirements section."""
        if not executor:
            return ""

        sections = [
            "CODE EXECUTION INSTRUCTIONS:",
            "------------------------------",
            f"Programming language: {executor.code_language}",
            f"Available packages: {', '.join(executor.dependencies)}",
        ]

        if executor.needs_gpu:
            sections.append("GPU access: Available")

        sections.append("You can install additional packages using pip3 if needed.")

        return "\n".join(sections)

    def _build_dataset_section(
        self,
        dataset: Dataset,
        workspace: Optional[str]
    ) -> str:
        """Build dataset instructions section."""
        sections = [
            f"DATASET: {dataset.dataset_name.upper()}",
            "------------------------------",
            f"Type: {dataset.kind}",
        ]

        if dataset.access_instructions:
            sections.append(f"Access: {dataset.access_instructions}")

        if dataset.usage_instructions:
            sections.append(f"Usage: {dataset.usage_instructions}")

        if dataset.read_instructions:
            sections.append(f"Reading data: {dataset.read_instructions}")

        # Add type-specific info
        if hasattr(dataset, 'urls'):
            sections.append(f"URLs: {', '.join(dataset.urls)}")

        if workspace:
            sections.append(
                f"\nNote: Data has been pre-downloaded to {workspace}. "
                f"DO NOT download again."
            )

        return "\n".join(sections)

    def _build_browsing_section(
        self,
        browsing: BrowsingInstructions
    ) -> str:
        """Build browsing instructions section."""
        return (
            f"BROWSING INSTRUCTIONS:\n"
            f"------------------------------\n"
            f"URLs: {', '.join(browsing.browsing_urls)}\n"
            f"{browsing.browsing_text}"
        )

    def _build_additional_instructions(self, paper: Paper) -> str:
        """Build additional instructions section."""
        return (
            f"ADDITIONAL INSTRUCTIONS:\n"
            f"{paper.other_instructions}"
        )

    def _build_blacklist_section(self, packages: List[str]) -> str:
        """Build forbidden packages section."""
        return (
            f"FORBIDDEN PACKAGES:\n"
            f"You must NOT use or install: {', '.join(packages)}\n"
            f"If you need functionality from these packages, implement it yourself."
        )

    def build_task_prompt(self, task: Task) -> str:
        """Build prompt for a single task."""
        sections = [
            f"TASK_ID: {task.task_id}",
            f"TYPE: {task.kind}",
            f"DESCRIPTION: {task.description}",
            "",
            "INSTRUCTIONS:",
        ]

        # Instructions are now always a list
        for i, instruction in enumerate(task.instructions, 1):
            if len(task.instructions) > 1:
                sections.append(f"{i}. {instruction}")
            else:
                sections.append(instruction)

        # Add output format hint
        sections.append("")
        sections.append(self._get_output_format_hint(task.expected_output))

        return "\n".join(sections)

    def _get_output_format_hint(self, expected_output: Any) -> str:
        """Get hint about expected output format."""
        if isinstance(expected_output, list):
            return f"EXPECTED OUTPUT: A list with {len(expected_output)} elements"
        elif isinstance(expected_output, dict):
            keys = ', '.join(expected_output.keys())
            return f"EXPECTED OUTPUT: A dictionary with keys: {keys}"
        elif isinstance(expected_output, float):
            return "EXPECTED OUTPUT: A floating point number"
        elif isinstance(expected_output, int):
            return "EXPECTED OUTPUT: An integer"
        else:
            return "EXPECTED OUTPUT: Match the format specified in instructions"
```

---

## 4. Usage Examples

### Loading Data (Clean and Simple)

```python
from dataset.dataloader import Dataloader

# Simple loading
loader = Dataloader(
    papers_dir="./papers",
    tasks_dir="./tasks",
    paper_ids=["gw_nsbh", "bayes_cal"],  # Optional filter
    task_types=["numeric"]  # Optional filter
)

# Access loaded data
paper = loader.papers["gw_nsbh"]
tasks = loader.tasks["gw_nsbh"]

# Load text separately (decoupled from paper loading)
paper_with_text = loader.get_paper_with_text("gw_nsbh", masked=True)

# Filter papers by criteria
expert_papers = loader.filter_papers(source="expert")
papers_with_code = loader.filter_papers(code_available=True)
```

### Setting Up Evaluation (Clean Separation)

```python
from dataset.dataloader import Dataloader
from evaluation.evaluation_config import EvaluationConfig
from evaluation.prompt_builder import PromptBuilder

# Load data
loader = Dataloader(papers_dir="./papers", tasks_dir="./tasks")

# Get paper with text
paper = loader.get_paper_with_text("gw_nsbh", masked=True)
tasks = loader.tasks["gw_nsbh"]

# Create evaluation config (scoring metadata)
eval_config = EvaluationConfig.from_tasks("gw_nsbh", tasks)

# Build prompts (separate from data)
builder = PromptBuilder()
paper_prompt = builder.build_paper_info(paper, workspace="/path/to/data")

task_prompts = [
    builder.build_task_prompt(task)
    for task in tasks.values()
]

# Use eval_config for scoring
expected_outputs = eval_config.expected_outputs
tolerances = eval_config.tolerances
```

### Downloading Datasets

```python
from dataset.dataloader import Dataloader

loader = Dataloader()
paper = loader.papers["gw_nsbh"]

# Download all datasets for a paper
workspace = "/path/to/workspace/gw_nsbh"
for dataset in paper.datasets:
    dataset.download(workspace)
```

---

## Key Improvements

### 1. **Single Responsibility**
- Data models: Just hold data
- Dataloader: Just load data
- PromptBuilder: Just build prompts
- EvaluationConfig: Just hold scoring metadata

### 2. **Cleaner Interfaces**
```python
# Old (confusing)
loader = Dataloader(mask=True, filter_on={"source": "showyourwork"})

# New (clear)
loader = Dataloader(paper_ids=["gw_nsbh"])
paper = loader.get_paper_with_text("gw_nsbh", masked=True)
expert_papers = loader.filter_papers(source="expert")
```

### 3. **Helper Functions**
Instead of monster `__init__` methods:
```python
# Old: Everything in __init__
def __init__(self, **kwargs):
    # 50 lines of parsing logic...

# New: Small helpers
def __init__(self, **kwargs):
    self.datasets = _parse_datasets(kwargs.get('dataset', []))
    self.executor = _parse_executor(kwargs.get('execution_requirements'))
    # ...

def _parse_datasets(data):
    """Parse dataset configuration."""
    # Focused, testable logic
```

### 4. **Decoupled Text Loading**
```python
# Old: Text loading coupled with Paper
paper = Paper.from_json("paper.json", mask=True)  # Loads text inside

# New: Separate concerns
paper = Paper.from_json("paper.json")  # Just metadata
text = load_paper_text(papers_dir, paper.paper_id, masked=True)
paper.full_text = text
```

### 5. **Type Normalization**
```python
# Old: Task.instructions can be str OR list
instructions = task.instructions
if isinstance(instructions, str):
    instructions = [instructions]

# New: Always normalized to list
instructions = task.instructions  # Always a list
```

---

## Migration Path

1. **Phase 1**: Create new modules alongside old ones
2. **Phase 2**: Update evaluation/ to use new structure
3. **Phase 3**: Remove old code
4. **Phase 4**: Add tests for new structure

Would you like me to implement any of these refactored files?
