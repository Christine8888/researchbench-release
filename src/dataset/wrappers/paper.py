from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
from .dataset import Dataset
from .executor import Executor
from .task import Task

@dataclass
class Paper:
    """Paper with metadata, text, and tasks."""
    paper_id: str
    title: str
    abstract: str
    publication_date: datetime

    paper_link: str = ""
    code_available: bool = False
    code_link: Optional[str] = None
    source: str = "expert"

    datasets: List['Dataset'] = field(default_factory=list)
    execution_requirements: Optional['Executor'] = None

    other_instructions: Optional[str] = None
    blacklist_packages: List[str] = field(default_factory=list)

    full_text: str = ""
    tasks: Dict[str, 'Task'] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_path: str) -> 'Paper':
        """Load paper metadata from JSON file."""
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
            datasets=_parse_datasets(data.get('dataset', []), data['paper_id']),
            execution_requirements=_parse_executor(data.get('execution_requirements')),
            other_instructions=data.get('other_instructions'),
            blacklist_packages=data.get('blacklist_packages', [])
        )

    def get_output(self) -> Dict[str, Any]:
        """Get expected output for all tasks as a dict."""
        return {task_id: task.expected_output for task_id, task in self.tasks.items()}

    def get_output_tolerance(self) -> Dict[str, Any]:
        """Get tolerance for all tasks as a dict."""
        return {task_id: task.tolerance for task_id, task in self.tasks.items()}

    def get_blank_output(self, fill: Any = 0) -> Dict[str, Any]:
        """Get blank output template with all values zeroed out."""
        from evaluation.core.utils import recursive_zero_out
        output = self.get_output()
        return recursive_zero_out(output, fill=fill)

    def to_dict(self, include_text: bool = True, include_tasks: bool = True) -> dict:
        """Export to dictionary."""
        data = {
            'paper_id': self.paper_id,
            'title': self.title,
            'abstract': self.abstract,
            'publication_date': self.publication_date.strftime('%Y-%m-%d'),
            'paper_link': self.paper_link,
            'code_available': self.code_available,
            'code_link': self.code_link,
            'source': self.source,
            'other_instructions': self.other_instructions,
            'blacklist_packages': self.blacklist_packages,
        }

        if include_text:
            data['full_text'] = self.full_text

        if include_tasks:
            data['tasks'] = {
                task_id: {
                    'task_id': task.task_id,
                    'paper_id': task.paper_id,
                    'kind': task.kind,
                    'difficulty': task.difficulty,
                    'description': task.description,
                    'instructions': task.instructions,
                    'expected_output': task.expected_output,
                    'tolerance': task.tolerance,
                    'parents': task.parents
                }
                for task_id, task in self.tasks.items()
            }

        return data


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _parse_datasets(dataset_data, paper_id: str) -> List['Dataset']:
    if isinstance(dataset_data, dict):
        dataset_data['paper_id'] = paper_id
        return [Dataset.create(**dataset_data)]
    elif isinstance(dataset_data, list):
        datasets = []
        for item in dataset_data:
            item['paper_id'] = paper_id
            datasets.append(Dataset.create(**item))
        return datasets
    return []


def _parse_executor(exec_data) -> Optional['Executor']:
    if exec_data is None:
        return None

    return Executor(**exec_data)