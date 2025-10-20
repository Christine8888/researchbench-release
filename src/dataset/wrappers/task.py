from dataclasses import dataclass
from typing import Any, List, Optional
import json


@dataclass
class Task:
    """Task with instructions and expected output."""
    task_id: str
    paper_id: str
    kind: str
    difficulty: int

    description: str
    instructions: List[str]

    expected_output: Any
    tolerance: Any

    parents: Optional[List[str]] = None

    @classmethod
    def from_json(cls, json_path: str) -> 'Task':
        """Load from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)

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
        """Check tolerance structure matches expected_output."""
        return _validate_structure(self.expected_output, self.tolerance)


def _validate_structure(output: Any, tolerance: Any) -> bool:
    """Recursively validate tolerance matches output structure."""
    if isinstance(output, dict):
        if not isinstance(tolerance, dict):
            return False
        return all(
            key in tolerance and _validate_structure(val, tolerance[key])
            for key, val in output.items()
        )

    if isinstance(output, list):
        if not isinstance(tolerance, list):
            return False
        if len(output) != len(tolerance):
            return False
        return all(
            _validate_structure(out_item, tol_item)
            for out_item, tol_item in zip(output, tolerance)
        )

    return True
