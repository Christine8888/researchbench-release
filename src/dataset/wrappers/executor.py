from dataclasses import dataclass
from typing import List


@dataclass
class Executor:
    """Execution environment requirements."""
    code_language: str
    dependencies: List[str]
    needs_gpu: bool = False
