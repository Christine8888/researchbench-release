from dataclasses import dataclass
from typing import List, Any
import warnings


@dataclass
class Executor:
    """Execution environment requirements."""
    code_language: str
    dependencies: List[str]
    needs_gpu: bool = False

    def __init__(self, **kwargs: Any) -> None:
        """Initialize executor from a dictionary, accepting any fields."""
        # Ignore browsing-related fields (deprecated)
        ignored_fields = {'needs_browsing', 'browsing_instructions'}

        # Warn about browsing fields
        found_browsing = ignored_fields & set(kwargs.keys())
        if found_browsing:
            warnings.warn(
                f"Executor: Ignoring deprecated browsing fields: {found_browsing}"
            )

        for key, value in kwargs.items():
            if key not in ignored_fields:
                setattr(self, key, value)

        if "code_language" not in kwargs:
            raise ValueError("code_language is a required field")
        if "dependencies" not in kwargs:
            raise ValueError("dependencies is a required field")
