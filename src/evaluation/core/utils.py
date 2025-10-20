"""Utility functions for evaluation."""

from typing import Any, Dict, List


def recursive_zero_out(value: Any, fill: Any = 0) -> Any:
    """Recursively zero out all values in a nested structure.

    Args:
        value: The value or structure to zero out
        fill: The value to replace with (default: 0)

    Returns:
        The zeroed out value or structure
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return fill

    if isinstance(value, str):
        return fill if isinstance(fill, str) else str(fill) if fill is not None else ""

    if isinstance(value, list):
        return [recursive_zero_out(item, fill) for item in value]

    if isinstance(value, tuple):
        return tuple(recursive_zero_out(item, fill) for item in value)

    if isinstance(value, dict):
        return {key: recursive_zero_out(val, fill) for key, val in value.items()}

    return fill
