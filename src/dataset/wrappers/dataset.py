from dataclasses import dataclass, field
from typing import List, Optional, Dict
import warnings


@dataclass
class Dataset:
    """Dataset access information."""
    paper_id: str
    dataset_name: str
    kind: str

    data_instructions: Dict[str, str] = field(default_factory=dict)

    def __init__(self, **kwargs):
        """Initialize dataset, accepting any extra fields."""
        self.paper_id = kwargs.get('paper_id')
        self.dataset_name = kwargs.get('dataset_name')
        self.kind = kwargs.get('kind')
        self.data_instructions = kwargs.get('data_instructions', {})

        # Warn about unexpected fields
        expected_fields = {'paper_id', 'dataset_name', 'kind', 'data_instructions'}
        extra_fields = set(kwargs.keys()) - expected_fields
        if extra_fields:
            warnings.warn(
                f"Dataset '{self.dataset_name}' ({self.paper_id}): Ignoring unexpected fields: {extra_fields}"
            )

    @classmethod
    def create(cls, **kwargs):
        """Factory for dataset subclasses."""
        kind = kwargs.get("kind")

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


@dataclass
class NoneDataset(Dataset):
    """No dataset required."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@dataclass
class APIDataset(Dataset):
    """Dataset via API."""
    api_key: Optional[str] = None
    api_url: Optional[str] = None

    def __init__(self, **kwargs):
        # Extract API-specific fields first
        self.api_key = kwargs.pop('api_key', None)
        self.api_url = kwargs.pop('api_url', None)
        # Initialize parent with remaining kwargs
        super().__init__(**kwargs)


@dataclass
class LocalDataset(Dataset):
    """Dataset bundled with benchmark."""
    data_path: List[str] = field(default_factory=list)
    size: List[float] = field(default_factory=list)

    def __init__(self, **kwargs):
        # Extract local-specific fields first
        self.data_path = kwargs.pop('data_path', [])
        self.size = kwargs.pop('size', [])
        # Initialize parent with remaining kwargs
        super().__init__(**kwargs)


@dataclass
class WgetDataset(Dataset):
    """Dataset via wget."""
    url: List[str] = field(default_factory=list)
    size: List[float] = field(default_factory=list)

    def __init__(self, **kwargs):
        # Extract wget-specific fields first
        self.url = kwargs.pop('url', [])
        self.size = kwargs.pop('size', [])
        # Initialize parent with remaining kwargs
        super().__init__(**kwargs)


@dataclass
class HuggingFaceDataset(Dataset):
    """Dataset via HuggingFace Hub."""
    hf_name: List[str] = field(default_factory=list)
    hf_split: List[str] = field(default_factory=list)
    hf_type: List[str] = field(default_factory=list)

    def __init__(self, **kwargs):
        # Extract HF-specific fields first
        self.hf_name = kwargs.pop('hf_name', [])
        self.hf_split = kwargs.pop('hf_split', [])
        self.hf_type = kwargs.pop('hf_type', [])

        # Set default split if not provided
        if not self.hf_split and self.hf_name:
            self.hf_split = ["train"] * len(self.hf_name)

        # Check for fields that belong to other dataset types
        conflicting_fields = {'url', 'size', 'data_path', 'api_key', 'api_url'}
        found_conflicts = conflicting_fields & set(kwargs.keys())
        if found_conflicts:
            paper_id = kwargs.get('paper_id', 'unknown')
            warnings.warn(
                f"HuggingFaceDataset ({paper_id}): Found fields for other dataset types: {found_conflicts}. "
                f"These will be ignored. Check if 'kind' is set correctly."
            )

        # Initialize parent with remaining kwargs (which may include extra fields to ignore)
        super().__init__(**kwargs)
