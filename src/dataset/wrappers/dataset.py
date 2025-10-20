from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Dataset:
    """Dataset access information."""
    paper_id: str
    dataset_name: str
    kind: str

    access_instructions: str = ""
    usage_instructions: str = ""
    read_instructions: str = ""

    @classmethod
    def create(cls, **kwargs):
        """Factory for dataset subclasses."""
        kind = kwargs.get("kind")

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


@dataclass
class NoneDataset(Dataset):
    """No dataset required."""
    pass


@dataclass
class APIDataset(Dataset):
    """Dataset via API."""
    api_key: Optional[str] = None
    api_url: Optional[str] = None


@dataclass
class LocalDataset(Dataset):
    """Dataset bundled with benchmark."""
    data_path: List[str] = field(default_factory=list)
    size: List[float] = field(default_factory=list)


@dataclass
class WgetDataset(Dataset):
    """Dataset via wget."""
    url: List[str] = field(default_factory=list)
    size: List[float] = field(default_factory=list)


@dataclass
class HuggingFaceDataset(Dataset):
    """Dataset via HuggingFace Hub."""
    hf_name: List[str] = field(default_factory=list)
    hf_split: List[str] = field(default_factory=list)
    hf_type: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.hf_split and self.hf_name:
            self.hf_split = ["train"] * len(self.hf_name)
