"""Load paper text content."""

import os


def load_paper_text(manuscripts_dir: str, paper_id: str, masked: bool = True) -> str:
    """Load text content for a paper."""
    text_subdir = "masked" if masked else "unmasked"
    text_path = os.path.join(manuscripts_dir, text_subdir, f"{paper_id}.txt")

    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")

    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read()


def has_masked_text(manuscripts_dir: str, paper_id: str) -> bool:
    """Check if masked text exists."""
    text_path = os.path.join(manuscripts_dir, "masked", f"{paper_id}.txt")
    return os.path.exists(text_path)


def has_unmasked_text(manuscripts_dir: str, paper_id: str) -> bool:
    """Check if unmasked text exists."""
    text_path = os.path.join(manuscripts_dir, "unmasked", f"{paper_id}.txt")
    return os.path.exists(text_path)
