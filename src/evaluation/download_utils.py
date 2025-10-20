"""Utilities for downloading datasets."""

import os
import shutil
import zipfile
import tarfile
import logging

logger = logging.getLogger(__name__)


def download_dataset(dataset, workspace: str) -> None:
    """Download dataset to workspace based on its type.

    Args:
        dataset: Dataset object (LocalDataset, WgetDataset, HuggingFaceDataset, etc.)
        workspace: Destination directory for downloaded data
    """
    from dataset.wrappers.dataset import (
        LocalDataset,
        WgetDataset,
        HuggingFaceDataset,
        NoneDataset,
        APIDataset
    )

    if isinstance(dataset, NoneDataset):
        logger.info(f"No data needed for {dataset.dataset_name}")
        return

    elif isinstance(dataset, LocalDataset):
        _download_local(dataset, workspace)

    elif isinstance(dataset, WgetDataset):
        _download_wget(dataset, workspace)

    elif isinstance(dataset, HuggingFaceDataset):
        _download_huggingface(dataset, workspace)

    elif isinstance(dataset, APIDataset):
        logger.info(f"API dataset {dataset.dataset_name} is accessed real-time via code")

    else:
        logger.warning(f"Unknown dataset type for {dataset.dataset_name}")


def _download_local(dataset, workspace: str) -> None:
    """Copy local dataset files to workspace."""
    logger.info(f"Copying local data for {dataset.paper_id}")
    os.makedirs(workspace, exist_ok=True)

    for path in dataset.data_path:
        if path.startswith("data"):
            basename = os.path.basename(path)
            dest_path = os.path.join(workspace, basename)

            if os.path.exists(dest_path):
                logger.info(f"{basename} already exists, skipping")
                continue

            if os.path.isdir(path):
                logger.info(f"Copying directory {basename}")
                shutil.copytree(path, dest_path)
            else:
                logger.info(f"Copying file {basename}")
                shutil.copy(path, workspace)
        else:
            logger.warning(f"Path {path} is not in data/ directory, skipping")


def _download_wget(dataset, workspace: str) -> None:
    """Download dataset from URLs."""
    import wget

    logger.info(f"Downloading data for {dataset.paper_id}")
    os.makedirs(workspace, exist_ok=True)

    for url in dataset.url:
        filename = wget.detect_filename(url)
        filepath = os.path.join(workspace, filename)

        if os.path.exists(filepath):
            logger.info(f"{filename} already exists, skipping")
            continue

        logger.info(f"Downloading {filename} from {url}")
        wget.download(url, workspace)

        if filename.endswith('.zip'):
            _extract_zip(filepath, workspace)
        elif filename.endswith('.tar.gz'):
            _extract_tar(filepath, workspace, mode='r:gz')
        elif filename.endswith('.tar'):
            _extract_tar(filepath, workspace, mode='r:')


def _download_huggingface(dataset, workspace: str) -> None:
    """Download dataset from HuggingFace Hub."""
    from huggingface_hub import snapshot_download
    from datasets import load_dataset

    logger.info(f"Downloading HuggingFace data for {dataset.paper_id}")
    os.makedirs(workspace, exist_ok=True)

    for hf_name, hf_split, hf_type in zip(dataset.hf_name, dataset.hf_split, dataset.hf_type):
        save_path = os.path.join(workspace, f"{hf_name}_{hf_split}")

        if hf_type == "snapshot":
            logger.info(f"Downloading snapshot {hf_name}")
            try:
                snapshot_download(repo_id=hf_name, local_dir=save_path, repo_type="dataset")
            except Exception as e:
                logger.error(f"Error downloading {hf_name}: {e}")
                raise
        else:
            logger.info(f"Downloading dataset {hf_name} split {hf_split}")
            try:
                ds = load_dataset(hf_name, split=hf_split)
                ds.save_to_disk(save_path)
            except Exception as e:
                logger.error(f"Error downloading {hf_name} {hf_split}: {e}")
                raise


def _extract_zip(filepath: str, dest_dir: str) -> None:
    """Extract zip archive."""
    logger.info(f"Extracting {os.path.basename(filepath)}")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)


def _extract_tar(filepath: str, dest_dir: str, mode: str = 'r:') -> None:
    """Extract tar archive."""
    logger.info(f"Extracting {os.path.basename(filepath)}")
    with tarfile.open(filepath, mode) as tar:
        tar.extractall(dest_dir)
