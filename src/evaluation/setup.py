"""Setup functions for paper evaluation environments."""

import os
import subprocess
import logging
from typing import List, Optional
from evaluation.download_utils import download_dataset

logger = logging.getLogger(__name__)


def setup_paper_data(paper, workspace_dir: str) -> None:
    """Download all datasets for a paper to workspace.

    Args:
        paper: Paper object with datasets
        workspace_dir: Directory to download data to
    """
    logger.info(f"Setting up data for paper {paper.paper_id}")
    os.makedirs(workspace_dir, exist_ok=True)

    for dataset in paper.datasets:
        try:
            download_dataset(dataset, workspace_dir)
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset.dataset_name}: {e}")
            raise


def install_dependencies(
    dependencies: List[str],
    method: str = "pip",
    target_dir: Optional[str] = None
) -> None:
    """Install Python dependencies.

    Args:
        dependencies: List of package specifiers
        method: Installation method ('pip' or 'conda')
        target_dir: Target directory for installation (optional)
    """
    if not dependencies:
        logger.info("No dependencies to install")
        return

    logger.info(f"Installing {len(dependencies)} dependencies")

    if method == "pip":
        cmd = ["python3", "-m", "pip", "install"]
        if target_dir:
            cmd.extend(["--target", target_dir])
        cmd.extend(dependencies)

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Successfully installed all packages")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Batch install failed, trying one by one: {e.stderr}")
            _install_dependencies_individually(dependencies, target_dir)
    else:
        logger.error(f"Unsupported installation method: {method}")
        raise ValueError(f"Unsupported installation method: {method}")


def _install_dependencies_individually(
    dependencies: List[str],
    target_dir: Optional[str] = None
) -> None:
    """Install dependencies one at a time."""
    for pkg in dependencies:
        cmd = ["python3", "-m", "pip", "install"]
        if target_dir:
            cmd.extend(["--target", target_dir])
        cmd.append(pkg)

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully installed {pkg}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Could not install {pkg}: {e.stderr}")


def setup_paper_environment(
    paper,
    workspace_dir: str,
    download_data: bool = True,
    install_deps: bool = False,
    deps_target: Optional[str] = None
) -> None:
    """Complete setup for a paper evaluation environment.

    Args:
        paper: Paper object
        workspace_dir: Workspace directory for this paper
        download_data: Whether to download datasets
        install_deps: Whether to install dependencies
        deps_target: Target directory for dependency installation
    """
    logger.info(f"Setting up environment for {paper.paper_id}")

    if download_data:
        setup_paper_data(paper, workspace_dir)

    if install_deps and paper.execution_requirements:
        dependencies = paper.execution_requirements.dependencies
        install_dependencies(dependencies, target_dir=deps_target)

    logger.info(f"Environment setup complete for {paper.paper_id}")
