"""Workspace management for cluster environments."""

import os
import subprocess
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def setup_paper_workspace(
    paper_id: str,
    workspace_base: str,
    overlay_size_mb: int = 500
) -> str:
    """Set up workspace directory for a paper.

    Args:
        paper_id: Paper ID
        workspace_base: Base workspace directory
        overlay_size_mb: Size of overlay (unused, kept for compatibility)

    Returns:
        Path to paper workspace directory
    """
    paper_workspace = os.path.join(workspace_base, paper_id)
    os.makedirs(paper_workspace, exist_ok=True)

    overlay_dir = os.path.join(paper_workspace, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    logger.info(f"Created workspace for {paper_id} at {paper_workspace}")
    return paper_workspace


def install_packages_in_overlay(
    packages: List[str],
    overlay_dir: str,
    singularity_image: str
) -> None:
    """Install Python packages into Singularity overlay.

    Args:
        packages: List of package specifications
        overlay_dir: Path to overlay directory
        singularity_image: Path to Singularity image
    """
    if not packages:
        logger.info("No packages to install")
        return

    logger.info(f"Installing {len(packages)} packages to overlay")

    install_cmd = [
        "singularity", "exec",
        "--overlay", overlay_dir,
        singularity_image,
        "bash", "-lc",
        f"python3 -m pip install {' '.join(packages)}"
    ]

    try:
        subprocess.run(install_cmd, check=True, capture_output=True, text=True)
        logger.info("Successfully installed all packages")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Batch install failed: {e.stderr}")
        _install_packages_individually(packages, overlay_dir, singularity_image)


def _install_packages_individually(
    packages: List[str],
    overlay_dir: str,
    singularity_image: str
) -> None:
    """Install packages one at a time."""
    for pkg in packages:
        install_cmd = [
            "singularity", "exec",
            "--overlay", overlay_dir,
            singularity_image,
            "bash", "-lc",
            f'python3 -m pip install "{pkg}"'
        ]

        try:
            subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully installed {pkg}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Could not install {pkg}: {e.stderr}")


def prepare_workspace_for_evaluation(
    paper,
    workspace_base: str,
    singularity_image: str,
    download_data: bool = True,
    install_deps: bool = True
) -> str:
    """Prepare complete workspace for paper evaluation.

    Args:
        paper: Paper object
        workspace_base: Base workspace directory
        singularity_image: Path to Singularity image
        download_data: Whether to download datasets
        install_deps: Whether to install dependencies

    Returns:
        Path to paper workspace
    """
    from evaluation.setup import setup_paper_data

    paper_workspace = setup_paper_workspace(paper.paper_id, workspace_base)

    if download_data:
        setup_paper_data(paper, paper_workspace)

    if install_deps and paper.execution_requirements:
        overlay_dir = os.path.join(paper_workspace, "overlay")
        dependencies = paper.execution_requirements.dependencies
        install_packages_in_overlay(dependencies, overlay_dir, singularity_image)

    return paper_workspace
