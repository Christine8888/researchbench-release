"""Setup workspace with data and dependencies for papers."""

import argparse
import logging
from pathlib import Path
from cluster_config import DEFAULT_CONFIG
from dataset.dataloader import Dataloader
from evaluation.cluster.workspace import prepare_workspace_for_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Setup paper workspaces")
    parser.add_argument("--papers", nargs="+", help="Paper IDs to setup")
    parser.add_argument("--data_only", action="store_true", help="Only download data")
    parser.add_argument("--deps_only", action="store_true", help="Only install dependencies")
    parser.add_argument("--task_types", nargs="+", help="Task types filter")
    parser.add_argument("--filter_source", help="Restrict only to specified source (e.g., showyourwork)")
    args = parser.parse_args()

    cluster_config = DEFAULT_CONFIG

    filters = {}
    if args.filter_source:
        filters["source"] = args.filter_source

    loader = Dataloader(
        paper_ids=args.papers,
        task_types=args.task_types,
        filters=filters if filters else None,
        load_text=False
    )

    paper_ids = list(loader.papers.keys())

    logger.info(f"Setting up {len(paper_ids)} papers")

    for paper_id in paper_ids:
        logger.info(f"Setting up {paper_id}")
        paper = loader.papers[paper_id]

        download_data = not args.deps_only
        install_deps = not args.data_only

        try:
            prepare_workspace_for_evaluation(
                paper=paper,
                workspace_base=cluster_config.workspace_base,
                singularity_image=cluster_config.singularity_image,
                download_data=download_data,
                install_deps=install_deps
            )
            logger.info(f"Successfully setup {paper_id}")
        except Exception as e:
            logger.error(f"Failed to setup {paper_id}: {e}")

    logger.info("Workspace setup complete")


if __name__ == "__main__":
    main()