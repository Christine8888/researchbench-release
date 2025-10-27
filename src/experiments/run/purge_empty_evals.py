"""Purge empty .eval files from log directories.

An .eval file is considered empty if it only contains _journal/start.json
and no actual evaluation results (samples, summaries, etc.).
"""

import argparse
import logging
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def is_empty_eval(eval_path: Path) -> bool:
    """Check if .eval file is empty (only contains start.json).

    Args:
        eval_path: Path to .eval file

    Returns:
        True if file is empty/incomplete, False if it has results
    """
    if not eval_path.exists():
        return False

    if eval_path.stat().st_size == 0:
        return True

    try:
        with zipfile.ZipFile(eval_path, 'r') as zf:
            namelist = zf.namelist()

            # Empty if only has start.json
            if len(namelist) == 1 and namelist[0] == '_journal/start.json':
                return True

            # Also check for presence of actual results
            has_samples = any('samples/' in name for name in namelist)
            has_summaries = 'summaries.json' in namelist

            # If it has more files but no results, consider it empty
            if not has_samples and not has_summaries:
                return True

            return False

    except (zipfile.BadZipFile, OSError) as e:
        logger.warning(f"Could not read {eval_path}: {e}")
        return True


def purge_empty_evals(directories: list[str], dry_run: bool = True) -> tuple[int, int]:
    """Purge empty .eval files from directories.

    Args:
        directories: List of directory paths to search
        dry_run: If True, only report what would be deleted

    Returns:
        Tuple of (empty_count, total_count)
    """
    empty_files = []
    total_files = 0

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            continue

        for eval_file in dir_path.rglob('*.eval'):
            total_files += 1
            if is_empty_eval(eval_file):
                empty_files.append(eval_file)

    logger.info(f"Found {len(empty_files)} empty .eval files out of {total_files} total")

    if empty_files:
        logger.info("\nEmpty files:")
        for f in empty_files:
            logger.info(f"  {f}")

    if not dry_run and empty_files:
        logger.info("\nDeleting empty files...")
        for f in empty_files:
            f.unlink()
            logger.info(f"  Deleted: {f}")
    elif dry_run and empty_files:
        logger.info("\n(Dry run - no files deleted. Use --execute to actually delete)")

    return len(empty_files), total_files


def main():
    parser = argparse.ArgumentParser(
        description="Purge empty .eval files from log directories"
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="Directory or directories to search for .eval files"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (default is dry-run)"
    )

    args = parser.parse_args()

    empty_count, total_count = purge_empty_evals(
        args.directories,
        dry_run=not args.execute
    )

    logger.info(f"\nSummary: {empty_count} empty / {total_count} total .eval files")


if __name__ == "__main__":
    main()
