"""Input/Output utilities for experiment management.

This module provides utilities for file system operations, particularly
handling of experiment output directories and checkpoint management.

Notes
-----
The `remove_non_empty_dir` function is available in `src.utils`.
"""

import shutil
from pathlib import Path


def remove_non_empty_dir(path: str | Path) -> None:
    """Remove a non-empty directory and all its contents.

    Parameters
    ----------
    path : str | Path
        Path to the directory to remove.
    """
    shutil.rmtree(path)
