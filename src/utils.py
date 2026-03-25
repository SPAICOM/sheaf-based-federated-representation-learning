"""Utility functions for experiment management.

This module provides helper functions for managing experiment directories
and file system operations.

Functions
---------
remove_non_empty_dir : Remove directory and all its contents recursively.

Notes
-----
Use with caution - this operation is irreversible.
"""

import shutil
from pathlib import Path


# ================================================================
#
#                        Methods Definition
#
# ================================================================
def remove_non_empty_dir(path: str) -> None:
    """Remove a non-empty directory and all its contents.

    Parameters
    ----------
    path : str
        Path to the directory to remove.

    Raises
    ------
    NotADirectoryError
        If the path is not a directory.
    Exception
        For any other error during deletion.
    """
    """
    Removes a non-empty directory given its path as a string.

    Parameters:
        path : str
            Path to the directory to remove.

    Raises:
        NotADirectoryError: If the path is not a directory.
        Exception: For any other error during deletion.
    """
    # Convert string path to Path object for filesystem operations
    dir_path = Path(path)

    # Check if path exists
    if not dir_path.exists():
        print(f"The path '{path}' does not exist.")
        return None

    # Verify it's actually a directory
    if not dir_path.is_dir():
        raise NotADirectoryError(f"The path '{path}' is not a directory.")

    # Recursively remove directory and all contents
    try:
        shutil.rmtree(dir_path)
        print(f'Successfully removed directory: {dir_path}')
    except Exception as e:
        raise Exception(f'Error while removing directory: {e}')

    return None


if __name__ == '__main__':
    pass
