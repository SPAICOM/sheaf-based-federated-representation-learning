import shutil
from pathlib import Path


def remove_non_empty_dir(path: str) -> None:
    """
    Removes a non-empty directory given its path as a string.

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
    dir_path = Path(path)

    if not dir_path.exists():
        print(f"The path '{path}' does not exist.")
        return None

    if not dir_path.is_dir():
        raise NotADirectoryError(f"The path '{path}' is not a directory.")

    try:
        shutil.rmtree(dir_path)
        print(f'Successfully removed directory: {dir_path}')
    except Exception as e:
        raise Exception(f'Error while removing directory: {e}')

    return None
