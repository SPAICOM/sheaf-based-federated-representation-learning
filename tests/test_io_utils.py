import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils.io import remove_non_empty_dir


class IoUtilsTests(unittest.TestCase):
    def test_remove_non_empty_dir_ignores_missing_path(self):
        with TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / 'missing-dir'
            remove_non_empty_dir(missing)
            self.assertFalse(missing.exists())

    def test_remove_non_empty_dir_removes_existing_directory(self):
        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / 'to-remove'
            target.mkdir()
            (target / 'artifact.txt').write_text('payload', encoding='utf-8')

            remove_non_empty_dir(target)

            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
