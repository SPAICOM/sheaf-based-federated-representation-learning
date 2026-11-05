"""Pytest configuration for tests."""

import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Tests never create wandb runs.
os.environ['WANDB_MODE'] = 'disabled'
