"""House plotting style for every permanent script in ``scripts/``.

    from src.utils.plotting import use_style

    use_style()  # before creating any figure

Figure-specific tweaks (e.g. smaller fonts for multi-panel rows) go on top via
``plt.rcParams`` in the script, never by editing the shared style file. The style
typesets text with LaTeX; on a machine without ``latex`` it falls back to
matplotlib's mathtext, so every figure script still runs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt

STYLE = Path(__file__).resolve().parents[2] / 'config/plotting/plt.mplstyle'


def use_style() -> None:
    """Apply the house style (mathtext instead of LaTeX when ``latex`` is missing)."""
    plt.style.use(STYLE)
    if plt.rcParams['text.usetex'] and shutil.which('latex') is None:
        plt.rcParams['text.usetex'] = False
