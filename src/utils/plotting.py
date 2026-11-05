"""House plotting style for every permanent script in ``scripts/``.

    from src.utils.plotting import use_style

    use_style()  # before creating any figure

Figure-specific tweaks (e.g. smaller fonts for multi-panel rows) go on top via
``plt.rcParams`` in the script, never by editing the shared style file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

STYLE = Path(__file__).resolve().parents[2] / 'config/plotting/plt.mplstyle'


def use_style() -> None:
    plt.style.use(STYLE)
