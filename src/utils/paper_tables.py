r"""Helpers shared by the scripts that write the paper's tables (scripts/table_*.py).

Ranking rule of the paper's tables: per column, the best remaining mean anchors a group made of
every remaining row whose mean ± s.d. interval overlaps the anchor's; the first group is set in
bold (``\rankA``), the second underlined (``\rankB``) and the third in italics (``\rankC``), the
macros of paper/macros.tex.
"""

from __future__ import annotations

import math

import numpy as np

MACRO = {0: r'\rankA', 1: r'\rankB', 2: r'\rankC'}
MD_MARK = {0: '**{}**', 1: '<u>{}</u>', 2: '*{}*'}


def mean_sd(xs) -> tuple[float, float]:
    """Mean and sample s.d. (nan for a single value) of the non-nan values."""
    a = np.asarray([x for x in xs if x == x], dtype=float)
    if a.size == 0:
        return math.nan, math.nan
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else math.nan


def rank_groups(
    stats: list[tuple[float, float]], higher: bool, levels: int = 3
) -> list[int | None]:
    """Group index (0 = best) of every (mean, sd) row, None past ``levels`` groups or for nan means."""
    sign = 1 if higher else -1
    left = [i for i, (m, _) in enumerate(stats) if m == m]
    rank: list[int | None] = [None] * len(stats)
    for level in range(levels):
        if not left:
            break
        a = max(left, key=lambda i: sign * stats[i][0])
        sd_a = stats[a][1] if stats[a][1] == stats[a][1] else 0.0
        lo, hi = stats[a][0] - sd_a, stats[a][0] + sd_a
        group = []
        for i in left:
            sd = stats[i][1] if stats[i][1] == stats[i][1] else 0.0
            if stats[i][0] + sd >= lo and stats[i][0] - sd <= hi:
                group.append(i)
        for i in group:
            rank[i] = level
        left = [i for i in left if i not in group]
    return rank


def sd_text(s: float, decimals: int) -> str:
    """s.d. with ``decimals`` decimals and no leading zero (.12, 1.04), as in the paper."""
    t = f'{s:.{decimals}f}'
    return t[1:] if t.startswith('0.') else t


def tex_cell(body: str, rank: int | None) -> str:
    """``$body$``, or the rank macro around it."""
    return f'{MACRO[rank]}{{{body}}}' if rank is not None else f'${body}$'


def md_cell(text: str, rank: int | None) -> str:
    """``text`` with the markdown mark of its rank group (bold, underline, italics)."""
    return MD_MARK[rank].format(text) if rank is not None else text
