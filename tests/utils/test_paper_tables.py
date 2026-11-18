"""Ranking rule of the paper's tables (src/utils/paper_tables.py)."""

import math

from src.utils.paper_tables import mean_sd, rank_groups, sd_text, tex_cell


def test_groups_of_overlapping_intervals():
    """Groups follow the overlap of mean ± s.d. with the anchor."""
    # anchor 82.1 ± .42: nothing overlaps; then 79.5 ± .54; then the 73.x cluster
    stats = [
        (73.0, 0.11),
        (73.1, 0.11),
        (56.9, 2.2),
        (79.5, 0.54),
        (82.1, 0.42),
    ]
    assert rank_groups(stats, higher=True) == [2, 2, None, 1, 0]


def test_lower_is_better_and_nan():
    """Lower-is-better columns rank the smallest first; nan rows stay unranked."""
    stats = [
        (0.92, 0.014),
        (1.07, 0.018),
        (math.nan, math.nan),
        (0.99, 0.019),
        (3.7, 0.04),
    ]
    assert rank_groups(stats, higher=False) == [0, 2, None, 1, None]


def test_overlap_joins_the_anchor_group():
    """Every row overlapping the anchor joins its group."""
    assert rank_groups(
        [(99.2, 0.02), (99.2, 0.03), (99.1, 0.09)], higher=True
    ) == [0, 0, 0]


def test_formatting():
    """Cell formatting helpers."""
    assert sd_text(0.214, 2) == '.21'
    assert sd_text(1.024, 2) == '1.02'
    assert tex_cell('73.0_{\\pm .11}', 2) == r'\rankC{73.0_{\pm .11}}'
    assert tex_cell('8.6', None) == '$8.6$'
    m, s = mean_sd([1.0, 2.0, 3.0])
    assert (m, s) == (2.0, 1.0)
    assert math.isnan(mean_sd([4.0])[1])
