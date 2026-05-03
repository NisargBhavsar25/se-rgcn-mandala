"""Tests for COW Trade 4.0 loader."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.trade import COW_TRADE_DEFAULT_PATH, load_cow_trade_edges


@pytest.fixture(scope="module")
def trade() -> pd.DataFrame:
    if not COW_TRADE_DEFAULT_PATH.exists():
        pytest.skip(f"{COW_TRADE_DEFAULT_PATH} not found.")
    return load_cow_trade_edges(years=range(1950, 2015))


def test_canonical_dyad_ordering(trade):
    assert (trade.gwcode_i < trade.gwcode_j).all()


def test_year_window(trade):
    assert trade.year.min() >= 1950
    assert trade.year.max() <= 2014


def test_no_germany_255_post_1949(trade):
    bad = trade[
        ((trade.gwcode_i == 255) | (trade.gwcode_j == 255))
        & (trade.year >= 1949)
    ]
    assert len(bad) == 0


def test_no_negative_trade_values(trade):
    """Smoothflow -9 sentinels must have been replaced; total_trade should
    therefore be non-negative."""
    assert (trade.total_trade >= 0).all()


def test_known_dyad_has_positive_trade():
    """USA-Canada (2-20) had substantial trade throughout 1950-2014."""
    df = load_cow_trade_edges(years=range(1980, 1981))
    us_can = df[(df.gwcode_i == 2) & (df.gwcode_j == 20)]
    assert len(us_can) == 1
    assert us_can.iloc[0].total_trade > 1000  # billions, easily
