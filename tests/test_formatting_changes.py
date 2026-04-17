"""Pytest conversion of test_formatting_changes.py."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Data Sourcing"))
from sec_file_sourcer import SECFileSourcer


@pytest.fixture
def sourcer():
    return SECFileSourcer()


@pytest.fixture
def sample_model():
    return {
        'annual_income_statement': pd.DataFrame({
            'Revenue': [1_000_000, 900_000],
            'CostOfGoodsSold': [600_000, 540_000],
            'OperatingIncome': [200_000, 180_000],
            'NetIncome': [150_000, 135_000],
        }, index=['2023-12-31', '2022-12-31']),
        'annual_balance_sheet': pd.DataFrame({
            'TotalAssets': [750_000, 675_000],
            'TotalLiabilities': [280_000, 252_000],
            'TotalStockholdersEquity': [470_000, 423_000],
        }, index=['2023-12-31', '2022-12-31']),
        'annual_cash_flow': pd.DataFrame({
            'OperatingCashFlow': [180_000, 162_000],
        }, index=['2023-12-31', '2022-12-31']),
    }


def test_remove_formatting_columns(sourcer):
    test_df = pd.DataFrame({
        'Line Item': ['Revenue'],
        'Value': [1000],
        'statement_type': ['INCOME'],
        'is_section_heading': [False],
        'is_aggregate': [False],
        'parent': [None],
    })
    cleaned = sourcer._remove_formatting_columns(test_df)
    formatting_cols = {'statement_type', 'is_section_heading', 'is_aggregate', 'parent'}
    assert (set(cleaned.columns) & formatting_cols) == set()


def test_stringify_date_columns(sourcer):
    dates = pd.date_range('2022-01-01', periods=2, freq='YE')
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=['x', 'y'])
    df.columns = dates
    result = sourcer._stringify_date_columns(df)
    for col in result.columns:
        assert isinstance(col, str)


def test_stacked_statement_structure(sourcer, sample_model):
    stacked = sourcer._create_vertically_stacked_statement(
        sample_model['annual_income_statement'],
        sample_model['annual_balance_sheet'],
        sample_model['annual_cash_flow'],
        'Annual'
    )
    assert 'Line Item' in stacked.columns
    assert len(stacked) > 0


def test_terminology_parent_map(sourcer):
    parent_map = sourcer._create_user_friendly_parent_map()
    assert isinstance(parent_map, dict)
    assert len(parent_map) > 0
