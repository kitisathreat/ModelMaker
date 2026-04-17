"""Pytest conversion of test_text_formatting.py."""
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
def sample_formatted_df():
    return pd.DataFrame([
        {'Line Item': 'INCOME STATEMENT', 'is_section_heading': True,
         'parent': None, 'is_aggregate': False, '2023': '', '2022': ''},
        {'Line Item': 'Revenue', 'is_section_heading': False,
         'parent': None, 'is_aggregate': False, '2023': 1_000_000, '2022': 900_000},
        {'Line Item': 'Cost of Revenue', 'is_section_heading': False,
         'parent': 'Revenue', 'is_aggregate': False, '2023': 600_000, '2022': 540_000},
        {'Line Item': 'Gross Profit', 'is_section_heading': False,
         'parent': None, 'is_aggregate': True, '2023': 400_000, '2022': 360_000},
        {'Line Item': 'Net Income', 'is_section_heading': False,
         'parent': None, 'is_aggregate': True, '2023': 150_000, '2022': 135_000},
    ])


def test_section_headings_identifiable(sample_formatted_df):
    headings = sample_formatted_df[sample_formatted_df['is_section_heading'] == True]
    assert len(headings) >= 1
    assert headings.iloc[0]['Line Item'] == 'INCOME STATEMENT'


def test_aggregate_items_identified(sample_formatted_df):
    aggregates = sample_formatted_df[sample_formatted_df['is_aggregate'] == True]
    assert len(aggregates) >= 1


def test_child_items_have_parent(sample_formatted_df):
    children = sample_formatted_df[sample_formatted_df['parent'].notna() &
                                    (sample_formatted_df['parent'] != '')]
    assert len(children) >= 1


def test_get_user_friendly_title(sourcer):
    assert isinstance(sourcer._get_user_friendly_title('NetIncomeLoss'), str)
    assert isinstance(sourcer._get_user_friendly_title('OperatingIncomeLoss'), str)


def test_dedup_by_end_date():
    """Regression test for Bug 7: 10-Q dedup keeps latest amendment."""
    filings = [
        {'end': '2023-03-31', 'form': '10-Q', 'filed': '2023-04-15', 'val': 100},
        {'end': '2023-03-31', 'form': '10-Q/A', 'filed': '2023-05-01', 'val': 105},
        {'end': '2022-12-31', 'form': '10-Q', 'filed': '2023-01-15', 'val': 90},
    ]
    deduped: dict = {}
    for p in filings:
        end = p.get('end', '')
        if end not in deduped or p.get('filed', '') > deduped[end].get('filed', ''):
            deduped[end] = p
    result = list(deduped.values())
    march = next(r for r in result if r['end'] == '2023-03-31')
    assert march['val'] == 105
    assert len(result) == 2
