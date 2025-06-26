#!/usr/bin/env python3
"""
Test script to demonstrate alternative text formatting for line item titles.
This script shows how the new formatting applies text formatting directly to titles
instead of using Excel cell-level formatting.
"""

import pandas as pd
import sys
import os

# Add the Data Sourcing directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Data Sourcing'))

from sec_file_sourcer import SECFileSourcer

def create_sample_financial_data():
    """Create sample financial data to demonstrate the text formatting."""
    
    # Sample data with proper flags
    sample_data = [
        # Section headings (bold)
        {'Line Item': 'INCOME STATEMENT', 'statement_type': 'INCOME', 'is_section_heading': True, 'parent': None, 'is_aggregate': False, '2023': '', '2022': '', '2021': ''},
        
        # Top-level line items (no parent, no aggregation)
        {'Line Item': 'Revenue', 'statement_type': 'INCOME', 'is_section_heading': False, 'parent': None, 'is_aggregate': False, '2023': 1000000, '2022': 900000, '2021': 800000},
        {'Line Item': 'Cost of Revenue', 'statement_type': 'INCOME', 'is_section_heading': False, 'parent': 'Revenue', 'is_aggregate': False, '2023': 600000, '2022': 540000, '2021': 480000},
        {'Line Item': 'Gross Profit', 'statement_type': 'INCOME', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 400000, '2022': 360000, '2021': 320000},
        
        # Parent aggregator with children
        {'Line Item': 'Operating Expenses', 'statement_type': 'INCOME', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 200000, '2022': 180000, '2021': 160000},
        {'Line Item': 'Research & Development', 'statement_type': 'INCOME', 'is_section_heading': False, 'parent': 'OperatingExpenses', 'is_aggregate': False, '2023': 80000, '2022': 72000, '2021': 64000},
        {'Line Item': 'Selling, General & Administrative', 'statement_type': 'INCOME', 'is_section_heading': False, 'parent': 'OperatingExpenses', 'is_aggregate': False, '2023': 120000, '2022': 108000, '2021': 96000},
        
        {'Line Item': 'Operating Income', 'statement_type': 'INCOME', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 200000, '2022': 180000, '2021': 160000},
        {'Line Item': 'Net Income', 'statement_type': 'INCOME', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 150000, '2022': 135000, '2021': 120000},
        
        # Section heading for next statement
        {'Line Item': 'BALANCE SHEET', 'statement_type': 'BALANCE', 'is_section_heading': True, 'parent': None, 'is_aggregate': False, '2023': '', '2022': '', '2021': ''},
        
        # Balance sheet items
        {'Line Item': 'Current Assets', 'statement_type': 'BALANCE', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 500000, '2022': 450000, '2021': 400000},
        {'Line Item': 'Cash & Cash Equivalents', 'statement_type': 'BALANCE', 'is_section_heading': False, 'parent': 'CurrentAssets', 'is_aggregate': False, '2023': 200000, '2022': 180000, '2021': 160000},
        {'Line Item': 'Accounts Receivable', 'statement_type': 'BALANCE', 'is_section_heading': False, 'parent': 'CurrentAssets', 'is_aggregate': False, '2023': 150000, '2022': 135000, '2021': 120000},
        {'Line Item': 'Inventory', 'statement_type': 'BALANCE', 'is_section_heading': False, 'parent': 'CurrentAssets', 'is_aggregate': False, '2023': 100000, '2022': 90000, '2021': 80000},
        {'Line Item': 'Other Current Assets', 'statement_type': 'BALANCE', 'is_section_heading': False, 'parent': 'CurrentAssets', 'is_aggregate': False, '2023': 50000, '2022': 45000, '2021': 40000},
        
        {'Line Item': 'Total Assets', 'statement_type': 'BALANCE', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 1000000, '2022': 900000, '2021': 800000},
        
        # Section heading for cash flow
        {'Line Item': 'CASH FLOW STATEMENT', 'statement_type': 'CASH', 'is_section_heading': True, 'parent': None, 'is_aggregate': False, '2023': '', '2022': '', '2021': ''},
        
        # Cash flow items
        {'Line Item': 'Net Income', 'statement_type': 'CASH', 'is_section_heading': False, 'parent': None, 'is_aggregate': False, '2023': 150000, '2022': 135000, '2021': 120000},
        {'Line Item': 'Operating Activities', 'statement_type': 'CASH', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 180000, '2022': 162000, '2021': 144000},
        {'Line Item': 'Depreciation & Amortization', 'statement_type': 'CASH', 'is_section_heading': False, 'parent': 'OperatingAdjustments', 'is_aggregate': False, '2023': 30000, '2022': 27000, '2021': 24000},
        {'Line Item': 'Operating Adjustments', 'statement_type': 'CASH', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 30000, '2022': 27000, '2021': 24000},
        {'Line Item': 'Net Cash from Operating Activities', 'statement_type': 'CASH', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 180000, '2022': 162000, '2021': 144000},
    ]
    
    return pd.DataFrame(sample_data)

def test_text_formatting_styles():
    """Test different text formatting styles."""
    
    print("Creating sample financial data...")
    sample_df = create_sample_financial_data()
    
    print("Sample data structure:")
    print(sample_df[['Line Item', 'is_section_heading', 'parent', 'is_aggregate']].head(10))
    print("\n" + "="*80)
    
    # Create SECFileSourcer instance
    sourcer = SECFileSourcer()
    
    # Test different formatting styles
    formatting_styles = ['markers', 'html', 'rich_text', 'unicode']
    
    for style in formatting_styles:
        print(f"\nTesting {style.upper()} formatting style:")
        print("-" * 50)
        
        # Apply formatting to a subset of data for demonstration
        test_df = sample_df.head(15).copy()
        formatted_df = sourcer.apply_text_formatting_to_titles(test_df, style)
        
        # Show the formatted titles
        for idx, row in formatted_df.iterrows():
            line_item = row['Line Item']
            is_section = row['is_section_heading']
            is_aggregate = row['is_aggregate']
            has_parent = row['parent'] is not None and str(row['parent']).strip() != ''
            
            # Add indicators for the formatting applied
            indicators = []
            if is_section:
                indicators.append("SECTION")
            if is_aggregate:
                indicators.append("AGGREGATE")
            if has_parent:
                indicators.append("CHILD")
            
            indicator_str = f" [{', '.join(indicators)}]" if indicators else ""
            print(f"{line_item}{indicator_str}")
    
    return sample_df

def test_excel_export_with_formatted_titles():
    """Test the new Excel export method with formatted titles."""
    
    print("\n" + "="*80)
    print("Testing Excel export with formatted titles...")
    print("="*80)
    
    # Create sample data
    sample_df = create_sample_financial_data()
    
    # Create SECFileSourcer instance
    sourcer = SECFileSourcer()
    
    # Create a mock financial model
    financial_model = {
        'annual_income_statement': sample_df[sample_df['statement_type'] == 'INCOME'].copy(),
        'annual_balance_sheet': sample_df[sample_df['statement_type'] == 'BALANCE'].copy(),
        'annual_cash_flow': sample_df[sample_df['statement_type'] == 'CASH'].copy(),
    }
    
    # Create a mock sensitivity model
    sensitivity_model = {
        'case_summary': pd.DataFrame({
            'Scenario': ['Base Case', 'Optimistic', 'Pessimistic'],
            'Revenue Growth': ['5%', '10%', '0%'],
            'Margin': ['20%', '25%', '15%']
        })
    }
    
    # Test different formatting styles
    formatting_styles = ['markers', 'html', 'rich_text', 'unicode']
    
    for style in formatting_styles:
        print(f"\nTesting Excel export with {style.upper()} formatting...")
        
        try:
            # Export to Excel with formatted titles
            filepath = sourcer.export_to_excel_with_formatted_titles(
                financial_model=financial_model,
                sensitivity_model=sensitivity_model,
                ticker='TEST',
                filename=f'test_text_formatting_{style}.xlsx',
                formatting_style=style,
                progress_callback=print
            )
            
            print(f"✓ Successfully created Excel file: {filepath}")
            print(f"  Formatting style: {style}")
            
        except Exception as e:
            print(f"✗ Error during {style} formatting: {str(e)}")
    
    print("\n" + "="*80)
    print("Text formatting demonstration completed!")
    print("="*80)
    print("\nFormatting styles explained:")
    print("• MARKERS: Uses ** for bold and * for italic (e.g., **INCOME STATEMENT**, *Gross Profit*)")
    print("• HTML: Uses HTML tags (e.g., <b>INCOME STATEMENT</b>, <i>Gross Profit</i>)")
    print("• RICH_TEXT: Uses custom markers (e.g., [BOLD]INCOME STATEMENT[/BOLD])")
    print("• UNICODE: Uses Unicode characters and icons for visual distinction")

if __name__ == "__main__":
    print("Alternative Text Formatting for Line Item Titles")
    print("=" * 60)
    
    # Test different formatting styles
    test_text_formatting_styles()
    
    # Test Excel export with formatted titles
    test_excel_export_with_formatted_titles()
    
    print("\nTest completed successfully!")
    print("Check the generated Excel files in the Storage directory.") 