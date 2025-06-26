#!/usr/bin/env python3
"""
Test script to verify the three formatting changes:
1. Formatting columns are removed before Excel export
2. Blank rows are added between sections
3. Consistent terminology is used for parent-child tracking
"""

import pandas as pd
import sys
import os

# Add the Data Sourcing directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Data Sourcing'))

from sec_file_sourcer import SECFileSourcer

def test_formatting_changes():
    """Test the three formatting changes."""
    
    print("Testing formatting changes...")
    
    # Create sample financial data
    sample_data = {
        'annual_income_statement': pd.DataFrame({
            'Revenue': [1000000, 900000],
            'CostOfGoodsSold': [600000, 540000],
            'GrossProfit': [400000, 360000],
            'ResearchAndDevelopmentExpense': [80000, 72000],
            'SellingGeneralAndAdministrativeExpense': [120000, 108000],
            'OperatingExpenses': [200000, 180000],
            'OperatingIncome': [200000, 180000],
            'NetIncome': [150000, 135000]
        }, index=['2023-12-31', '2022-12-31']),
        
        'annual_balance_sheet': pd.DataFrame({
            'CashAndCashEquivalents': [200000, 180000],
            'AccountsReceivable': [150000, 135000],
            'Inventory': [100000, 90000],
            'CurrentAssets': [450000, 405000],
            'PropertyPlantAndEquipmentNet': [300000, 270000],
            'TotalAssets': [750000, 675000],
            'AccountsPayable': [80000, 72000],
            'CurrentLiabilities': [80000, 72000],
            'LongTermDebt': [200000, 180000],
            'TotalLiabilities': [280000, 252000],
            'CommonStock': [100000, 90000],
            'RetainedEarnings': [370000, 333000],
            'TotalStockholdersEquity': [470000, 423000]
        }, index=['2023-12-31', '2022-12-31']),
        
        'annual_cash_flow': pd.DataFrame({
            'NetIncome': [150000, 135000],
            'DepreciationAndAmortization': [30000, 27000],
            'OperatingAdjustments': [30000, 27000],
            'NetCashFromOperatingActivities': [180000, 162000],
            'CapitalExpenditures': [-50000, -45000],
            'NetCashFromInvestingActivities': [-50000, -45000],
            'DividendsPaid': [-20000, -18000],
            'NetCashFromFinancingActivities': [-20000, -18000],
            'NetChangeInCash': [110000, 99000],
            'CashAtEndOfPeriod': [200000, 180000]
        }, index=['2023-12-31', '2022-12-31'])
    }
    
    # Create sensitivity model
    sensitivity_model = {
        'case_summary': pd.DataFrame({
            'Scenario': ['Base Case', 'Optimistic', 'Pessimistic'],
            'Revenue Growth': ['5%', '10%', '0%'],
            'Margin': ['20%', '25%', '15%']
        })
    }
    
    # Create SECFileSourcer instance
    sourcer = SECFileSourcer()
    
    print("1. Testing Excel export with formatting changes...")
    
    try:
        # Export to Excel
        filepath = sourcer.export_to_excel(
            financial_model=sample_data,
            sensitivity_model=sensitivity_model,
            ticker='TEST',
            filename='test_formatting_changes.xlsx',
            progress_callback=print
        )
        
        print(f"✓ Excel file created: {filepath}")
        
        # Verify the file exists
        if os.path.exists(filepath):
            print("✓ File exists and can be opened")
            
            # Read the Excel file to verify formatting columns are removed
            df = pd.read_excel(filepath, sheet_name='Annual Financial Statements')
            print(f"✓ Excel sheet has {len(df.columns)} columns")
            
            # Check that formatting columns are removed
            formatting_columns = ['is_section_heading', 'parent', 'is_aggregate', 'statement_type']
            for col in formatting_columns:
                if col in df.columns:
                    print(f"✗ Formatting column '{col}' still exists - should be removed")
                else:
                    print(f"✓ Formatting column '{col}' correctly removed")
            
            # Check for blank rows between sections
            line_items = df['Line Item'].tolist()
            section_indices = []
            for i, item in enumerate(line_items):
                if item.upper() in ['INCOME STATEMENT', 'BALANCE SHEET', 'CASH FLOW STATEMENT']:
                    section_indices.append(i)
            
            print(f"✓ Found {len(section_indices)} section headings")
            
            # Check for blank rows between sections
            blank_rows_found = 0
            for i in range(len(section_indices) - 1):
                current_section = section_indices[i]
                next_section = section_indices[i + 1]
                
                # Check if there's a blank row between sections
                if next_section - current_section > 1:
                    # Look for blank row
                    for j in range(current_section + 1, next_section):
                        if line_items[j] == '':
                            blank_rows_found += 1
                            break
            
            if blank_rows_found == len(section_indices) - 1:
                print("✓ Blank rows correctly added between all sections")
            else:
                print(f"✗ Only {blank_rows_found} blank rows found, expected {len(section_indices) - 1}")
            
        else:
            print("✗ Excel file was not created")
            
    except Exception as e:
        print(f"✗ Error during Excel export: {str(e)}")
    
    print("\n2. Testing terminology consistency...")
    
    # Test the parent map consistency
    parent_map = sourcer._create_user_friendly_parent_map()
    
    # Check that key parent-child relationships use consistent terminology
    key_relationships = [
        ('Research & Development', 'Operating Expenses'),
        ('Selling, General & Administrative', 'Operating Expenses'),
        ('Cash & Cash Equivalents', 'Current Assets'),
        ('Accounts Receivable', 'Current Assets'),
        ('Accounts Payable', 'Current Liabilities'),
        ('Common Stock', 'Stockholders\' Equity')
    ]
    
    for child, parent in key_relationships:
        if child in parent_map and parent_map[child] == parent:
            print(f"✓ Consistent terminology: '{child}' → '{parent}'")
        else:
            print(f"✗ Inconsistent terminology: '{child}' → '{parent_map.get(child, 'NOT FOUND')}'")
    
    print("\n3. Testing _remove_formatting_columns method...")
    
    # Create a test DataFrame with formatting columns
    test_df = pd.DataFrame({
        'Line Item': ['Revenue', 'Cost of Revenue', 'Gross Profit'],
        'is_section_heading': [False, False, False],
        'parent': [None, 'Revenue', None],
        'is_aggregate': [False, False, True],
        'statement_type': ['INCOME', 'INCOME', 'INCOME'],
        '2023': [1000000, 600000, 400000],
        '2022': [900000, 540000, 360000]
    })
    
    print(f"Original DataFrame has {len(test_df.columns)} columns: {list(test_df.columns)}")
    
    # Remove formatting columns
    cleaned_df = sourcer._remove_formatting_columns(test_df)
    
    print(f"Cleaned DataFrame has {len(cleaned_df.columns)} columns: {list(cleaned_df.columns)}")
    
    # Check that formatting columns are removed
    formatting_columns = ['is_section_heading', 'parent', 'is_aggregate', 'statement_type']
    removed_count = 0
    for col in formatting_columns:
        if col not in cleaned_df.columns:
            removed_count += 1
    
    if removed_count == len(formatting_columns):
        print("✓ All formatting columns correctly removed")
    else:
        print(f"✗ Only {removed_count}/{len(formatting_columns)} formatting columns removed")
    
    print("\n" + "="*60)
    print("FORMATTING CHANGES TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_formatting_changes() 