#!/usr/bin/env python3
"""
Test script to demonstrate the performance difference between fast preview and full formatting Excel generation.
"""

import time
import pandas as pd
import os
import sys

# Add the Data Sourcing directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Data Sourcing'))

from sec_file_sourcer import SECFileSourcer

def create_sample_data():
    """Create sample financial data for testing."""
    # Create sample annual income statement
    annual_income = pd.DataFrame({
        'Revenue': [1000000, 1100000, 1200000, 1300000],
        'CostOfGoodsSold': [600000, 660000, 720000, 780000],
        'GrossProfit': [400000, 440000, 480000, 520000],
        'OperatingExpenses': [200000, 220000, 240000, 260000],
        'OperatingIncome': [200000, 220000, 240000, 260000],
        'NetIncome': [150000, 165000, 180000, 195000]
    }, index=['2020', '2021', '2022', '2023'])
    
    # Create sample balance sheet
    annual_balance = pd.DataFrame({
        'CashAndCashEquivalents': [50000, 55000, 60000, 65000],
        'AccountsReceivable': [80000, 88000, 96000, 104000],
        'Inventory': [120000, 132000, 144000, 156000],
        'TotalAssets': [500000, 550000, 600000, 650000],
        'AccountsPayable': [60000, 66000, 72000, 78000],
        'TotalLiabilities': [200000, 220000, 240000, 260000],
        'StockholdersEquity': [300000, 330000, 360000, 390000]
    }, index=['2020', '2021', '2022', '2023'])
    
    # Create sample cash flow
    annual_cash_flow = pd.DataFrame({
        'NetIncome': [150000, 165000, 180000, 195000],
        'DepreciationAndAmortization': [30000, 33000, 36000, 39000],
        'NetCashFromOperatingActivities': [180000, 198000, 216000, 234000],
        'CapitalExpenditures': [-40000, -44000, -48000, -52000],
        'NetCashFromInvestingActivities': [-40000, -44000, -48000, -52000],
        'NetCashFromFinancingActivities': [-90000, -99000, -108000, -117000],
        'NetChangeInCash': [5000, 5500, 6000, 6500]
    }, index=['2020', '2021', '2022', '2023'])
    
    # Create sample sensitivity data
    sensitivity_model = {
        'case_summary': pd.DataFrame({
            'Scenario': ['Base Case', 'Optimistic', 'Pessimistic'],
            'Revenue Growth': [10, 15, 5],
            'Operating Margin': [20, 25, 15],
            'Net Income': [195000, 243750, 146250]
        }),
        'kpi_summary': pd.DataFrame({
            'KPI': ['Net Profit Margin', 'Operating Margin', 'Gross Margin'],
            'Value': [15.0, 20.0, 40.0],
            'Unit': ['%', '%', '%']
        }),
        'financial_model': pd.DataFrame({
            'Metric': ['Revenue', 'Net Income', 'EPS'],
            '2023': [1300000, 195000, 1.95],
            '2024': [1430000, 214500, 2.15],
            '2025': [1573000, 235950, 2.36]
        })
    }
    
    financial_model = {
        'annual_income_statement': annual_income,
        'annual_balance_sheet': annual_balance,
        'annual_cash_flow': annual_cash_flow,
        'quarterly_income_statement': annual_income,  # Using same data for simplicity
        'quarterly_balance_sheet': annual_balance,
        'quarterly_cash_flow': annual_cash_flow
    }
    
    return financial_model, sensitivity_model

def test_performance():
    """Test the performance difference between fast preview and full formatting."""
    print("🚀 Testing Excel Generation Performance")
    print("=" * 50)
    
    # Create sample data
    financial_model, sensitivity_model = create_sample_data()
    sourcer = SECFileSourcer()
    ticker = "TEST"
    
    # Test 1: Fast Preview (no formatting)
    print("\n📊 Test 1: Fast Preview (No Formatting)")
    print("-" * 40)
    
    start_time = time.time()
    fast_file = sourcer.export_to_excel_fast_preview(
        financial_model, 
        sensitivity_model, 
        ticker, 
        "test_fast_preview.xlsx"
    )
    fast_time = time.time() - start_time
    
    print(f"✅ Fast preview completed in {fast_time:.2f} seconds")
    print(f"📁 File saved: {fast_file}")
    
    # Test 2: Full Formatting
    print("\n🎨 Test 2: Full Formatting")
    print("-" * 40)
    
    start_time = time.time()
    full_file = sourcer.export_to_excel(
        financial_model, 
        sensitivity_model, 
        ticker, 
        filename="test_full_formatting.xlsx"
    )
    full_time = time.time() - start_time
    
    print(f"✅ Full formatting completed in {full_time:.2f} seconds")
    print(f"📁 File saved: {full_file}")
    
    # Test 3: Apply formatting to existing file
    print("\n🔧 Test 3: Apply Formatting to Existing File")
    print("-" * 40)
    
    start_time = time.time()
    formatted_file = sourcer.apply_excel_formatting(fast_file)
    format_time = time.time() - start_time
    
    print(f"✅ Formatting applied in {format_time:.2f} seconds")
    print(f"📁 File updated: {formatted_file}")
    
    # Performance comparison
    print("\n📈 Performance Comparison")
    print("=" * 50)
    print(f"Fast Preview:           {fast_time:.2f}s")
    print(f"Full Formatting:        {full_time:.2f}s")
    print(f"Apply to Existing:      {format_time:.2f}s")
    print(f"Speedup (Fast vs Full): {full_time/fast_time:.1f}x faster")
    print(f"Formatting overhead:    {format_time/fast_time:.1f}x the base time")
    
    # File size comparison
    fast_size = os.path.getsize(fast_file) / 1024  # KB
    full_size = os.path.getsize(full_file) / 1024  # KB
    
    print(f"\n📦 File Size Comparison")
    print(f"Fast Preview:    {fast_size:.1f} KB")
    print(f"Full Formatting: {full_size:.1f} KB")
    print(f"Size difference: {full_size - fast_size:.1f} KB")
    
    # Cleanup
    print(f"\n🧹 Cleaning up test files...")
    try:
        os.remove(fast_file)
        os.remove(full_file)
        print("✅ Test files cleaned up")
    except Exception as e:
        print(f"⚠️  Could not clean up all files: {e}")
    
    print("\n🎉 Performance test completed!")

if __name__ == "__main__":
    test_performance() 