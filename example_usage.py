#!/usr/bin/env python3
"""
Example usage of the SEC File Sourcer class.

This script demonstrates how to:
1. Find SEC filings for a company
2. Create a financial model
3. Perform sensitivity analysis
4. Export results to Excel
"""

import sys
import os

# Add the Data Sourcing directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Data Sourcing'))

from sec_file_sourcer import SECFileSourcer

def main():
    """Main function to demonstrate SEC File Sourcer usage."""
    
    print("=== SEC File Sourcer Example ===\n")
    
    # Initialize the SEC File Sourcer
    print("Initializing SEC File Sourcer...")
    sourcer = SECFileSourcer()
    
    # Example ticker (Apple Inc.)
    ticker = "AAPL"
    print(f"Analyzing company: {ticker}\n")
    
    # Step 1: Find SEC filings
    print("Step 1: Finding SEC filings...")
    filings = sourcer.find_sec_filings(ticker, filing_types=['10-K', '10-Q'])
    
    if filings.empty:
        print(f"No filings found for {ticker}. Please check the ticker symbol.")
        return
    
    print(f"Found {len(filings)} filings")
    print("\nRecent filings:")
    print(filings[['form', 'filingDate', 'description']].head())
    print()
    
    # Step 2: Create financial model
    print("Step 2: Creating financial model...")
    financial_model = sourcer.create_financial_model(ticker)
    
    # Check what data is available
    print("\nFinancial model components:")
    for sheet_name, df in financial_model.items():
        if not df.empty:
            print(f"  {sheet_name}: {len(df)} data points")
        else:
            print(f"  {sheet_name}: No data available")
    print()
    
    # Step 3: Create sensitivity analysis
    print("Step 3: Creating sensitivity analysis...")
    sensitivity_model = sourcer.create_sensitivity_model(financial_model, ticker)
    
    # Display case summary if available
    case_summary = sensitivity_model['case_summary']
    if not case_summary.empty:
        print("\nOperating Leverage Scenarios:")
        print(case_summary[['Scenario', 'Revenue', 'Operating Income', 'Operating Leverage']].to_string(index=False))
        print()
    
    # Display KPI summary if available
    kpi_summary = sensitivity_model['kpi_summary']
    if not kpi_summary.empty:
        print("Key Performance Indicators:")
        print(kpi_summary.to_string(index=False))
        print()
    
    # Step 4: Export to Excel
    print("Step 4: Exporting to Excel...")
    excel_file = sourcer.export_to_excel(financial_model, sensitivity_model, ticker)
    
    if excel_file:
        print(f"✓ Financial model exported to: {excel_file}")
        print("\nExcel file contains the following sheets:")
        print("  - Annual Income Statement")
        print("  - Annual Balance Sheet") 
        print("  - Annual Cash Flow")
        print("  - Quarterly Income Statement")
        print("  - Quarterly Balance Sheet")
        print("  - Quarterly Cash Flow")
        print("  - Case Summary (Operating Leverage Scenarios)")
        print("  - Financial Model (Historical + Forecasted)")
        print("  - KPI Summary (Key Performance Indicators)")
        print("  - Summary (Model Metadata)")
    else:
        print("✗ Failed to export to Excel")
    
    print("\n=== Analysis Complete ===")

def analyze_multiple_companies():
    """Example function to analyze multiple companies."""
    
    print("\n=== Multiple Company Analysis ===\n")
    
    sourcer = SECFileSourcer()
    companies = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in companies:
        print(f"Analyzing {ticker}...")
        
        # Find filings
        filings = sourcer.find_sec_filings(ticker)
        print(f"  Found {len(filings)} filings")
        
        # Create financial model
        financial_model = sourcer.create_financial_model(ticker)
        
        # Count available data points
        total_data_points = sum(len(df) for df in financial_model.values() if not df.empty)
        print(f"  Total data points: {total_data_points}")
        
        # Create sensitivity analysis
        sensitivity_model = sourcer.create_sensitivity_model(financial_model, ticker)
        
        # Export to Excel
        excel_file = sourcer.export_to_excel(financial_model, sensitivity_model, ticker)
        if excel_file:
            print(f"  ✓ Exported to: {excel_file}")
        else:
            print(f"  ✗ Export failed")
        
        print()

if __name__ == "__main__":
    # Run the main example
    main()
    
    # Uncomment the line below to run multiple company analysis
    # analyze_multiple_companies() 