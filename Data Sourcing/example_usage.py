#!/usr/bin/env python3
"""
Example usage of SECFileSourcer with different quarter configurations.

This script demonstrates how to use the new quarters parameter to specify
how much historical data you want to analyze.
"""

from sec_file_sourcer import SECFileSourcer

def main():
    # Initialize the sourcer
    sourcer = SECFileSourcer()
    
    # Show available configurations
    sourcer.print_quarter_configurations()
    
    # Example ticker
    ticker = "AAPL"
    
    # Example 1: Short-term analysis (1 year)
    print(f"\n{'='*60}")
    print(f"EXAMPLE 1: Short-term analysis for {ticker}")
    print(f"{'='*60}")
    configs = sourcer.get_quarter_configurations()
    quarters = configs["short_term"]["quarters"]
    
    print(f"Creating financial model with {quarters} quarters of data...")
    financial_model = sourcer.create_financial_model(ticker, quarters=quarters)
    
    if any(not df.empty for df in financial_model.values()):
        sensitivity_model = sourcer.create_sensitivity_model(financial_model, ticker, quarters=quarters)
        filename = f"{ticker}_short_term_{quarters}q.xlsx"
        excel_file = sourcer.export_to_excel(financial_model, sensitivity_model, ticker, filename)
        print(f"Short-term model saved to: {excel_file}")
    
    # Example 2: Long-term analysis (3 years)
    print(f"\n{'='*60}")
    print(f"EXAMPLE 2: Long-term analysis for {ticker}")
    print(f"{'='*60}")
    quarters = configs["long_term"]["quarters"]
    
    print(f"Creating financial model with {quarters} quarters of data...")
    financial_model = sourcer.create_financial_model(ticker, quarters=quarters)
    
    if any(not df.empty for df in financial_model.values()):
        sensitivity_model = sourcer.create_sensitivity_model(financial_model, ticker, quarters=quarters)
        filename = f"{ticker}_long_term_{quarters}q.xlsx"
        excel_file = sourcer.export_to_excel(financial_model, sensitivity_model, ticker, filename)
        print(f"Long-term model saved to: {excel_file}")
    
    # Example 3: Custom quarters
    print(f"\n{'='*60}")
    print(f"EXAMPLE 3: Custom quarters for {ticker}")
    print(f"{'='*60}")
    custom_quarters = 6  # 1.5 years
    
    print(f"Creating financial model with {custom_quarters} quarters of data...")
    financial_model = sourcer.create_financial_model(ticker, quarters=custom_quarters)
    
    if any(not df.empty for df in financial_model.values()):
        sensitivity_model = sourcer.create_sensitivity_model(financial_model, ticker, quarters=custom_quarters)
        filename = f"{ticker}_custom_{custom_quarters}q.xlsx"
        excel_file = sourcer.export_to_excel(financial_model, sensitivity_model, ticker, filename)
        print(f"Custom model saved to: {excel_file}")

def demonstrate_quarter_calculation():
    """
    Demonstrate how the quarter calculation works.
    """
    print("\n" + "="*60)
    print("QUARTER CALCULATION DEMONSTRATION")
    print("="*60)
    
    # Show how the calculation works
    quarter_examples = [4, 8, 12, 16, 20]
    
    for quarters in quarter_examples:
        years_needed = max(1, (quarters + 3) // 4)  # Round up to get full years needed
        k_filings_needed = years_needed  # One 10-K per year
        q_filings_needed = quarters  # One 10-Q per quarter
        
        print(f"{quarters:2d} quarters: {years_needed} years → {k_filings_needed} 10-K filings, {q_filings_needed} 10-Q filings")

if __name__ == "__main__":
    # Show how quarter calculation works
    demonstrate_quarter_calculation()
    
    # Run the main examples
    main() 