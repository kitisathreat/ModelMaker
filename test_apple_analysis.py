#!/usr/bin/env python3
"""
Test script to analyze Apple (AAPL) for 2 quarters using the SEC file sourcer.
"""

import sys
import os
import time
import pandas as pd

# Add the Data Sourcing directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Data Sourcing'))

from sec_file_sourcer import SECFileSourcer

def test_apple_analysis():
    """Test Apple (AAPL) analysis for 2 quarters."""
    print("Apple (AAPL) Analysis Test - 2 Quarters")
    print("=" * 50)
    
    # Initialize the sourcer
    sourcer = SECFileSourcer()
    ticker = "AAPL"
    quarters = 2
    
    print(f"Analyzing {ticker} for {quarters} quarters")
    print()
    
    # Track timing
    start_time = time.time()
    
    try:
        # Create financial model
        print("Creating financial model...")
        financial_model = sourcer.create_financial_model(
            ticker=ticker,
            quarters=quarters,
            progress_callback=lambda msg: print(f"   {msg}")
        )
        
        model_time = time.time() - start_time
        print(f"Financial model created in {model_time:.1f} seconds")
        
        # Display results summary
        print("\nFinancial Model Summary:")
        for key, df in financial_model.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                print(f"   {key}: {len(df)} periods, {len(df.columns)} line items")
        
        # Create sensitivity analysis
        print("\nCreating sensitivity analysis...")
        sensitivity_model = sourcer.create_sensitivity_model(
            financial_model=financial_model,
            ticker=ticker,
            quarters=quarters,
            progress_callback=lambda msg: print(f"   {msg}")
        )
        
        # Export to Excel
        print("\nExporting to Excel...")
        excel_file = sourcer.export_to_excel_fast_preview(
            financial_model=financial_model,
            sensitivity_model=sensitivity_model,
            ticker=ticker,
            filename=f"apple_test_{int(time.time())}.xlsx",
            progress_callback=lambda msg: print(f"   {msg}")
        )
        
        total_time = time.time() - start_time
        print(f"\nTest complete! Total time: {total_time:.1f} seconds")
        print(f"Results saved to: {excel_file}")
        
        return financial_model, sensitivity_model, excel_file
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    test_apple_analysis() 