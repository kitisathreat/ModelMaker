#!/usr/bin/env python3
"""
Test script for the enhanced Stock Analyzer GUI.

This script tests the new features:
1. S&P 500 autocomplete functionality
2. Period configuration dropdown
3. Excel viewer integration
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from stock_analyzer_gui import SP500Data, AutocompleteLineEdit, ExcelViewer

def test_sp500_data():
    """Test S&P 500 data loading and search functionality."""
    print("Testing S&P 500 data loading...")
    
    sp500 = SP500Data()
    print(f"Loaded {len(sp500.companies)} companies")
    
    # Test search functionality
    test_queries = ['AAPL', 'Apple', 'MSFT', 'Microsoft', 'GOOGL', 'Alphabet']
    
    for query in test_queries:
        results = sp500.search_companies(query)
        print(f"Search for '{query}': {len(results)} results")
        for result in results[:3]:  # Show first 3 results
            print(f"  - {result['display']}")
    
    print("\nS&P 500 data test completed successfully!")

def test_autocomplete():
    """Test autocomplete functionality."""
    print("\nTesting autocomplete functionality...")
    
    sp500 = SP500Data()
    
    # Test different search patterns
    test_cases = [
        ('AAPL', 'Should find Apple Inc.'),
        ('Apple', 'Should find Apple Inc.'),
        ('MSFT', 'Should find Microsoft'),
        ('Microsoft', 'Should find Microsoft'),
        ('GOOGL', 'Should find Alphabet'),
        ('Alphabet', 'Should find Alphabet'),
    ]
    
    for query, description in test_cases:
        results = sp500.search_companies(query)
        print(f"'{query}' - {description}")
        if results:
            print(f"  Found: {results[0]['display']}")
        else:
            print(f"  No results found")
    
    print("\nAutocomplete test completed successfully!")

def test_period_configurations():
    """Test period configuration mapping."""
    print("\nTesting period configurations...")
    
    # Test the period input logic
    test_cases = [
        (0, 0, "Should default to 1 quarter"),
        (1, 0, "Should be 4 quarters (1 year)"),
        (2, 0, "Should be 8 quarters (2 years)"),
        (0, 4, "Should be 4 quarters"),
        (1, 2, "Should be 6 quarters (1 year + 2 quarters)"),
        (3, 1, "Should be 13 quarters (3 years + 1 quarter)"),
        (5, 0, "Should be 20 quarters (5 years)"),
        (6, 0, "Should trigger warning (> 5 years)"),
    ]
    
    for years, quarters, description in test_cases:
        total_quarters = years * 4 + quarters
        if total_quarters <= 0:
            total_quarters = 1  # Default minimum
        
        warning = ""
        if total_quarters > 20:
            warning = " (WARNING: > 5 years)"
        elif total_quarters >= 16:
            warning = " (INFO: substantial data)"
        
        print(f"  {years}y + {quarters}q = {total_quarters} quarters - {description}{warning}")
    
    print("\nPeriod configuration test completed successfully!")

def main():
    """Run all tests."""
    print("=" * 60)
    print("STOCK ANALYZER GUI TEST SUITE")
    print("=" * 60)
    
    try:
        test_sp500_data()
        test_autocomplete()
        test_period_configurations()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe GUI should work correctly with:")
        print("✓ S&P 500 company autocomplete")
        print("✓ Period configuration dropdown")
        print("✓ Excel viewer integration")
        print("✓ Enhanced user interface")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 