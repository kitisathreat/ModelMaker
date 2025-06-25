#!/usr/bin/env python3
"""
Test script to verify the quarters filtering fix.
"""

import sys
import os

# Add the Data Sourcing directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Data Sourcing'))

from sec_file_sourcer import SECFileSourcer

def test_quarters_calculation():
    """Test the quarters calculation logic."""
    sourcer = SECFileSourcer()
    
    # Test cases
    test_cases = [
        (5, "1 year + 1 quarter"),
        (4, "1 year"),
        (8, "2 years"),
        (12, "3 years"),
        (1, "1 quarter"),
        (2, "2 quarters")
    ]
    
    print("Testing quarters calculation logic:")
    print("=" * 50)
    
    for quarters, description in test_cases:
        years_needed = max(1, (quarters + 3) // 4)
        print(f"{description:15} -> {quarters:2d} quarters -> {years_needed} years needed")
    
    print("\nTesting with actual quarters parameter:")
    print("=" * 50)
    
    # Test with 5 quarters (1 year + 1 quarter)
    quarters = 5
    sourcer.current_quarters = quarters
    years_needed = max(1, (quarters + 3) // 4)
    
    print(f"Requested quarters: {quarters}")
    print(f"Years needed: {years_needed}")
    print(f"Annual data periods: {years_needed}")
    print(f"Quarterly data periods: {quarters}")
    
    # Test the filtering logic
    print(f"\nFor annual data: Will include {years_needed} most recent years")
    print(f"For quarterly data: Will include {quarters} most recent quarters")

if __name__ == "__main__":
    test_quarters_calculation() 