#!/usr/bin/env python3
"""
Test script to verify the us-gaap: prefix handling fix.
"""

import sys
import os

# Add the Data Sourcing directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Data Sourcing'))

from sec_file_sourcer import SECFileSourcer

def test_prefix_handling():
    """Test the us-gaap: prefix handling."""
    print("🧪 Testing us-gaap: Prefix Handling Fix")
    print("=" * 50)
    
    sourcer = SECFileSourcer()
    
    # Test cases with us-gaap: prefix
    test_cases = [
        ('us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenue'),
        ('us-gaap:CostOfGoodsSold', 'Cost of Goods Sold'),
        ('us-gaap:ResearchAndDevelopmentExpense', 'Research & Development'),
        ('us-gaap:NetIncomeLoss', 'Net Income'),
        ('us-gaap:CashAndCashEquivalentsAtCarryingValue', 'Cash & Cash Equivalents'),
        ('us-gaap:PropertyPlantAndEquipmentNet', 'Property, Plant & Equipment (Net)'),
        ('us-gaap:LongTermDebt', 'Long-Term Debt'),
        ('us-gaap:DepreciationAndAmortization', 'Depreciation & Amortization'),
    ]
    
    all_passed = True
    
    for concept_name, expected_title in test_cases:
        actual_title = sourcer._get_user_friendly_title(concept_name)
        status = "✅ PASS" if actual_title == expected_title else "❌ FAIL"
        print(f"{status} | {concept_name}")
        print(f"     Expected: {expected_title}")
        print(f"     Actual:   {actual_title}")
        print()
        
        if actual_title != expected_title:
            all_passed = False
    
    if all_passed:
        print("🎉 All tests passed! The us-gaap: prefix handling is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    test_prefix_handling() 