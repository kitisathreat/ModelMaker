#!/usr/bin/env python3
"""
Test script to demonstrate the improved financial statement classification system.
"""

import sys
import os

# Add the Data Sourcing directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Data Sourcing'))

from sec_file_sourcer import SECFileSourcer

def test_classification_system():
    """Test the financial statement classification system with various line items."""
    print("🧪 Testing Financial Statement Classification System")
    print("=" * 60)
    
    sourcer = SECFileSourcer()
    
    # Test line items for each statement type
    test_items = {
        'Income Statement': [
            'Revenue',
            'SalesRevenueNet',
            'CostOfRevenue',
            'CostOfGoodsSold',
            'GrossProfit',
            'ResearchAndDevelopmentExpense',
            'SellingGeneralAndAdministrativeExpense',
            'OperatingExpenses',
            'OperatingIncomeLoss',
            'InterestExpense',
            'InterestIncome',
            'IncomeTaxExpenseBenefit',
            'NetIncomeLoss',
            'EarningsPerShareBasic',
            'EarningsPerShareDiluted',
            'DepreciationAndAmortization',
            'StockBasedCompensationExpense',
            'RestructuringCharges',
            'ImpairmentCharges',
            'GainLossOnSaleOfAssets',
            'ForeignCurrencyGainLoss',
            'OtherIncomeExpenseNet'
        ],
        
        'Balance Sheet': [
            'CashAndCashEquivalentsAtCarryingValue',
            'AccountsReceivableNetCurrent',
            'InventoryNet',
            'PrepaidExpenseAndOtherAssetsCurrent',
            'ShortTermInvestments',
            'AssetsCurrent',
            'PropertyPlantAndEquipmentNet',
            'Goodwill',
            'IntangibleAssetsNetExcludingGoodwill',
            'LongTermInvestments',
            'DeferredTaxAssetsNet',
            'OtherAssetsNoncurrent',
            'AssetsNoncurrent',
            'Assets',
            'AccountsPayableCurrent',
            'AccruedLiabilitiesCurrent',
            'ContractWithCustomerLiabilityCurrent',
            'ShortTermBorrowings',
            'OtherLiabilitiesCurrent',
            'LiabilitiesCurrent',
            'LongTermDebtNoncurrent',
            'DeferredTaxLiabilitiesNet',
            'OtherLiabilitiesNoncurrent',
            'LiabilitiesNoncurrent',
            'Liabilities',
            'CommonStockValue',
            'AdditionalPaidInCapital',
            'RetainedEarningsAccumulatedDeficit',
            'AccumulatedOtherComprehensiveIncomeLossNetOfTax',
            'TreasuryStockValue',
            'StockholdersEquity',
            'WorkingCapital'
        ],
        
        'Cash Flow Statement': [
            'NetCashProvidedByUsedInOperatingActivities',
            'NetCashProvidedByUsedInInvestingActivities',
            'NetCashProvidedByUsedInFinancingActivities',
            'DepreciationAndAmortization',
            'StockBasedCompensationExpense',
            'DeferredIncomeTaxExpenseBenefit',
            'IncreaseDecreaseInAccountsReceivable',
            'IncreaseDecreaseInInventories',
            'IncreaseDecreaseInAccountsPayable',
            'IncreaseDecreaseInContractWithCustomerLiability',
            'OtherOperatingActivitiesCashFlowStatement',
            'PaymentsToAcquirePropertyPlantAndEquipment',
            'PaymentsToAcquireBusinessesNetOfCashAcquired',
            'PaymentsToAcquireInvestments',
            'ProceedsFromSaleMaturityAndCollectionsOfInvestments',
            'OtherInvestingActivitiesCashFlowStatement',
            'ProceedsFromIssuanceOfLongTermDebt',
            'RepaymentsOfLongTermDebt',
            'PaymentsOfDividends',
            'PaymentsForRepurchaseOfCommonStock',
            'ProceedsFromIssuanceOfCommonStock',
            'OtherFinancingActivitiesCashFlowStatement',
            'EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
            'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect',
            'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodBeginningBalance',
            'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodEndingBalance'
        ]
    }
    
    # Test each category
    for statement_type, items in test_items.items():
        print(f"\n📊 {statement_type}")
        print("-" * 40)
        
        correct_classifications = 0
        total_items = len(items)
        
        for item in items:
            classification = sourcer._classify_financial_line_item(item)
            
            # Determine expected statement type
            expected_statement = statement_type.lower().replace(' ', '_')
            if expected_statement == 'cash_flow_statement':
                expected_statement = 'cash_flow'
            
            # Check if classification is correct
            is_correct = classification['statement'] == expected_statement
            if is_correct:
                correct_classifications += 1
            
            # Display result
            status = "✅" if is_correct else "❌"
            confidence_level = "HIGH" if classification['confidence'] >= 0.8 else "MEDIUM" if classification['confidence'] >= 0.6 else "LOW"
            
            print(f"{status} {item:<50} → {classification['statement']:<15} ({confidence_level} {classification['confidence']:.2f})")
        
        # Calculate accuracy
        accuracy = (correct_classifications / total_items) * 100
        print(f"\n📈 Accuracy: {correct_classifications}/{total_items} ({accuracy:.1f}%)")
    
    # Test edge cases and ambiguous items
    print(f"\n🔍 Edge Cases and Ambiguous Items")
    print("-" * 40)
    
    edge_cases = [
        'NetIncomeLoss',  # Appears in both Income Statement and Cash Flow
        'DepreciationAndAmortization',  # Appears in both Income Statement and Cash Flow
        'StockBasedCompensationExpense',  # Appears in both Income Statement and Cash Flow
        'DeferredIncomeTaxExpenseBenefit',  # Appears in both Income Statement and Cash Flow
        'OtherIncomeExpenseNet',  # Could be ambiguous
        'OtherAssetsNoncurrent',  # Generic category
        'OtherLiabilitiesNoncurrent',  # Generic category
        'OtherOperatingActivitiesCashFlowStatement',  # Generic category
        'OtherInvestingActivitiesCashFlowStatement',  # Generic category
        'OtherFinancingActivitiesCashFlowStatement',  # Generic category
    ]
    
    for item in edge_cases:
        classification = sourcer._classify_financial_line_item(item)
        confidence_level = "HIGH" if classification['confidence'] >= 0.8 else "MEDIUM" if classification['confidence'] >= 0.6 else "LOW"
        
        print(f"🔍 {item:<50} → {classification['statement']:<15} ({confidence_level} {classification['confidence']:.2f})")
    
    # Test fuzzy matching with similar concepts
    print(f"\n🎯 Fuzzy Matching Test")
    print("-" * 40)
    
    fuzzy_tests = [
        ('Revenue', 'Revenues'),  # Similar concepts
        ('NetIncome', 'NetIncomeLoss'),  # Similar concepts
        ('CashAndCashEquivalents', 'CashAndCashEquivalentsAtCarryingValue'),  # Similar concepts
        ('AccountsReceivable', 'AccountsReceivableNetCurrent'),  # Similar concepts
        ('Inventory', 'InventoryNet'),  # Similar concepts
    ]
    
    for original, similar in fuzzy_tests:
        original_class = sourcer._classify_financial_line_item(original)
        similar_class = sourcer._classify_financial_line_item(similar)
        
        print(f"🎯 {original:<30} → {original_class['statement']:<15} ({original_class['confidence']:.2f})")
        print(f"   {similar:<30} → {similar_class['statement']:<15} ({similar_class['confidence']:.2f})")
        
        if original_class['statement'] == similar_class['statement']:
            print("   ✅ Consistent classification")
        else:
            print("   ⚠️  Inconsistent classification")
        print()

def test_validation_system():
    """Test the validation system for statement classifications."""
    print("\n🔍 Testing Validation System")
    print("=" * 40)
    
    sourcer = SECFileSourcer()
    
    # Test validation scenarios
    validation_tests = [
        ('Revenue', 'income_statement', True),  # Should be valid
        ('CashAndCashEquivalents', 'balance_sheet', True),  # Should be valid
        ('NetCashFromOperatingActivities', 'cash_flow', True),  # Should be valid
        ('Revenue', 'balance_sheet', False),  # Should be invalid
        ('CashAndCashEquivalents', 'income_statement', False),  # Should be invalid
        ('NetCashFromOperatingActivities', 'income_statement', False),  # Should be invalid
    ]
    
    for line_item, proposed_statement, expected_valid in validation_tests:
        is_valid = sourcer._validate_statement_classification(line_item, proposed_statement)
        status = "✅" if is_valid == expected_valid else "❌"
        
        print(f"{status} {line_item:<30} → {proposed_statement:<15} (Valid: {is_valid}, Expected: {expected_valid})")

def main():
    """Run all classification tests."""
    print("🚀 Financial Statement Classification System Test")
    print("=" * 60)
    
    # Test the classification system
    test_classification_system()
    
    # Test the validation system
    test_validation_system()
    
    print("\n🎉 Classification system test completed!")
    print("\n📋 Summary:")
    print("- Rules-based classification with regex patterns")
    print("- Fuzzy matching for edge cases")
    print("- Confidence scoring for each classification")
    print("- Validation system for quality control")
    print("- Support for all three financial statements")
    print("- GAAP-compliant categorization")

if __name__ == "__main__":
    main() 