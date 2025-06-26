#!/usr/bin/env python3
"""
Demonstration script for alternative text formatting of line item titles.
This shows how to apply formatting directly to the text instead of using Excel cell formatting.
"""

import pandas as pd
import sys
import os

# Add the Data Sourcing directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Data Sourcing'))

from sec_file_sourcer import SECFileSourcer

def create_sample_data():
    """Create sample financial data with formatting flags."""
    
    sample_data = [
        # Section headings (bold)
        {'Line Item': 'INCOME STATEMENT', 'is_section_heading': True, 'parent': None, 'is_aggregate': False, '2023': '', '2022': '', '2021': ''},
        
        # Top-level line items
        {'Line Item': 'Revenue', 'is_section_heading': False, 'parent': None, 'is_aggregate': False, '2023': 1000000, '2022': 900000, '2021': 800000},
        {'Line Item': 'Cost of Revenue', 'is_section_heading': False, 'parent': 'Revenue', 'is_aggregate': False, '2023': 600000, '2022': 540000, '2021': 480000},
        {'Line Item': 'Gross Profit', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 400000, '2022': 360000, '2021': 320000},
        
        # Parent aggregator with children
        {'Line Item': 'Operating Expenses', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 200000, '2022': 180000, '2021': 160000},
        {'Line Item': 'Research & Development', 'is_section_heading': False, 'parent': 'OperatingExpenses', 'is_aggregate': False, '2023': 80000, '2022': 72000, '2021': 64000},
        {'Line Item': 'Selling, General & Administrative', 'is_section_heading': False, 'parent': 'OperatingExpenses', 'is_aggregate': False, '2023': 120000, '2022': 108000, '2021': 96000},
        
        {'Line Item': 'Operating Income', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 200000, '2022': 180000, '2021': 160000},
        {'Line Item': 'Net Income', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 150000, '2022': 135000, '2021': 120000},
        
        # Section heading for next statement
        {'Line Item': 'BALANCE SHEET', 'is_section_heading': True, 'parent': None, 'is_aggregate': False, '2023': '', '2022': '', '2021': ''},
        
        # Balance sheet items
        {'Line Item': 'Current Assets', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 500000, '2022': 450000, '2021': 400000},
        {'Line Item': 'Cash & Cash Equivalents', 'is_section_heading': False, 'parent': 'CurrentAssets', 'is_aggregate': False, '2023': 200000, '2022': 180000, '2021': 160000},
        {'Line Item': 'Accounts Receivable', 'is_section_heading': False, 'parent': 'CurrentAssets', 'is_aggregate': False, '2023': 150000, '2022': 135000, '2021': 120000},
        {'Line Item': 'Total Assets', 'is_section_heading': False, 'parent': None, 'is_aggregate': True, '2023': 1000000, '2022': 900000, '2021': 800000},
    ]
    
    return pd.DataFrame(sample_data)

def demonstrate_formatting_styles():
    """Demonstrate different text formatting styles."""
    
    print("Alternative Text Formatting for Line Item Titles")
    print("=" * 60)
    
    # Create sample data
    sample_df = create_sample_data()
    sourcer = SECFileSourcer()
    
    print("Original data structure:")
    print(sample_df[['Line Item', 'is_section_heading', 'parent', 'is_aggregate']].head(10))
    print("\n" + "="*80)
    
    # Test all formatting styles
    styles = ['markers', 'html', 'rich_text', 'unicode']
    
    for style in styles:
        print(f"\n{style.upper()} FORMATTING STYLE:")
        print("-" * 40)
        
        # Apply formatting
        formatted_df = sourcer.apply_text_formatting_to_titles(sample_df, style)
        
        # Display results
        for idx, row in formatted_df.iterrows():
            line_item = row['Line Item']
            is_section = row['is_section_heading']
            is_aggregate = row['is_aggregate']
            has_parent = row['parent'] is not None and str(row['parent']).strip() != ''
            
            # Add formatting indicators
            indicators = []
            if is_section:
                indicators.append("SECTION")
            if is_aggregate:
                indicators.append("AGGREGATE")
            if has_parent:
                indicators.append("CHILD")
            
            indicator_str = f" [{', '.join(indicators)}]" if indicators else ""
            print(f"{line_item}{indicator_str}")
    
    print("\n" + "="*80)
    print("FORMATTING STYLES EXPLAINED:")
    print("="*80)
    print("• MARKERS: Uses ** for bold and * for italic")
    print("  Example: **INCOME STATEMENT**, *Gross Profit*")
    print()
    print("• HTML: Uses HTML tags for formatting")
    print("  Example: <b>INCOME STATEMENT</b>, <i>Gross Profit</i>")
    print()
    print("• RICH_TEXT: Uses custom formatting markers")
    print("  Example: [BOLD]INCOME STATEMENT[/BOLD], [ITALIC]Gross Profit[/ITALIC]")
    print()
    print("• UNICODE: Uses Unicode characters and icons")
    print("  Example: 🔹 INCOME STATEMENT, 📊 Gross Profit, ➤ Cash & Cash Equivalents")

def demonstrate_excel_export():
    """Demonstrate Excel export with formatted titles."""
    
    print("\n" + "="*80)
    print("EXCEL EXPORT WITH FORMATTED TITLES")
    print("="*80)
    
    # Create sample data
    sample_df = create_sample_data()
    sourcer = SECFileSourcer()
    
    # Create mock financial model
    financial_model = {
        'annual_income_statement': sample_df.copy(),
        'annual_balance_sheet': sample_df.copy(),
        'annual_cash_flow': sample_df.copy(),
    }
    
    # Create mock sensitivity model
    sensitivity_model = {
        'case_summary': pd.DataFrame({
            'Scenario': ['Base Case', 'Optimistic', 'Pessimistic'],
            'Revenue Growth': ['5%', '10%', '0%'],
            'Margin': ['20%', '25%', '15%']
        })
    }
    
    # Test each formatting style
    styles = ['markers', 'html', 'rich_text', 'unicode']
    
    for style in styles:
        print(f"\nCreating Excel file with {style.upper()} formatting...")
        
        try:
            filepath = sourcer.export_to_excel_with_formatted_titles(
                financial_model=financial_model,
                sensitivity_model=sensitivity_model,
                ticker='DEMO',
                filename=f'demo_text_formatting_{style}.xlsx',
                formatting_style=style,
                progress_callback=lambda msg: print(f"  {msg}")
            )
            
            print(f"✓ Successfully created: {filepath}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    
    print("\n" + "="*80)
    print("Excel files created successfully!")
    print("Check the Storage directory for the generated files.")
    print("Each file contains the same financial data but with different text formatting styles.")

def show_usage_examples():
    """Show usage examples for the new functionality."""
    
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    print("1. Apply formatting to existing DataFrame:")
    print("""
    # Create SECFileSourcer instance
    sourcer = SECFileSourcer()
    
    # Apply formatting to your DataFrame
    formatted_df = sourcer.apply_text_formatting_to_titles(your_df, 'markers')
    
    # Export to Excel
    formatted_df.to_excel('output.xlsx', index=False)
    """)
    
    print("\n2. Export financial model with formatted titles:")
    print("""
    # Export with different formatting styles
    filepath = sourcer.export_to_excel_with_formatted_titles(
        financial_model=your_financial_model,
        sensitivity_model=your_sensitivity_model,
        ticker='AAPL',
        formatting_style='unicode',  # or 'markers', 'html', 'rich_text'
        progress_callback=print
    )
    """)
    
    print("\n3. Available formatting styles:")
    print("   • 'markers': **bold** and *italic* markers")
    print("   • 'html': <b>bold</b> and <i>italic</i> HTML tags")
    print("   • 'rich_text': [BOLD]text[/BOLD] and [ITALIC]text[/ITALIC]")
    print("   • 'unicode': 🔹 📊 ➤ • Unicode characters and icons")

if __name__ == "__main__":
    # Demonstrate formatting styles
    demonstrate_formatting_styles()
    
    # Demonstrate Excel export
    demonstrate_excel_export()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED!")
    print("="*80)
    print("\nThis alternative approach applies formatting directly to the text")
    print("instead of relying on Excel cell-level formatting. This makes the")
    print("formatting visible in any text viewer and more portable across")
    print("different applications and platforms.") 