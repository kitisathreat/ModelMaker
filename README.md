# SEC File Sourcer

A comprehensive Python class for sourcing SEC filings and creating financial models with sensitivity analysis.

## Features

- **SEC Filing Discovery**: Find and download SEC 10-K and 10-Q filings for any publicly traded company
- **Financial Model Creation**: Generate traditional three-statement financial models (Income Statement, Balance Sheet, Cash Flow Statement)
- **Multi-Period Views**: Support for both annual and quarterly financial data
- **Sensitivity Analysis**: Create operating leverage scenarios and impact analysis
- **KPI Summary**: Generate key performance indicators and financial ratios
- **Excel Export**: Export all models to Excel with multiple sheets

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from Data_Sourcing.sec_file_sourcer import SECFileSourcer

# Initialize the sourcer
sourcer = SECFileSourcer()

# Find SEC filings for a company
ticker = "AAPL"
filings = sourcer.find_sec_filings(ticker)
print(f"Found {len(filings)} filings")

# Create financial model
financial_model = sourcer.create_financial_model(ticker)

# Create sensitivity analysis
sensitivity_model = sourcer.create_sensitivity_model(financial_model, ticker)

# Export to Excel
excel_file = sourcer.export_to_excel(financial_model, sensitivity_model, ticker)
```

### Detailed Example

```python
from Data_Sourcing.sec_file_sourcer import SECFileSourcer

# Initialize
sourcer = SECFileSourcer()

# Step 1: Find SEC filings
ticker = "MSFT"
filings = sourcer.find_sec_filings(ticker, filing_types=['10-K', '10-Q'])

if not filings.empty:
    print("Recent filings found:")
    print(filings[['form', 'filingDate', 'description']].head())
    
    # Step 2: Create comprehensive financial model
    financial_model = sourcer.create_financial_model(ticker)
    
    # Check what data is available
    for sheet_name, df in financial_model.items():
        if not df.empty:
            print(f"{sheet_name}: {len(df)} data points")
    
    # Step 3: Create sensitivity analysis
    sensitivity_model = sourcer.create_sensitivity_model(financial_model, ticker)
    
    # View case summary (operating leverage scenarios)
    case_summary = sensitivity_model['case_summary']
    if not case_summary.empty:
        print("\nOperating Leverage Scenarios:")
        print(case_summary[['Scenario', 'Revenue', 'Operating Income', 'Operating Leverage']])
    
    # View KPI summary
    kpi_summary = sensitivity_model['kpi_summary']
    if not kpi_summary.empty:
        print("\nKey Performance Indicators:")
        print(kpi_summary)
    
    # Step 4: Export to Excel
    excel_file = sourcer.export_to_excel(financial_model, sensitivity_model, ticker)
    print(f"\nModel exported to: {excel_file}")
```

## Class Methods

### `find_sec_filings(ticker, filing_types=['10-K', '10-Q'])`
- **Purpose**: Find SEC filings for a given stock ticker
- **Returns**: DataFrame with filing information sorted by date
- **Parameters**:
  - `ticker` (str): Stock ticker symbol
  - `filing_types` (List[str]): Types of filings to search for

### `create_financial_model(ticker)`
- **Purpose**: Create traditional three-statement financial model
- **Returns**: Dictionary containing 6 DataFrames for annual and quarterly views

### `create_sensitivity_model(financial_model, ticker)`
- **Purpose**: Create sensitivity analysis with operating leverage impacts
- **Returns**: Dictionary containing case summary, financial model, and KPI summary

### `export_to_excel(financial_model, sensitivity_model, ticker, filename=None)`
- **Purpose**: Export all models to Excel file with multiple sheets
- **Returns**: Path to the exported Excel file

## Excel Output Structure

The exported Excel file contains sheets for:
- Annual and quarterly financial statements
- Case summary with operating leverage scenarios
- Enhanced financial model with historical and forecasted data
- KPI summary with key performance indicators

## Sensitivity Analysis Features

Includes scenarios for:
- Base Case
- Optimistic (+20%)
- Pessimistic (-20%)
- High Growth (+50%)
- Recession (-30%)

Each scenario calculates revenue impact, cost changes, operating income impact, and operating leverage ratios.

## Key Performance Indicators

The KPI summary includes:

- **Profitability Ratios**: Net profit margin, operating margin, gross margin
- **Efficiency Ratios**: Return on assets (ROA), return on equity (ROE)
- **Liquidity Ratios**: Current ratio
- **Cash Flow Ratios**: Operating cash flow to capital expenditures
- **Growth Metrics**: Year-over-year revenue growth

## Data Sources

- **SEC EDGAR**: Primary source for financial filings
- **SEC API**: Company facts and submissions data
- **XBRL Data**: Structured financial data from SEC filings

## Error Handling

The class includes comprehensive error handling for:
- Network connectivity issues
- Invalid ticker symbols
- Missing or incomplete data
- API rate limiting
- File export errors

## Limitations

- Requires internet connection to access SEC data
- Data availability depends on SEC filing requirements
- Some companies may have limited historical data
- API rate limits may apply for high-frequency usage

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the functionality of the SEC File Sourcer.

## License

This project is open source and available under the MIT License. 