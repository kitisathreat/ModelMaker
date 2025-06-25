# SEC File Sourcer

A comprehensive Python class for sourcing SEC filings and creating financial models with sensitivity analysis.

## Features

- **SEC Filing Discovery**: Find and download SEC 10-K and 10-Q filings for any publicly traded company
- **Financial Model Creation**: Generate traditional three-statement financial models (Income Statement, Balance Sheet, Cash Flow Statement)
- **Multi-Period Views**: Support for both annual and quarterly financial data
- **Flexible Data Periods**: Specify exactly how many quarters of historical data you want (1-20 quarters)
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

# Create financial model with default 8 quarters (2 years)
financial_model = sourcer.create_financial_model(ticker)

# Create sensitivity analysis
sensitivity_model = sourcer.create_sensitivity_model(financial_model, ticker)

# Export to Excel
excel_file = sourcer.export_to_excel(financial_model, sensitivity_model, ticker)
```

### Specifying Data Periods

The `create_financial_model()` method now accepts a `quarters` parameter to specify how much historical data you want:

```python
# Show available configurations
sourcer.print_quarter_configurations()

# Get predefined configurations
configs = sourcer.get_quarter_configurations()

# Short-term analysis (1 year = 4 quarters)
financial_model = sourcer.create_financial_model(ticker, quarters=4)

# Medium-term analysis (2 years = 8 quarters) - DEFAULT
financial_model = sourcer.create_financial_model(ticker, quarters=8)

# Long-term analysis (3 years = 12 quarters)
financial_model = sourcer.create_financial_model(ticker, quarters=12)

# Extended analysis (4 years = 16 quarters)
financial_model = sourcer.create_financial_model(ticker, quarters=16)

# Custom period (e.g., 6 quarters = 1.5 years)
financial_model = sourcer.create_financial_model(ticker, quarters=6)
```

### Available Quarter Configurations

The system provides predefined configurations for common analysis periods:

- **short_term**: 4 quarters (1 year) - Good for recent performance analysis
- **medium_term**: 8 quarters (2 years) - Balanced view for most analyses (default)
- **long_term**: 12 quarters (3 years) - Good for trend analysis
- **extended**: 16 quarters (4 years) - Comprehensive historical view
- **maximum**: 20 quarters (5 years) - Maximum recommended for performance

### Detailed Example

```python
from Data_Sourcing.sec_file_sourcer import SECFileSourcer

# Initialize
sourcer = SECFileSourcer()

# Show available configurations
sourcer.print_quarter_configurations()

# Step 1: Find SEC filings
ticker = "MSFT"
filings = sourcer.find_sec_filings(ticker, filing_types=['10-K', '10-Q'])

if not filings.empty:
    print("Recent filings found:")
    print(filings[['form', 'filingDate', 'description']].head())
    
    # Step 2: Create comprehensive financial model with 12 quarters (3 years)
    financial_model = sourcer.create_financial_model(ticker, quarters=12)
    
    # Check what data is available
    for sheet_name, df in financial_model.items():
        if not df.empty:
            print(f"{sheet_name}: {len(df)} data points")
    
    # Step 3: Create sensitivity analysis
    sensitivity_model = sourcer.create_sensitivity_model(financial_model, ticker, quarters=12)
    
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

### `create_financial_model(ticker, quarters=8)`
- **Purpose**: Create traditional three-statement financial model
- **Returns**: Dictionary containing 6 DataFrames for annual and quarterly views
- **Parameters**:
  - `ticker` (str): Stock ticker symbol
  - `quarters` (int): Number of quarters of data to retrieve (default: 8, which is 2 years)

### `create_sensitivity_model(financial_model, ticker, quarters=8)`
- **Purpose**: Create sensitivity analysis with operating leverage impacts
- **Returns**: Dictionary containing case summary, financial model, and KPI summary
- **Parameters**:
  - `financial_model` (Dict): Base financial model
  - `ticker` (str): Stock ticker symbol
  - `quarters` (int): Number of quarters used in the financial model (for reference)

### `get_quarter_configurations()`
- **Purpose**: Get predefined quarter configurations for different analysis periods
- **Returns**: Dictionary of configuration names to their details

### `print_quarter_configurations()`
- **Purpose**: Print available quarter configurations with descriptions

### `export_to_excel(financial_model, sensitivity_model, ticker, filename=None)`
- **Purpose**: Export all models to Excel file with multiple sheets
- **Returns**: Path to the exported Excel file

## How Quarter Calculation Works

The system automatically calculates how many 10-K and 10-Q filings to process based on your quarters parameter:

- **Years needed** = `max(1, (quarters + 3) // 4)` (rounds up to get full years)
- **10-K filings** = Years needed (one 10-K per year)
- **10-Q filings** = Quarters specified (one 10-Q per quarter)

Examples:
- 4 quarters → 1 year → 1 10-K, 4 10-Q filings
- 8 quarters → 2 years → 2 10-K, 8 10-Q filings  
- 12 quarters → 3 years → 3 10-K, 12 10-Q filings
- 16 quarters → 4 years → 4 10-K, 16 10-Q filings

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
- Processing time increases with more quarters of data

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the functionality of the SEC File Sourcer.

## License

This project is open source and available under the MIT License. 