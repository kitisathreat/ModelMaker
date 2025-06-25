# Stock Analyzer GUI

A PyQt5-based desktop application for financial analysis that provides a user-friendly interface to analyze stock data using SEC filings.

## Features

- **Stock Ticker Input**: Enter any stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
- **Real-time Analysis**: Runs SEC data sourcing functions in the background
- **Spreadsheet-like Interface**: View results in organized tabs with sortable tables
- **Multiple Data Views**: 
  - Annual Financial Statements (Income Statement, Balance Sheet, Cash Flow)
  - Quarterly Financial Statements
  - Operating Leverage Scenarios
  - Key Performance Indicators (KPIs)
  - Enhanced Financial Model with forecasts
- **Progress Tracking**: Real-time progress updates during analysis
- **Error Handling**: User-friendly error messages and validation

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Make sure you have PyQt5 installed:
   ```bash
   pip install PyQt5
   ```

## Usage

### Running the GUI

From the project root directory:
```bash
python run_gui.py
```

Or directly from the Visualization directory:
```bash
cd Visualization
python stock_analyzer_gui.py
```

### Using the Application

1. **Enter Stock Ticker**: Type a valid stock ticker symbol in the input field
2. **Click "Analyze Stock"**: The application will start fetching and processing data
3. **Monitor Progress**: Watch the progress bar and status messages
4. **View Results**: Once complete, results will appear in tabs at the bottom
5. **Navigate Tabs**: Click on different tabs to view various financial data sheets

### Available Data Tabs

- **Annual Income Statement**: Revenue, costs, and profitability metrics
- **Annual Balance Sheet**: Assets, liabilities, and equity
- **Annual Cash Flow**: Operating, investing, and financing cash flows
- **Quarterly Statements**: Same as annual but for quarterly periods
- **Scenarios**: Operating leverage analysis with different scenarios
- **KPIs**: Key performance indicators and ratios
- **Financial Model**: Historical data with forecasted projections
- **Summary**: Overview of available data and analysis metadata

## Technical Details

### Architecture

- **Main Window**: `StockAnalyzerGUI` - Main application window
- **Worker Thread**: `AnalysisWorker` - Background processing to prevent UI freezing
- **Spreadsheet Widget**: `SpreadsheetTab` - Reusable table widget for data display
- **Integration**: Uses the existing `SECFileSourcer` class from the Data Sourcing module

### Data Flow

1. User inputs stock ticker
2. Worker thread fetches SEC filings
3. Financial model is created from SEC data
4. Sensitivity analysis is performed
5. Results are displayed in organized tabs

### Error Handling

- Invalid ticker symbols
- Network connectivity issues
- Missing or incomplete data
- SEC API rate limiting

## Troubleshooting

### Common Issues

1. **PyQt5 not installed**:
   ```bash
   pip install PyQt5
   ```

2. **Import errors**: Make sure you're running from the correct directory

3. **No data found**: Verify the stock ticker symbol is correct

4. **Network errors**: Check your internet connection and SEC API availability

### Performance Notes

- Analysis may take 30-60 seconds depending on data availability
- Large datasets are automatically formatted for readability (K, M suffixes)
- Tables are sortable by clicking column headers

## Development

The GUI is built using PyQt5 and follows standard Qt patterns:

- Signal/slot connections for thread communication
- Proper separation of UI and business logic
- Responsive design with progress indicators
- Modern styling with CSS-like stylesheets

## Dependencies

- PyQt5 >= 5.15.0
- pandas >= 1.5.0
- requests >= 2.28.0
- openpyxl >= 3.0.0
- numpy >= 1.21.0 