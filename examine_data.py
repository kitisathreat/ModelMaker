import pandas as pd

# Load the Excel file
xl = pd.ExcelFile('Storage/preview_AAPL_20250625_155802.xlsx')
print("Available sheets:", xl.sheet_names)

# Examine Annual Financial Statements
print("\n=== ANNUAL FINANCIAL STATEMENTS ===")
annual_df = pd.read_excel(xl, 'Annual Financial Statements')
print(f"Shape: {annual_df.shape}")
print(f"Number of columns: {len(annual_df.columns)}")
print(f"First 10 columns: {list(annual_df.columns[:10])}")
print(f"Last 10 columns: {list(annual_df.columns[-10:])}")

# Count non-null values in each column
print("\nNon-null values per column (first 20):")
for i, col in enumerate(annual_df.columns[:20]):
    non_null_count = annual_df[col].notna().sum()
    print(f"{i+1:2d}. {col}: {non_null_count} non-null values")

# Show sample data
print("\nSample data (first 3 rows, first 5 columns):")
print(annual_df.iloc[:3, :5])

# Examine Quarterly Financial Statements
print("\n=== QUARTERLY FINANCIAL STATEMENTS ===")
quarterly_df = pd.read_excel(xl, 'Quarterly Financial Statements')
print(f"Shape: {quarterly_df.shape}")
print(f"Number of columns: {len(quarterly_df.columns)}")
print(f"First 10 columns: {list(quarterly_df.columns[:10])}")

# Count non-null values in quarterly data
print("\nNon-null values per column (first 20):")
for i, col in enumerate(quarterly_df.columns[:20]):
    non_null_count = quarterly_df[col].notna().sum()
    print(f"{i+1:2d}. {col}: {non_null_count} non-null values")

# Show sample quarterly data
print("\nSample quarterly data (first 3 rows, first 5 columns):")
print(quarterly_df.iloc[:3, :5]) 