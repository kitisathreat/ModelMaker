#!/usr/bin/env python3
"""
Stock Analyzer GUI - A PyQt-based desktop application for financial analysis.

This application provides a user-friendly interface to:
1. Input stock ticker symbols
2. Run SEC data sourcing functions
3. Display results in a spreadsheet-like interface with tabs
4. View financial models and sensitivity analysis
"""

import sys
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

# Add the Data Sourcing directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Data Sourcing'))

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem,
                             QProgressBar, QMessageBox, QSplitter, QFrame,
                             QHeaderView, QComboBox, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

from sec_file_sourcer import SECFileSourcer

class AnalysisWorker(QThread):
    """Worker thread for running financial analysis in the background."""
    
    progress_updated = pyqtSignal(str)
    analysis_complete = pyqtSignal(dict, dict, str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, ticker: str):
        super().__init__()
        self.ticker = ticker
        self.sourcer = SECFileSourcer()
    
    def run(self):
        """Run the financial analysis."""
        try:
            # Step 1: Find SEC filings
            self.progress_updated.emit("Finding SEC filings...")
            filings = self.sourcer.find_sec_filings(self.ticker, filing_types=['10-K', '10-Q'])
            
            if filings.empty:
                self.error_occurred.emit(f"No filings found for {self.ticker}. Please check the ticker symbol.")
                return
            
            self.progress_updated.emit(f"Found {len(filings)} filings. Creating financial model...")
            
            # Step 2: Create financial model
            financial_model = self.sourcer.create_financial_model(self.ticker)
            
            self.progress_updated.emit("Creating sensitivity analysis...")
            
            # Step 3: Create sensitivity analysis
            sensitivity_model = self.sourcer.create_sensitivity_model(financial_model, self.ticker)
            
            self.progress_updated.emit("Analysis complete!")
            
            # Emit results
            self.analysis_complete.emit(financial_model, sensitivity_model, self.ticker)
            
        except Exception as e:
            self.error_occurred.emit(f"Error during analysis: {str(e)}")

class SpreadsheetTab(QWidget):
    """A spreadsheet-like widget for displaying financial data."""
    
    def __init__(self, data: pd.DataFrame, title: str):
        super().__init__()
        self.data = data
        self.title = title
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the spreadsheet interface."""
        layout = QVBoxLayout()
        
        # Title label
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Table widget
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        
        # Populate table
        self.populate_table()
        
        layout.addWidget(self.table)
        self.setLayout(layout)
    
    def populate_table(self):
        """Populate the table with data."""
        if self.data.empty:
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("No data available"))
            return
        
        # Set table dimensions
        self.table.setRowCount(len(self.data))
        self.table.setColumnCount(len(self.data.columns))
        
        # Set headers
        self.table.setHorizontalHeaderLabels(self.data.columns)
        self.table.setVerticalHeaderLabels(self.data.index.astype(str))
        
        # Populate data
        for i, row in enumerate(self.data.itertuples()):
            for j, value in enumerate(row[1:]):  # Skip index
                if pd.isna(value):
                    item = QTableWidgetItem("")
                else:
                    # Format numbers appropriately
                    if isinstance(value, (int, float)):
                        if abs(value) >= 1e6:
                            formatted_value = f"{value/1e6:.2f}M"
                        elif abs(value) >= 1e3:
                            formatted_value = f"{value/1e3:.2f}K"
                        else:
                            formatted_value = f"{value:.2f}"
                        item = QTableWidgetItem(formatted_value)
                    else:
                        item = QTableWidgetItem(str(value))
                
                self.table.setItem(i, j, item)
        
        # Resize columns to content
        self.table.resizeColumnsToContents()
        
        # Make headers bold
        header_font = QFont()
        header_font.setBold(True)
        self.table.horizontalHeader().setFont(header_font)
        self.table.verticalHeader().setFont(header_font)

class StockAnalyzerGUI(QMainWindow):
    """Main application window for the Stock Analyzer."""
    
    def __init__(self):
        super().__init__()
        self.financial_model = {}
        self.sensitivity_model = {}
        self.current_ticker = ""
        self.setup_ui()
        self.setup_styles()
    
    def setup_ui(self):
        """Setup the main user interface."""
        self.setWindowTitle("Stock Analyzer - Financial Model Generator")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Input section
        self.create_input_section(main_layout)
        
        # Progress section
        self.create_progress_section(main_layout)
        
        # Results section
        self.create_results_section(main_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready to analyze stocks")
    
    def create_input_section(self, parent_layout):
        """Create the input section for stock ticker."""
        input_group = QGroupBox("Stock Analysis Input")
        input_layout = QGridLayout()
        
        # Ticker input
        ticker_label = QLabel("Stock Ticker:")
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter stock ticker (e.g., AAPL, MSFT, GOOGL)")
        self.ticker_input.returnPressed.connect(self.start_analysis)
        
        # Analysis button
        self.analyze_button = QPushButton("Analyze Stock")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setMinimumHeight(40)
        
        # Add to layout
        input_layout.addWidget(ticker_label, 0, 0)
        input_layout.addWidget(self.ticker_input, 0, 1)
        input_layout.addWidget(self.analyze_button, 0, 2)
        
        input_group.setLayout(input_layout)
        parent_layout.addWidget(input_group)
    
    def create_progress_section(self, parent_layout):
        """Create the progress section."""
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        self.status_text.setVisible(False)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_text)
        
        progress_group.setLayout(progress_layout)
        parent_layout.addWidget(progress_group)
    
    def create_results_section(self, parent_layout):
        """Create the results section with tabs."""
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        # Tab widget for results
        self.results_tabs = QTabWidget()
        self.results_tabs.setVisible(False)
        
        results_layout.addWidget(self.results_tabs)
        
        results_group.setLayout(results_layout)
        parent_layout.addWidget(results_group)
    
    def setup_styles(self):
        """Setup application styles."""
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLineEdit {
                padding: 8px;
                border: 2px solid #cccccc;
                border-radius: 4px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
            QTableWidget {
                gridline-color: #cccccc;
                background-color: white;
                alternate-background-color: #f9f9f9;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 4px;
                border: 1px solid #cccccc;
                font-weight: bold;
            }
        """)
    
    def start_analysis(self):
        """Start the financial analysis."""
        ticker = self.ticker_input.text().strip().upper()
        
        if not ticker:
            QMessageBox.warning(self, "Input Error", "Please enter a stock ticker symbol.")
            return
        
        # Disable input during analysis
        self.ticker_input.setEnabled(False)
        self.analyze_button.setEnabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_text.setVisible(True)
        self.status_text.clear()
        
        # Create and start worker thread
        self.worker = AnalysisWorker(ticker)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.analysis_complete.connect(self.handle_analysis_complete)
        self.worker.error_occurred.connect(self.handle_analysis_error)
        self.worker.start()
    
    def update_progress(self, message: str):
        """Update progress message."""
        self.status_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.statusBar().showMessage(message)
    
    def handle_analysis_complete(self, financial_model: Dict, sensitivity_model: Dict, ticker: str):
        """Handle completed analysis."""
        self.financial_model = financial_model
        self.sensitivity_model = sensitivity_model
        self.current_ticker = ticker
        
        # Re-enable input
        self.ticker_input.setEnabled(True)
        self.analyze_button.setEnabled(True)
        
        # Hide progress
        self.progress_bar.setVisible(False)
        
        # Display results
        self.display_results()
        
        self.statusBar().showMessage(f"Analysis complete for {ticker}")
    
    def handle_analysis_error(self, error_message: str):
        """Handle analysis error."""
        # Re-enable input
        self.ticker_input.setEnabled(True)
        self.analyze_button.setEnabled(True)
        
        # Hide progress
        self.progress_bar.setVisible(False)
        
        # Show error message
        QMessageBox.critical(self, "Analysis Error", error_message)
        self.statusBar().showMessage("Analysis failed")
    
    def display_results(self):
        """Display analysis results in tabs."""
        self.results_tabs.clear()
        self.results_tabs.setVisible(True)
        
        # Add financial model tabs
        self.add_financial_model_tabs()
        
        # Add sensitivity analysis tabs
        self.add_sensitivity_analysis_tabs()
        
        # Add summary tab
        self.add_summary_tab()
    
    def add_financial_model_tabs(self):
        """Add tabs for financial model data."""
        # Annual statements
        annual_tabs = [
            ("Annual Income Statement", self.financial_model.get('annual_income_statement', pd.DataFrame())),
            ("Annual Balance Sheet", self.financial_model.get('annual_balance_sheet', pd.DataFrame())),
            ("Annual Cash Flow", self.financial_model.get('annual_cash_flow', pd.DataFrame()))
        ]
        
        for title, data in annual_tabs:
            if not data.empty:
                tab = SpreadsheetTab(data, title)
                self.results_tabs.addTab(tab, title)
        
        # Quarterly statements
        quarterly_tabs = [
            ("Quarterly Income Statement", self.financial_model.get('quarterly_income_statement', pd.DataFrame())),
            ("Quarterly Balance Sheet", self.financial_model.get('quarterly_balance_sheet', pd.DataFrame())),
            ("Quarterly Cash Flow", self.financial_model.get('quarterly_cash_flow', pd.DataFrame()))
        ]
        
        for title, data in quarterly_tabs:
            if not data.empty:
                tab = SpreadsheetTab(data, title)
                self.results_tabs.addTab(tab, title)
    
    def add_sensitivity_analysis_tabs(self):
        """Add tabs for sensitivity analysis."""
        # Case summary
        case_summary = self.sensitivity_model.get('case_summary', pd.DataFrame())
        if not case_summary.empty:
            tab = SpreadsheetTab(case_summary, "Operating Leverage Scenarios")
            self.results_tabs.addTab(tab, "Scenarios")
        
        # KPI summary
        kpi_summary = self.sensitivity_model.get('kpi_summary', pd.DataFrame())
        if not kpi_summary.empty:
            tab = SpreadsheetTab(kpi_summary, "Key Performance Indicators")
            self.results_tabs.addTab(tab, "KPIs")
        
        # Enhanced financial model
        enhanced_model = self.sensitivity_model.get('enhanced_financial_model', pd.DataFrame())
        if not enhanced_model.empty:
            tab = SpreadsheetTab(enhanced_model, "Financial Model (Historical + Forecasted)")
            self.results_tabs.addTab(tab, "Financial Model")
    
    def add_summary_tab(self):
        """Add a summary tab with analysis information."""
        summary_widget = QWidget()
        summary_layout = QVBoxLayout()
        
        # Summary information
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        
        summary_info = f"""
        <h2>Analysis Summary for {self.current_ticker}</h2>
        <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Stock Ticker:</strong> {self.current_ticker}</p>
        
        <h3>Available Data:</h3>
        <ul>
        """
        
        # Count available data
        for sheet_name, df in self.financial_model.items():
            if not df.empty:
                summary_info += f"<li>{sheet_name.replace('_', ' ').title()}: {len(df)} data points</li>"
        
        for sheet_name, df in self.sensitivity_model.items():
            if not df.empty:
                summary_info += f"<li>{sheet_name.replace('_', ' ').title()}: {len(df)} data points</li>"
        
        summary_info += "</ul>"
        
        summary_text.setHtml(summary_info)
        summary_layout.addWidget(summary_text)
        
        summary_widget.setLayout(summary_layout)
        self.results_tabs.addTab(summary_widget, "Summary")

def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Stock Analyzer")
    app.setApplicationVersion("1.0")
    
    # Create and show the main window
    window = StockAnalyzerGUI()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 