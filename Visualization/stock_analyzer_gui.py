#!/usr/bin/env python3
"""
Stock Analyzer GUI - A PyQt-based desktop application for financial analysis.

This application provides a user-friendly interface to:
1. Input stock ticker symbols with autocomplete from S&P 500 companies
2. Configure analysis periods (quarters/years)
3. Run SEC data sourcing functions
4. Display results in a spreadsheet-like interface with tabs
5. View financial models and sensitivity analysis
6. Display Excel files within the GUI
"""

import sys
import os
import pandas as pd
import requests
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import numpy as np
from PyQt5.QtWidgets import QSizePolicy

# Add the Data Sourcing directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Data Sourcing'))

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem,
                             QProgressBar, QMessageBox, QSplitter, QFrame,
                             QHeaderView, QComboBox, QGroupBox, QGridLayout,
                             QListWidget, QListWidgetItem, QScrollArea, QFileDialog,
                             QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QIcon, QPainter, QBrush, QPainterPath, QLinearGradient

from sec_file_sourcer import SECFileSourcer

class SP500Data:
    """Class to manage S&P 500 company data for autocomplete."""
    
    def __init__(self):
        self.companies = []
        self.load_sp500_data()
    
    def load_sp500_data(self):
        """Load S&P 500 company data from a reliable source."""
        try:
            # Try to load from a local cache first
            cache_file = os.path.join(os.path.dirname(__file__), 'sp500_cache.json')
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    self.companies = json.load(f)
                return
            
            # If no cache, try to fetch from a reliable source
            # Using a simple API that provides S&P 500 data
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Parse CSV data
                lines = response.text.strip().split('\n')[1:]  # Skip header
                self.companies = []
                
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        symbol = parts[0].strip()
                        name = parts[1].strip()
                        self.companies.append({
                            'symbol': symbol,
                            'name': name,
                            'display': f"{symbol} - {name}"
                        })
                
                # Cache the data
                with open(cache_file, 'w') as f:
                    json.dump(self.companies, f)
            else:
                # Fallback: Create a basic list of major companies
                self.companies = self._get_fallback_companies()
                
        except Exception as e:
            print(f"Error loading S&P 500 data: {e}")
            self.companies = self._get_fallback_companies()
    
    def _get_fallback_companies(self):
        """Fallback list of major companies if API fails."""
        return [
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'display': 'AAPL - Apple Inc.'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'display': 'MSFT - Microsoft Corporation'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'display': 'GOOGL - Alphabet Inc.'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'display': 'AMZN - Amazon.com Inc.'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'display': 'TSLA - Tesla Inc.'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'display': 'META - Meta Platforms Inc.'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'display': 'NVDA - NVIDIA Corporation'},
            {'symbol': 'BRK.A', 'name': 'Berkshire Hathaway Inc.', 'display': 'BRK.A - Berkshire Hathaway Inc.'},
            {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'display': 'JNJ - Johnson & Johnson'},
            {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.', 'display': 'JPM - JPMorgan Chase & Co.'},
            {'symbol': 'V', 'name': 'Visa Inc.', 'display': 'V - Visa Inc.'},
            {'symbol': 'PG', 'name': 'Procter & Gamble Co.', 'display': 'PG - Procter & Gamble Co.'},
            {'symbol': 'UNH', 'name': 'UnitedHealth Group Inc.', 'display': 'UNH - UnitedHealth Group Inc.'},
            {'symbol': 'HD', 'name': 'Home Depot Inc.', 'display': 'HD - Home Depot Inc.'},
            {'symbol': 'MA', 'name': 'Mastercard Inc.', 'display': 'MA - Mastercard Inc.'},
        ]
    
    def search_companies(self, query: str) -> List[Dict]:
        """Search companies by symbol or name."""
        if not query:
            return []
        
        query = query.upper()
        results = []
        
        for company in self.companies:
            if (query in company['symbol'].upper() or 
                query in company['name'].upper()):
                results.append(company)
        
        return results[:10]  # Limit to 10 results

class AutocompleteLineEdit(QLineEdit):
    """Enhanced line edit with autocomplete functionality."""
    
    def __init__(self, sp500_data: SP500Data, parent=None):
        super().__init__(parent)
        self.sp500_data = sp500_data
        self.suggestions = []
        self.current_suggestion_index = -1
        
        # Create suggestion list
        self.suggestion_list = QListWidget()
        self.suggestion_list.setVisible(False)
        self.suggestion_list.setMaximumHeight(200)
        self.suggestion_list.itemClicked.connect(self.select_suggestion)
        
        # Connect signals
        self.textChanged.connect(self.on_text_changed)
        
        # Timer for debouncing
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.perform_search)
    
    def on_text_changed(self, text):
        """Handle text changes and trigger search."""
        self.search_timer.stop()
        self.search_timer.start(300)  # 300ms delay
    
    def perform_search(self):
        """Perform the actual search."""
        query = self.text().strip()
        self.suggestions = self.sp500_data.search_companies(query)
        self.show_suggestions()
    
    def show_suggestions(self):
        """Show suggestion list."""
        self.suggestion_list.clear()
        
        if not self.suggestions:
            self.suggestion_list.setVisible(False)
            return
        
        for company in self.suggestions:
            item = QListWidgetItem(company['display'])
            item.setData(Qt.UserRole, company['symbol'])
            self.suggestion_list.addItem(item)
        
        self.suggestion_list.setVisible(True)
        self.suggestion_list.setCurrentRow(0)
    
    def select_suggestion(self, item):
        """Select a suggestion from the list."""
        symbol = item.data(Qt.UserRole)
        self.setText(symbol)
        self.suggestion_list.setVisible(False)
        self.clearFocus()
    
    def keyPressEvent(self, event):
        """Handle key press events for navigation."""
        if self.suggestion_list.isVisible():
            if event.key() == Qt.Key_Down:
                current_row = self.suggestion_list.currentRow()
                if current_row < self.suggestion_list.count() - 1:
                    self.suggestion_list.setCurrentRow(current_row + 1)
                return
            elif event.key() == Qt.Key_Up:
                current_row = self.suggestion_list.currentRow()
                if current_row > 0:
                    self.suggestion_list.setCurrentRow(current_row - 1)
                return
            elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                current_item = self.suggestion_list.currentItem()
                if current_item:
                    self.select_suggestion(current_item)
                return
            elif event.key() == Qt.Key_Escape:
                self.suggestion_list.setVisible(False)
                return
        
        super().keyPressEvent(event)
    
    def focusOutEvent(self, event):
        """Hide suggestions when focus is lost."""
        super().focusOutEvent(event)
        # Use a timer to delay hiding to allow for clicks
        QTimer.singleShot(100, lambda: self.suggestion_list.setVisible(False))

class ExcelViewer(QWidget):
    """Widget to display Excel files within the GUI."""
    
    def __init__(self, file_path: str = None):
        super().__init__()
        self.file_path = file_path
        self.setup_ui()
        if file_path:
            self.load_excel_file(file_path)
    
    def setup_ui(self):
        """Setup the Excel viewer interface."""
        layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        self.file_label = QLabel("No file loaded")
        self.file_label.setFont(QFont("Arial", 10, QFont.Bold))
        header_layout.addWidget(self.file_label)
        
        # Sheet selector
        self.sheet_combo = QComboBox()
        self.sheet_combo.currentTextChanged.connect(self.on_sheet_changed)
        header_layout.addWidget(QLabel("Sheet:"))
        header_layout.addWidget(self.sheet_combo)
        
        layout.addLayout(header_layout)
        
        # Table widget
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table)
        
        self.setLayout(layout)
    
    def load_excel_file(self, file_path: str):
        """Load an Excel file and display its contents."""
        try:
            self.file_path = file_path
            self.file_label.setText(f"File: {os.path.basename(file_path)}")
            
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            self.sheets = {}
            
            # Load all sheets
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    self.sheets[sheet_name] = df
                except Exception as e:
                    print(f"Error loading sheet {sheet_name}: {e}")
            
            # Populate sheet selector
            self.sheet_combo.clear()
            self.sheet_combo.addItems(self.sheets.keys())
            
            # Display first sheet
            if self.sheets:
                first_sheet = list(self.sheets.keys())[0]
                self.display_sheet(first_sheet)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load Excel file: {str(e)}")
    
    def display_sheet(self, sheet_name: str):
        """Display a specific sheet in the table."""
        if sheet_name not in self.sheets:
            return
        
        df = self.sheets[sheet_name]
        
        # Set table dimensions
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        
        # Set headers
        self.table.setHorizontalHeaderLabels([str(col) for col in df.columns])
        self.table.setVerticalHeaderLabels([str(idx) for idx in df.index])
        
        # Populate data
        for i, row in enumerate(df.itertuples()):
            for j, value in enumerate(row[1:]):  # Skip index
                if pd.isna(value):
                    item = QTableWidgetItem("")
                else:
                    # Format numbers appropriately
                    if isinstance(value, (int, float)):
                        if abs(value) >= 1e9:
                            formatted_value = f"{value/1e9:.2f}B"
                        elif abs(value) >= 1e6:
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
    
    def on_sheet_changed(self, sheet_name: str):
        """Handle sheet selection change."""
        if sheet_name in self.sheets:
            self.display_sheet(sheet_name)

class AnalysisWorker(QThread):
    """Worker thread for running financial analysis in the background."""
    
    progress_updated = pyqtSignal(str)
    analysis_complete = pyqtSignal(dict, dict, str, str)  # Added file_path
    error_occurred = pyqtSignal(str)
    preview_ready = pyqtSignal(str)  # New signal for preview
    
    def __init__(self, ticker: str, quarters: int, schmoove_mode: bool = False, enhanced_fuzzy_matching: bool = True, fast_preview: bool = True):
        super().__init__()
        self.ticker = ticker
        self.quarters = quarters
        self.schmoove_mode = schmoove_mode
        self.enhanced_fuzzy_matching = enhanced_fuzzy_matching
        self.fast_preview = fast_preview
        self.sourcer = SECFileSourcer()
    
    def run(self):
        """Run the financial analysis."""
        try:
            import time
            start_time = time.time()
            
            self.progress_updated.emit(f"Starting financial analysis for {self.ticker}...")
            self.progress_updated.emit(f"Analysis period: {self.quarters} quarters")
            if self.schmoove_mode:
                self.progress_updated.emit("Schmoove mode: ENABLED (enhanced performance)")
            if self.enhanced_fuzzy_matching:
                self.progress_updated.emit("Enhanced fuzzy matching: ENABLED (includes non-GAAP to GAAP mapping)")
            else:
                self.progress_updated.emit("Enhanced fuzzy matching: DISABLED (standard GAAP matching only)")
            
            self.progress_updated.emit("📊 Using standard three-statement financial modeling principles")
            
            # Provide helpful information about what to expect
            if self.quarters > 12:
                self.progress_updated.emit("⚠️  This is a large analysis - it may take several minutes to complete.")
            elif self.quarters > 8:
                self.progress_updated.emit("ℹ️  This analysis will process multiple years of data - please be patient.")
            else:
                self.progress_updated.emit("ℹ️  Analysis in progress - this typically takes 30-60 seconds.")
            
            self.progress_updated.emit("")
            
            # Step 1: Find SEC filings
            step_start = time.time()
            self.progress_updated.emit("Step 1/4: Finding SEC filings...")
            self.progress_updated.emit("  • Converting ticker to CIK number...")
            filings = self.sourcer.find_sec_filings(self.ticker, filing_types=['10-K', '10-Q'])
            
            if filings.empty:
                self.error_occurred.emit(f"No filings found for {self.ticker}. Please check the ticker symbol.")
                return
            
            step_time = time.time() - step_start
            self.progress_updated.emit(f"  ✓ Found {len(filings)} SEC filings (took {step_time:.1f}s)")
            
            # Step 2: Create financial model with specified quarters
            step_start = time.time()
            self.progress_updated.emit("Step 2/4: Creating financial model...")
            self.progress_updated.emit("  • Analyzing filing structure and data availability...")
            
            # Calculate expected processing time based on quarters
            years_needed = max(1, (self.quarters + 3) // 4)
            k_filings_needed = years_needed
            q_filings_needed = min(self.quarters, 20)
            
            self.progress_updated.emit(f"  • Processing {k_filings_needed} 10-K filings and {q_filings_needed} 10-Q filings...")
            self.progress_updated.emit("  • Extracting XBRL financial data...")
            
            # Pass progress callback to create_financial_model
            financial_model = self.sourcer.create_financial_model(
                self.ticker, 
                quarters=self.quarters, 
                progress_callback=self.progress_updated.emit,
                enhanced_fuzzy_matching=self.enhanced_fuzzy_matching
            )
            
            # Check what data was successfully extracted
            data_points = sum(len(df) for df in financial_model.values() if not df.empty)
            step_time = time.time() - step_start
            self.progress_updated.emit(f"  ✓ Financial model created with {data_points} data points (took {step_time:.1f}s)")
            
            # Step 3: Create sensitivity analysis
            step_start = time.time()
            self.progress_updated.emit("Step 3/4: Creating sensitivity analysis...")
            self.progress_updated.emit("  • Analyzing operating leverage scenarios...")
            
            # Pass progress callback to create_sensitivity_model
            sensitivity_model = self.sourcer.create_sensitivity_model(
                financial_model, 
                self.ticker, 
                quarters=self.quarters,
                progress_callback=self.progress_updated.emit
            )
            
            self.progress_updated.emit("  • Generating KPI summary and enhanced financial model...")
            step_time = time.time() - step_start
            self.progress_updated.emit(f"  ✓ Sensitivity analysis completed (took {step_time:.1f}s)")
            
            # Step 4: Export to Excel (preview first, then final)
            step_start = time.time()
            self.progress_updated.emit("Step 4/4: Exporting to Excel...")
            self.progress_updated.emit("  • Creating preview file for immediate display...")
            
            import os
            from datetime import datetime
            preview_filename = f"preview_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            preview_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Storage', preview_filename)
            
            # Pass progress callback to export_to_excel
            if self.fast_preview:
                self.sourcer.export_to_excel_fast_preview(
                    financial_model, 
                    sensitivity_model, 
                    self.ticker, 
                    preview_filename, 
                    progress_callback=self.progress_updated.emit
                )
            else:
                self.sourcer.export_to_excel(
                    financial_model, 
                    sensitivity_model, 
                    self.ticker, 
                    filename=preview_filename, 
                    schmoove_mode=self.schmoove_mode,
                    progress_callback=self.progress_updated.emit
                )
            self.preview_ready.emit(preview_path)
            
            self.progress_updated.emit("  • Creating final Excel file with professional formatting...")
            excel_file = self.sourcer.export_to_excel(
                financial_model, 
                sensitivity_model, 
                self.ticker, 
                schmoove_mode=self.schmoove_mode,
                progress_callback=self.progress_updated.emit
            )
            
            step_time = time.time() - step_start
            self.progress_updated.emit(f"  ✓ Excel export completed (took {step_time:.1f}s)")
            
            total_time = time.time() - start_time
            self.progress_updated.emit("")
            self.progress_updated.emit(f"🎉 Analysis complete! Total time: {total_time:.1f}s")
            self.progress_updated.emit(f"📊 Results available in {len(financial_model)} financial statements and {len(sensitivity_model)} analysis sheets")
            
            # Emit results
            self.analysis_complete.emit(financial_model, sensitivity_model, self.ticker, excel_file)
            
            # Optionally clean up preview file
            try:
                if os.path.exists(preview_path):
                    os.remove(preview_path)
            except Exception:
                pass
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

class WavyLabel(QWidget):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.letters = []
        self.animations = []
        self.base_y = 0
        self.text = text
        
        # Set size policy to allow expansion and prevent shrinking
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        
        layout = QHBoxLayout(self)
        layout.setSpacing(24)  # We'll use fixed width for spacing
        layout.setContentsMargins(0, 10, 10, 10)
        font = QFont("Consolas", 36, QFont.Bold)
        
        for i, char in enumerate(text):
            lbl = QLabel(char)
            lbl.setFont(font)
            lbl.setStyleSheet("color: #4CAF50;")
            lbl.setFixedWidth(36)
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)
            self.letters.append(lbl)
            anim = QPropertyAnimation(lbl, b"pos", self)
            anim.setDuration(1200)
            anim.setLoopCount(-1)
            anim.setEasingCurve(QEasingCurve.InOutSine)
            phase = (i / max(1, len(text)-1))
            anim.setKeyValueAt(0, QPoint(lbl.x(), 0))
            anim.setKeyValueAt(0.2, QPoint(lbl.x(), -6 * (0.5 + 0.5 * np.sin(2 * np.pi * phase))))
            anim.setKeyValueAt(0.5, QPoint(lbl.x(), -12 * (0.5 + 0.5 * np.sin(2 * np.pi * phase + np.pi))))
            anim.setKeyValueAt(0.8, QPoint(lbl.x(), -6 * (0.5 + 0.5 * np.sin(2 * np.pi * phase))))
            anim.setKeyValueAt(1, QPoint(lbl.x(), 0))
            self.animations.append(anim)
        
        # Calculate and set minimum width to ensure all letters fit
        min_width = len(text) * (36 + 24) + 20  # letter width + spacing + margins
        self.setMinimumWidth(min_width)
        self.setFixedHeight(60)  # Set a fixed height to prevent vertical issues
        
        # Force the widget to be at least the minimum width
        self.resize(min_width, 60)
        
        # Add a timer to periodically check size
        self.size_timer = QTimer(self)
        self.size_timer.timeout.connect(self.enforce_minimum_size)
        self.size_timer.start(100)  # Check every 100ms
        
        self.start_wave()
    
    def enforce_minimum_size(self):
        """Enforce minimum size to prevent collapsing."""
        min_width = len(self.text) * (36 + 24) + 20
        if self.width() < min_width:
            self.setFixedWidth(min_width)
    
    def start_wave(self):
        for anim in self.animations:
            anim.start()
    
    def showEvent(self, event):
        """Ensure proper sizing when widget is shown."""
        super().showEvent(event)
        self.updateGeometry()
        self.adjustSize()
    
    def resizeEvent(self, event):
        """Ensure the widget maintains its proper size."""
        super().resizeEvent(event)
        # Force the minimum width
        min_width = len(self.text) * (36 + 24) + 20
        if self.width() < min_width:
            self.setFixedWidth(min_width)

class RedTintOverlay(QWidget):
    """Overlay widget that applies a red tint to the entire window with opacity tied to CPU usage."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cpu_percent = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_cpu_usage)
        self.timer.start(50)  # Update every 100ms
        self.setVisible(False)
        
        # Set up the overlay to cover the entire parent window
        self.setAttribute(Qt.WA_NoSystemBackground)  # Transparent background
        self.setAttribute(Qt.WA_TranslucentBackground)  # Allow transparency
        
        # Enable mouse tracking for exclusion zone
        self.setMouseTracking(True)
        
        # Track mouse position for exclusion zone
        self.mouse_pos = QPoint(0, 0)
        self.exclusion_radius = 100  # 100 pixel radius exclusion zone
        self.mouse_in_window = False  # Track if mouse is in window
        
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None
    
    def update_cpu_usage(self):
        """Update CPU usage percentage."""
        if self.psutil:
            self.cpu_percent = self.psutil.cpu_percent(interval=None)
        else:
            self.cpu_percent = 0
        self.update()
    
    def mouseMoveEvent(self, event):
        """Track mouse position for exclusion zone."""
        if self.mouse_in_window:
            self.mouse_pos = event.pos()
            self.update()  # Redraw to update exclusion zone
        # Pass the event to parent widgets
        event.ignore()
    
    def enterEvent(self, event):
        """Handle mouse entering the window."""
        self.mouse_in_window = True
        self.update()  # Redraw to show exclusion zone
        event.accept()
    
    def leaveEvent(self, event):
        """Handle mouse leaving the window."""
        self.mouse_in_window = False
        self.update()  # Redraw to hide exclusion zone
        event.accept()
    
    def mousePressEvent(self, event):
        """Pass mouse press events through to underlying widgets."""
        event.ignore()
    
    def mouseReleaseEvent(self, event):
        """Pass mouse release events through to underlying widgets."""
        event.ignore()
    
    def mouseDoubleClickEvent(self, event):
        """Pass mouse double-click events through to underlying widgets."""
        event.ignore()
    
    def wheelEvent(self, event):
        """Pass wheel events through to underlying widgets."""
        event.ignore()
    
    def paintEvent(self, event):
        """Paint the red tint overlay with exponential scaling and mouse exclusion zone."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate opacity with exponential scaling
        if self.cpu_percent >= 95:
            # 100% opacity when CPU is 95% or above
            base_opacity = 1.0
        else:
            # Exponential scaling: (cpu_percent/100)^2 for more dramatic effect
            # This creates a curve that starts slow and accelerates
            normalized_cpu = self.cpu_percent / 100.0
            base_opacity = normalized_cpu ** 2  # Exponential scaling
            # Scale to max 0.8 opacity for CPU < 95%
            base_opacity = min(0.8, base_opacity)
        
        # Create exclusion zone around mouse cursor with 100% transparency
        if base_opacity > 0 and self.mouse_in_window:
            # Create a circular path for the exclusion zone
            exclusion_path = QPainterPath()
            exclusion_path.addEllipse(self.mouse_pos, self.exclusion_radius, self.exclusion_radius)
            
            # Create a mask that excludes the circular area
            mask = QPainterPath()
            mask.addRect(self.rect())
            mask = mask.subtracted(exclusion_path)
            
            # Fill the masked area with red tint
            red_tint = QColor(255, 0, 0, int(255 * base_opacity))
            painter.fillPath(mask, red_tint)
        else:
            # Fill the entire widget with the red tint if no exclusion needed
            red_tint = QColor(255, 0, 0, int(255 * base_opacity))
            painter.fillRect(self.rect(), red_tint)
        
        painter.end()
    
    def showEvent(self, event):
        """Handle show event to ensure overlay covers the entire parent window."""
        super().showEvent(event)
        self.resize_to_parent()
    
    def resizeEvent(self, event):
        """Handle resize events to maintain full coverage."""
        super().resizeEvent(event)
        self.resize_to_parent()
    
    def resize_to_parent(self):
        """Resize the overlay to cover the entire parent window."""
        if self.parent():
            parent_rect = self.parent().rect()
            self.setGeometry(parent_rect)

class StockAnalyzerGUI(QMainWindow):
    """Main application window for the Stock Analyzer."""
    
    def __init__(self):
        super().__init__()
        self.financial_model = {}
        self.sensitivity_model = {}
        self.current_ticker = ""
        self.current_excel_file = ""
        self.sp500_data = SP500Data()
        self.setup_ui()
        self.setup_styles()
    
    def setup_ui(self):
        """Setup the main user interface."""
        self.setWindowTitle("Stock Analyzer - Financial Model Generator")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Input section
        self.create_input_section(main_layout)
        
        # Progress section
        self.create_progress_section(main_layout)
        
        # Results section with splitter
        self.create_results_section(main_layout)
        
        # Create red tint overlay as a child of the central widget
        self.red_tint_overlay = RedTintOverlay(central_widget)
        self.red_tint_overlay.raise_()  # Ensure it's on top
        
        # Status bar
        self.statusBar().showMessage("Ready to analyze stocks")
    
    def resizeEvent(self, event):
        """Handle window resize events to ensure red tint overlay covers the entire window."""
        super().resizeEvent(event)
        if hasattr(self, 'red_tint_overlay') and self.red_tint_overlay.isVisible():
            self.red_tint_overlay.resize_to_parent()
    
    def create_input_section(self, parent_layout):
        """Create the input section for stock ticker and period configuration."""
        input_group = QGroupBox("Stock Analysis Configuration")
        input_layout = QGridLayout()
        
        # Stock selection
        ticker_label = QLabel("Stock Selection:")
        self.ticker_input = AutocompleteLineEdit(self.sp500_data)
        self.ticker_input.setPlaceholderText("Enter stock ticker or company name (e.g., AAPL, Apple)")
        
        # Add suggestion list to layout
        input_layout.addWidget(ticker_label, 0, 0)
        input_layout.addWidget(self.ticker_input, 0, 1)
        input_layout.addWidget(self.ticker_input.suggestion_list, 1, 1)
        
        # Period configuration
        period_label = QLabel("Analysis Period:")
        
        # Create horizontal layout for period inputs
        period_layout = QHBoxLayout()
        
        # Years input
        self.years_input = QLineEdit()
        self.years_input.setPlaceholderText("Years")
        self.years_input.setMaximumWidth(80)
        self.years_input.setText("2")  # Default to 2 years
        self.years_input.textChanged.connect(self.on_period_changed)
        
        years_label = QLabel("y")
        years_label.setMaximumWidth(20)
        
        # Quarters input
        self.quarters_input = QLineEdit()
        self.quarters_input.setPlaceholderText("Quarters")
        self.quarters_input.setMaximumWidth(80)
        self.quarters_input.setText("0")  # Default to 0 quarters
        self.quarters_input.textChanged.connect(self.on_period_changed)
        
        quarters_label = QLabel("q")
        quarters_label.setMaximumWidth(20)
        
        # Add to period layout
        period_layout.addWidget(self.years_input)
        period_layout.addWidget(years_label)
        period_layout.addWidget(QLabel("+"))
        period_layout.addWidget(self.quarters_input)
        period_layout.addWidget(quarters_label)
        period_layout.addStretch()  # Add stretch to push warning to the right
        
        input_layout.addWidget(period_label, 2, 0)
        input_layout.addLayout(period_layout, 2, 1)
        
        # Warning label for long periods
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: #ff6b35; font-style: italic; font-size: 10px;")
        self.warning_label.setWordWrap(True)
        input_layout.addWidget(self.warning_label, 3, 1)
        
        # Schmoove Mode Checkbox
        schmoove_layout = QHBoxLayout()
        self.schmoove_checkbox = QCheckBox()
        self.schmoove_checkbox.setChecked(False)
        self.schmoove_label = WavyLabel("Schmoove Mode")
        
        # Add widgets with proper spacing
        schmoove_layout.addWidget(self.schmoove_checkbox)
        schmoove_layout.addSpacing(20)  # Fixed spacing instead of stretch
        schmoove_layout.addWidget(self.schmoove_label, 1)  # Give it stretch factor of 1
        schmoove_layout.addStretch()  # Only one stretch at the end
        
        input_layout.addLayout(schmoove_layout, 5, 0, 1, 2)
        
        # Enhanced Fuzzy Matching Checkbox
        fuzzy_layout = QHBoxLayout()
        self.fuzzy_checkbox = QCheckBox()
        self.fuzzy_checkbox.setChecked(True)  # Default to enabled
        fuzzy_label = QLabel("🔍 Enhanced Fuzzy Matching")
        fuzzy_label.setStyleSheet("font-weight: bold; color: #2E86AB; font-size: 11px;")
        
        # Add tooltip for explanation
        fuzzy_tooltip = ("When enabled, includes non-GAAP to GAAP mapping for better data coverage.\n"
                        "When disabled, uses only standard fuzzy matching for exact GAAP concepts.\n\n"
                        "Enhanced mode may find more data but could include less precise matches.")
        fuzzy_label.setToolTip(fuzzy_tooltip)
        self.fuzzy_checkbox.setToolTip(fuzzy_tooltip)
        
        # Add widgets with proper spacing
        fuzzy_layout.addWidget(self.fuzzy_checkbox)
        fuzzy_layout.addSpacing(20)  # Fixed spacing instead of stretch
        fuzzy_layout.addWidget(fuzzy_label, 1)  # Give it stretch factor of 1
        fuzzy_layout.addStretch()  # Only one stretch at the end
        
        input_layout.addLayout(fuzzy_layout, 6, 0, 1, 2)
        
        # Fast Preview Formatting Checkbox
        fast_preview_layout = QHBoxLayout()
        self.fast_preview_checkbox = QCheckBox()
        self.fast_preview_checkbox.setChecked(True)  # Default to enabled for faster preview
        fast_preview_label = QLabel("⚡ Fast Preview (Skip Formatting)")
        fast_preview_label.setStyleSheet("font-weight: bold; color: #FF6B35; font-size: 11px;")
        
        # Add tooltip for explanation
        fast_preview_tooltip = ("When enabled, creates preview files without professional formatting for faster generation.\n"
                               "When disabled, applies full formatting to preview files (slower but more professional).\n\n"
                               "Final Excel files always include full formatting regardless of this setting.")
        fast_preview_label.setToolTip(fast_preview_tooltip)
        self.fast_preview_checkbox.setToolTip(fast_preview_tooltip)
        
        # Add widgets with proper spacing
        fast_preview_layout.addWidget(self.fast_preview_checkbox)
        fast_preview_layout.addSpacing(20)  # Fixed spacing instead of stretch
        fast_preview_layout.addWidget(fast_preview_label, 1)  # Give it stretch factor of 1
        fast_preview_layout.addStretch()  # Only one stretch at the end
        
        input_layout.addLayout(fast_preview_layout, 7, 0, 1, 2)
        
        # Analysis button
        self.analyze_button = QPushButton("Analyze Stock")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setMinimumHeight(40)
        
        input_layout.addWidget(self.analyze_button, 8, 0, 1, 2)
        
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
        """Create the results section with splitter for tabs and Excel viewer."""
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        # Create splitter for tabs and Excel viewer
        self.results_splitter = QSplitter(Qt.Vertical)
        
        # Tab widget for results
        self.results_tabs = QTabWidget()
        self.results_tabs.setVisible(False)
        
        # Excel viewer
        self.excel_viewer = ExcelViewer()
        self.excel_viewer.setVisible(False)
        
        # Add widgets to splitter
        self.results_splitter.addWidget(self.results_tabs)
        self.results_splitter.addWidget(self.excel_viewer)
        
        # Set initial splitter sizes (70% tabs, 30% Excel viewer)
        self.results_splitter.setSizes([700, 300])
        
        results_layout.addWidget(self.results_splitter)
        
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
            QComboBox {
                padding: 8px;
                border: 2px solid #cccccc;
                border-radius: 4px;
                font-size: 12px;
            }
            QComboBox:focus {
                border-color: #4CAF50;
            }
            QListWidget {
                border: 2px solid #cccccc;
                border-radius: 4px;
                background-color: white;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #4CAF50;
                color: white;
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
    
    def on_period_changed(self):
        """Handle period input changes and show warnings."""
        try:
            years = int(self.years_input.text() or "0")
            quarters = int(self.quarters_input.text() or "0")
            
            total_quarters = years * 4 + quarters
            
            # Clear previous warning
            self.warning_label.setText("")
            
            # Show warning for periods greater than 5 years (20 quarters)
            if total_quarters > 20:
                self.warning_label.setText("⚠️ Warning: Analysis periods greater than 5 years will take a significant amount of time to process.")
            elif total_quarters <= 0:
                self.warning_label.setText("⚠️ Please enter a valid period (at least 1 quarter).")
                self.warning_label.setStyleSheet("color: #ff6b35; font-style: italic; font-size: 10px;")
            else:
                # Show helpful info for reasonable periods
                if total_quarters >= 16:
                    self.warning_label.setText("ℹ️ This will analyze a substantial amount of historical data.")
                    self.warning_label.setStyleSheet("color: #4CAF50; font-style: italic; font-size: 10px;")
                else:
                    self.warning_label.setStyleSheet("color: #ff6b35; font-style: italic; font-size: 10px;")
                    
        except ValueError:
            self.warning_label.setText("⚠️ Please enter valid numbers for years and quarters.")
            self.warning_label.setStyleSheet("color: #ff6b35; font-style: italic; font-size: 10px;")
    
    def get_quarters_from_period(self) -> int:
        """Get the number of quarters from the years and quarters inputs."""
        try:
            years = int(self.years_input.text() or "0")
            quarters = int(self.quarters_input.text() or "0")
            total_quarters = years * 4 + quarters
            
            # Ensure minimum of 1 quarter
            return max(1, total_quarters)
        except ValueError:
            # Default to 8 quarters (2 years) if invalid input
            return 8
    
    def start_analysis(self):
        """Start the financial analysis."""
        ticker = self.ticker_input.text().strip().upper()
        
        if not ticker:
            QMessageBox.warning(self, "Input Error", "Please enter a stock ticker symbol.")
            return
        
        quarters = self.get_quarters_from_period()
        schmoove_mode = self.schmoove_checkbox.isChecked()
        enhanced_fuzzy_matching = self.fuzzy_checkbox.isChecked()
        fast_preview = self.fast_preview_checkbox.isChecked()
        
        # Disable input during analysis
        self.ticker_input.setEnabled(False)
        self.years_input.setEnabled(False)
        self.quarters_input.setEnabled(False)
        self.schmoove_checkbox.setEnabled(False)
        self.fuzzy_checkbox.setEnabled(False)
        self.fast_preview_checkbox.setEnabled(False)
        self.analyze_button.setEnabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_text.setVisible(True)
        self.status_text.clear()
        
        # Show flames if schmoove mode
        self.red_tint_overlay.setVisible(schmoove_mode)
        
        # Create and start worker thread
        self.worker = AnalysisWorker(ticker, quarters, schmoove_mode=schmoove_mode, enhanced_fuzzy_matching=enhanced_fuzzy_matching, fast_preview=fast_preview)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.analysis_complete.connect(self.handle_analysis_complete)
        self.worker.error_occurred.connect(self.handle_analysis_error)
        self.worker.preview_ready.connect(self.handle_preview_ready)  # Connect new signal
        self.worker.start()
    
    def update_progress(self, message: str):
        """Update progress message."""
        self.status_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.statusBar().showMessage(message)
    
    def handle_analysis_complete(self, financial_model: Dict, sensitivity_model: Dict, ticker: str, excel_file: str):
        """Handle completed analysis."""
        self.financial_model = financial_model
        self.sensitivity_model = sensitivity_model
        self.current_ticker = ticker
        self.current_excel_file = excel_file
        
        # Re-enable input
        self.ticker_input.setEnabled(True)
        self.years_input.setEnabled(True)
        self.quarters_input.setEnabled(True)
        self.schmoove_checkbox.setEnabled(True)
        self.fuzzy_checkbox.setEnabled(True)
        self.fast_preview_checkbox.setEnabled(True)
        self.analyze_button.setEnabled(True)
        
        # Hide progress
        self.progress_bar.setVisible(False)
        
        # Hide flames
        self.red_tint_overlay.setVisible(False)
        
        # Display results
        self.display_results()
        
        self.statusBar().showMessage(f"Analysis complete for {ticker}")
    
    def handle_analysis_error(self, error_message: str):
        """Handle analysis error."""
        # Re-enable input
        self.ticker_input.setEnabled(True)
        self.years_input.setEnabled(True)
        self.quarters_input.setEnabled(True)
        self.schmoove_checkbox.setEnabled(True)
        self.fuzzy_checkbox.setEnabled(True)
        self.fast_preview_checkbox.setEnabled(True)
        self.analyze_button.setEnabled(True)
        
        # Hide progress
        self.progress_bar.setVisible(False)
        
        # Hide flames
        self.red_tint_overlay.setVisible(False)
        
        # Show error message
        QMessageBox.critical(self, "Analysis Error", error_message)
        self.statusBar().showMessage("Analysis failed")
    
    def handle_preview_ready(self, preview_path):
        """Handle preview Excel file ready for visualization."""
        if os.path.exists(preview_path):
            self.excel_viewer.load_excel_file(preview_path)
            self.excel_viewer.setVisible(True)
    
    def display_results(self):
        """Display analysis results in tabs and Excel viewer."""
        self.results_tabs.clear()
        self.results_tabs.setVisible(True)
        
        # Add financial model tabs
        self.add_financial_model_tabs()
        
        # Add sensitivity analysis tabs
        self.add_sensitivity_analysis_tabs()
        
        # Add summary tab
        self.add_summary_tab()
        
        # Load Excel file in viewer
        if self.current_excel_file and os.path.exists(self.current_excel_file):
            self.excel_viewer.load_excel_file(self.current_excel_file)
            self.excel_viewer.setVisible(True)
    
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
        enhanced_model = self.sensitivity_model.get('financial_model', pd.DataFrame())
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
        
        quarters = self.get_quarters_from_period()
        years = quarters // 4
        remaining_quarters = quarters % 4
        
        # Format period display
        if years > 0 and remaining_quarters > 0:
            period_display = f"{years} year{'s' if years > 1 else ''} + {remaining_quarters} quarter{'s' if remaining_quarters > 1 else ''}"
        elif years > 0:
            period_display = f"{years} year{'s' if years > 1 else ''}"
        else:
            period_display = f"{remaining_quarters} quarter{'s' if remaining_quarters > 1 else ''}"
        
        summary_info = f"""
        <h2>Analysis Summary for {self.current_ticker}</h2>
        <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Stock Ticker:</strong> {self.current_ticker}</p>
        <p><strong>Analysis Period:</strong> {period_display} ({quarters} quarters total)</p>
        
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
        
        if self.current_excel_file:
            summary_info += f"<p><strong>Excel File:</strong> {os.path.basename(self.current_excel_file)}</p>"
        
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