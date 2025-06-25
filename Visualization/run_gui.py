#!/usr/bin/env python3
"""
Launcher script for the Stock Analyzer GUI application.

This script provides an easy way to run the GUI application from the project root.
"""

import sys
import os

# Add the Visualization directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Visualization'))

def main():
    """Launch the GUI application."""
    try:
        from stock_analyzer_gui import main as gui_main
        print("Starting Stock Analyzer GUI...")
        gui_main()
    except ImportError as e:
        print(f"Error importing GUI module: {e}")
        print("Please make sure PyQt5 is installed: pip install PyQt5")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 