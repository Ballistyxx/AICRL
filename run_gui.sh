#!/bin/bash

# AICRL GUI Runner Script
# This script ensures the GUI runs with the proper virtual environment

echo "🚀 Starting AICRL GUI with Virtual Environment Support"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "gui_main.py" ]; then
    echo "❌ Error: gui_main.py not found!"
    echo "   Please run this script from the AICRL directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "AICRL" ]; then
    echo "❌ Error: AICRL virtual environment not found!"
    echo "   Please ensure the virtual environment is properly set up"
    exit 1
fi

# Run the test first to verify everything is working
echo "🔍 Running pre-flight checks..."
python3 test_complete_fix.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All checks passed! Launching GUI..."
    echo ""
    
    # Launch the GUI
    python3 gui_main.py
else
    echo ""
    echo "❌ Pre-flight checks failed!"
    echo "   Please fix the issues above before running the GUI"
    exit 1
fi

echo ""
echo "🏁 GUI session completed"