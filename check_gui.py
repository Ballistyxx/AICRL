#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test basic imports
try:
    import tkinter as tk
    print("✓ tkinter available")
except ImportError:
    print("✗ tkinter not available")

try:
    from modules.schematic_import import SchematicImportModule
    print("✓ SchematicImportModule imported")
except Exception as e:
    print(f"✗ SchematicImportModule error: {e}")

try:
    from modules.training_runner import TrainingRunnerModule
    print("✓ TrainingRunnerModule imported")
except Exception as e:
    print(f"✗ TrainingRunnerModule error: {e}")

try:
    from modules.magic_export import MagicExportModule
    print("✓ MagicExportModule imported")
except Exception as e:
    print(f"✗ MagicExportModule error: {e}")

try:
    from gui_main import AICRLMainGUI
    print("✓ AICRLMainGUI imported")
except Exception as e:
    print(f"✗ AICRLMainGUI error: {e}")

print("Import test complete.")