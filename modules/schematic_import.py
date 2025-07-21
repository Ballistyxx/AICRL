"""
Schematic Import Module for AICRL GUI.

This module handles the import and parsing of schematic files for IC layout generation.
Supports various schematic formats and provides validation and preprocessing capabilities.

Author: AICRL Team
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Any


class SchematicImportModule:
    """Module for handling schematic file imports."""
    
    def __init__(self):
        """Initialize the schematic import module."""
        self.current_file = None
        self.schematic_data = None
        self.supported_formats = [
            ("SPICE Netlist", "*.sp *.spice *.cir"),
            ("Verilog", "*.v *.vh"),
            ("JSON Schematic", "*.json"),
            ("All Files", "*.*")
        ]
        
    def create_gui(self, parent_frame):
        """Create the GUI components for the schematic import module."""
        # Configure parent frame
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(0, weight=1)
        
        # Main content frame
        content_frame = ttk.Frame(parent_frame)
        content_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(3, weight=1)
        
        # File selection section
        self.create_file_selection_section(content_frame)
        
        # File info section
        self.create_file_info_section(content_frame)
        
        # Format options section
        self.create_format_options_section(content_frame)
        
        # Preview section
        self.create_preview_section(content_frame)
        
    def create_file_selection_section(self, parent):
        """Create the file selection section."""
        # File selection frame
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # Browse button
        ttk.Button(file_frame, text="Browse...", 
                  command=self.browse_file).grid(row=0, column=0, padx=(0, 10))
        
        # File path display
        self.file_path_var = tk.StringVar(value="No file selected")
        self.file_path_label = ttk.Label(file_frame, textvariable=self.file_path_var,
                                        foreground='#7F8C8D', font=('Arial', 9))
        self.file_path_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
    def create_file_info_section(self, parent):
        """Create the file information section."""
        # File info frame
        info_frame = ttk.LabelFrame(parent, text="File Information", padding="10")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        
        # File type
        ttk.Label(info_frame, text="Type:").grid(row=0, column=0, sticky=(tk.W,), padx=(0, 10))
        self.file_type_var = tk.StringVar(value="Unknown")
        ttk.Label(info_frame, textvariable=self.file_type_var).grid(row=0, column=1, sticky=(tk.W,))
        
        # File size
        ttk.Label(info_frame, text="Size:").grid(row=1, column=0, sticky=(tk.W,), padx=(0, 10))
        self.file_size_var = tk.StringVar(value="Unknown")
        ttk.Label(info_frame, textvariable=self.file_size_var).grid(row=1, column=1, sticky=(tk.W,))
        
        # Component count
        ttk.Label(info_frame, text="Components:").grid(row=2, column=0, sticky=(tk.W,), padx=(0, 10))
        self.component_count_var = tk.StringVar(value="Unknown")
        ttk.Label(info_frame, textvariable=self.component_count_var).grid(row=2, column=1, sticky=(tk.W,))
        
    def create_format_options_section(self, parent):
        """Create the format options section."""
        # Format options frame
        options_frame = ttk.LabelFrame(parent, text="Import Options", padding="10")
        options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Auto-detect format
        self.auto_detect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Auto-detect file format",
                       variable=self.auto_detect_var).grid(row=0, column=0, sticky=(tk.W,))
        
        # Validate components
        self.validate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Validate component compatibility",
                       variable=self.validate_var).grid(row=1, column=0, sticky=(tk.W,))
        
        # Preprocessing
        self.preprocess_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Apply preprocessing optimizations",
                       variable=self.preprocess_var).grid(row=2, column=0, sticky=(tk.W,))
        
    def create_preview_section(self, parent):
        """Create the preview section."""
        # Preview frame
        preview_frame = ttk.LabelFrame(parent, text="Preview", padding="10")
        preview_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Preview text widget with scrollbar
        text_frame = ttk.Frame(preview_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.preview_text = tk.Text(text_frame, height=8, wrap=tk.WORD, 
                                   state=tk.DISABLED, font=('Courier', 9))
        self.preview_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for preview
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.preview_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.preview_text.configure(yscrollcommand=scrollbar.set)
        
        # Initially show placeholder text
        self.update_preview("No file selected for preview.")
        
    def browse_file(self):
        """Open file browser dialog to select schematic file."""
        filename = filedialog.askopenfilename(
            title="Select Schematic File",
            filetypes=self.supported_formats,
            initialdir=os.getcwd()
        )
        
        if filename:
            self.current_file = filename
            self.file_path_var.set(filename)
            self.analyze_file()
            
    def analyze_file(self):
        """Analyze the selected file and update UI."""
        if not self.current_file or not os.path.exists(self.current_file):
            return
            
        try:
            # Get file info
            file_path = Path(self.current_file)
            file_size = file_path.stat().st_size
            
            # Update file type
            file_extension = file_path.suffix.lower()
            if file_extension in ['.sp', '.spice', '.cir']:
                file_type = "SPICE Netlist"
            elif file_extension in ['.v', '.vh']:
                file_type = "Verilog"
            elif file_extension == '.json':
                file_type = "JSON Schematic"
            else:
                file_type = "Unknown"
                
            self.file_type_var.set(file_type)
            self.file_size_var.set(f"{file_size:,} bytes")
            
            # Analyze file content for component count
            component_count = self.count_components()
            self.component_count_var.set(str(component_count))
            
            # Update preview
            self.update_file_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze file: {str(e)}")
            
    def count_components(self) -> int:
        """Count the number of components in the schematic file."""
        if not self.current_file:
            return 0
            
        try:
            with open(self.current_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            file_extension = Path(self.current_file).suffix.lower()
            
            if file_extension in ['.sp', '.spice', '.cir']:
                # Count SPICE components (lines starting with component identifiers)
                lines = content.split('\n')
                count = 0
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith(('*', '.', ';')):
                        # Check if line starts with component identifier
                        if line[0].upper() in 'RCQLMXD':
                            count += 1
                return count
                
            elif file_extension in ['.v', '.vh']:
                # Count Verilog module instances
                import re
                # Simple pattern for module instantiation
                pattern = r'^\s*\w+\s+\w+\s*\('
                matches = re.findall(pattern, content, re.MULTILINE)
                return len(matches)
                
            elif file_extension == '.json':
                # Count JSON components
                data = json.loads(content)
                if isinstance(data, dict) and 'components' in data:
                    return len(data['components'])
                    
        except Exception:
            pass
            
        return 0
        
    def update_file_preview(self):
        """Update the preview text with file content."""
        if not self.current_file:
            self.update_preview("No file selected for preview.")
            return
            
        try:
            with open(self.current_file, 'r', encoding='utf-8') as f:
                # Read first 1000 characters for preview
                content = f.read(1000)
                if len(content) == 1000:
                    content += "\n\n... [file truncated for preview]"
                    
            self.update_preview(content)
            
        except Exception as e:
            self.update_preview(f"Error reading file: {str(e)}")
            
    def update_preview(self, text: str):
        """Update the preview text widget."""
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(1.0, text)
        self.preview_text.config(state=tk.DISABLED)
        
    def import_schematic(self) -> bool:
        """Import the selected schematic file."""
        if not self.current_file:
            messagebox.showwarning("Warning", "Please select a schematic file first.")
            return False
            
        try:
            # Parse the schematic file
            self.schematic_data = self.parse_schematic_file()
            
            if self.schematic_data is None:
                messagebox.showerror("Error", "Failed to parse schematic file.")
                return False
                
            # Validate if requested
            if self.validate_var.get():
                if not self.validate_schematic():
                    messagebox.showerror("Error", "Schematic validation failed.")
                    return False
                    
            # Preprocess if requested
            if self.preprocess_var.get():
                self.preprocess_schematic()
                
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import schematic: {str(e)}")
            return False
            
    def parse_schematic_file(self) -> Optional[Dict[str, Any]]:
        """Parse the schematic file based on its format."""
        if not self.current_file:
            return None
            
        file_extension = Path(self.current_file).suffix.lower()
        
        try:
            if file_extension in ['.sp', '.spice', '.cir']:
                return self.parse_spice_file()
            elif file_extension in ['.v', '.vh']:
                return self.parse_verilog_file()
            elif file_extension == '.json':
                return self.parse_json_file()
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            print(f"Error parsing schematic: {e}")
            return None
            
    def parse_spice_file(self) -> Dict[str, Any]:
        """Parse a SPICE netlist file."""
        # Placeholder implementation
        return {
            "format": "spice",
            "file": self.current_file,
            "components": [],
            "nets": [],
            "parsed": True
        }
        
    def parse_verilog_file(self) -> Dict[str, Any]:
        """Parse a Verilog file."""
        # Placeholder implementation
        return {
            "format": "verilog",
            "file": self.current_file,
            "modules": [],
            "instances": [],
            "parsed": True
        }
        
    def parse_json_file(self) -> Dict[str, Any]:
        """Parse a JSON schematic file."""
        with open(self.current_file, 'r') as f:
            data = json.load(f)
        data["format"] = "json"
        data["file"] = self.current_file
        data["parsed"] = True
        return data
        
    def validate_schematic(self) -> bool:
        """Validate the parsed schematic data."""
        if not self.schematic_data:
            return False
            
        # Basic validation - check if we have components
        if self.schematic_data.get("format") == "spice":
            return len(self.schematic_data.get("components", [])) > 0
        elif self.schematic_data.get("format") == "verilog":
            return len(self.schematic_data.get("instances", [])) > 0
        elif self.schematic_data.get("format") == "json":
            return "components" in self.schematic_data
            
        return True
        
    def preprocess_schematic(self):
        """Apply preprocessing optimizations to the schematic data."""
        if not self.schematic_data:
            return
            
        # Placeholder for preprocessing logic
        self.schematic_data["preprocessed"] = True
        
    def get_schematic_data(self) -> Optional[Dict[str, Any]]:
        """Get the current schematic data."""
        return self.schematic_data
        
    def clear_schematic(self):
        """Clear the current schematic data."""
        self.current_file = None
        self.schematic_data = None
        self.file_path_var.set("No file selected")
        self.file_type_var.set("Unknown")
        self.file_size_var.set("Unknown")
        self.component_count_var.set("Unknown")
        self.update_preview("No file selected for preview.")