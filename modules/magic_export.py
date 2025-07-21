"""
Magic VLSI Export Module for AICRL GUI.

This module handles the export of generated IC layouts to Magic VLSI format.
Supports various export options and formats compatible with Magic tools.

Author: AICRL Team
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple


class MagicExportModule:
    """Module for exporting layouts to Magic VLSI format."""
    
    def __init__(self):
        """Initialize the Magic export module."""
        self.current_layout = None
        self.export_settings = {
            "format": "magic",
            "technology": "sky130",
            "scale": "lambda",
            "include_labels": True,
            "include_ports": True,
            "layer_mapping": "default",
            "grid_snap": True,
            "units": "microns"
        }
        
        # Supported export formats
        self.export_formats = [
            ("Magic Layout", "*.mag"),
            ("GDS II", "*.gds"),
            ("LEF", "*.lef"),
            ("DEF", "*.def"),
            ("All Files", "*.*")
        ]
        
        # Technology files
        self.tech_files = [
            "sky130",
            "sky90",
            "gf180mcu",
            "custom"
        ]
        
    def create_gui(self, parent_frame):
        """Create the GUI components for the Magic export module."""
        # Configure parent frame
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(0, weight=1)
        
        # Main content frame
        content_frame = ttk.Frame(parent_frame)
        content_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(3, weight=1)
        
        # Layout source section
        self.create_source_section(content_frame)
        
        # Export format section
        self.create_format_section(content_frame)
        
        # Export options section
        self.create_options_section(content_frame)
        
        # Export status section
        self.create_status_section(content_frame)
        
    def create_source_section(self, parent):
        """Create the layout source selection section."""
        # Source frame
        source_frame = ttk.LabelFrame(parent, text="Layout Source", padding="10")
        source_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        source_frame.columnconfigure(1, weight=1)
        
        # Source type selection
        self.source_type_var = tk.StringVar(value="current")
        ttk.Radiobutton(source_frame, text="Use current generated layout",
                       variable=self.source_type_var, value="current").grid(
                           row=0, column=0, columnspan=2, sticky=(tk.W,))
        
        ttk.Radiobutton(source_frame, text="Load from file:",
                       variable=self.source_type_var, value="file",
                       command=self.toggle_file_selection).grid(
                           row=1, column=0, sticky=(tk.W,), pady=(5, 0))
        
        # File selection
        self.layout_file_var = tk.StringVar(value="No file selected")
        self.layout_file_label = ttk.Label(source_frame, textvariable=self.layout_file_var,
                                          foreground='#7F8C8D', font=('Arial', 9))
        self.layout_file_label.grid(row=1, column=1, sticky=(tk.W, tk.E), 
                                   padx=(10, 0), pady=(5, 0))
        
        # Browse button
        self.browse_layout_btn = ttk.Button(source_frame, text="Browse...",
                                           command=self.browse_layout_file,
                                           state=tk.DISABLED)
        self.browse_layout_btn.grid(row=2, column=1, sticky=(tk.W,), 
                                   padx=(10, 0), pady=(5, 0))
        
    def create_format_section(self, parent):
        """Create the export format section."""
        # Format frame
        format_frame = ttk.LabelFrame(parent, text="Export Format", padding="10")
        format_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        format_frame.columnconfigure(1, weight=1)
        format_frame.columnconfigure(3, weight=1)
        
        # Output format
        ttk.Label(format_frame, text="Format:").grid(row=0, column=0, sticky=(tk.W,), padx=(0, 10))
        self.format_var = tk.StringVar(value="magic")
        format_combo = ttk.Combobox(format_frame, textvariable=self.format_var,
                                   values=["magic", "gds", "lef", "def"], 
                                   state="readonly", width=15)
        format_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 20))
        format_combo.bind('<<ComboboxSelected>>', self.on_format_change)
        
        # Technology file
        ttk.Label(format_frame, text="Technology:").grid(row=0, column=2, sticky=(tk.W,), padx=(0, 10))
        self.tech_var = tk.StringVar(value="sky130")
        tech_combo = ttk.Combobox(format_frame, textvariable=self.tech_var,
                                 values=self.tech_files, state="readonly", width=15)
        tech_combo.grid(row=0, column=3, sticky=(tk.W, tk.E))
        
        # Scale/Units
        ttk.Label(format_frame, text="Units:").grid(row=1, column=0, sticky=(tk.W,), padx=(0, 10))
        self.units_var = tk.StringVar(value="microns")
        units_combo = ttk.Combobox(format_frame, textvariable=self.units_var,
                                  values=["microns", "lambda", "nanometers"], 
                                  state="readonly", width=15)
        units_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 20), pady=(5, 0))
        
        # Grid snap
        self.grid_snap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(format_frame, text="Snap to grid",
                       variable=self.grid_snap_var).grid(
                           row=1, column=2, columnspan=2, sticky=(tk.W,), pady=(5, 0))
        
    def create_options_section(self, parent):
        """Create the export options section."""
        # Options frame
        options_frame = ttk.LabelFrame(parent, text="Export Options", padding="10")
        options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        options_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)
        
        # Left column options
        left_frame = ttk.Frame(options_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.include_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Include component labels",
                       variable=self.include_labels_var).grid(row=0, column=0, sticky=(tk.W,))
        
        self.include_ports_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Include port definitions",
                       variable=self.include_ports_var).grid(row=1, column=0, sticky=(tk.W,), pady=(5, 0))
        
        self.optimize_layout_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Optimize for area",
                       variable=self.optimize_layout_var).grid(row=2, column=0, sticky=(tk.W,), pady=(5, 0))
        
        # Right column options
        right_frame = ttk.Frame(options_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.include_routing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right_frame, text="Include routing layers",
                       variable=self.include_routing_var).grid(row=0, column=0, sticky=(tk.W,))
        
        self.validate_drc_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right_frame, text="Validate DRC rules",
                       variable=self.validate_drc_var).grid(row=1, column=0, sticky=(tk.W,), pady=(5, 0))
        
        self.generate_report_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right_frame, text="Generate export report",
                       variable=self.generate_report_var).grid(row=2, column=0, sticky=(tk.W,), pady=(5, 0))
        
    def create_status_section(self, parent):
        """Create the export status section."""
        # Status frame
        status_frame = ttk.LabelFrame(parent, text="Export Status", padding="10")
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(1, weight=1)
        
        # Status label
        self.export_status_var = tk.StringVar(value="Ready to export")
        status_label = ttk.Label(status_frame, textvariable=self.export_status_var,
                                font=('Arial', 10, 'bold'))
        status_label.grid(row=0, column=0, sticky=(tk.W,), pady=(0, 10))
        
        # Export log
        log_frame = ttk.Frame(status_frame)
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.export_log_text = tk.Text(log_frame, height=4, wrap=tk.WORD,
                                      font=('Courier', 9), state=tk.DISABLED)
        self.export_log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Log scrollbar
        export_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, 
                                        command=self.export_log_text.yview)
        export_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.export_log_text.configure(yscrollcommand=export_scrollbar.set)
        
        # Initial log message
        self.add_export_log("Magic export module initialized. Ready to export layouts.")
        
    def toggle_file_selection(self):
        """Toggle the file selection interface."""
        if self.source_type_var.get() == "file":
            self.browse_layout_btn.config(state=tk.NORMAL)
        else:
            self.browse_layout_btn.config(state=tk.DISABLED)
            
    def browse_layout_file(self):
        """Browse for a layout file to export."""
        filename = filedialog.askopenfilename(
            title="Select Layout File",
            filetypes=[
                ("Layout Files", "*.json *.pkl"),
                ("Image Files", "*.png *.jpg *.jpeg"),
                ("All Files", "*.*")
            ],
            initialdir="./output"
        )
        
        if filename:
            self.layout_file_var.set(filename)
            self.add_export_log(f"Selected layout file: {os.path.basename(filename)}")
            
    def on_format_change(self, event):
        """Handle format selection change."""
        format_type = self.format_var.get()
        self.add_export_log(f"Export format changed to: {format_type}")
        
        # Update export settings based on format
        self.export_settings["format"] = format_type
        
    def add_export_log(self, message: str):
        """Add a message to the export log."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.export_log_text.config(state=tk.NORMAL)
        self.export_log_text.insert(tk.END, log_entry)
        self.export_log_text.see(tk.END)
        self.export_log_text.config(state=tk.DISABLED)
        
    def update_export_settings(self):
        """Update export settings from GUI values."""
        self.export_settings.update({
            "format": self.format_var.get(),
            "technology": self.tech_var.get(),
            "units": self.units_var.get(),
            "include_labels": self.include_labels_var.get(),
            "include_ports": self.include_ports_var.get(),
            "include_routing": self.include_routing_var.get(),
            "grid_snap": self.grid_snap_var.get(),
            "optimize_layout": self.optimize_layout_var.get(),
            "validate_drc": self.validate_drc_var.get(),
            "generate_report": self.generate_report_var.get()
        })
        
    def export_to_magic(self) -> bool:
        """Export the layout to Magic VLSI format."""
        try:
            self.export_status_var.set("Preparing export...")
            self.add_export_log("Starting Magic VLSI export process...")
            
            # Update settings
            self.update_export_settings()
            
            # Get the layout to export
            layout_data = self.get_layout_data()
            if not layout_data:
                messagebox.showwarning("Warning", "No layout data available for export.")
                return False
                
            # Choose output file
            output_file = self.choose_output_file()
            if not output_file:
                self.add_export_log("Export cancelled by user")
                return False
                
            # Perform the export
            self.export_status_var.set("Exporting...")
            success = self.perform_export(layout_data, output_file)
            
            if success:
                self.export_status_var.set("Export completed successfully!")
                self.add_export_log(f"Export completed: {output_file}")
                
                # Generate report if requested
                if self.export_settings["generate_report"]:
                    self.generate_export_report(output_file)
                    
                return True
            else:
                self.export_status_var.set("Export failed")
                return False
                
        except Exception as e:
            self.export_status_var.set("Export failed")
            self.add_export_log(f"Export error: {str(e)}")
            messagebox.showerror("Error", f"Export failed: {str(e)}")
            return False
            
    def get_layout_data(self) -> Optional[Dict[str, Any]]:
        """Get the layout data to export."""
        if self.source_type_var.get() == "current":
            # Try to get current layout from the generated results
            return self.get_current_layout()
        else:
            # Load from selected file
            layout_file = self.layout_file_var.get()
            if layout_file == "No file selected":
                return None
            return self.load_layout_from_file(layout_file)
            
    def get_current_layout(self) -> Optional[Dict[str, Any]]:
        """Get the current generated layout."""
        # Placeholder implementation - would integrate with actual layout data
        self.add_export_log("Loading current generated layout...")
        
        # Mock layout data
        return {
            "format": "aicrl_layout",
            "grid_size": 20,
            "components": [
                {"id": 1, "name": "nfet_d1", "type": "nfet", "x": 2, "y": 3, "width": 2, "height": 2},
                {"id": 2, "name": "nfet_d2", "type": "nfet", "x": 5, "y": 3, "width": 2, "height": 2},
                {"id": 3, "name": "pfet_m1", "type": "pfet", "x": 2, "y": 6, "width": 2, "height": 2},
            ],
            "connections": [(1, 2), (2, 3)],
            "metadata": {
                "created": time.time(),
                "algorithm": "PPO",
                "quality_score": 85.5
            }
        }
        
    def load_layout_from_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load layout data from a file."""
        try:
            self.add_export_log(f"Loading layout from: {os.path.basename(filepath)}")
            
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:
                # Handle other formats
                self.add_export_log("File format not directly supported, using conversion...")
                return None
                
        except Exception as e:
            self.add_export_log(f"Error loading layout file: {str(e)}")
            return None
            
    def choose_output_file(self) -> Optional[str]:
        """Choose the output file for export."""
        format_type = self.export_settings["format"]
        
        # Set default extension based on format
        if format_type == "magic":
            default_ext = ".mag"
            filetypes = [("Magic Layout", "*.mag"), ("All Files", "*.*")]
        elif format_type == "gds":
            default_ext = ".gds"
            filetypes = [("GDS II", "*.gds"), ("All Files", "*.*")]
        elif format_type == "lef":
            default_ext = ".lef"
            filetypes = [("LEF", "*.lef"), ("All Files", "*.*")]
        elif format_type == "def":
            default_ext = ".def"
            filetypes = [("DEF", "*.def"), ("All Files", "*.*")]
        else:
            default_ext = ".mag"
            filetypes = self.export_formats
            
        # Create output directory if it doesn't exist
        output_dir = "./magic_export"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = filedialog.asksaveasfilename(
            title="Save Magic Export As",
            defaultextension=default_ext,
            filetypes=filetypes,
            initialdir=output_dir,
            initialname=f"layout_export_{int(time.time())}{default_ext}"
        )
        
        return filename if filename else None
        
    def perform_export(self, layout_data: Dict[str, Any], output_file: str) -> bool:
        """Perform the actual export process."""
        try:
            format_type = self.export_settings["format"]
            
            self.add_export_log(f"Converting layout to {format_type.upper()} format...")
            
            if format_type == "magic":
                return self.export_magic_format(layout_data, output_file)
            elif format_type == "gds":
                return self.export_gds_format(layout_data, output_file)
            elif format_type == "lef":
                return self.export_lef_format(layout_data, output_file)
            elif format_type == "def":
                return self.export_def_format(layout_data, output_file)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            self.add_export_log(f"Export conversion error: {str(e)}")
            return False
            
    def export_magic_format(self, layout_data: Dict[str, Any], output_file: str) -> bool:
        """Export to Magic .mag format."""
        self.add_export_log("Generating Magic layout file...")
        
        try:
            with open(output_file, 'w') as f:
                # Write Magic header
                f.write("magic\n")
                f.write("tech " + self.export_settings["technology"] + "\n")
                f.write("timestamp " + str(int(time.time())) + "\n")
                f.write("<< end >>\n\n")
                
                # Write components
                for comp in layout_data.get("components", []):
                    if self.export_settings["include_labels"]:
                        f.write(f"<< {comp['type']} >>\n")
                        f.write(f"rect {comp['x']} {comp['y']} ")
                        f.write(f"{comp['x'] + comp['width']} {comp['y'] + comp['height']}\n")
                        f.write(f"<< {comp['type']}_label >>\n")
                        f.write(f"rlabel {comp['type']} {comp['x']} {comp['y']} ")
                        f.write(f"{comp['x'] + comp['width']} {comp['y'] + comp['height']} 0 {comp['name']}\n")
                        f.write("<< end >>\n\n")
                
                # Write connections if routing is included
                if self.export_settings["include_routing"]:
                    f.write("<< metal1 >>\n")
                    for conn in layout_data.get("connections", []):
                        f.write(f"# Connection from component {conn[0]} to {conn[1]}\n")
                    f.write("<< end >>\n")
                    
            self.add_export_log("Magic file written successfully")
            return True
            
        except Exception as e:
            self.add_export_log(f"Error writing Magic file: {str(e)}")
            return False
            
    def export_gds_format(self, layout_data: Dict[str, Any], output_file: str) -> bool:
        """Export to GDS II format."""
        self.add_export_log("GDS export requires additional libraries (gdspy/gdstk)")
        self.add_export_log("Creating simplified GDS representation...")
        
        # Placeholder for GDS export
        try:
            with open(output_file, 'w') as f:
                f.write("# GDS II export placeholder\n")
                f.write("# This would require gdspy or gdstk library\n")
                f.write(f"# Layout data: {json.dumps(layout_data, indent=2)}\n")
            return True
        except Exception as e:
            self.add_export_log(f"Error creating GDS file: {str(e)}")
            return False
            
    def export_lef_format(self, layout_data: Dict[str, Any], output_file: str) -> bool:
        """Export to LEF format."""
        self.add_export_log("Generating LEF file...")
        
        try:
            with open(output_file, 'w') as f:
                f.write("VERSION 5.8 ;\n")
                f.write("BUSBITCHARS \"[]\" ;\n")
                f.write("DIVIDERCHAR \"/\" ;\n\n")
                
                f.write("UNITS\n")
                f.write("  DATABASE MICRONS 1000 ;\n")
                f.write("END UNITS\n\n")
                
                # Write macro for each component
                for comp in layout_data.get("components", []):
                    f.write(f"MACRO {comp['name']}\n")
                    f.write(f"  CLASS CORE ;\n")
                    f.write(f"  SIZE {comp['width']} BY {comp['height']} ;\n")
                    f.write(f"  ORIGIN 0 0 ;\n")
                    f.write(f"END {comp['name']}\n\n")
                    
            return True
            
        except Exception as e:
            self.add_export_log(f"Error writing LEF file: {str(e)}")
            return False
            
    def export_def_format(self, layout_data: Dict[str, Any], output_file: str) -> bool:
        """Export to DEF format."""
        self.add_export_log("Generating DEF file...")
        
        try:
            with open(output_file, 'w') as f:
                f.write("VERSION 5.8 ;\n")
                f.write("DIVIDERCHAR \"/\" ;\n")
                f.write("BUSBITCHARS \"[]\" ;\n\n")
                
                f.write(f"DESIGN {Path(output_file).stem} ;\n\n")
                
                f.write("UNITS DISTANCE MICRONS 1000 ;\n\n")
                
                grid_size = layout_data.get("grid_size", 20)
                f.write(f"DIEAREA ( 0 0 ) ( {grid_size * 1000} {grid_size * 1000} ) ;\n\n")
                
                # Write components
                components = layout_data.get("components", [])
                if components:
                    f.write(f"COMPONENTS {len(components)} ;\n")
                    for comp in components:
                        f.write(f"  - {comp['name']} {comp['type']}\n")
                        f.write(f"    + PLACED ( {comp['x'] * 1000} {comp['y'] * 1000} ) N ;\n")
                    f.write("END COMPONENTS\n\n")
                
                f.write("END DESIGN\n")
                
            return True
            
        except Exception as e:
            self.add_export_log(f"Error writing DEF file: {str(e)}")
            return False
            
    def generate_export_report(self, output_file: str):
        """Generate an export report."""
        try:
            report_file = output_file.replace(Path(output_file).suffix, "_report.txt")
            
            with open(report_file, 'w') as f:
                f.write("Magic VLSI Export Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Export Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output File: {output_file}\n")
                f.write(f"Format: {self.export_settings['format'].upper()}\n")
                f.write(f"Technology: {self.export_settings['technology']}\n")
                f.write(f"Units: {self.export_settings['units']}\n\n")
                
                f.write("Export Settings:\n")
                f.write("-" * 20 + "\n")
                for key, value in self.export_settings.items():
                    f.write(f"{key}: {value}\n")
                    
            self.add_export_log(f"Export report saved: {os.path.basename(report_file)}")
            
        except Exception as e:
            self.add_export_log(f"Error generating report: {str(e)}")
            
    def get_export_settings(self) -> Dict[str, Any]:
        """Get the current export settings."""
        self.update_export_settings()
        return self.export_settings.copy()