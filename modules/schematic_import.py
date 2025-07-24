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
            file_extension = Path(self.current_file).suffix.lower()
            
            if file_extension in ['.sp', '.spice', '.cir']:
                # Use the enhanced SPICE parser for accurate counting
                parsed_data = self.parse_spice_file()
                if parsed_data and parsed_data.get("parsed", False):
                    return len(parsed_data.get("components", []))
                else:
                    # Fallback to simple counting
                    return self._count_spice_components_simple()
                    
            elif file_extension in ['.v', '.vh']:
                # Count Verilog module instances
                return self._count_verilog_components()
                
            elif file_extension == '.json':
                # Count JSON components
                return self._count_json_components()
                    
        except Exception as e:
            print(f"Error counting components: {e}")
            
        return 0
        
    def _count_spice_components_simple(self) -> int:
        """Simple SPICE component counting as fallback."""
        try:
            with open(self.current_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.split('\n')
            count = 0
            in_subckt = False
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('.subckt'):
                    in_subckt = True
                elif line.startswith('.ends'):
                    in_subckt = False
                elif in_subckt and line and not line.startswith(('*', '.', ';')):
                    # Check if line starts with component identifier
                    if line[0].upper() in 'XMRCQLJDVI':
                        count += 1
                        
            return count
        except Exception:
            return 0
            
    def _count_verilog_components(self) -> int:
        """Count Verilog module instances."""
        try:
            with open(self.current_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            import re
            # Simple pattern for module instantiation
            pattern = r'^\s*\w+\s+\w+\s*\('
            matches = re.findall(pattern, content, re.MULTILINE)
            return len(matches)
        except Exception:
            return 0
            
    def _count_json_components(self) -> int:
        """Count JSON components."""
        try:
            with open(self.current_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
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
        try:
            with open(self.current_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the SPICE file
            parsed_data = {
                "format": "spice",
                "file": self.current_file,
                "subcircuits": [],
                "components": [],
                "nets": set(),
                "pin_info": {},
                "parsed": True,
                "statistics": {}
            }
            
            lines = content.split('\n')
            current_subckt = None
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines and comments (except .PININFO)
                if not line or (line.startswith('*') and not line.startswith('*.PININFO')):
                    i += 1
                    continue
                    
                # Handle line continuations (+ at start of line)
                while i + 1 < len(lines) and lines[i + 1].strip().startswith('+'):
                    i += 1
                    line += ' ' + lines[i].strip()[1:]  # Remove the + and add to current line
                
                # Parse different line types
                if line.startswith('.subckt'):
                    current_subckt = self._parse_subckt_definition(line)
                    parsed_data["subcircuits"].append(current_subckt)
                    
                elif line.startswith('.ends'):
                    current_subckt = None
                    
                elif line.startswith('*.PININFO'):
                    parsed_data["pin_info"] = self._parse_pininfo(line)
                    
                elif line.startswith(('.end', '.END')):
                    break
                    
                elif current_subckt and line and not line.startswith('.'):
                    # Parse component instances
                    component = self._parse_spice_component(line)
                    if component:
                        parsed_data["components"].append(component)
                        # Add nets to the set
                        for net in component.get("nets", []):
                            if net not in ['vdd', 'vss', 'gnd', '0']:  # Filter out power/ground
                                parsed_data["nets"].add(net)
                                
                i += 1
            
            # Convert nets set to list for JSON serialization
            parsed_data["nets"] = list(parsed_data["nets"])
            
            # Calculate statistics
            parsed_data["statistics"] = self._calculate_spice_statistics(parsed_data)
            
            return parsed_data
            
        except Exception as e:
            print(f"Error parsing SPICE file: {e}")
            return {
                "format": "spice",
                "file": self.current_file,
                "components": [],
                "nets": [],
                "parsed": False,
                "error": str(e)
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
            
        format_type = self.schematic_data.get("format", "")
        
        if format_type == "spice":
            return self._validate_spice_data()
        elif format_type == "verilog":
            return self._validate_verilog_data()
        elif format_type == "json":
            return self._validate_json_data()
            
        return True
        
    def _validate_spice_data(self) -> bool:
        """Validate SPICE schematic data."""
        try:
            # Check if parsing was successful
            if not self.schematic_data.get("parsed", False):
                print("SPICE parsing failed")
                return False
                
            # Check for components
            components = self.schematic_data.get("components", [])
            if not components:
                print("No components found in SPICE file")
                return False
                
            # Check for subcircuits
            subcircuits = self.schematic_data.get("subcircuits", [])
            if not subcircuits:
                print("Warning: No subcircuit definitions found")
                
            # Validate component structure
            invalid_components = []
            for i, comp in enumerate(components):
                if not self._validate_spice_component(comp):
                    invalid_components.append(i)
                    
            if invalid_components:
                print(f"Found {len(invalid_components)} invalid components")
                return False
                
            # Check for reasonable net connectivity
            nets = self.schematic_data.get("nets", [])
            if len(nets) < 2:
                print("Warning: Very few nets found, connectivity may be limited")
                
            print(f"SPICE validation passed: {len(components)} components, {len(nets)} nets")
            return True
            
        except Exception as e:
            print(f"SPICE validation error: {e}")
            return False
            
    def _validate_spice_component(self, component: Dict[str, Any]) -> bool:
        """Validate a single SPICE component."""
        required_fields = ["instance_name", "type", "model"]
        
        for field in required_fields:
            if field not in component:
                print(f"Component missing required field: {field}")
                return False
                
        # Check instance name format
        instance_name = component.get("instance_name", "")
        if not instance_name or len(instance_name) < 2:
            print(f"Invalid instance name: {instance_name}")
            return False
            
        # Validate component type
        valid_types = ["subcircuit", "mosfet", "resistor", "capacitor", "inductor", 
                      "diode", "bjt", "jfet", "vsource", "isource", "unknown"]
        comp_type = component.get("type", "")
        if comp_type not in valid_types:
            print(f"Unknown component type: {comp_type}")
            return False
            
        # Check nets
        nets = component.get("nets", [])
        if comp_type in ["mosfet", "bjt"] and len(nets) < 3:
            print(f"Transistor {instance_name} has insufficient nets: {len(nets)}")
            return False
        elif comp_type in ["resistor", "capacitor", "inductor"] and len(nets) < 2:
            print(f"Two-terminal component {instance_name} has insufficient nets: {len(nets)}")
            return False
            
        return True
        
    def _validate_verilog_data(self) -> bool:
        """Validate Verilog schematic data."""
        # Basic validation for Verilog
        instances = self.schematic_data.get("instances", [])
        return len(instances) > 0
        
    def _validate_json_data(self) -> bool:
        """Validate JSON schematic data."""
        # Basic validation for JSON
        return "components" in self.schematic_data
        
    def preprocess_schematic(self):
        """Apply preprocessing optimizations to the schematic data."""
        if not self.schematic_data:
            return
            
        # Placeholder for preprocessing logic
        self.schematic_data["preprocessed"] = True
        
    def get_schematic_data(self) -> Optional[Dict[str, Any]]:
        """Get the current schematic data."""
        return self.schematic_data
        
    def get_rl_compatible_data(self) -> Optional[Dict[str, Any]]:
        """Convert schematic data to RL-compatible format for the layout environment."""
        if not self.schematic_data:
            return None
            
        try:
            # Transform parsed SPICE data into RL environment format
            rl_data = self._transform_to_rl_format(self.schematic_data)
            return rl_data
            
        except Exception as e:
            print(f"Error converting to RL format: {e}")
            return None
            
    def _transform_to_rl_format(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform parsed schematic data into RL environment format."""
        
        # Extract components from SPICE data
        spice_components = parsed_data.get("components", [])
        nets = parsed_data.get("nets", [])
        
        # Convert SPICE components to RL format
        rl_components = []
        rl_connections = []
        component_id_counter = 1
        
        # Component type mapping from SPICE to RL
        type_mapping = {
            'mosfet': self._determine_mosfet_type,  # Will check nfet vs pfet
            'subcircuit': 'custom',
            'resistor': 'resistor', 
            'capacitor': 'cap',
            'inductor': 'inductor',
            'diode': 'diode',
            'bjt': 'bjt',
            'vsource': 'vsource',
            'isource': 'isource'
        }
        
        # Color mapping for visualization
        color_mapping = {
            'nfet': 'red',
            'pfet': 'blue', 
            'cap': 'green',
            'resistor': 'orange',
            'inductor': 'purple',
            'diode': 'yellow',
            'bjt': 'brown',
            'vsource': 'pink',
            'isource': 'cyan',
            'custom': 'gray'
        }
        
        # Size mapping based on component type
        size_mapping = {
            'nfet': (2, 2),
            'pfet': (2, 2),
            'cap': (1, 1),
            'resistor': (1, 2),
            'inductor': (2, 1),
            'diode': (1, 1),
            'bjt': (2, 2),
            'vsource': (1, 1),
            'isource': (1, 1),
            'custom': (2, 2)
        }
        
        # Group components for symmetry matching
        component_groups = {}
        net_to_components = {}  # Track which components connect to each net
        
        for spice_comp in spice_components:
            instance_name = spice_comp.get("instance_name", "")
            spice_type = spice_comp.get("type", "unknown")
            model = spice_comp.get("model", "")
            parameters = spice_comp.get("parameters", {})
            nets = spice_comp.get("nets", [])
            
            # Determine RL component type
            if spice_type == 'mosfet':
                rl_type = self._determine_mosfet_type(model, parameters)
            else:
                rl_type = type_mapping.get(spice_type, 'custom')
            
            # Get component dimensions and properties
            width, height = size_mapping.get(rl_type, (2, 2))
            color = color_mapping.get(rl_type, 'gray')
            
            # Determine if components can overlap (usually only passive components)
            can_overlap = rl_type in ['cap', 'resistor', 'inductor']
            
            # Create match group for symmetry (group similar components)
            match_group = self._determine_match_group(rl_type, model, parameters)
            
            # Create RL component
            rl_component = {
                "name": instance_name,
                "id": component_id_counter,
                "width": width,
                "height": height, 
                "color": color,
                "can_overlap": can_overlap,
                "type": rl_type,
                "match_group": match_group,
                "spice_model": model,
                "parameters": parameters.copy(),
                "nets": nets.copy(),
                "original_spice": spice_comp
            }
            
            rl_components.append(rl_component)
            
            # Track net connections for generating RL connections
            for net in nets:
                if net not in net_to_components:
                    net_to_components[net] = []
                net_to_components[net].append(component_id_counter)
            
            # Add to component groups for match group processing
            if match_group:
                if match_group not in component_groups:
                    component_groups[match_group] = []
                component_groups[match_group].append(component_id_counter)
            
            component_id_counter += 1
        
        # Generate connections based on net connectivity
        rl_connections = self._generate_rl_connections(net_to_components)
        
        # Create final RL-compatible data structure
        rl_format = {
            "format": "rl_compatible",
            "source_format": parsed_data.get("format", "unknown"),
            "source_file": parsed_data.get("file", "unknown"),
            "components": rl_components,
            "connections": rl_connections,
            "component_groups": component_groups,
            "nets": list(net_to_components.keys()),
            "net_connectivity": net_to_components,
            "statistics": {
                "total_components": len(rl_components),
                "component_types": list(set(comp["type"] for comp in rl_components)),
                "total_nets": len(net_to_components),
                "total_connections": len(rl_connections),
                "match_groups": len(component_groups)
            },
            "schematic_constraints": {
                "placement_hints": self._generate_placement_hints(rl_components, net_to_components),
                "symmetry_requirements": self._generate_symmetry_requirements(component_groups),
                "connectivity_requirements": self._generate_connectivity_requirements(net_to_components)
            }
        }
        
        return rl_format
        
    def _determine_mosfet_type(self, model: str, parameters: Dict[str, Any]) -> str:
        """Determine if MOSFET is NFET or PFET based on model name and parameters."""
        model_lower = model.lower()
        
        # Check model name for type indicators
        if 'nfet' in model_lower or 'nmos' in model_lower:
            return 'nfet'
        elif 'pfet' in model_lower or 'pmos' in model_lower:
            return 'pfet'
        
        # Check parameters for type indicators
        if 'type' in parameters:
            param_type = str(parameters['type']).lower()
            if param_type in ['n', 'nmos', 'nfet']:
                return 'nfet'
            elif param_type in ['p', 'pmos', 'pfet']:
                return 'pfet'
        
        # Default to nfet if cannot determine
        return 'nfet'
        
    def _determine_match_group(self, component_type: str, model: str, parameters: Dict[str, Any]) -> Optional[str]:
        """Determine the match group for symmetry constraints."""
        # Group similar components for symmetric placement
        if component_type in ['nfet', 'pfet']:
            # Group transistors by type and size
            width = parameters.get('w', parameters.get('width', 'default'))
            length = parameters.get('l', parameters.get('length', 'default'))
            return f"{component_type}_{width}_{length}"
        elif component_type == 'cap':
            # Group capacitors by value
            value = parameters.get('c', parameters.get('value', 'default'))
            return f"cap_{value}"
        elif component_type == 'resistor':
            # Group resistors by value
            value = parameters.get('r', parameters.get('value', 'default'))
            return f"resistor_{value}"
        else:
            # Other components grouped by type
            return component_type
            
    def _generate_rl_connections(self, net_to_components: Dict[str, List[int]]) -> List[tuple]:
        """Generate RL connections based on net connectivity."""
        connections = []
        
        # For each net, connect all components on that net
        for net, component_ids in net_to_components.items():
            if len(component_ids) >= 2:  # Need at least 2 components to make a connection
                # Connect each component to every other component on the same net
                for i in range(len(component_ids)):
                    for j in range(i + 1, len(component_ids)):
                        connection = (component_ids[i], component_ids[j])
                        if connection not in connections:
                            connections.append(connection)
        
        return connections
        
    def _generate_placement_hints(self, components: List[Dict], net_connectivity: Dict[str, List[int]]) -> Dict[str, Any]:
        """Generate placement hints for the RL model."""
        hints = {
            "component_priorities": [],
            "preferred_regions": {},
            "avoid_regions": {}
        }
        
        # Prioritize components with more connections
        component_connection_count = {}
        for comp in components:
            comp_id = comp["id"]
            connection_count = 0
            for net, comp_ids in net_connectivity.items():
                if comp_id in comp_ids:
                    connection_count += len(comp_ids) - 1  # Subtract self
            component_connection_count[comp_id] = connection_count
        
        # Sort by connection count (descending)
        hints["component_priorities"] = sorted(
            component_connection_count.keys(),
            key=lambda x: component_connection_count[x],
            reverse=True
        )
        
        return hints
        
    def _generate_symmetry_requirements(self, component_groups: Dict[str, List[int]]) -> Dict[str, Any]:
        """Generate symmetry requirements for matched components."""
        requirements = {
            "symmetric_pairs": [],
            "symmetric_groups": [],
            "symmetry_axes": []
        }
        
        for group_name, component_ids in component_groups.items():
            if len(component_ids) == 2:
                # Symmetric pair
                requirements["symmetric_pairs"].append(component_ids)
            elif len(component_ids) > 2:
                # Symmetric group
                requirements["symmetric_groups"].append(component_ids)
        
        return requirements
        
    def _generate_connectivity_requirements(self, net_connectivity: Dict[str, List[int]]) -> Dict[str, Any]:
        """Generate connectivity requirements for the RL model."""
        requirements = {
            "critical_nets": [],
            "high_connectivity_components": [],
            "minimum_distances": {},
            "maximum_distances": {}
        }
        
        # Identify critical nets (high fan-out)
        for net, comp_ids in net_connectivity.items():
            if len(comp_ids) >= 3:  # Nets with 3+ components are critical
                requirements["critical_nets"].append({
                    "net": net,
                    "components": comp_ids,
                    "fanout": len(comp_ids)
                })
        
        # Identify high connectivity components
        component_net_count = {}
        for net, comp_ids in net_connectivity.items():
            for comp_id in comp_ids:
                component_net_count[comp_id] = component_net_count.get(comp_id, 0) + 1
        
        for comp_id, net_count in component_net_count.items():
            if net_count >= 3:  # Components connected to 3+ nets
                requirements["high_connectivity_components"].append({
                    "component": comp_id,
                    "net_count": net_count
                })
        
        return requirements
        
    def clear_schematic(self):
        """Clear the current schematic data."""
        self.current_file = None
        self.schematic_data = None
        self.file_path_var.set("No file selected")
        self.file_type_var.set("Unknown")
        self.file_size_var.set("Unknown")
        self.component_count_var.set("Unknown")
        self.update_preview("No file selected for preview.")
        
    def _parse_subckt_definition(self, line: str) -> Dict[str, Any]:
        """Parse a .subckt definition line."""
        parts = line.split()
        if len(parts) < 2:
            return {"name": "unknown", "ports": []}
            
        subckt_name = parts[1]
        ports = parts[2:] if len(parts) > 2 else []
        
        return {
            "name": subckt_name,
            "ports": ports,
            "port_count": len(ports)
        }
        
    def _parse_pininfo(self, line: str) -> Dict[str, str]:
        """Parse a .PININFO comment line."""
        # Format: *.PININFO vdd3v3:B vss3v3:B porb_h:O porb_l:O por_l:O vdd1v8:B vss1v8:B
        pin_info = {}
        
        if '*.PININFO' in line:
            info_part = line.split('*.PININFO')[1].strip()
            pin_specs = info_part.split()
            
            for pin_spec in pin_specs:
                if ':' in pin_spec:
                    pin_name, pin_type = pin_spec.split(':')
                    pin_info[pin_name] = pin_type
                    
        return pin_info
        
    def _parse_spice_component(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a SPICE component instance line."""
        try:
            parts = line.split()
            if len(parts) < 2:
                return None
                
            instance_name = parts[0]
            
            # Determine component type from instance name prefix
            component_type = self._get_component_type_from_name(instance_name)
            
            # Parse nets and model/subcircuit name
            nets = []
            model_name = None
            parameters = {}
            
            # Find the model/subcircuit name (usually after nets, before parameters)
            param_start_idx = None
            for i, part in enumerate(parts[1:], 1):
                if '=' in part:
                    param_start_idx = i
                    break
                    
            if param_start_idx:
                # Everything before parameters are nets, last non-net is model
                net_parts = parts[1:param_start_idx]
                if net_parts:
                    nets = net_parts[:-1]
                    model_name = net_parts[-1]
                    
                # Parse parameters
                for param_part in parts[param_start_idx:]:
                    if '=' in param_part:
                        key, value = param_part.split('=', 1)
                        parameters[key] = self._convert_parameter_value(value)
            else:
                # No parameters, everything after instance name are nets except last (model)
                if len(parts) > 2:
                    nets = parts[1:-1]
                    model_name = parts[-1]
                elif len(parts) == 2:
                    model_name = parts[1]
                    
            return {
                "instance_name": instance_name,
                "type": component_type,
                "model": model_name,
                "nets": nets,
                "parameters": parameters,
                "net_count": len(nets)
            }
            
        except Exception as e:
            print(f"Error parsing component line '{line}': {e}")
            return None
            
    def _get_component_type_from_name(self, instance_name: str) -> str:
        """Determine component type from instance name."""
        if not instance_name:
            return "unknown"
            
        prefix = instance_name[0].upper()
        
        # Standard SPICE prefixes
        type_map = {
            'X': 'subcircuit',  # Subcircuit instance
            'M': 'mosfet',      # MOSFET
            'R': 'resistor',    # Resistor
            'C': 'capacitor',   # Capacitor
            'L': 'inductor',    # Inductor
            'D': 'diode',       # Diode
            'Q': 'bjt',         # BJT
            'J': 'jfet',        # JFET
            'V': 'vsource',     # Voltage source
            'I': 'isource',     # Current source
        }
        
        return type_map.get(prefix, 'unknown')
        
    def _convert_parameter_value(self, value_str: str) -> Any:
        """Convert parameter value string to appropriate type."""
        try:
            # Handle engineering notation (e.g., 1.2e-6, 500n, 30u)
            value_str = value_str.lower()
            
            # Engineering suffixes
            suffixes = {
                'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6,
                'm': 1e-3, 'k': 1e3, 'meg': 1e6, 'g': 1e9
            }
            
            # Check for suffix
            for suffix, multiplier in suffixes.items():
                if value_str.endswith(suffix):
                    numeric_part = value_str[:-len(suffix)]
                    return float(numeric_part) * multiplier
                    
            # Try direct conversion
            try:
                # Try integer first
                return int(value_str)
            except ValueError:
                # Try float
                return float(value_str)
                
        except ValueError:
            # Return as string if can't convert
            return value_str
            
    def _calculate_spice_statistics(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for the parsed SPICE data."""
        components = parsed_data.get("components", [])
        
        # Count components by type
        type_counts = {}
        total_components = len(components)
        
        for comp in components:
            comp_type = comp.get("type", "unknown")
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
            
        # Count unique nets
        net_count = len(parsed_data.get("nets", []))
        
        # Count subcircuits
        subcircuit_count = len(parsed_data.get("subcircuits", []))
        
        return {
            "total_components": total_components,
            "component_types": type_counts,
            "net_count": net_count,
            "subcircuit_count": subcircuit_count,
            "most_common_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "none"
        }