#!/usr/bin/env python3
"""
Main GUI application for AI Chip Layout Reinforcement Learning (AICRL).

This application provides a user-friendly interface for:
1. Importing schematic files
2. Running training and layout generation
3. Exporting results for Magic VLSI

Author: AICRL Team
Version: 1.0.0
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from pathlib import Path

# Add the current directory to Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Ensure access to virtual environment packages
def ensure_venv_access():
    """Ensure access to virtual environment packages."""
    try:
        import stable_baselines3
    except ImportError:
        # Try to add venv site-packages to path
        venv_site_packages = Path(current_dir) / "AICRL" / "lib" / "python3.10" / "site-packages"
        
        if venv_site_packages.exists():
            sys.path.insert(0, str(venv_site_packages))
            print(f"✓ Added venv path for GUI: {venv_site_packages}")
        else:
            print(f"⚠️  Virtual environment not found at: {venv_site_packages}")

# Ensure venv access on startup
ensure_venv_access()

# Import our custom modules
from modules.schematic_import import SchematicImportModule
from modules.training_runner import TrainingRunnerModule
from modules.magic_export import MagicExportModule


class AICRLMainGUI:
    """Main GUI application for AICRL."""
    
    def __init__(self, root):
        """Initialize the main GUI."""
        self.root = root
        self.setup_window()
        self.setup_styles()
        self.create_modules()
        self.create_main_layout()
        
    def setup_window(self):
        """Configure the main window."""
        self.root.title("AI Chip Layout Reinforcement Learning - Main Interface")
        self.root.geometry("1200x700")
        self.root.minsize(900, 500)
        
        # Center the window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"1200x700+{x}+{y}")
        
        # Configure window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_styles(self):
        """Configure ttk styles for a professional appearance."""
        style = ttk.Style()
        
        # Use a modern theme if available
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'vista' in available_themes:
            style.theme_use('vista')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Arial', 24, 'bold'), foreground='#2C3E50')
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'italic'), foreground='#7F8C8D')
        style.configure('ModuleTitle.TLabel', font=('Arial', 14, 'bold'), foreground='#34495E')
        style.configure('Module.TFrame', relief='raised', borderwidth=2)
        style.configure('ModuleButton.TButton', font=('Arial', 11, 'bold'), padding=(10, 5))
        
    def create_modules(self):
        """Initialize the three main modules."""
        self.schematic_module = SchematicImportModule()
        self.training_module = TrainingRunnerModule()
        self.export_module = MagicExportModule()
        
    def create_main_layout(self):
        """Create the main layout with three side-by-side boxes."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsive design
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title section
        self.create_title_section(main_frame)
        
        # Three main modules side by side
        self.create_schematic_box(main_frame, row=1, column=0)
        self.create_training_box(main_frame, row=1, column=1)
        self.create_export_box(main_frame, row=1, column=2)
        
        # Status bar at bottom
        self.create_status_bar(main_frame)
        
    def create_title_section(self, parent):
        """Create the title section at the top."""
        title_frame = ttk.Frame(parent)
        title_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        title_frame.columnconfigure(0, weight=1)
        
        # Main title
        title_label = ttk.Label(title_frame, text="AI Chip Layout Reinforcement Learning", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0)
        
        # Subtitle
        subtitle_label = ttk.Label(title_frame, 
                                  text="Automated IC Layout Generation using Machine Learning", 
                                  style='Subtitle.TLabel')
        subtitle_label.grid(row=1, column=0, pady=(5, 0))
        
    def create_schematic_box(self, parent, row, column):
        """Create the schematic import box."""
        # Module frame
        module_frame = ttk.LabelFrame(parent, text="1. Schematic Import", 
                                     style='Module.TFrame', padding="15")
        module_frame.grid(row=row, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), 
                         padx=(0, 10))
        module_frame.columnconfigure(0, weight=1)
        module_frame.rowconfigure(1, weight=1)
        
        # Module description
        desc_label = ttk.Label(module_frame, 
                              text="Import and parse schematic files\nfor layout generation",
                              justify=tk.CENTER, style='ModuleTitle.TLabel')
        desc_label.grid(row=0, column=0, pady=(0, 15))
        
        # Content area
        content_frame = ttk.Frame(module_frame)
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Schematic module content
        self.schematic_module.create_gui(content_frame)
        
        # Action buttons
        button_frame = ttk.Frame(module_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        button_frame.columnconfigure(0, weight=1)
        
        import_btn = ttk.Button(button_frame, text="Import Schematic",
                               command=self.handle_schematic_import,
                               style='ModuleButton.TButton')
        import_btn.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
    def create_training_box(self, parent, row, column):
        """Create the training/running box."""
        # Module frame
        module_frame = ttk.LabelFrame(parent, text="2. Training & Execution", 
                                     style='Module.TFrame', padding="15")
        module_frame.grid(row=row, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), 
                         padx=5)
        module_frame.columnconfigure(0, weight=1)
        module_frame.rowconfigure(1, weight=1)
        
        # Module description
        desc_label = ttk.Label(module_frame, 
                              text="Train AI model and generate\noptimal IC layouts",
                              justify=tk.CENTER, style='ModuleTitle.TLabel')
        desc_label.grid(row=0, column=0, pady=(0, 15))
        
        # Content area
        content_frame = ttk.Frame(module_frame)
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Training module content
        self.training_module.create_gui(content_frame)
        
        # Action buttons
        button_frame = ttk.Frame(module_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        train_btn = ttk.Button(button_frame, text="Start Training",
                              command=self.handle_training_start,
                              style='ModuleButton.TButton')
        train_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        run_btn = ttk.Button(button_frame, text="Generate Layout",
                            command=self.handle_layout_generation,
                            style='ModuleButton.TButton')
        run_btn.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
    def create_export_box(self, parent, row, column):
        """Create the export box."""
        # Module frame
        module_frame = ttk.LabelFrame(parent, text="3. Magic VLSI Export", 
                                     style='Module.TFrame', padding="15")
        module_frame.grid(row=row, column=column, sticky=(tk.W, tk.E, tk.N, tk.S), 
                         padx=(10, 0))
        module_frame.columnconfigure(0, weight=1)
        module_frame.rowconfigure(1, weight=1)
        
        # Module description
        desc_label = ttk.Label(module_frame, 
                              text="Export generated layouts\nfor Magic VLSI tools",
                              justify=tk.CENTER, style='ModuleTitle.TLabel')
        desc_label.grid(row=0, column=0, pady=(0, 15))
        
        # Content area
        content_frame = ttk.Frame(module_frame)
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Export module content
        self.export_module.create_gui(content_frame)
        
        # Action buttons
        button_frame = ttk.Frame(module_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        button_frame.columnconfigure(0, weight=1)
        
        export_btn = ttk.Button(button_frame, text="Export to Magic",
                               command=self.handle_magic_export,
                               style='ModuleButton.TButton')
        export_btn.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
    def create_status_bar(self, parent):
        """Create the status bar at the bottom."""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(20, 0))
        status_frame.columnconfigure(1, weight=1)
        
        # Status label
        ttk.Label(status_frame, text="Status:", font=('Arial', 9, 'bold')).grid(row=0, column=0)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     font=('Arial', 9), foreground='#27AE60')
        self.status_label.grid(row=0, column=1, sticky=(tk.W,), padx=(10, 0))
        
        # Progress bar (initially hidden)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                          maximum=100, length=200)
        # Don't grid it initially - will show when needed
        
    def handle_schematic_import(self):
        """Handle schematic import button click."""
        try:
            self.update_status("Importing schematic...")
            result = self.schematic_module.import_schematic()
            if result:
                self.update_status("Schematic imported successfully!")
                
                # Pass schematic data to training module
                schematic_data = self.schematic_module.get_schematic_data()
                if schematic_data:
                    # Convert to RL-compatible format
                    rl_compatible_data = self.schematic_module.get_rl_compatible_data()
                    if rl_compatible_data:
                        # Pass both raw and RL-compatible data
                        self.training_module.set_schematic_data(rl_compatible_data)
                        
                        # Get statistics for user feedback
                        stats = rl_compatible_data.get("statistics", {})
                        components = stats.get("total_components", 0)
                        nets = stats.get("total_nets", 0)
                        connections = stats.get("total_connections", 0)
                        
                        info_message = (
                            f"Schematic imported and processed successfully!\n\n"
                            f"📋 Components: {components}\n"
                            f"🔗 Nets: {nets}\n" 
                            f"⚡ Connections: {connections}\n"
                            f"🎯 Component Types: {', '.join(stats.get('component_types', []))}"
                        )
                        messagebox.showinfo("Success", info_message)
                    else:
                        # Fallback to raw schematic data
                        self.training_module.set_schematic_data(schematic_data)
                        components = len(schematic_data.get("components", []))
                        messagebox.showinfo("Success", 
                                          f"Schematic imported successfully!\n{components} components detected.")
                else:
                    messagebox.showinfo("Success", "Schematic file imported successfully!")
            else:
                self.update_status("Schematic import cancelled")
        except Exception as e:
            self.update_status("Schematic import failed")
            messagebox.showerror("Error", f"Failed to import schematic: {str(e)}")
            
    def handle_training_start(self):
        """Handle training start button click."""
        try:
            self.update_status("Starting training...")
            self.show_progress_bar()
            result = self.training_module.start_training(progress_callback=self.update_progress)
            self.hide_progress_bar()
            if result:
                self.update_status("Training completed successfully!")
                messagebox.showinfo("Success", "Training completed successfully!")
            else:
                self.update_status("Training cancelled")
        except Exception as e:
            self.hide_progress_bar()
            self.update_status("Training failed")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            
    def handle_layout_generation(self):
        """Handle layout generation button click."""
        try:
            self.update_status("Generating layout...")
            result = self.training_module.generate_layout()
            if result:
                self.update_status("Layout generated successfully!")
                messagebox.showinfo("Success", 
                                  "Layout generated successfully!\nCheck the output directory for results.")
                
                # Notify export module that layout is ready
                self.export_module.set_layout_available(True)
            else:
                self.update_status("Layout generation cancelled")
        except Exception as e:
            self.update_status("Layout generation failed")
            messagebox.showerror("Error", f"Failed to generate layout: {str(e)}")
            
    def handle_magic_export(self):
        """Handle Magic VLSI export button click."""
        try:
            self.update_status("Exporting to Magic VLSI...")
            result = self.export_module.export_to_magic()
            if result:
                self.update_status("Export completed successfully!")
                messagebox.showinfo("Success", "Layout exported to Magic VLSI format!")
            else:
                self.update_status("Export cancelled")
        except Exception as e:
            self.update_status("Export failed")
            messagebox.showerror("Error", f"Failed to export to Magic: {str(e)}")
            
    def update_status(self, message):
        """Update the status bar message."""
        self.status_var.set(message)
        self.root.update_idletasks()
        
    def show_progress_bar(self):
        """Show the progress bar."""
        self.progress_bar.grid(row=0, column=2, padx=(10, 0))
        
    def hide_progress_bar(self):
        """Hide the progress bar."""
        self.progress_bar.grid_remove()
        
    def update_progress(self, value):
        """Update the progress bar value."""
        self.progress_var.set(value)
        self.root.update_idletasks()
        
    def on_closing(self):
        """Handle application closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.root.destroy()


def main():
    """Main entry point for the application."""
    # Create the main window
    root = tk.Tk()
    
    # Create and run the application
    app = AICRLMainGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {str(e)}")
    finally:
        try:
            root.destroy()
        except:
            pass


if __name__ == "__main__":
    main()