# AICRL GUI Application

A comprehensive graphical user interface for the AI Chip Layout Reinforcement Learning system.

## Overview

The AICRL GUI provides an intuitive interface for the complete IC layout generation workflow:

1. **Schematic Import** - Import and parse schematic files
2. **Training & Execution** - Train AI models and generate layouts  
3. **Magic VLSI Export** - Export layouts for Magic VLSI tools

## Usage

### Starting the Application

```bash
python3 gui_main.py
```

### Requirements

- Python 3.10+
- tkinter (usually included with Python)
- All dependencies from requirements.txt

## Module Architecture

The application is built with a modular architecture for maintainability:

### `/modules/` Directory Structure

```
modules/
├── __init__.py
├── schematic_import.py    # Handles schematic file imports
├── training_runner.py     # Manages AI training and layout generation
└── magic_export.py        # Exports layouts for Magic VLSI
```

### Main Components

#### 1. Schematic Import Module
- **Purpose**: Import and validate schematic files
- **Supported Formats**: SPICE (.sp, .spice, .cir), Verilog (.v, .vh), JSON (.json)
- **Features**:
  - File format auto-detection
  - Component validation
  - Preview functionality
  - Preprocessing options

#### 2. Training Runner Module  
- **Purpose**: AI model training and layout generation
- **Features**:
  - PPO algorithm configuration
  - Real-time training progress
  - Model parameter tuning
  - Layout generation with trained models
  - Comprehensive logging

#### 3. Magic Export Module
- **Purpose**: Export layouts for Magic VLSI tools
- **Supported Formats**: Magic (.mag), GDS II (.gds), LEF (.lef), DEF (.def)
- **Features**:
  - Technology file selection (sky130, sky90, gf180mcu)
  - Export options (labels, ports, routing)
  - DRC validation
  - Export reporting

## GUI Features

### Professional Styling
- Modern ttk widgets with professional appearance
- Responsive design that adapts to window resizing
- Consistent color scheme and typography
- Progress bars and status indicators

### User Experience
- Intuitive three-panel layout
- Real-time feedback and logging
- Error handling with informative messages
- Context-sensitive help and validation

### Integration
- Seamless integration with existing AICRL codebase
- Uses established training pipelines
- Compatible with current visualization tools
- Maintains existing file formats and structures

## Workflow

1. **Import Schematic**
   - Click "Browse..." to select schematic file
   - Review file information and preview
   - Configure import options
   - Click "Import Schematic"

2. **Train & Generate**
   - Configure model parameters (algorithm, grid size, etc.)
   - Set training parameters (timesteps, learning rate, etc.)
   - Click "Start Training" to begin AI training
   - Monitor progress in real-time
   - Click "Generate Layout" to create optimal layout

3. **Export to Magic**
   - Select export format and technology file
   - Configure export options
   - Choose output location
   - Click "Export to Magic"

## Testing

Run the test suite to verify GUI functionality:

```bash
python3 test_gui.py
```

Or check basic imports:

```bash
python3 check_gui.py
```

## Integration with Existing Code

The GUI integrates with existing AICRL components:

- `AnalogICLayoutEnv` - Environment for training
- `enhanced_visualizer` - Professional visualizations  
- `layout_analyzer` - Quality metrics
- `stable_baselines3` - PPO training algorithms

## Future Enhancements

- Real-time 3D layout visualization
- Advanced schematic parsing for more formats
- Integration with additional VLSI tools
- Batch processing capabilities
- Cloud training integration
- Advanced DRC checking

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **GUI Not Starting**: Check tkinter installation
3. **Training Fails**: Verify CUDA/GPU setup if using acceleration
4. **Export Issues**: Check file permissions and output directory

### Debug Mode

For debugging, set environment variable:
```bash
export AICRL_DEBUG=1
python3 gui_main.py
```

## Contributing

The modular architecture makes it easy to extend functionality:

1. Add new modules to `/modules/` directory
2. Follow existing patterns for GUI integration
3. Update this README with new features
4. Add appropriate tests

## License

Part of the AICRL project. See main project LICENSE for details.