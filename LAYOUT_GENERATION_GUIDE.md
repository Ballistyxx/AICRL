# Layout Generation Guide - AICRL GUI

## Overview

The AICRL GUI now supports automated IC layout generation using pre-trained reinforcement learning models. The system can load existing trained models from the `/models` directory and generate optimized component layouts.

## Features

### 🤖 **Automated Model Discovery**
- Automatically discovers trained models in the `/models` directory
- Supports both zipped (.zip) and unzipped model formats
- Displays model information (size, modification date, version)
- Compatible with Stable-Baselines3 PPO models

### 📐 **Intelligent Layout Generation** 
- Uses pre-trained PPO models for optimal component placement
- Generates multiple layout attempts and selects the best one
- Incorporates schematic constraints when available
- Provides comprehensive quality metrics and analysis

### 🔗 **Schematic Integration**
- Automatically incorporates imported schematic data
- Uses component information to guide layout generation
- Considers connectivity requirements from SPICE netlists
- Reports schematic compatibility in generation logs

### 📊 **Professional Output**
- High-quality layout visualizations with metrics
- Detailed generation reports with statistics
- JSON export of layout data for further processing
- Organized output structure with timestamps

## Usage

### Step 1: Model Selection
1. In the **Training & Execution** panel, ensure "Use existing trained model" is checked
2. Select a model from the dropdown (best_model, simple_ic_layout_model, etc.)
3. View model information including size and compatibility

### Step 2: Optional - Import Schematic
1. In the **Schematic Import** panel, browse and select a SPICE file
2. Click "Import Schematic" to parse the netlist
3. The system will automatically pass component info to the layout generator

### Step 3: Configure Parameters
- **Grid Size**: Set the layout grid dimensions (default: 20x20)
- **Algorithm**: PPO (automatically configured for loaded models)

### Step 4: Generate Layout
1. Click **"Generate Layout"** in the Training & Execution panel
2. Monitor progress in the log window
3. The system will:
   - Load the selected model
   - Generate 3 layout attempts
   - Select the best scoring layout
   - Create visualizations and reports

### Step 5: View Results
Generated layouts are saved to `./output/layout_[timestamp]/`:
- **`layout_professional.png`** - High-quality layout visualization
- **`generation_report.txt`** - Detailed analysis and statistics  
- **`layout_data.json`** - Complete layout data for export

## Available Models

### best_model (Recommended)
- **Location**: `/models/best_model/` or `/models/best_model.zip`
- **Type**: Unzipped PPO model (preferred format)
- **Version**: Stable-Baselines3 2.6.0
- **Description**: Optimized model with best performance metrics
- **Use Case**: Production-quality layout generation

### simple_ic_layout_model  
- **Location**: `/models/simple_ic_layout_model.zip`
- **Type**: Zipped PPO model
- **Description**: Basic trained model for simple layouts
- **Use Case**: Testing and development

## Output Structure

```
output/
└── layout_[timestamp]/
    ├── layout_professional.png    # Main visualization
    ├── generation_report.txt      # Detailed report
    └── layout_data.json          # Complete data export
```

### Sample Generation Report
```
Layout Generation Report
==================================================

Generation Time: 2024-01-15 14:30:25
Model Used: best_model
Grid Size: 20
Generation Steps: 89
Total Reward: 245.67
Best Attempt: 2

Schematic Information:
-------------------------
Source File: sky130_ef_ip__simple_por.sp
Format: spice
Components: 19
Nets: 12

Layout Analysis:
--------------------
compactness_score: 85.30
connectivity_score: 78.45
completion_rate: 1.00
```

## Quality Metrics

The system evaluates layouts using multiple criteria:

- **Compactness Score**: How efficiently components are placed
- **Connectivity Score**: How well connections are optimized
- **Completion Rate**: Percentage of components successfully placed
- **Overall Score**: Combined metric for layout selection

## Integration with Export

Generated layouts can be exported to Magic VLSI format:

1. After successful generation, the **Magic VLSI Export** panel shows "Layout ready for export"
2. Configure export settings (format, technology, options)
3. Click **"Export to Magic"** to create VLSI-compatible files

## Troubleshooting

### Common Issues

**"No trained model available"**
- Ensure models exist in `/models` directory
- Check model file permissions
- Verify Stable-Baselines3 compatibility

**"Layout generation failed"**
- Check model compatibility with current environment
- Verify grid size settings (10-50 recommended)
- Review logs for specific error messages

**"No components placed"**
- Model may need retraining for current configuration
- Try different grid sizes
- Check if schematic data is properly formatted

### Debug Information

Enable detailed logging by monitoring the log window in the GUI. Key information includes:
- Model loading status
- Generation step progress
- Quality scores for each attempt
- Final placement statistics

## Advanced Features

### Multiple Generation Attempts
- System generates 3 different layouts per request
- Automatically selects highest-scoring result
- Uses stochastic model predictions for variety

### Schematic Constraint Integration
- Incorporates component counts from imported schematics
- Uses net information to guide placement decisions
- Reports compatibility between schematic and generated layout

### Professional Visualizations
- High-resolution PNG output (300 DPI)
- Color-coded component types
- Quality metrics overlay
- Grid and measurement indicators

## Performance Tips

### Optimal Settings
- **Grid Size**: 20x20 for most circuits, 30x30+ for complex designs
- **Model Selection**: Use `best_model` for production work
- **Generation Time**: Expect 30-60 seconds per layout

### Best Practices
1. Import schematics before generation for better results
2. Use larger grid sizes for complex circuits
3. Review quality metrics to assess layout quality
4. Save high-scoring layouts for future reference

---

**Status**: ✅ **Ready for Production Use**

The layout generation system is fully integrated and ready for use with existing trained models. For questions or issues, refer to the GUI log messages and generation reports.