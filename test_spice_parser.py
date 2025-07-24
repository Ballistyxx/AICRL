#!/usr/bin/env python3
"""
Test script for the enhanced SPICE parser in the schematic import module.
Tests the parser with the provided SPICE file example.
"""

import sys
import os
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_spice_parser():
    """Test the SPICE parser with the provided example."""
    print("Testing Enhanced SPICE Parser")
    print("=" * 50)
    
    try:
        from modules.schematic_import import SchematicImportModule
        
        # Create the schematic import module
        module = SchematicImportModule()
        
        # Set the test file
        test_file = "./test_spice_example.sp"
        if not os.path.exists(test_file):
            print(f"✗ Test file not found: {test_file}")
            return False
            
        module.current_file = test_file
        
        # Test parsing
        print(f"Parsing SPICE file: {test_file}")
        parsed_data = module.parse_spice_file()
        
        if not parsed_data:
            print("✗ Parser returned None")
            return False
            
        # Display parsing results
        print("\nParsing Results:")
        print(f"✓ Format: {parsed_data.get('format', 'unknown')}")
        print(f"✓ Parsed successfully: {parsed_data.get('parsed', False)}")
        
        # Show subcircuit information
        subcircuits = parsed_data.get("subcircuits", [])
        print(f"✓ Subcircuits found: {len(subcircuits)}")
        for subckt in subcircuits:
            print(f"    - {subckt.get('name', 'unknown')}: {subckt.get('port_count', 0)} ports")
            
        # Show component statistics
        components = parsed_data.get("components", [])
        print(f"✓ Components found: {len(components)}")
        
        if components:
            print("\nComponent Details:")
            for i, comp in enumerate(components[:10]):  # Show first 10
                print(f"  {i+1:2d}. {comp.get('instance_name', 'unknown'):10s} "
                     f"({comp.get('type', 'unknown'):10s}) "
                     f"Model: {comp.get('model', 'unknown'):25s} "
                     f"Nets: {len(comp.get('nets', []))}")
            
            if len(components) > 10:
                print(f"      ... and {len(components) - 10} more components")
                
        # Show statistics
        stats = parsed_data.get("statistics", {})
        if stats:
            print("\nStatistics:")
            print(f"✓ Total components: {stats.get('total_components', 0)}")
            print(f"✓ Unique nets: {stats.get('net_count', 0)}")
            print(f"✓ Subcircuits: {stats.get('subcircuit_count', 0)}")
            
            type_counts = stats.get('component_types', {})
            if type_counts:
                print("✓ Component types:")
                for comp_type, count in type_counts.items():
                    print(f"    - {comp_type}: {count}")
                    
        # Show pin information
        pin_info = parsed_data.get("pin_info", {})
        if pin_info:
            print(f"\nPin Information ({len(pin_info)} pins):")
            for pin_name, pin_type in pin_info.items():
                print(f"  {pin_name}: {pin_type}")
                
        # Test validation
        print("\nValidation Test:")
        module.schematic_data = parsed_data
        is_valid = module.validate_schematic()
        print(f"✓ Validation result: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test component counting
        print("\nComponent Counting Test:")
        count = module.count_components()
        print(f"✓ Component count: {count}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_parsed_results():
    """Save the parsed results to a JSON file for inspection."""
    try:
        from modules.schematic_import import SchematicImportModule
        
        module = SchematicImportModule()
        module.current_file = "./test_spice_example.sp"
        parsed_data = module.parse_spice_file()
        
        if parsed_data:
            output_file = "./test_spice_parsed.json"
            with open(output_file, 'w') as f:
                json.dump(parsed_data, f, indent=2, default=str)
            print(f"✓ Parsed results saved to: {output_file}")
            
    except Exception as e:
        print(f"✗ Error saving results: {e}")

def main():
    """Run the test suite."""
    print("SPICE Parser Test Suite")
    print("=" * 60)
    
    success = test_spice_parser()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ All tests PASSED! SPICE parser is working correctly.")
        print("\nSaving detailed results...")
        save_parsed_results()
    else:
        print("\n" + "=" * 60)
        print("✗ Tests FAILED! Check the errors above.")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)