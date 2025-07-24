#!/usr/bin/env python3
"""
GUI Test for the SPICE parser - tests the import and validation functionality
through the actual GUI module.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_spice_import_gui():
    """Test SPICE import through the GUI module."""
    print("Testing SPICE Import GUI Module")
    print("=" * 40)
    
    try:
        from modules.schematic_import import SchematicImportModule
        
        # Create module instance
        module = SchematicImportModule()
        print("✓ Schematic import module created")
        
        # Test file path
        test_file = os.path.abspath("./test_spice_example.sp")
        if not os.path.exists(test_file):
            print(f"✗ Test file not found: {test_file}")
            return False
            
        print(f"✓ Test file found: {os.path.basename(test_file)}")
        
        # Simulate setting the file (as GUI would do)
        module.current_file = test_file
        
        # Test component counting
        print("\n1. Testing component counting...")
        count = module.count_components()
        print(f"   Component count: {count}")
        
        # Test file analysis 
        print("\n2. Testing file analysis...")
        module.analyze_file()
        
        # Test parsing
        print("\n3. Testing schematic parsing...")
        parsed_data = module.parse_schematic_file()
        if parsed_data:
            print(f"   ✓ Parsing successful")
            print(f"   ✓ Format: {parsed_data.get('format', 'unknown')}")
            print(f"   ✓ Components found: {len(parsed_data.get('components', []))}")
        else:
            print("   ✗ Parsing failed")
            return False
            
        # Test validation
        print("\n4. Testing validation...")
        module.schematic_data = parsed_data
        is_valid = module.validate_schematic()
        print(f"   Validation result: {'✓ PASSED' if is_valid else '✗ FAILED'}")
        
        # Test import process
        print("\n5. Testing full import process...")
        success = module.import_schematic()
        print(f"   Import result: {'✓ SUCCESS' if success else '✗ FAILED'}")
        
        if success:
            # Show detailed results
            data = module.get_schematic_data()
            if data:
                print("\n6. Import Results Summary:")
                stats = data.get("statistics", {})
                print(f"   Total components: {stats.get('total_components', 0)}")
                print(f"   Component types: {stats.get('component_types', {})}")
                print(f"   Nets: {stats.get('net_count', 0)}")
                print(f"   Subcircuits: {stats.get('subcircuit_count', 0)}")
                
        return success
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the GUI test."""
    success = test_spice_import_gui()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests PASSED!")
        print("The SPICE parser can successfully handle the provided file format.")
    else:
        print("✗ Tests FAILED!")
        print("There are issues with parsing the SPICE file.")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)