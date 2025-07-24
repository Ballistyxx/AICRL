#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Quick SPICE Parser Test")
print("-" * 30)

try:
    # Test the import
    from modules.schematic_import import SchematicImportModule
    print("✓ Module imported successfully")
    
    # Create instance
    module = SchematicImportModule()
    print("✓ Module instance created")
    
    # Check if test file exists
    test_file = "./test_spice_example.sp"
    if os.path.exists(test_file):
        print(f"✓ Test file exists: {test_file}")
        
        # Set current file and test parsing
        module.current_file = test_file
        result = module.parse_spice_file()
        
        print(f"✓ Parser result: {type(result)}")
        print(f"✓ Components found: {len(result.get('components', []))}")
        print(f"✓ Parsing success: {result.get('parsed', False)}")
        
        if result.get("components"):
            print("✓ First component:", result["components"][0]["instance_name"])
            
    else:
        print(f"✗ Test file not found: {test_file}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")