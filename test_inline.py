import sys
import os
sys.path.append('.')

# Create test module instance
from modules.schematic_import import SchematicImportModule
module = SchematicImportModule()
module.current_file = "test_spice_example.sp"

# Quick test
try:
    result = module.parse_spice_file()
    print("Parse result:", result is not None)
    print("Components:", len(result.get("components", [])))
    print("Parsed flag:", result.get("parsed", False))
    
    # Show first component if exists
    if result and result.get("components"):
        comp = result["components"][0]
        print("First component:", comp.get("instance_name"), comp.get("type"))
        
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()