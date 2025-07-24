#!/usr/bin/env python3
"""
Test script to verify the venv fix works.
"""

import sys
import os
from pathlib import Path

print("Testing virtual environment fix...")
print("-" * 40)

# Add current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Test the import fix from training_runner
try:
    print("Testing training_runner module import...")
    from modules.training_runner import TrainingRunnerModule
    print("✓ training_runner imported successfully")
    
    # Test creating the module 
    print("Testing module creation...")
    module = TrainingRunnerModule()
    print("✓ TrainingRunnerModule created successfully")
    
    print(f"✓ Found {len(module.available_models)} available models")
    
    # Test stable_baselines3 import directly
    print("Testing stable_baselines3 import...")
    from stable_baselines3 import PPO
    print("✓ stable_baselines3 imported successfully")
    
    print("\n🎉 All tests passed! Virtual environment fix is working.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print(f"\nCurrent Python path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")