#!/usr/bin/env python3
"""
Initial setup script for cattle recognition project.
This script can be run without external dependencies to set up the basic project structure.
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_project_structure():
    """Verify project directory structure."""
    project_root = Path(__file__).parent.parent
    required_dirs = [
        "data/detection",
        "data/identification", 
        "models/detection",
        "models/identification",
        "models/optimized",
        "src/data_preparation",
        "src/training",
        "src/inference", 
        "src/tracking",
        "tests",
        "notebooks",
        "deployment"
    ]
    
    print("\n📁 Checking project structure...")
    all_good = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - creating...")
            full_path.mkdir(parents=True, exist_ok=True)
            all_good = False
    
    return all_good

def create_init_files():
    """Create __init__.py files for Python packages."""
    project_root = Path(__file__).parent.parent
    package_dirs = [
        "src",
        "src/data_preparation", 
        "src/training",
        "src/inference",
        "src/tracking"
    ]
    
    print("\n🐍 Creating Python package files...")
    for pkg_dir in package_dirs:
        init_file = project_root / pkg_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"✅ Created {pkg_dir}/__init__.py")

def check_roboflow_datasets():
    """Check if Roboflow datasets have been downloaded."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "detection"
    
    print("\n📦 Checking for downloaded datasets...")
    
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    datasets = [d for d in data_dir.iterdir() if d.is_dir()]
    if len(datasets) == 0:
        print("❌ No datasets found")
        print("   Run: python src/data_preparation/download_datasets.py")
        return False
    
    print(f"✅ Found {len(datasets)} datasets:")
    for dataset in datasets:
        print(f"   - {dataset.name}")
    
    return True

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🚀 CATTLE RECOGNITION PROJECT SETUP")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("\n1. Install Required Packages:")
    print("   pip install roboflow ultralytics opencv-python numpy pandas")
    print("   pip install matplotlib seaborn jupyter notebook")
    
    print("\n2. Download Datasets:")
    print("   python src/data_preparation/download_datasets.py")
    
    print("\n3. Analyze Data:")
    print("   jupyter notebook notebooks/01_dataset_analysis.ipynb")
    
    print("\n4. Or run data analysis script:")
    print("   python src/data_preparation/data_utils.py")
    
    print("\n🎯 PHASE 1.1 OBJECTIVES:")
    print("   ✅ Project structure created")
    print("   ⏳ Download detection datasets (Roboflow)")
    print("   ⏳ Analyze dataset properties")
    print("   ⏳ Prepare data for training")
    
    print("\n📊 DATASETS TO DOWNLOAD:")
    print("   - cattle-detection-v1 (UAV images)")
    print("   - cattle-detection-v2 (UAV images)")
    print("   - cattle-detection-v3 (UAV images)")  
    print("   - cow-counting-v3 (drone images)")
    
    print("\n💡 TIP: The datasets provide detection data for Phase 1.1")
    print("    Later phases will use this data to train cattle detection models")
    
    print("\n" + "="*60)

def main():
    """Main setup function."""
    print("🐄 Cattle Recognition Project Setup")
    print("===================================")
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check/create project structure
    check_project_structure()
    
    # Create Python package files
    create_init_files()
    
    # Check for datasets
    datasets_ready = check_roboflow_datasets()
    
    # Print next steps
    print_next_steps()
    
    if datasets_ready:
        print("\n✅ Setup complete! Ready to analyze data.")
    else:
        print("\n⚠️  Setup complete! Next: download datasets.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())