#!/usr/bin/env python3
"""
Dataset Download Script for Cattle Recognition Project

This script downloads the cattle detection datasets from Roboflow.
It downloads all 4 versions mentioned in the project requirements.
"""

import os
import sys
from pathlib import Path

def download_roboflow_datasets():
    """Download all cattle detection datasets from Roboflow."""
    
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Error: roboflow package not installed. Please run: pip install roboflow")
        return False
    
    # API key
    api_key = "5frUSqpECrA2NkmCyRC5"
    rf = Roboflow(api_key=api_key)
    
    # Create data directory structure
    data_dir = Path(__file__).parent.parent / "data" / "detection"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to data directory for downloads
    os.chdir(data_dir)
    
    datasets_info = [
        {
            "name": "cattle-detection-v1", 
            "workspace": "lairg-project",
            "project": "cattle-detection-with-uav-image",
            "version": 1
        },
        {
            "name": "cattle-detection-v2",
            "workspace": "lairg-project", 
            "project": "cattle-detection-with-uav-image",
            "version": 2
        },
        {
            "name": "cattle-detection-v3",
            "workspace": "lairg-project",
            "project": "cattle-detection-with-uav-image", 
            "version": 3
        },
        {
            "name": "cow-counting-v3",
            "workspace": "drone-hnsge",
            "project": "cow-counting-x8bwh",
            "version": 3
        }
    ]
    
    for dataset_info in datasets_info:
        print(f"\n{'='*50}")
        print(f"Downloading {dataset_info['name']}...")
        print(f"{'='*50}")
        
        try:
            # Get project
            project = rf.workspace(dataset_info["workspace"]).project(dataset_info["project"])
            version = project.version(dataset_info["version"])
            
            # Download dataset
            dataset = version.download("yolov11", location=dataset_info["name"])
            
            print(f"✅ Successfully downloaded {dataset_info['name']}")
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_info['name']}: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print("Dataset download completed!")
    print(f"All datasets saved to: {data_dir}")
    print(f"{'='*50}")
    
    return True

def main():
    """Main function."""
    print("Cattle Recognition Dataset Downloader")
    print("=====================================")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "CLAUDE.md").exists():
        print("Warning: Please run this script from the project root directory")
        print(f"Current directory: {current_dir}")
    
    # Download datasets
    success = download_roboflow_datasets()
    
    if success:
        print("\n✅ All operations completed successfully!")
        print("\nNext steps:")
        print("1. Explore the downloaded datasets in data/detection/")
        print("2. Run data analysis notebooks to understand the data structure")
        print("3. Begin training the detection model")
    else:
        print("\n❌ Some operations failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())