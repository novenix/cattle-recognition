# Cattle Recognition Project

This project implements an automated cattle recognition system using computer vision and machine learning techniques.

## Project Structure

```
cattle-recognition/
├── data/                   # Dataset storage
│   ├── detection/         # Detection training data
│   └── identification/    # Individual cattle images
├── models/                # Trained models
├── src/                   # Source code
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
└── deployment/            # Deployment scripts
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Datasets**
   ```bash
   python src/data_preparation/download_datasets.py
   ```

3. **Analyze Data**
   ```bash
   jupyter notebook notebooks/01_dataset_analysis.ipynb
   ```

## Phase 1.1: Dataset Creation ✅

The project includes datasets from Roboflow with proper YOLO format annotations for cattle detection.

### Available Datasets:
- cattle-detection-v1, v2, v3 (UAV images)
- cow-counting-v3 (drone images)

## Development Phases

1. **Phase 1**: Dataset Creation (Current)
2. **Phase 2**: System Capability Training
3. **Phase 3**: Real-World Optimization
4. **Phase 4**: Tracking Logic Development
5. **Phase 5**: Field Testing and Calibration

## Technologies Used

- Python 3.12
- PyTorch/YOLOv11
- OpenCV
- Roboflow
- Jupyter Notebooks

## License

This project is for research and educational purposes.
