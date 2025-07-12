# Data Directory

This directory contains the datasets for the cattle recognition project.

## Structure

```
data/
├── detection/          # YOLO datasets for cattle detection
│   ├── cattle-detection-v1/
│   ├── cattle-detection-v2/
│   ├── cattle-detection-v3/
│   └── cow-counting-v3/
└── identification/     # Individual cattle images for identification
```

## Getting the Data

The datasets are **NOT** included in this repository due to their large size.

### Automatic Download

Run the download script to get all datasets:

```bash
python src/data_preparation/download_datasets.py
```

### Manual Download

The datasets are hosted on Roboflow and can be downloaded using:

1. **cattle-detection-v1**: UAV images for cattle detection
2. **cattle-detection-v2**: UAV images for cattle detection  
3. **cattle-detection-v3**: UAV images for cattle detection
4. **cow-counting-v3**: Drone images for cow counting

### Dataset Information

- **Format**: YOLO (images + text annotations)
- **Classes**: Single class (cattle/cow) with class ID = 0
- **Splits**: train/valid/test
- **Total size**: ~Several GB when downloaded
- **Source**: Roboflow workspace

### Data Analysis

After downloading, analyze the data using:

```bash
jupyter notebook notebooks/01_dataset_analysis.ipynb
```

Or run the analysis script:

```bash
python src/data_preparation/data_utils.py
```

## Important Notes

⚠️ **Never commit dataset files to git** - they are too large and are ignored by .gitignore

✅ **Always use the download scripts** to ensure consistency across team members

📊 **Run data analysis** before training to understand dataset properties