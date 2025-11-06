# Training Model - Quick Guide

## How to Build/Train the Model

You have **two options** for training your autonomous driving model:

### Option 1: Run Python Script (Recommended for Production)

```bash
# Make sure you're in the project directory
cd /Users/selector/Documents/projects/autopilot

# Run the training script
python src/train.py
```

The script will:

1. Load data from `data/driving_log/*.csv`
2. Balance the dataset (remove steering angle bias)
3. Preprocess all images (crop, convert to YUV, resize, normalize)
4. Split into 80% training / 20% validation
5. Train NVIDIA architecture model
6. Save model to `models/model.h5`
7. Save training plot to `models/training_history.png`

### Option 2: Use Jupyter Notebook (Interactive Training)

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/train_model.ipynb
# Run cells step by step
```

## Quick Test After Training

```bash
# Run the autonomous driving server
python src/models/__init__.py

# Or with custom model path
MODEL_PATH=models/model.h5 python src/models/__init__.py
```

Then connect to the simulator at `http://localhost:4567`

## Training Parameters

Edit in `src/train.py` (bottom of file):

- `epochs=30` - Number of training cycles
- `batch_size=100` - Samples per batch
- `samples_per_bin=200` - Max samples per steering angle bin

## Data Requirements

Place your data in:

```
data/
  driving_log/
    driving_log_git.csv
    driving_log_my_sim.csv
  IMG/
    center_*.jpg
    left_*.jpg
    right_*.jpg
```

The training script automatically finds all `.csv` files in `data/driving_log/`.
