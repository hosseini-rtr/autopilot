# Quick Start Guide

## 🚀 Getting Started with Autopilot

### Prerequisites

- Python 3.8 or higher
- Git
- ~2GB free space (for dependencies)

---

## Installation

### Step 1: Clone & Navigate

```bash
cd /path/to/autopilot
```

### Step 2: Create Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Upgrade pip & Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Setup Project

```bash
python setup.py
```

---

## Basic Usage

### Lane Detection Example

```python
import cv2
from src.utils import canny_edge_detection, region_of_interest, detect_lanes

# Load image
image = cv2.imread('image.jpg')

# Detect edges
edges = canny_edge_detection(image)

# Apply region of interest
roi = region_of_interest(edges)

# Detect lane lines
lines = detect_lanes(roi)
```

### Image Preprocessing

```python
from src.utils import preprocess_image

# Preprocess for neural network
processed = preprocess_image(image)  # Returns normalized 66x200x3
```

### Running Autonomous Driving

```bash
python src/drive.py
```

The server will start on `localhost:4567` and listen for telemetry data.

**Note:** Ensure `models/model.h5` exists or set `MODEL_PATH` environment variable:

```bash
export MODEL_PATH=/path/to/your/model.h5
python src/drive.py
```

---

## Project Structure Reference

```
src/                    → Main application code
├── drive.py           → Real-time driving app (run this!)
├── models/            → Neural network models
└── utils/             → Helper functions
    ├── image_processor.py    → Image preprocessing
    └── lane_detection.py     → Lane detection algorithms

notebooks/            → Jupyter notebooks for experimentation
data/                 → Dataset storage
config/               → Configuration files
tests/                → Unit tests
```

---

## Common Tasks

### Train a New Model

1. Prepare driving data in `data/driving_log/`
2. Create notebook in `notebooks/train_model.ipynb`
3. Use the CNN architecture from `src/models/__init__.py`

### Process Video

```python
import cv2
from src.utils import canny_edge_detection, average_slope_intercept

cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    edges = canny_edge_detection(frame)
    # Process edges...
```

### View Configuration

Edit `config/config.yml` to adjust:

- Image dimensions
- Lane detection parameters
- Server settings
- Model paths

---

## Troubleshooting

### Import Errors

```bash
# Verify packages installed
pip list | grep -E "tensorflow|opencv|keras"

# Reinstall if needed
pip install --force-reinstall -r requirements.txt
```

### Model Not Found

```bash
# Ensure model.h5 exists in project root
ls -la model.h5

# Or set MODEL_PATH environment variable
export MODEL_PATH=/path/to/model.h5
```

### CUDA Issues (GPU)

```bash
# For GPU support, install TensorFlow-GPU
pip install tensorflow[and-cuda]
```

---

## Development

### Add New Utility

1. Create function in `src/utils/`
2. Add docstring and type hints
3. Export in `src/utils/__init__.py`
4. Add tests in `tests/`

### Run Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Check style
pylint src/
```

---

## Resources

- 📚 **TensorFlow Docs**: https://www.tensorflow.org/
- 🖼️ **OpenCV Docs**: https://docs.opencv.org/
- 📖 **NVIDIA Paper**: https://arxiv.org/abs/1604.07316
- 📝 **Full README**: See README.md

---

## Getting Help

- Check `/CLEANUP_SUMMARY.md` for detailed changes
- Review docstrings: `python -c "from src.utils import preprocess_image; help(preprocess_image)"`
- Check config in `config/config.yml`

---

**Happy coding! 🤖🏎️**
