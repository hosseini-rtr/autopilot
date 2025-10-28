# Autonomous Driving - End-to-End Learning

An end-to-end autonomous driving system combining real-time lane detection and neural network-based steering control.

## Overview

This project implements an autonomous race car system using computer vision and deep learning. It combines two main components:

1. **Lane Detection**: Real-time lane boundary detection using Canny edge detection and Hough transform
2. **Autonomous Steering**: Neural network-based steering control trained on driving data

## Features

- ✅ Real-time video processing for lane detection
- ✅ Convolutional Neural Network (CNN) for steering prediction
- ✅ WebSocket-based communication with driving simulator
- ✅ Image preprocessing pipeline (YUV color space, normalization)
- ✅ Data augmentation for robust model training
- ✅ End-to-end learning without manual feature engineering

## Project Structure

```
autopilot/
├── src/                          # Main source code
│   ├── drive.py                 # Real-time driving application
│   ├── models/                  # Model definitions and training
│   │   └── __init__.py
│   └── utils/                   # Utility functions
│       ├── image_processor.py   # Image preprocessing
│       └── lane_detection.py    # Lane detection algorithms
├── data/                         # Data storage
│   └── driving_log/             # Driving log CSVs
│       ├── driving_log_git.csv
│       └── driving_log_my_sim.csv
├── models/                       # Trained model weights
│   └── model.h5
├── notebooks/                    # Jupyter notebooks for experimentation
├── tests/                        # Unit tests
├── config/                       # Configuration files
│   └── config.yml
├── backup/                       # Legacy code backup (optional, for reference)
├── requirements.txt              # Python dependencies
├── setup.py                      # Setup automation
└── README.md                     # This file
```

## Requirements

- Python 3.8+
- TensorFlow/Keras
- OpenCV
- NumPy, Pandas, Matplotlib
- Flask with Socket.IO

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/hosseini-rtr/autopilot.git
cd autopilot
```

2. **Create virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Running the Autonomous Driving System

```bash
cd src
python drive.py
```

The application will:

- Load the pre-trained model from `model.h5`
- Start a WebSocket server on `0.0.0.0:4567`
- Listen for telemetry data and send steering commands

### Lane Detection

See `finding_lanes/run.ipynb` for interactive lane detection examples.

## Training a New Model

Use the notebooks in the `notebooks/` directory to:

1. Load and explore driving data
2. Preprocess and augment images
3. Train the CNN model
4. Evaluate performance

## Architecture

### CNN Model for Steering

Based on NVIDIA's end-to-end learning approach:

```
Input: 66x200x3 (YUV)
  ↓
Conv2D(24, 5x5) → ReLU → MaxPool
Conv2D(36, 5x5) → ReLU → MaxPool
Conv2D(48, 5x5) → ReLU → MaxPool
Conv2D(64, 3x3) → ReLU
Conv2D(64, 3x3) → ReLU
  ↓
Flatten
Dense(100) → ReLU → Dropout(0.5)
Dense(50) → ReLU → Dropout(0.5)
Dense(10) → ReLU
  ↓
Output: Steering Angle
```

### Image Preprocessing

- Crop input image (remove sky and hood)
- Convert RGB → YUV color space
- Apply Gaussian blur
- Resize to 200x66 pixels
- Normalize to [0, 1]

## Data

Driving data is located in:

- `driving/driving_log_git.csv` - Git-tracked data
- `driving/driving_log_my_sim.csv` - Simulator data

CSV format: `center, left, right, steering, throttle, reverse, speed`

## Configuration

Environment variables:

- `MODEL_PATH`: Path to trained model (default: `model.h5`)
- `SPEED_LIMIT`: Maximum speed for throttle control (default: `10`)

## Performance

- Real-time inference: ~100 FPS (depending on hardware)
- Model size: ~50 MB
- Training time: 1-2 hours on GPU

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

- Follow PEP 8
- Use type hints
- Document functions with docstrings

### Contributing

1. Create a feature branch
2. Make changes and add tests
3. Submit a pull request

## Acknowledgments

- NVIDIA's end-to-end learning for autonomous driving
- Udacity's self-driving car nanodegree project structure
- OpenCV and TensorFlow communities

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## References

- [NVIDIA: End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- [OpenCV Lane Detection](https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)
