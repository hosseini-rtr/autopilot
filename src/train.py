"""Training script for autonomous driving model."""

import ntpath
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


def path_leaf(path):
    """Extract filename from path and prepend data directory."""
    head, tail = ntpath.split(path)
    return f"data/IMG/{tail}"


def load_data(data_dir="data/driving_log"):
    """
    Load driving log data from CSV files.

    Args:
        data_dir: Directory containing CSV files

    Returns:
        Combined pandas DataFrame
    """
    logger.info(f"Loading data from {data_dir}")

    columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]

    # Load all CSV files from data directory
    csv_files = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            csv_path = os.path.join(data_dir, file)
            logger.info(f"Reading {csv_path}")
            df = pd.read_csv(csv_path, names=columns)
            csv_files.append(df)

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    # Combine all dataframes
    data = pd.concat(csv_files, ignore_index=True)

    # Fix image paths
    data["center"] = data["center"].apply(path_leaf)
    data["right"] = data["right"].apply(path_leaf)
    data["left"] = data["left"].apply(path_leaf)

    logger.info(f"Total data loaded: {len(data)} samples")
    return data


def balance_data(data, samples_per_bin=200, num_bins=25):
    """
    Balance steering angle distribution to avoid bias.

    Args:
        data: DataFrame with steering data
        samples_per_bin: Maximum samples per bin
        num_bins: Number of bins for histogram

    Returns:
        Balanced DataFrame
    """
    logger.info("Balancing dataset...")

    hist, bins = np.histogram(data["steering"], num_bins)
    logger.info(f"Original data: {len(data)} samples")

    remove_list = []
    for j in range(num_bins):
        list_ = []
        for i in range(len(data["steering"])):
            if bins[j] <= data["steering"][i] <= bins[j + 1]:
                list_.append(i)
        list_ = shuffle(list_)
        list_ = list_[samples_per_bin:]
        remove_list.extend(list_)

    data.drop(data.index[remove_list], inplace=True)

    logger.info(f"Removed: {len(remove_list)} samples")
    logger.info(f"Remaining: {len(data)} samples")

    return data


def load_img_steering(data):
    """
    Extract image paths and steering angles from DataFrame.

    Args:
        data: DataFrame with image and steering data

    Returns:
        image_paths, steerings (numpy arrays)
    """
    image_paths = []
    steerings = []

    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center = indexed_data[0]
        image_paths.append(center.strip())
        steerings.append(float(indexed_data[3]))

    return np.asarray(image_paths), np.asarray(steerings)


def img_preprocess(img):
    """
    Preprocess image for neural network input.

    Args:
        img: Image path or numpy array

    Returns:
        Preprocessed image array
    """
    # Load image if path is provided
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop to region of interest (remove sky and hood)
    img = img[60:135, :, :]

    # Convert to YUV color space (better for lane detection)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Resize to NVIDIA model input size
    img = cv2.resize(img, (200, 66))

    # Normalize to [0, 1]
    img = img / 255.0

    return img


def nvidia_model():
    """
    Create NVIDIA end-to-end learning model architecture.

    Original paper: "End to End Learning for Self-Driving Cars"

    Returns:
        Compiled Keras model
    """
    model = Sequential()

    # Convolutional layers
    model.add(
        Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation="elu")
    )
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))

    model.add(Dropout(0.5))

    # Flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))  # Output: steering angle

    # Compile with Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    return model


def train_model(epochs=30, batch_size=100, samples_per_bin=200):
    """
    Complete training pipeline.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        samples_per_bin: Maximum samples per bin for balancing
    """
    logger.info("=" * 60)
    logger.info("Starting model training pipeline")
    logger.info("=" * 60)

    # Load and balance data
    data = load_data()
    data = balance_data(data, samples_per_bin=samples_per_bin)

    # Extract image paths and steering angles
    image_paths, steerings = load_img_steering(data)
    logger.info(f"Extracted {len(image_paths)} images with steering angles")

    # Split into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        image_paths, steerings, test_size=0.2, random_state=6
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_valid)} samples")

    # Preprocess images
    logger.info("Preprocessing training images...")
    X_train = np.array(list(map(img_preprocess, X_train)))

    logger.info("Preprocessing validation images...")
    X_valid = np.array(list(map(img_preprocess, X_valid)))

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_valid.shape}")

    # Create model
    logger.info("Building NVIDIA model architecture...")
    model = nvidia_model()
    logger.info("Model summary:")
    model.summary()

    # Train model
    logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_valid, y_valid),
        batch_size=batch_size,
        verbose=1,
        shuffle=True,
    )

    # Save model
    model_path = "models/model.h5"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    logger.info(f"✅ Model saved to {model_path}")

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)

    plot_path = "models/training_history.png"
    plt.savefig(plot_path)
    logger.info(f"✅ Training history plot saved to {plot_path}")

    # Print final metrics
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Final Training Loss: {final_train_loss:.6f}")
    logger.info(f"Final Validation Loss: {final_val_loss:.6f}")
    logger.info("=" * 60)

    return model, history


if __name__ == "__main__":
    # Train the model
    model, history = train_model(epochs=30, batch_size=100)
