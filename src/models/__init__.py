"""Real-time driving application using neural network steering control."""

import os
from base64 import b64decode
from io import BytesIO

import eventlet
import numpy as np
import socketio
from flask import Flask
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from PIL import Image

from logging_config import get_logger
from utils import preprocess_image

# Initialize logger
logger = get_logger(__name__)

# Initialize Socket.IO and Flask app
sio = socketio.Server(async_mode="eventlet", cors_allowed_origins="*")
app = Flask(__name__)

# Configuration
SPEED_LIMIT = 10
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.h5")


def create_model(input_shape: tuple = (66, 200, 3)) -> Sequential:
    """
    Create a CNN model for end-to-end autonomous driving.

    Based on NVIDIA's end-to-end learning architecture.

    Args:
        input_shape: Input image shape (height, width, channels)

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential(
        [
            # Normalization layer (handled in preprocessing)
            Conv2D(24, (5, 5), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(36, (5, 5), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(48, (5, 5), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            Conv2D(64, (3, 3), activation="relu"),
            Flatten(),
            Dense(100, activation="relu"),
            Dropout(0.5),
            Dense(50, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="relu"),
            Dense(1),  # Output steering angle
        ]
    )

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

    return model


def load_trained_model(model_path: str):
    """Load pre-trained Keras model."""
    try:
        model = load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


@sio.on("telemetry")
def telemetry(sid, data):
    """Handle telemetry data and send steering commands."""
    try:
        speed = float(data["speed"])

        # Decode and process image
        image_data = data["image"]
        image = Image.open(BytesIO(b64decode(image_data)))
        image = np.asarray(image)

        # Preprocess image for model input
        processed_image = preprocess_image(image)
        processed_image = np.array([processed_image])

        # Predict steering angle
        steering_angle = float(model.predict(processed_image, verbose=0))

        # Calculate throttle based on speed
        throttle = 1.0 - speed / SPEED_LIMIT

        # Log telemetry
        logger.debug(
            f"Steering: {steering_angle:.4f} | "
            f"Throttle: {throttle:.4f} | Speed: {speed:.2f}"
        )

        # Send control signals
        send_control(steering_angle, throttle)

    except Exception as e:
        logger.error(f"Error processing telemetry: {e}")


@sio.on("connect")
def connect(sid, environ):
    """Handle client connection."""
    logger.info(f"Client connected: {sid}")
    send_control(0, 0)


@sio.on("disconnect")
def disconnect(sid):
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {sid}")


def send_control(steering_angle: float, throttle: float):
    """Send control commands to vehicle."""
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle),
        },
        skip_sid=True,
    )


if __name__ == "__main__":
    # Load model
    model = load_trained_model(MODEL_PATH)

    if model is None:
        logger.error("Failed to load model. Exiting.")
        exit(1)

    # Create WSGI application
    app = socketio.Middleware(sio, app)

    # Start server
    logger.info("Starting autonomous driving server on 0.0.0.0:4567")
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
