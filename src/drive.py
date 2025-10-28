"""Training and inference utilities for neural network models."""

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam


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
