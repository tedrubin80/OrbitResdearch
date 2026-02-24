"""TensorFlow/Keras training for orbit prediction models."""

import numpy as np


def train_tf_model(
    model,
    train_data: tuple,
    val_data: tuple,
    epochs: int = 100,
    batch_size: int = 64,
    patience: int = 15,
    model_name: str = "tf_model",
):
    """Train a Keras model with early stopping.

    Args:
        model: Compiled Keras model
        train_data: (inputs, targets) numpy arrays
        val_data: (inputs, targets) numpy arrays
        epochs: Max training epochs
        batch_size: Batch size
        patience: Early stopping patience
        model_name: Name for checkpoint

    Returns:
        Keras History object
    """
    try:
        from tensorflow import keras
    except ImportError:
        print("TensorFlow not available")
        return None

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            f"checkpoints/{model_name}_best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    train_inputs, train_targets = train_data
    val_inputs, val_targets = val_data

    history = model.fit(
        train_inputs,
        train_targets,
        validation_data=(val_inputs, val_targets),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return history
