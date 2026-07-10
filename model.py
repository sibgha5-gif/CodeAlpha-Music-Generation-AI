"""
model.py
==========================================================
Defines the LSTM-based deep learning architecture used to
learn musical patterns and generate new music.

Architecture:
    Embedding -> LSTM(512) -> Dropout -> LSTM(512) -> Dropout
    -> Dense(256, relu) -> Dropout -> Dense(vocab_size, softmax)

Compiled with Adam optimizer and categorical crossentropy,
tracking accuracy.
==========================================================
"""

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from keras import layers

import config
import utils

logger = utils.get_logger(__name__)


def build_model(vocab_size: int, sequence_length: int = config.SEQUENCE_LENGTH) -> keras.Model:
    """
    Build and compile the LSTM music generation model.

    Args:
        vocab_size: Number of unique tokens (notes/chords/rests) in the dataset.
        sequence_length: Number of timesteps fed into the model per sample.

    Returns:
        A compiled Keras model ready for training.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer.")

    model = keras.Sequential(name="AI_Music_Generation_LSTM")

    # Embedding layer: turns integer token IDs into dense vectors.
    model.add(
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=config.EMBEDDING_DIM,
            input_length=sequence_length,
            name="token_embedding",
        )
    )

    # First LSTM block: learns short/medium-term musical patterns.
    model.add(
        layers.LSTM(
            config.LSTM_UNITS_1,
            return_sequences=True,
            name="lstm_1",
        )
    )
    model.add(layers.Dropout(config.DROPOUT_RATE, name="dropout_1"))

    # Second LSTM block: learns longer-term structure/phrasing.
    model.add(
        layers.LSTM(
            config.LSTM_UNITS_2,
            return_sequences=False,
            name="lstm_2",
        )
    )
    model.add(layers.Dropout(config.DROPOUT_RATE, name="dropout_2"))

    # Dense block for final feature mixing before classification.
    model.add(layers.Dense(config.DENSE_UNITS, activation="relu", name="dense_1"))
    model.add(layers.Dropout(config.DROPOUT_RATE / 2, name="dropout_3"))

    # Output layer: probability distribution over the vocabulary.
    model.add(layers.Dense(vocab_size, activation="softmax", name="output_softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info(f"Model built: vocab_size={vocab_size}, sequence_length={sequence_length}")
    return model


def load_trained_model(model_path: Optional[str] = None) -> keras.Model:
    """
    Load a previously trained model from disk.

    Args:
        model_path: Optional explicit path. Defaults to the best
            checkpoint saved during training.

    Returns:
        The loaded Keras model.

    Raises:
        FileNotFoundError: If no trained model exists yet.
    """
    path = model_path or config.BEST_MODEL_FILE

    import os
    if not os.path.exists(path):
        # Fall back to the final model if "best" checkpoint is missing.
        if os.path.exists(config.FINAL_MODEL_FILE):
            path = config.FINAL_MODEL_FILE
        else:
            raise FileNotFoundError(
                "No trained model found. Please train the model first "
                "using the 'Train Model' panel on the dashboard."
            )

    logger.info(f"Loading trained model from {path}")
    return keras.models.load_model(path)


def get_model_summary_text(model: keras.Model) -> str:
    """Capture model.summary() output as a plain string (for the dashboard)."""
    lines = []
    model.summary(print_fn=lambda line: lines.append(line))
    return "\n".join(lines)


def count_parameters(model: keras.Model) -> dict:
    """Return trainable / non-trainable parameter counts for display."""
    trainable = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    non_trainable = int(sum(tf.size(w).numpy() for w in model.non_trainable_weights))
    return {
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "total_params": trainable + non_trainable,
    }
