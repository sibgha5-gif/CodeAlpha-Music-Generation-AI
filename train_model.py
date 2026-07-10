"""
train_model.py
==========================================================
Trains the LSTM model on the preprocessed note sequences.

Responsibilities:
- Load processed sequences + vocabulary mappings
- Build the model (model.py)
- Train with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
  and TensorBoard callbacks
- Save training history as JSON
- Save loss / accuracy / learning-rate graphs as PNG
- Save model_info.json for the dashboard
==========================================================
"""

import os
import time
import json
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for Flask/server use
import matplotlib.pyplot as plt

from tensorflow import keras

import config
import utils
import model as model_module
from utils import AppError

logger = utils.get_logger(__name__)


# ----------------------------------------------------------
# CUSTOM CALLBACK: live progress written to disk for the dashboard
# ----------------------------------------------------------
class DashboardProgressCallback(keras.callbacks.Callback):
    """
    Writes training progress to a JSON file after every epoch so the
    Flask dashboard can poll it via the /training-progress route.
    """

    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self.start_time
        epochs_done = epoch + 1
        avg_epoch_time = elapsed / epochs_done
        remaining_epochs = max(self.total_epochs - epochs_done, 0)
        eta_seconds = avg_epoch_time * remaining_epochs

        progress = {
            "status": "training",
            "epoch": epochs_done,
            "total_epochs": self.total_epochs,
            "loss": float(logs.get("loss", 0.0)),
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
            "learning_rate": float(self.model.optimizer.learning_rate.numpy()),
            "elapsed_time": utils.format_duration(elapsed),
            "eta": utils.format_duration(eta_seconds),
            "progress_percent": round((epochs_done / self.total_epochs) * 100, 1),
        }
        utils.save_json(progress, os.path.join(config.TRAINING_LOGS_DIR, "progress.json"))
        logger.info(
            f"Epoch {epochs_done}/{self.total_epochs} | "
            f"loss={progress['loss']:.4f} | acc={progress['accuracy']:.4f} | "
            f"val_loss={progress['val_loss']:.4f} | val_acc={progress['val_accuracy']:.4f} | "
            f"ETA={progress['eta']}"
        )


# ----------------------------------------------------------
# CALLBACK FACTORY
# ----------------------------------------------------------
def build_callbacks(total_epochs: int) -> list:
    """Assemble the full list of Keras callbacks used during training."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=config.BEST_MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=config.TENSORBOARD_LOG_DIR,
        ),
        DashboardProgressCallback(total_epochs=total_epochs),
    ]
    return callbacks


# ----------------------------------------------------------
# GRAPH SAVING
# ----------------------------------------------------------
def save_training_graphs(history: dict) -> None:
    """Save loss, accuracy, and learning-rate curves as PNG files."""
    epochs_range = range(1, len(history.get("loss", [])) + 1)

    # --- Loss graph ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.get("loss", []), label="Training Loss", color="#6C63FF")
    plt.plot(epochs_range, history.get("val_loss", []), label="Validation Loss", color="#EF4444")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.LOSS_GRAPH_FILE, dpi=120)
    plt.close()

    # --- Accuracy graph ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.get("accuracy", []), label="Training Accuracy", color="#22C55E")
    plt.plot(epochs_range, history.get("val_accuracy", []), label="Validation Accuracy", color="#00D4FF")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.ACCURACY_GRAPH_FILE, dpi=120)
    plt.close()

    # --- Learning rate graph (only if recorded) ---
    if "lr" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, history.get("lr", []), label="Learning Rate", color="#7F5AF0")
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(config.LR_GRAPH_FILE, dpi=120)
        plt.close()

    logger.info("Training graphs saved to training_logs/")


# ----------------------------------------------------------
# MAIN TRAINING PIPELINE
# ----------------------------------------------------------
def run_training(epochs: Optional[int] = None, batch_size: Optional[int] = None) -> dict:
    """
    Full training pipeline. Assumes preprocess.run_preprocessing()
    has already been run (checked via file existence).

    Returns:
        A dict summary of the completed training run.
    """
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE

    if not os.path.exists(config.SEQUENCES_FILE) or not os.path.exists(config.MAPPINGS_FILE):
        raise AppError(
            "No preprocessed data found. Please run preprocessing "
            "(click 'Prepare Dataset') before training the model."
        )

    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 60)

    # --- Load processed data ---
    sequences = np.load(config.SEQUENCES_FILE)
    X_train, y_train = sequences["X_train"], sequences["y_train"]
    X_val, y_val = sequences["X_val"], sequences["y_val"]

    mappings = utils.load_pickle(config.MAPPINGS_FILE)
    vocab_size = len(mappings["token_to_int"])

    # --- Build model ---
    net_model = model_module.build_model(vocab_size=vocab_size, sequence_length=X_train.shape[1])
    callbacks = build_callbacks(total_epochs=epochs)

    utils.save_json(
        {"status": "training", "epoch": 0, "total_epochs": epochs, "progress_percent": 0},
        os.path.join(config.TRAINING_LOGS_DIR, "progress.json"),
    )

    start_time = time.time()

    try:
        history_obj = net_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )
    except Exception as exc:
        utils.save_json(
            {"status": "error", "message": str(exc)},
            os.path.join(config.TRAINING_LOGS_DIR, "progress.json"),
        )
        logger.error(f"Training failed: {exc}")
        raise AppError(f"Training failed: {exc}")

    total_time = time.time() - start_time

    # --- Save final model + history ---
    net_model.save(config.FINAL_MODEL_FILE)
    history_dict = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}
    utils.save_json(history_dict, config.TRAINING_HISTORY_FILE)
    save_training_graphs(history_dict)

    final_epoch = len(history_dict.get("loss", []))
    model_info = {
        "vocab_size": vocab_size,
        "sequence_length": int(X_train.shape[1]),
        "total_params": model_module.count_parameters(net_model)["total_params"],
        "epochs_trained": final_epoch,
        "requested_epochs": epochs,
        "batch_size": batch_size,
        "training_time": utils.format_duration(total_time),
        "final_loss": history_dict["loss"][-1] if history_dict.get("loss") else None,
        "final_accuracy": history_dict["accuracy"][-1] if history_dict.get("accuracy") else None,
        "final_val_loss": history_dict["val_loss"][-1] if history_dict.get("val_loss") else None,
        "final_val_accuracy": history_dict["val_accuracy"][-1] if history_dict.get("val_accuracy") else None,
        "trained_at": utils.timestamp_now(),
    }
    utils.save_json(model_info, config.MODEL_INFO_FILE)

    utils.save_json(
        {
            "status": "completed",
            "epoch": final_epoch,
            "total_epochs": epochs,
            "progress_percent": 100,
            "training_time": model_info["training_time"],
        },
        os.path.join(config.TRAINING_LOGS_DIR, "progress.json"),
    )

    logger.info(f"Training complete in {model_info['training_time']}. Model saved.")
    return model_info


def get_training_progress() -> dict:
    """Read the latest training progress snapshot (used by the dashboard poll)."""
    return utils.load_json(
        os.path.join(config.TRAINING_LOGS_DIR, "progress.json"),
        default={"status": "idle", "progress_percent": 0},
    )


if __name__ == "__main__":
    run_training()
