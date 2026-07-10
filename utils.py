"""
utils.py
==========================================================
Shared utility functions used across the project:
- logging helpers
- JSON read/write helpers
- pickle read/write helpers
- MIDI validation
- misc formatting helpers

Keeping these in one place avoids duplicating logic in
preprocess.py, train_model.py, generate_music.py, etc.
==========================================================
"""

import os
import json
import pickle
import logging
import time
from typing import Any, Optional

import config


# ----------------------------------------------------------
# LOGGING SETUP
# ----------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Create (or fetch) a configured logger that writes to both
    the console and a persistent training_log.txt file.

    Args:
        name: Name of the logger (usually __name__ of the caller).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        try:
            file_handler = logging.FileHandler(config.TRAINING_LOG_TXT)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            # If the log directory isn't writable yet, skip file logging
            # rather than crashing the whole app.
            pass

    return logger


# ----------------------------------------------------------
# JSON HELPERS
# ----------------------------------------------------------
def save_json(data: Any, filepath: str) -> None:
    """Save a Python object as a pretty-printed JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def load_json(filepath: str, default: Optional[Any] = None) -> Any:
    """Load JSON data from disk, returning `default` if the file doesn't exist."""
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


# ----------------------------------------------------------
# PICKLE HELPERS
# ----------------------------------------------------------
def save_pickle(data: Any, filepath: str) -> None:
    """Save any Python object to disk using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath: str, default: Optional[Any] = None) -> Any:
    """Load a pickled Python object from disk."""
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError):
        return default


# ----------------------------------------------------------
# FILE VALIDATION
# ----------------------------------------------------------
def allowed_midi_file(filename: str) -> bool:
    """Check whether a filename has an allowed MIDI extension."""
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in config.ALLOWED_MIDI_EXTENSIONS


def list_midi_files(directory: str) -> list:
    """Recursively list every MIDI file inside a directory."""
    midi_files = []
    if not os.path.isdir(directory):
        return midi_files

    for root, _dirs, files in os.walk(directory):
        for filename in files:
            if allowed_midi_file(filename):
                midi_files.append(os.path.join(root, filename))
    return midi_files


# ----------------------------------------------------------
# FORMATTING HELPERS
# ----------------------------------------------------------
def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string (e.g. '2m 13s')."""
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def timestamp_now() -> str:
    """Return the current time as a filesystem-safe timestamp string."""
    return time.strftime("%Y%m%d_%H%M%S")


def human_readable_size(num_bytes: float) -> str:
    """Convert a byte count into a human-readable string (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


class AppError(Exception):
    """
    Base exception class for application-level errors that should be
    shown to the user as a friendly message instead of a stack trace.
    """

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
