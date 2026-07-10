"""
config.py
==========================================================
Central configuration file for AI Music Generation Studio.

All paths, hyperparameters, and constants used across the
project are defined here so that every module stays in
sync and nothing is hard-coded in multiple places.
==========================================================
"""

import os

# ----------------------------------------------------------
# BASE PATHS
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
GENERATED_MUSIC_DIR = os.path.join(BASE_DIR, "generated_music")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TRAINING_LOGS_DIR = os.path.join(BASE_DIR, "training_logs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure all required directories exist at import time.
for directory in [
    DATASET_DIR,
    PROCESSED_DATA_DIR,
    GENERATED_MUSIC_DIR,
    UPLOADS_DIR,
    MODEL_DIR,
    TRAINING_LOGS_DIR,
]:
    os.makedirs(directory, exist_ok=True)

# ----------------------------------------------------------
# FILE NAMES
# ----------------------------------------------------------
NOTES_FILE = os.path.join(PROCESSED_DATA_DIR, "notes.pkl")
SEQUENCES_FILE = os.path.join(PROCESSED_DATA_DIR, "sequences.npz")
MAPPINGS_FILE = os.path.join(PROCESSED_DATA_DIR, "mappings.pkl")
DATASET_STATS_FILE = os.path.join(PROCESSED_DATA_DIR, "dataset_stats.json")

BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.keras")
FINAL_MODEL_FILE = os.path.join(MODEL_DIR, "final_model.keras")
MODEL_INFO_FILE = os.path.join(MODEL_DIR, "model_info.json")

TRAINING_HISTORY_FILE = os.path.join(TRAINING_LOGS_DIR, "history.json")
LOSS_GRAPH_FILE = os.path.join(TRAINING_LOGS_DIR, "loss_graph.png")
ACCURACY_GRAPH_FILE = os.path.join(TRAINING_LOGS_DIR, "accuracy_graph.png")
LR_GRAPH_FILE = os.path.join(TRAINING_LOGS_DIR, "lr_graph.png")
TRAINING_LOG_TXT = os.path.join(TRAINING_LOGS_DIR, "training_log.txt")
TENSORBOARD_LOG_DIR = os.path.join(TRAINING_LOGS_DIR, "tensorboard")

GENERATION_HISTORY_FILE = os.path.join(GENERATED_MUSIC_DIR, "generation_history.json")

# ----------------------------------------------------------
# ALLOWED FILE TYPES
# ----------------------------------------------------------
ALLOWED_MIDI_EXTENSIONS = {"mid", "midi"}
MAX_UPLOAD_SIZE_MB = 20

# ----------------------------------------------------------
# PREPROCESSING SETTINGS
# ----------------------------------------------------------
SEQUENCE_LENGTH = 100          # Number of notes fed to the model per training sample
TRAIN_VAL_SPLIT = 0.9          # 90% train / 10% validation
MIN_DATASET_SONGS = 1          # Minimum number of MIDI files required to train

# Supported musical "styles" -> subfolder name inside dataset/
MUSIC_STYLES = {
    "classical": "classical",
    "jazz": "jazz",
    "piano": "piano",
    "mixed": "mixed",
    "custom": "custom",
}
DEFAULT_STYLE = "mixed"

# ----------------------------------------------------------
# MODEL / TRAINING HYPERPARAMETERS
# ----------------------------------------------------------
EMBEDDING_DIM = 128
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 128
DROPOUT_RATE = 0.3
DENSE_UNITS = 128

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 5

EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

# ----------------------------------------------------------
# GENERATION SETTINGS
# ----------------------------------------------------------
GENERATION_LENGTHS = [100, 200, 500, 1000]
DEFAULT_GENERATION_LENGTH = 1000

TEMPERATURE_OPTIONS = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
DEFAULT_TEMPERATURE = 0.6

DEFAULT_TEMPO = 120  # BPM for generated MIDI output

# ----------------------------------------------------------
# FLASK APP SETTINGS
# ----------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
DEBUG = True
HOST = "0.0.0.0"
PORT = 5000

MAX_CONTENT_LENGTH = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # bytes
