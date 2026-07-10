"""
preprocess.py
==========================================================
Handles everything related to turning raw MIDI files into
numeric training data for the LSTM model:

1. Load every MIDI file from the dataset folder (or a
   style-specific subfolder).
2. Parse notes, chords, and rests using music21.
3. Build a vocabulary of unique musical "tokens".
4. Encode each song as a sequence of integers.
5. Build sliding-window (input, output) training sequences.
6. Split into train/validation sets.
7. Save everything to processed_data/ for train_model.py.
==========================================================
"""

import os
import glob
from typing import List, Tuple, Dict

import numpy as np
from music21 import converter, instrument, note, chord, stream

import config
import utils
from utils import AppError

logger = utils.get_logger(__name__)


# ----------------------------------------------------------
# STEP 1: LOAD MIDI FILES
# ----------------------------------------------------------
def get_dataset_path(style: str = config.DEFAULT_STYLE) -> str:
    """
    Resolve the folder to load MIDI files from, based on the
    selected music style. Falls back to the root dataset
    folder if no style-specific subfolder exists.
    """
    style_folder = config.MUSIC_STYLES.get(style, style)
    style_path = os.path.join(config.DATASET_DIR, style_folder)

    if os.path.isdir(style_path) and utils.list_midi_files(style_path):
        return style_path

    # Fallback: use the whole dataset directory (mixed).
    return config.DATASET_DIR


def load_midi_paths(style: str = config.DEFAULT_STYLE) -> List[str]:
    """Return a list of all MIDI file paths for the chosen style."""
    dataset_path = get_dataset_path(style)
    midi_paths = utils.list_midi_files(dataset_path)

    if len(midi_paths) < config.MIN_DATASET_SONGS:
        raise AppError(
            f"No MIDI files found in '{dataset_path}'. "
            f"Please add .mid/.midi files to the dataset folder before preprocessing."
        )

    logger.info(f"Found {len(midi_paths)} MIDI file(s) in {dataset_path}")
    return midi_paths


# ----------------------------------------------------------
# STEP 2: PARSE NOTES / CHORDS / RESTS
# ----------------------------------------------------------
def extract_tokens_from_midi(midi_path: str) -> List[str]:
    """
    Parse a single MIDI file and convert it into a list of string
    tokens representing notes, chords, and rests, e.g.:
        "C4"          -> a single note
        "C4.E4.G4"    -> a chord
        "REST"        -> a rest
    """
    tokens: List[str] = []

    try:
        midi_stream = converter.parse(midi_path)
    except Exception as exc:  # music21 raises various error types
        logger.warning(f"Skipping unreadable MIDI file '{midi_path}': {exc}")
        return tokens

    try:
        parts = instrument.partitionByInstrument(midi_stream)
        notes_to_parse = parts.parts[0].recurse() if parts else midi_stream.flat.notesAndRests
    except Exception:
        notes_to_parse = midi_stream.flat.notesAndRests

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            tokens.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            tokens.append(".".join(str(n) for n in element.normalOrder))
        elif isinstance(element, note.Rest):
            tokens.append("REST")

    return tokens


def extract_all_tokens(midi_paths: List[str]) -> List[str]:
    """Parse every MIDI file and concatenate all tokens into one long sequence."""
    all_tokens: List[str] = []

    for index, midi_path in enumerate(midi_paths, start=1):
        logger.info(f"[{index}/{len(midi_paths)}] Parsing {os.path.basename(midi_path)}")
        tokens = extract_tokens_from_midi(midi_path)
        all_tokens.extend(tokens)

    if not all_tokens:
        raise AppError(
            "No valid notes could be extracted from the dataset. "
            "Please check that your MIDI files are not corrupted."
        )

    logger.info(f"Extracted {len(all_tokens)} total tokens from {len(midi_paths)} files.")
    return all_tokens


# ----------------------------------------------------------
# STEP 3: BUILD VOCABULARY + ENCODE
# ----------------------------------------------------------
def build_vocabulary(tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build token<->integer mappings from a list of tokens."""
    unique_tokens = sorted(set(tokens))
    token_to_int = {token: i for i, token in enumerate(unique_tokens)}
    int_to_token = {i: token for token, i in token_to_int.items()}

    logger.info(f"Vocabulary size: {len(unique_tokens)} unique tokens")
    return token_to_int, int_to_token


# ----------------------------------------------------------
# STEP 4: CREATE INPUT/OUTPUT SEQUENCES
# ----------------------------------------------------------
def create_sequences(
    tokens: List[str],
    token_to_int: Dict[str, int],
    sequence_length: int = config.SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window (X, y) sequences for training:
        X[i] = tokens[i : i+sequence_length]   (encoded as ints)
        y[i] = tokens[i+sequence_length]       (encoded as int)
    """
    encoded = [token_to_int[t] for t in tokens]

    network_input = []
    network_output = []

    for i in range(len(encoded) - sequence_length):
        network_input.append(encoded[i:i + sequence_length])
        network_output.append(encoded[i + sequence_length])

    if not network_input:
        raise AppError(
            "Dataset is too small to build even a single training sequence. "
            "Add more MIDI data or reduce SEQUENCE_LENGTH in config.py."
        )

    X = np.array(network_input, dtype=np.int32)
    y = np.array(network_output, dtype=np.int32)

    logger.info(f"Built {len(X)} training sequences of length {sequence_length}")
    return X, y


def train_val_split(
    X: np.ndarray, y: np.ndarray, split: float = config.TRAIN_VAL_SPLIT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split sequences into training and validation sets."""
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    split_index = int(len(X) * split)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    logger.info(f"Train sequences: {len(X_train)} | Validation sequences: {len(X_val)}")
    return X_train, y_train, X_val, y_val


# ----------------------------------------------------------
# STEP 5: DATASET STATISTICS
# ----------------------------------------------------------
def compute_dataset_stats(midi_paths: List[str], tokens: List[str], vocab_size: int) -> dict:
    """Compute summary statistics about the dataset for the dashboard."""
    total_size_bytes = sum(os.path.getsize(p) for p in midi_paths if os.path.exists(p))

    stats = {
        "num_songs": len(midi_paths),
        "vocabulary_size": vocab_size,
        "total_tokens": len(tokens),
        "average_tokens_per_song": round(len(tokens) / max(len(midi_paths), 1), 2),
        "dataset_size_readable": utils.human_readable_size(total_size_bytes),
        "dataset_size_bytes": total_size_bytes,
    }
    return stats


# ----------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------
def run_preprocessing(style: str = config.DEFAULT_STYLE) -> dict:
    """
    Full preprocessing pipeline, callable from the CLI or from
    Flask's /train route. Returns the computed dataset statistics.
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    midi_paths = load_midi_paths(style)
    tokens = extract_all_tokens(midi_paths)
    token_to_int, int_to_token = build_vocabulary(tokens)

    X, y = create_sequences(tokens, token_to_int)
    X_train, y_train, X_val, y_val = train_val_split(X, y)

    # Persist everything needed by train_model.py / generate_music.py
    utils.save_pickle(tokens, config.NOTES_FILE)
    utils.save_pickle(
        {"token_to_int": token_to_int, "int_to_token": int_to_token},
        config.MAPPINGS_FILE,
    )
    np.savez_compressed(
        config.SEQUENCES_FILE,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )

    stats = compute_dataset_stats(midi_paths, tokens, len(token_to_int))
    stats["style"] = style
    utils.save_json(stats, config.DATASET_STATS_FILE)

    logger.info("Preprocessing complete.")
    logger.info(f"Stats: {stats}")
    return stats


def get_dataset_info(style: str = config.DEFAULT_STYLE) -> dict:
    """
    Lightweight dataset info for the dashboard that does NOT require
    running the full preprocessing pipeline. Falls back to cached
    stats if preprocessing has already been run.
    """
    cached_stats = utils.load_json(config.DATASET_STATS_FILE)
    if cached_stats:
        return cached_stats

    try:
        midi_paths = load_midi_paths(style)
        total_size_bytes = sum(os.path.getsize(p) for p in midi_paths if os.path.exists(p))
        return {
            "num_songs": len(midi_paths),
            "vocabulary_size": None,
            "total_tokens": None,
            "average_tokens_per_song": None,
            "dataset_size_readable": utils.human_readable_size(total_size_bytes),
            "dataset_size_bytes": total_size_bytes,
            "style": style,
            "preprocessed": False,
        }
    except AppError:
        return {
            "num_songs": 0,
            "vocabulary_size": 0,
            "total_tokens": 0,
            "average_tokens_per_song": 0,
            "dataset_size_readable": "0 B",
            "dataset_size_bytes": 0,
            "style": style,
            "preprocessed": False,
        }


if __name__ == "__main__":
    run_preprocessing()
