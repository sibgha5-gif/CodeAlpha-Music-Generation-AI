"""
generate_music.py
==========================================================
Uses the trained LSTM model to generate brand-new music.

Responsibilities:
- Load the trained model + vocabulary mappings
- Pick a seed sequence (random or user-provided)
- Autoregressively sample new tokens using temperature-controlled
  sampling
- Convert the generated token sequence back into a playable MIDI
  file using music21
- Save the result to generated_music/ and log it in generation_history.json
==========================================================
"""

import os
import random
from typing import List, Optional

import numpy as np
from music21 import stream, note, chord, tempo

import config
import utils
import model as model_module
from utils import AppError

logger = utils.get_logger(__name__)


# ----------------------------------------------------------
# SEED SELECTION
# ----------------------------------------------------------
def get_random_seed(sequence_length: int) -> List[int]:
    """Pick a random contiguous sequence from the training data as a seed."""
    sequences = np.load(config.SEQUENCES_FILE)
    X_train = sequences["X_train"]

    if len(X_train) == 0:
        raise AppError("No training sequences available to use as a seed.")

    random_index = random.randint(0, len(X_train) - 1)
    return X_train[random_index].tolist()


def encode_custom_seed(seed_tokens: List[str], token_to_int: dict, sequence_length: int) -> List[int]:
    """
    Encode a user-provided list of note names into integer tokens,
    padding/truncating to match the required sequence length.
    Unknown tokens are skipped with a warning.
    """
    encoded = []
    for token in seed_tokens:
        if token in token_to_int:
            encoded.append(token_to_int[token])
        else:
            logger.warning(f"Unknown seed token '{token}' skipped (not in vocabulary).")

    if not encoded:
        raise AppError("None of the provided seed notes were recognized. Using a random seed instead.")

    # Pad by repeating, or truncate, to match sequence_length exactly.
    while len(encoded) < sequence_length:
        encoded = encoded + encoded
    encoded = encoded[-sequence_length:]

    return encoded


# ----------------------------------------------------------
# TEMPERATURE-CONTROLLED SAMPLING
# ----------------------------------------------------------
def sample_with_temperature(predictions: np.ndarray, temperature: float = config.DEFAULT_TEMPERATURE) -> int:
    """
    Apply temperature scaling to a softmax probability distribution
    and sample the next token index.

    Lower temperature -> more conservative / repetitive output.
    Higher temperature -> more random / experimental output.
    """
    temperature = max(temperature, 1e-8)  # avoid divide-by-zero

    preds = np.asarray(predictions).astype("float64")
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    probabilities = np.random.multinomial(1, preds, 1)
    return int(np.argmax(probabilities))


# ----------------------------------------------------------
# CORE GENERATION LOOP
# ----------------------------------------------------------
def generate_token_sequence(
    seed: List[int],
    num_notes: int,
    temperature: float,
    trained_model=None,
    int_to_token: Optional[dict] = None,
) -> List[str]:
    """
    Autoregressively generate `num_notes` new tokens starting from `seed`.

    Returns:
        A list of decoded string tokens (notes/chords/rests).
    """
    if trained_model is None:
        trained_model = model_module.load_trained_model()

    mappings = utils.load_pickle(config.MAPPINGS_FILE)
    int_to_token = int_to_token or mappings["int_to_token"]
    sequence_length = trained_model.input_shape[1]

    pattern = list(seed)
    generated_tokens: List[str] = []

    logger.info(f"Generating {num_notes} notes (temperature={temperature})...")

    for _ in range(num_notes):
        input_array = np.reshape(pattern[-sequence_length:], (1, sequence_length))
        prediction = trained_model.predict(input_array, verbose=0)[0]

        next_index = sample_with_temperature(prediction, temperature)
        next_token = int_to_token.get(next_index, "REST")

        generated_tokens.append(next_token)
        pattern.append(next_index)

    logger.info(f"Generated {len(generated_tokens)} tokens.")
    return generated_tokens


# ----------------------------------------------------------
# TOKENS -> MIDI CONVERSION
# ----------------------------------------------------------
def tokens_to_midi_stream(tokens: List[str], tempo_bpm: int = config.DEFAULT_TEMPO) -> stream.Stream:
    """Convert a list of string tokens back into a music21 Stream (playable MIDI)."""
    output_stream = stream.Stream()
    output_stream.append(tempo.MetronomeMark(number=tempo_bpm))

    offset = 0.0
    for token in tokens:
        if token == "REST":
            new_element = note.Rest()
        elif "." in token:
        # Skip generated chord tokens that don't contain octave information.
        # This prevents invalid/empty MIDI output.
          continue
        else:
            try:
                new_element = note.Note(token)
            except Exception:
                new_element = note.Rest()

        new_element.offset = offset
        output_stream.append(new_element)
        offset += 0.5  # eighth-note spacing between generated events

    return output_stream


def save_midi_stream(midi_stream: stream.Stream, filename: str) -> str:
    """Write a music21 Stream to disk as a .mid file and return the full path."""
    filepath = os.path.join(config.GENERATED_MUSIC_DIR, filename)
    midi_stream.write("midi", fp=filepath)
    return filepath


# ----------------------------------------------------------
# GENERATION HISTORY LOG
# ----------------------------------------------------------
def log_generation(entry: dict) -> None:
    """Append a generation record to generation_history.json for the dashboard."""
    history = utils.load_json(config.GENERATION_HISTORY_FILE, default=[])
    history.insert(0, entry)  # newest first
    history = history[:50]    # keep the list bounded
    utils.save_json(history, config.GENERATION_HISTORY_FILE)


def get_generation_history() -> list:
    """Return the saved list of previously generated songs."""
    return utils.load_json(config.GENERATION_HISTORY_FILE, default=[])


# ----------------------------------------------------------
# MAIN ENTRY POINT (used by the Flask /generate route)
# ----------------------------------------------------------
def run_generation(
    num_notes: int = config.DEFAULT_GENERATION_LENGTH,
    temperature: float = config.DEFAULT_TEMPERATURE,
    custom_seed_tokens: Optional[List[str]] = None,
    style: str = config.DEFAULT_STYLE,
) -> dict:
    """
    Full generation pipeline: seed -> generate -> save MIDI -> log history.

    Returns:
        A dict describing the generated file (used to build the JSON API response).
    """
    import time
    start_time = time.time()

    trained_model = model_module.load_trained_model()
    mappings = utils.load_pickle(config.MAPPINGS_FILE)
    token_to_int = mappings["token_to_int"]
    int_to_token = mappings["int_to_token"]

    sequence_length = trained_model.input_shape[1]

    if custom_seed_tokens:
        seed = encode_custom_seed(custom_seed_tokens, token_to_int, sequence_length)
    else:
        seed = get_random_seed(sequence_length)

    generated_tokens = generate_token_sequence(
        seed=seed,
        num_notes=num_notes,
        temperature=temperature,
        trained_model=trained_model,
        int_to_token=int_to_token,
    )

    print("\n===== DEBUG =====")
    print("First 100 generated tokens:")
    print(generated_tokens[:100])

    print("\nUnique tokens:")
    print(len(set(generated_tokens)))

    print("\nSample unique tokens:")
    print(list(set(generated_tokens))[:30])
    print("=================\n")

    midi_stream = tokens_to_midi_stream(generated_tokens)
    filename = f"generated_{style}_{utils.timestamp_now()}.mid"
    filepath = save_midi_stream(midi_stream, filename)

    generation_time = time.time() - start_time

    entry = {
        "filename": filename,
        "filepath": filepath,
        "style": style,
        "num_notes": num_notes,
        "temperature": temperature,
        "generation_time": utils.format_duration(generation_time),
        "created_at": utils.timestamp_now(),
        "type": "generated",
    }
    log_generation(entry)

    logger.info(f"Generation complete: {filename} ({utils.format_duration(generation_time)})")
    return entry


if __name__ == "__main__":
    run_generation()
