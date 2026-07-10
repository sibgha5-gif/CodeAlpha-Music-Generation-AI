"""
continue_melody.py
==========================================================
Allows a user to upload an existing MIDI file and have the
trained model continue/extend the melody.

Pipeline:
1. Read the uploaded MIDI file with music21.
2. Extract its note/chord/rest tokens.
3. Take the final `sequence_length` tokens as the seed.
4. Generate a continuation using generate_music.py's engine.
5. Merge original melody + generated continuation into one Stream.
6. Save the merged MIDI file and log it in generation history.
==========================================================
"""

import os
from typing import List

import config
import utils
import model as model_module
import generate_music
from preprocess import extract_tokens_from_midi
from utils import AppError

logger = utils.get_logger(__name__)


def load_uploaded_tokens(midi_path: str) -> List[str]:
    """Extract tokens from an uploaded MIDI file, validating it first."""
    if not os.path.exists(midi_path):
        raise AppError("Uploaded MIDI file could not be found on the server.")

    tokens = extract_tokens_from_midi(midi_path)
    if not tokens:
        raise AppError(
            "Could not extract any notes from the uploaded MIDI file. "
            "Please make sure it is a valid, non-corrupted .mid file."
        )
    return tokens


def build_seed_from_upload(tokens: List[str], token_to_int: dict, sequence_length: int) -> List[int]:
    """
    Build a model-ready seed from the tail end of the uploaded melody.
    Unknown tokens (not seen during training) are mapped to the closest
    available fallback ('REST') rather than crashing the request.
    """
    tail_tokens = tokens[-sequence_length:] if len(tokens) >= sequence_length else tokens

    encoded = []
    for token in tail_tokens:
        if token in token_to_int:
            encoded.append(token_to_int[token])
        else:
            fallback = token_to_int.get("REST", 0)
            encoded.append(fallback)

    # Pad at the front if the upload was shorter than sequence_length.
    while len(encoded) < sequence_length:
        encoded = [token_to_int.get("REST", 0)] + encoded

    return encoded[-sequence_length:]


def run_melody_continuation(
    midi_path: str,
    num_notes: int = config.DEFAULT_GENERATION_LENGTH,
    temperature: float = config.DEFAULT_TEMPERATURE,
) -> dict:
    """
    Full melody-continuation pipeline, called from the Flask /continue route.

    Returns:
        A dict describing the generated (merged) MIDI file.
    """
    import time
    start_time = time.time()

    trained_model = model_module.load_trained_model()
    mappings = utils.load_pickle(config.MAPPINGS_FILE)
    token_to_int = mappings["token_to_int"]
    int_to_token = mappings["int_to_token"]
    sequence_length = trained_model.input_shape[1]

    logger.info(f"Reading uploaded MIDI: {midi_path}")
    original_tokens = load_uploaded_tokens(midi_path)

    seed = build_seed_from_upload(original_tokens, token_to_int, sequence_length)

    continuation_tokens = generate_music.generate_token_sequence(
        seed=seed,
        num_notes=num_notes,
        temperature=temperature,
        trained_model=trained_model,
        int_to_token=int_to_token,
    )

    # Merge original melody with the newly generated continuation.
    merged_tokens = original_tokens + continuation_tokens
    midi_stream = generate_music.tokens_to_midi_stream(merged_tokens)

    original_name = os.path.splitext(os.path.basename(midi_path))[0]
    filename = f"continued_{original_name}_{utils.timestamp_now()}.mid"
    filepath = generate_music.save_midi_stream(midi_stream, filename)

    generation_time = time.time() - start_time

    entry = {
        "filename": filename,
        "filepath": filepath,
        "source_file": os.path.basename(midi_path),
        "original_notes": len(original_tokens),
        "continuation_notes": len(continuation_tokens),
        "temperature": temperature,
        "generation_time": utils.format_duration(generation_time),
        "created_at": utils.timestamp_now(),
        "type": "continued",
    }
    generate_music.log_generation(entry)

    logger.info(f"Melody continuation complete: {filename}")
    return entry
