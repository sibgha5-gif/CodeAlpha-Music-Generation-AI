"""
app.py
==========================================================
Main Flask application for AI Music Generation Studio.

Wires together preprocessing, training, generation, and
melody continuation behind a set of JSON/AJAX-friendly
routes, and serves the dashboard UI.
==========================================================
"""

import os
import threading
import traceback

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, url_for
)
from werkzeug.utils import secure_filename

import config
import utils
import preprocess
import train_model
import generate_music
import continue_melody
import model as model_module
from utils import AppError

logger = utils.get_logger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Guards against starting two training runs at once.
training_lock = threading.Lock()
training_thread: threading.Thread = None


# ----------------------------------------------------------
# ERROR HANDLING
# ----------------------------------------------------------
@app.errorhandler(AppError)
def handle_app_error(error: AppError):
    logger.warning(f"AppError: {error.message}")
    return jsonify({"success": False, "error": error.message}), error.status_code


@app.errorhandler(413)
def handle_file_too_large(_error):
    return jsonify({
        "success": False,
        "error": f"File is too large. Maximum upload size is {config.MAX_UPLOAD_SIZE_MB}MB."
    }), 413


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    logger.error(f"Unexpected error: {error}\n{traceback.format_exc()}")
    return jsonify({
        "success": False,
        "error": "Something went wrong on the server. Please try again."
    }), 500


# ----------------------------------------------------------
# PAGE ROUTES
# ----------------------------------------------------------
@app.route("/")
def index():
    """Landing / hero page."""
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """Main application dashboard."""
    return render_template("dashboard.html")


# ----------------------------------------------------------
# API: DATASET INFO
# ----------------------------------------------------------
@app.route("/dataset-info")
def dataset_info():
    style = request.args.get("style", config.DEFAULT_STYLE)
    stats = preprocess.get_dataset_info(style)
    return jsonify({"success": True, "data": stats})


# ----------------------------------------------------------
# API: MODEL STATUS
# ----------------------------------------------------------
@app.route("/model-status")
def model_status():
    info = utils.load_json(config.MODEL_INFO_FILE)
    is_trained = os.path.exists(config.BEST_MODEL_FILE) or os.path.exists(config.FINAL_MODEL_FILE)

    return jsonify({
        "success": True,
        "data": {
            "is_trained": is_trained,
            "info": info,
        }
    })


# ----------------------------------------------------------
# API: TRAIN MODEL (runs in a background thread)
# ----------------------------------------------------------
def _background_train(epochs, batch_size, style):
    try:
        preprocess.run_preprocessing(style=style)
        train_model.run_training(epochs=epochs, batch_size=batch_size)
    except AppError as exc:
        logger.error(f"Training pipeline error: {exc.message}")
        utils.save_json(
            {"status": "error", "message": exc.message},
            os.path.join(config.TRAINING_LOGS_DIR, "progress.json"),
        )
    except Exception as exc:
        logger.error(f"Training pipeline crashed: {exc}\n{traceback.format_exc()}")
        utils.save_json(
            {"status": "error", "message": "Training crashed unexpectedly. Check server logs."},
            os.path.join(config.TRAINING_LOGS_DIR, "progress.json"),
        )
    finally:
        if training_lock.locked():
            training_lock.release()


@app.route("/train", methods=["POST"])
def train():
    """Kick off preprocessing + training in a background thread."""
    global training_thread

    if training_lock.locked():
        return jsonify({"success": False, "error": "A training run is already in progress."}), 409

    payload = request.get_json(silent=True) or {}
    epochs = int(payload.get("epochs", config.EPOCHS))
    batch_size = int(payload.get("batch_size", config.BATCH_SIZE))
    style = payload.get("style", config.DEFAULT_STYLE)

    training_lock.acquire()
    training_thread = threading.Thread(
        target=_background_train,
        args=(epochs, batch_size, style),
        daemon=True,
    )
    training_thread.start()

    return jsonify({"success": True, "message": "Training started."})


@app.route("/training-progress")
def training_progress():
    """Polled by the dashboard to update the training progress bar/graphs."""
    progress = train_model.get_training_progress()
    return jsonify({"success": True, "data": progress})


# ----------------------------------------------------------
# API: TRAINING HISTORY / GRAPHS
# ----------------------------------------------------------
@app.route("/history")
def history():
    generation_history = generate_music.get_generation_history()
    training_history = utils.load_json(config.TRAINING_HISTORY_FILE, default={})
    return jsonify({
        "success": True,
        "data": {
            "generation_history": generation_history,
            "training_history_available": bool(training_history),
        }
    })


@app.route("/training-graph/<graph_name>")
def training_graph(graph_name):
    """Serve a saved training graph image (loss, accuracy, or lr)."""
    valid_graphs = {
        "loss": config.LOSS_GRAPH_FILE,
        "accuracy": config.ACCURACY_GRAPH_FILE,
        "lr": config.LR_GRAPH_FILE,
    }
    filepath = valid_graphs.get(graph_name)
    if not filepath or not os.path.exists(filepath):
        raise AppError(f"Graph '{graph_name}' is not available yet.", status_code=404)

    directory, filename = os.path.split(filepath)
    return send_from_directory(directory, filename)


# ----------------------------------------------------------
# API: GENERATE MUSIC
# ----------------------------------------------------------
@app.route("/generate", methods=["POST"])
def generate():
    payload = request.get_json(silent=True) or {}

    num_notes = int(payload.get("num_notes", config.DEFAULT_GENERATION_LENGTH))
    temperature = float(payload.get("temperature", config.DEFAULT_TEMPERATURE))
    style = payload.get("style", config.DEFAULT_STYLE)
    custom_seed_raw = payload.get("custom_seed", "").strip()

    if num_notes not in config.GENERATION_LENGTHS:
        raise AppError(f"Invalid generation length. Choose from {config.GENERATION_LENGTHS}.")
    if temperature not in config.TEMPERATURE_OPTIONS:
        raise AppError(f"Invalid temperature. Choose from {config.TEMPERATURE_OPTIONS}.")

    custom_seed_tokens = None
    if custom_seed_raw:
        custom_seed_tokens = [t.strip() for t in custom_seed_raw.split(",") if t.strip()]

    entry = generate_music.run_generation(
        num_notes=num_notes,
        temperature=temperature,
        custom_seed_tokens=custom_seed_tokens,
        style=style,
    )

    entry["download_url"] = url_for("download_file", filename=entry["filename"])
    entry["play_url"] = url_for("play_file", filename=entry["filename"])

    return jsonify({"success": True, "data": entry})


# ----------------------------------------------------------
# API: UPLOAD MIDI (for melody continuation)
# ----------------------------------------------------------
@app.route("/upload-midi", methods=["POST"])
def upload_midi():
    if "file" not in request.files:
        raise AppError("No file was uploaded.")

    file = request.files["file"]
    if file.filename == "":
        raise AppError("No file was selected.")

    if not utils.allowed_midi_file(file.filename):
        raise AppError("Invalid file type. Please upload a .mid or .midi file.")

    filename = secure_filename(file.filename)
    unique_filename = f"{utils.timestamp_now()}_{filename}"
    filepath = os.path.join(config.UPLOADS_DIR, unique_filename)
    file.save(filepath)

    logger.info(f"Uploaded MIDI saved to {filepath}")
    return jsonify({
        "success": True,
        "data": {"filename": unique_filename, "filepath": filepath}
    })


# ----------------------------------------------------------
# API: CONTINUE MELODY
# ----------------------------------------------------------
@app.route("/continue", methods=["POST"])
def continue_route():
    payload = request.get_json(silent=True) or {}

    filename = payload.get("filename")
    num_notes = int(payload.get("num_notes", config.DEFAULT_GENERATION_LENGTH))
    temperature = float(payload.get("temperature", config.DEFAULT_TEMPERATURE))

    if not filename:
        raise AppError("No uploaded file specified. Please upload a MIDI file first.")

    filepath = os.path.join(config.UPLOADS_DIR, secure_filename(filename))
    entry = continue_melody.run_melody_continuation(
        midi_path=filepath,
        num_notes=num_notes,
        temperature=temperature,
    )

    entry["download_url"] = url_for("download_file", filename=entry["filename"])
    entry["play_url"] = url_for("play_file", filename=entry["filename"])

    return jsonify({"success": True, "data": entry})


# ----------------------------------------------------------
# API: PLAY / DOWNLOAD GENERATED MUSIC
# ----------------------------------------------------------
@app.route("/play/<path:filename>")
def play_file(filename):
    safe_name = secure_filename(filename)
    if not os.path.exists(os.path.join(config.GENERATED_MUSIC_DIR, safe_name)):
        raise AppError("Generated file not found.", status_code=404)
    return send_from_directory(config.GENERATED_MUSIC_DIR, safe_name, mimetype="audio/midi")


@app.route("/download/<path:filename>")
def download_file(filename):
    safe_name = secure_filename(filename)
    if not os.path.exists(os.path.join(config.GENERATED_MUSIC_DIR, safe_name)):
        raise AppError("Generated file not found.", status_code=404)
    return send_from_directory(config.GENERATED_MUSIC_DIR, safe_name, as_attachment=True)


# ----------------------------------------------------------
# API: EXPORT TRAINING REPORT
# ----------------------------------------------------------
@app.route("/export-report")
def export_report():
    model_info = utils.load_json(config.MODEL_INFO_FILE, default={})
    dataset_stats = utils.load_json(config.DATASET_STATS_FILE, default={})
    training_history = utils.load_json(config.TRAINING_HISTORY_FILE, default={})

    report = {
        "project": "AI Music Generation Studio",
        "generated_at": utils.timestamp_now(),
        "dataset_stats": dataset_stats,
        "model_info": model_info,
        "training_history_summary": {
            "epochs": len(training_history.get("loss", [])),
            "final_loss": training_history.get("loss", [None])[-1],
            "final_accuracy": training_history.get("accuracy", [None])[-1],
        },
    }

    report_path = os.path.join(config.TRAINING_LOGS_DIR, "training_report.json")
    utils.save_json(report, report_path)

    return send_from_directory(
        config.TRAINING_LOGS_DIR, "training_report.json", as_attachment=True
    )


# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
