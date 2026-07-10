# 🎵 AI Music Generation Studio

**Create original music using Artificial Intelligence.**

An end-to-end deep learning web application that trains an LSTM neural network on MIDI datasets and generates brand-new musical compositions — or continues melodies from your own uploaded MIDI files. Built entirely with open-source, local tooling (no paid APIs, no external LLMs).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## ✨ Features

- 🎹 **MIDI Dataset Processing** — automatically loads, parses, and tokenizes notes, chords, and rests using `music21`
- 🧠 **Deep LSTM Architecture** — a stacked LSTM(512) → LSTM(512) network learns musical structure and phrasing
- 🎼 **Music Style Selection** — Classical, Jazz, Piano, Mixed Dataset, or a Custom Uploaded Dataset
- 🌡 **Temperature-Controlled Generation** — tune creativity from conservative (0.3) to experimental (1.5)
- 🎵 **Adjustable Generation Length** — 100 / 200 / 500 / 1000 notes
- 🎹 **Melody Continuation** — upload a MIDI file and let the AI extend it in a matching style
- 📊 **Live Training Dashboard** — epoch, loss, accuracy, validation metrics, ETA, and learning rate
- 📈 **Saved Training Graphs** — loss, accuracy, and learning-rate curves saved as PNGs after training
- ▶ **In-Browser Playback** — powered by Tone.js with an animated piano keyboard
- 📂 **Drag-and-Drop Upload** — no page reloads, fully AJAX-driven
- 🌙 **Dark Glassmorphism UI** — premium, modern, responsive dashboard
- 🔔 **Toast Notifications & Generation Logs** — clear feedback for every action
- 📄 **Exportable Training Report** — download a JSON summary of dataset, model, and training results

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Deep Learning | TensorFlow, Keras |
| Music Processing | music21, PrettyMIDI, mido |
| Data | NumPy, Pandas |
| Visualization | Matplotlib (server-side graphs), Chart.js (dashboard) |
| Frontend | HTML5, CSS3, Bootstrap 5, JavaScript, Tone.js, Bootstrap Icons |

---

## 📁 Folder Structure

```
Music-Generation-AI/
├── app.py                  # Flask application & routes
├── config.py                # Central configuration
├── preprocess.py            # MIDI loading & tokenization
├── model.py                  # LSTM architecture
├── train_model.py            # Training loop & callbacks
├── generate_music.py         # Music generation engine
├── continue_melody.py        # Melody continuation logic
├── utils.py                  # Shared helper functions
├── requirements.txt
├── README.md
├── dataset/                  # Place your MIDI files here
│   ├── classical/
│   ├── jazz/
│   ├── piano/
│   └── custom/
├── processed_data/           # Auto-generated preprocessing output
├── generated_music/          # AI-generated MIDI output
├── uploads/                  # User-uploaded MIDI files
├── model/                    # Saved trained models
├── training_logs/            # History, graphs, progress files
└── static/
    ├── css/style.css
    ├── js/main.js
    └── images/
└── templates/
    ├── base.html
    ├── index.html
    └── dashboard.html
```

---

## ⚙️ Installation Guide

### 1. Clone or extract the project
```bash
cd Music-Generation-AI
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## 🎼 Dataset Setup

1. Download any MIDI dataset — e.g. [MAESTRO](https://magenta.tensorflow.org/datasets/maestro), [Nottingham](https://ifdo.ca/~seymour/nottingham/nottingham.html), or [JSB Chorales](https://github.com/czhuang/JSB-Chorales-dataset).
2. Place `.mid` / `.midi` files into `dataset/` — or into a style subfolder such as `dataset/classical/` to enable style-based selection.
3. Open the dashboard and choose your **Music Style** before training. If no style-specific folder is found, the app falls back to the full `dataset/` directory.

> The app needs at least one valid MIDI file to preprocess. Corrupted or unreadable files are automatically skipped with a warning in the logs.

---

## 🚀 Training Instructions

1. Open the dashboard at `/dashboard`.
2. Choose a **Music Style**, **Epochs**, and **Batch Size**.
3. Click **Start Training** — this runs preprocessing and training in a background thread so the UI stays responsive.
4. Watch live progress (epoch, loss, accuracy, ETA) update every few seconds.
5. Once training completes, the **Model Information** panel and **Training Graphs** update automatically.

Training artifacts are saved to:
- `model/best_model.keras` — best checkpoint (lowest validation loss)
- `model/final_model.keras` — final epoch model
- `training_logs/history.json`, `loss_graph.png`, `accuracy_graph.png`, `lr_graph.png`

---

## 🎵 Music Generation Guide

1. Adjust the **Music Length** slider (100–1000 notes).
2. Adjust the **Temperature** slider to control creativity (lower = safer, higher = more experimental).
3. Optionally provide **Custom Seed Notes** (comma-separated, e.g. `C4, E4, G4`).
4. Click **Generate Music**. The result appears in the player with playback and download options.

### Melody Continuation
1. Drag and drop (or click to browse) a `.mid` file into the **Continue a Melody** panel.
2. Click **Continue Melody** — the AI reads the final notes of your upload as a seed and extends it.
3. The merged (original + generated) MIDI file is saved and made available for playback/download.

---

## 🧩 API Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing page |
| `/dashboard` | GET | Main dashboard |
| `/dataset-info` | GET | Dataset statistics (JSON) |
| `/model-status` | GET | Current model training status |
| `/train` | POST | Start preprocessing + training |
| `/training-progress` | GET | Poll live training progress |
| `/generate` | POST | Generate new music |
| `/upload-midi` | POST | Upload a MIDI file |
| `/continue` | POST | Continue an uploaded melody |
| `/play/<filename>` | GET | Stream a generated MIDI file |
| `/download/<filename>` | GET | Download a generated MIDI file |
| `/history` | GET | Generation history |
| `/training-graph/<name>` | GET | Serve a saved training graph PNG |
| `/export-report` | GET | Download a JSON training report |

---

## 🩹 Error Handling

The app gracefully handles and reports (via toast notifications, not crashes):
- Missing or empty datasets
- Invalid/corrupted MIDI files
- Attempting to generate before training a model
- Failed uploads (wrong file type, oversized files)
- Playback errors (falls back to suggesting a direct download)
- Training crashes (surfaced via the progress panel with a clear message)

---

## 🖼️ Screenshots

> _Add screenshots of the landing page and dashboard here before publishing to GitHub._

`docs/screenshot-landing.png`
`docs/screenshot-dashboard.png`

---

## 🔮 Future Improvements

- Transformer-based generation (Music Transformer / GPT-style) as an alternative to LSTM
- Multi-instrument / multi-track generation
- Real-time streaming training updates via WebSockets
- User accounts to save personal generation history
- Fine-tuning on a single uploaded custom dataset end-to-end from the UI
- Export generated music to WAV/MP3 using a soundfont renderer

---

## 📄 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use, modify, and build on it.

---

## 🙌 Acknowledgements

Built with TensorFlow, Flask, music21, and Tone.js. Created as an internship portfolio project.
