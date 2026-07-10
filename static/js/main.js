/* ==========================================================
   AI Music Generation Studio — main.js
   Handles all AJAX interactions, charts, drag-and-drop upload,
   toast notifications, and Tone.js playback.
   Only runs dashboard-specific code if APP_URLS is defined
   (i.e. we're on the dashboard page).
   ========================================================== */

document.addEventListener("DOMContentLoaded", () => {
    if (typeof window.APP_URLS === "undefined") {
        return; // Not on the dashboard page — nothing to wire up.
    }

    const URLS = window.APP_URLS;

    // -----------------------------------------------------
    // STATE
    // -----------------------------------------------------
    const state = {
        uploadedFilename: null,
        lengthOptions: [100, 200, 500, 1000],
        tempOptions: [0.3, 0.5, 0.7, 1.0, 1.2, 1.5],
        currentTrackUrl: null,
        isPlaying: false,
        pollingTimer: null,
    };

    let synth = null;
    let currentPart = null;

    // -----------------------------------------------------
    // TOAST NOTIFICATIONS
    // -----------------------------------------------------
    function showToast(message, type = "info") {
        const iconMap = {
            success: "bi-check-circle-fill text-success",
            error: "bi-x-circle-fill text-danger",
            info: "bi-info-circle-fill text-info",
        };
        const container = document.getElementById("toastContainer");

        const toastEl = document.createElement("div");
        toastEl.className = "toast align-items-center border-0";
        toastEl.style.background = "rgba(15, 23, 42, 0.95)";
        toastEl.style.color = "#fff";
        toastEl.style.border = "1px solid rgba(255,255,255,0.15)";
        toastEl.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <i class="bi ${iconMap[type] || iconMap.info} me-2"></i>${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        container.appendChild(toastEl);
        const bsToast = new bootstrap.Toast(toastEl, { delay: 4000 });
        bsToast.show();
        toastEl.addEventListener("hidden.bs.toast", () => toastEl.remove());

        logToConsole(message, type);
    }

    function logToConsole(message, type = "info") {
        const consoleEl = document.getElementById("logConsole");
        if (!consoleEl) return;
        const line = document.createElement("div");
        line.className = `log-line ${type}`;
        const time = new Date().toLocaleTimeString();
        line.textContent = `[${time}] ${message}`;
        consoleEl.appendChild(line);
        consoleEl.scrollTop = consoleEl.scrollHeight;
    }

    // -----------------------------------------------------
    // FETCH HELPER
    // -----------------------------------------------------
    async function apiRequest(url, options = {}) {
        try {
            const response = await fetch(url, options);
            const data = await response.json();
            if (!response.ok || data.success === false) {
                throw new Error(data.error || "Request failed.");
            }
            return data;
        } catch (err) {
            showToast(err.message || "Network error. Please try again.", "error");
            throw err;
        }
    }

    // -----------------------------------------------------
    // DATASET INFO
    // -----------------------------------------------------
    async function loadDatasetInfo() {
        const style = document.getElementById("styleSelect")?.value || "mixed";
        try {
            const res = await apiRequest(`${URLS.datasetInfo}?style=${style}`);
            const data = res.data;
            document.getElementById("statSongs").textContent = data.num_songs ?? "--";
            document.getElementById("statVocab").textContent = data.vocabulary_size ?? "--";
        } catch (err) {
            // Already toasted.
        }
    }

    // -----------------------------------------------------
    // MODEL STATUS
    // -----------------------------------------------------
    async function loadModelStatus() {
        try {
            const res = await apiRequest(URLS.modelStatus);
            const { is_trained, info } = res.data;

            const statusEl = document.getElementById("statModelStatus");
            statusEl.textContent = is_trained ? "Trained" : "Not Trained";
            statusEl.style.color = is_trained ? "var(--success)" : "var(--danger)";

            const infoBody = document.getElementById("modelInfoBody");
            if (info) {
                document.getElementById("statAccuracy").textContent =
                    info.final_val_accuracy ? (info.final_val_accuracy * 100).toFixed(1) + "%" : "--";

                infoBody.innerHTML = `
                    <ul class="list-unstyled small mb-0">
                        <li><strong>Vocabulary Size:</strong> ${info.vocab_size ?? "--"}</li>
                        <li><strong>Sequence Length:</strong> ${info.sequence_length ?? "--"}</li>
                        <li><strong>Total Parameters:</strong> ${info.total_params?.toLocaleString() ?? "--"}</li>
                        <li><strong>Epochs Trained:</strong> ${info.epochs_trained ?? "--"}</li>
                        <li><strong>Training Time:</strong> ${info.training_time ?? "--"}</li>
                        <li><strong>Final Loss:</strong> ${info.final_loss?.toFixed(4) ?? "--"}</li>
                        <li><strong>Final Accuracy:</strong> ${info.final_accuracy ? (info.final_accuracy * 100).toFixed(2) + "%" : "--"}</li>
                        <li><strong>Trained At:</strong> ${info.trained_at ?? "--"}</li>
                    </ul>
                `;
            }
        } catch (err) {
            // Already toasted.
        }
    }

    // -----------------------------------------------------
    // TRAINING
    // -----------------------------------------------------
    document.getElementById("trainBtn")?.addEventListener("click", async () => {
        const epochs = parseInt(document.getElementById("epochsInput").value, 10);
        const batchSize = parseInt(document.getElementById("batchSizeInput").value, 10);
        const style = document.getElementById("styleSelect").value;

        const btn = document.getElementById("trainBtn");
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Starting...`;

        try {
            await apiRequest(URLS.train, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ epochs, batch_size: batchSize, style }),
            });
            showToast("Training started! This may take a while depending on dataset size.", "success");
            document.getElementById("trainingProgressWrap").classList.remove("d-none");
            startProgressPolling();
        } catch (err) {
            // Already toasted.
        } finally {
            btn.disabled = false;
            btn.innerHTML = `<i class="bi bi-play-fill"></i> Start Training`;
        }
    });

    function startProgressPolling() {
        if (state.pollingTimer) clearInterval(state.pollingTimer);

        state.pollingTimer = setInterval(async () => {
            try {
                const res = await fetch(URLS.trainingProgress);
                const data = await res.json();
                const progress = data.data;

                if (!progress || progress.status === "idle") return;

                if (progress.status === "error") {
                    showToast(`Training error: ${progress.message}`, "error");
                    clearInterval(state.pollingTimer);
                    return;
                }

                document.getElementById("trainingEpochLabel").textContent =
                    `Epoch ${progress.epoch ?? 0}/${progress.total_epochs ?? 0}`;
                document.getElementById("trainingEtaLabel").textContent = `ETA: ${progress.eta ?? "--"}`;
                document.getElementById("trainingProgressBar").style.width = `${progress.progress_percent ?? 0}%`;
                document.getElementById("metricLoss").textContent = progress.loss?.toFixed(4) ?? "--";
                document.getElementById("metricAcc").textContent = progress.accuracy ? (progress.accuracy * 100).toFixed(1) + "%" : "--";
                document.getElementById("metricValLoss").textContent = progress.val_loss?.toFixed(4) ?? "--";
                document.getElementById("metricValAcc").textContent = progress.val_accuracy ? (progress.val_accuracy * 100).toFixed(1) + "%" : "--";

                if (progress.status === "completed") {
                    clearInterval(state.pollingTimer);
                    showToast(`Training completed in ${progress.training_time}!`, "success");
                    loadModelStatus();
                    loadTrainingGraphs();
                }
            } catch (err) {
                // Silently ignore polling errors to avoid toast spam.
            }
        }, 3000);
    }

    // -----------------------------------------------------
    // TRAINING GRAPHS (Chart.js — rendered from saved PNGs' underlying data
    // isn't available client-side, so we display the saved images instead
    // by drawing them onto the canvases as a background, OR simply embed
    // the PNGs directly. Here we swap canvases for images for reliability.)
    // -----------------------------------------------------
    function loadTrainingGraphs() {
        replaceCanvasWithImage("lossChart", URLS.lossGraph);
        replaceCanvasWithImage("accuracyChart", URLS.accuracyGraph);
        replaceCanvasWithImage("lrChart", URLS.lrGraph);
    }

    function replaceCanvasWithImage(canvasId, imageUrl) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const img = document.createElement("img");
        img.src = `${imageUrl}?t=${Date.now()}`;
        img.className = "img-fluid rounded";
        img.alt = canvasId;
        img.onerror = () => { img.style.display = "none"; };
        canvas.replaceWith(img);
        img.id = canvasId;
    }

    // -----------------------------------------------------
    // GENERATE MUSIC — sliders
    // -----------------------------------------------------
    const lengthSlider = document.getElementById("lengthSlider");
    const lengthValueEl = document.getElementById("lengthValue");
    lengthSlider?.addEventListener("input", () => {
        lengthValueEl.textContent = state.lengthOptions[lengthSlider.value];
    });

    const tempSlider = document.getElementById("tempSlider");
    const tempValueEl = document.getElementById("tempValue");
    tempSlider?.addEventListener("input", () => {
        tempValueEl.textContent = state.tempOptions[tempSlider.value];
    });

    document.getElementById("generateBtn")?.addEventListener("click", async () => {
        const btn = document.getElementById("generateBtn");
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Generating...`;

        const numNotes = state.lengthOptions[lengthSlider.value];
        const temperature = state.tempOptions[tempSlider.value];
        const style = document.getElementById("styleSelect").value;
        const customSeed = document.getElementById("customSeedInput").value;

        try {
            const res = await apiRequest(URLS.generate, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    num_notes: numNotes,
                    temperature,
                    style,
                    custom_seed: customSeed,
                }),
            });
            showToast(`Generated "${res.data.filename}" in ${res.data.generation_time}!`, "success");
            loadTrack(res.data);
            loadHistory();
        } catch (err) {
            // Already toasted.
        } finally {
            btn.disabled = false;
            btn.innerHTML = `<i class="bi bi-stars"></i> Generate Music`;
        }
    });

    // -----------------------------------------------------
    // UPLOAD MIDI (drag & drop + click)
    // -----------------------------------------------------
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");

    dropZone?.addEventListener("click", () => fileInput.click());

    dropZone?.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone?.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

    dropZone?.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput?.addEventListener("change", () => {
        if (fileInput.files.length) {
            handleFileUpload(fileInput.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (!file.name.match(/\.(mid|midi)$/i)) {
            showToast("Please upload a valid .mid or .midi file.", "error");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await apiRequest(URLS.uploadMidi, { method: "POST", body: formData });
            state.uploadedFilename = res.data.filename;
            document.getElementById("uploadedFileName").innerHTML =
                `<i class="bi bi-file-earmark-check text-success"></i> ${file.name} uploaded successfully.`;
            document.getElementById("continueBtn").disabled = false;
            showToast("MIDI file uploaded!", "success");
        } catch (err) {
            // Already toasted.
        }
    }

    document.getElementById("continueBtn")?.addEventListener("click", async () => {
        if (!state.uploadedFilename) {
            showToast("Please upload a MIDI file first.", "error");
            return;
        }

        const btn = document.getElementById("continueBtn");
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Continuing...`;

        const numNotes = state.lengthOptions[lengthSlider.value];
        const temperature = state.tempOptions[tempSlider.value];

        try {
            const res = await apiRequest(URLS.continueMelody, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filename: state.uploadedFilename,
                    num_notes: numNotes,
                    temperature,
                }),
            });
            showToast(`Melody continued: "${res.data.filename}"`, "success");
            loadTrack(res.data);
            loadHistory();
        } catch (err) {
            // Already toasted.
        } finally {
            btn.disabled = false;
            btn.innerHTML = `<i class="bi bi-play-fill"></i> Continue Melody`;
        }
    });

    // -----------------------------------------------------
    // HISTORY LIST
    // -----------------------------------------------------
    async function loadHistory() {
        try {
            const res = await apiRequest(URLS.history);
            const items = res.data.generation_history || [];
            const listEl = document.getElementById("historyList");

            if (!items.length) {
                listEl.innerHTML = `<p class="text-muted-light">No music generated yet.</p>`;
                return;
            }

            listEl.innerHTML = items.map(item => `
                <div class="history-item">
                    <div>
                        <strong>${item.filename}</strong>
                        <div class="text-muted-light small">
                            ${item.type === "continued" ? "Continuation" : "Generated"} &middot;
                            ${item.num_notes ?? item.continuation_notes ?? "--"} notes &middot;
                            Temp ${item.temperature ?? "--"} &middot;
                            ${item.created_at}
                        </div>
                    </div>
                    <div class="d-flex gap-2">
                        <button class="btn btn-sm btn-outline-glass" data-play="${item.filename}"><i class="bi bi-play-fill"></i></button>
                        <a class="btn btn-sm btn-outline-glass" href="/download/${item.filename}" download><i class="bi bi-download"></i></a>
                    </div>
                </div>
            `).join("");

            listEl.querySelectorAll("[data-play]").forEach(btn => {
                btn.addEventListener("click", () => {
                    const filename = btn.getAttribute("data-play");
                    loadTrack({ filename, download_url: `/download/${filename}`, play_url: `/play/${filename}` });
                });
            });
        } catch (err) {
            // Already toasted.
        }
    }

    // -----------------------------------------------------
    // PLAYBACK (Tone.js) + ANIMATED PIANO KEYBOARD
    // -----------------------------------------------------
    function buildPianoKeyboard() {
        const keyboard = document.getElementById("pianoKeyboard");
        if (!keyboard || keyboard.childElementCount > 0) return;
        const notes = ["C", "D", "E", "F", "G", "A", "B"];
        for (let octave = 3; octave <= 5; octave++) {
            notes.forEach(n => {
                const key = document.createElement("div");
                key.className = "piano-key";
                key.dataset.note = `${n}${octave}`;
                keyboard.appendChild(key);
            });
        }
    }

    function flashKey(noteName) {
        const octaveMatch = noteName.match(/[A-G]#?\d/);
        if (!octaveMatch) return;
        const simplified = noteName.replace("#", ""); // approximate visual match
        const keyEl = document.querySelector(`.piano-key[data-note="${simplified}"]`);
        if (keyEl) {
            keyEl.classList.add("active");
            setTimeout(() => keyEl.classList.remove("active"), 200);
        }
    }

    function loadTrack(trackData) {
        state.currentTrackUrl = trackData.play_url || `/play/${trackData.filename}`;
        document.getElementById("nowPlayingLabel").textContent = trackData.filename;
        document.getElementById("playBtn").disabled = false;
        document.getElementById("stopBtn").disabled = false;

        const downloadBtn = document.getElementById("downloadBtn");
        downloadBtn.href = trackData.download_url || `/download/${trackData.filename}`;
        downloadBtn.classList.remove("disabled");
    }

    document.getElementById("playBtn")?.addEventListener("click", async () => {
        if (!state.currentTrackUrl) return;

        try {
            await Tone.start();
            if (!synth) synth = new Tone.PolySynth(Tone.Synth).toDestination();

            const midi = await Midi.fromUrl(state.currentTrackUrl);
            Tone.Transport.stop();
            Tone.Transport.cancel();

            midi.tracks.forEach(track => {
                track.notes.forEach(note => {
                    Tone.Transport.schedule((time) => {
                        synth.triggerAttackRelease(note.name, note.duration, time, note.velocity);
                        flashKey(note.name);
                    }, note.time);
                });
            });

            Tone.Transport.start();
            state.isPlaying = true;
            showToast("Playing generated music...", "info");
        } catch (err) {
    console.error(err);
    showToast("Playback failed. Try downloading the MIDI file instead.", "error");
}
    });

    document.getElementById("stopBtn")?.addEventListener("click", () => {
        Tone.Transport.stop();
        Tone.Transport.cancel();
        state.isPlaying = false;
    });

    // -----------------------------------------------------
    // STYLE SELECT -> refresh dataset info
    // -----------------------------------------------------
    document.getElementById("styleSelect")?.addEventListener("change", loadDatasetInfo);

    // -----------------------------------------------------
    // INITIAL LOAD
    // -----------------------------------------------------
    buildPianoKeyboard();
    loadDatasetInfo();
    loadModelStatus();
    loadTrainingGraphs();
    loadHistory();
});

/* ==========================================================
   Minimal @tonejs/midi loader shim.
   The Tone.js CDN bundle does not include the MIDI file parser,
   so we lazy-load it from CDN only when playback is triggered.
   ========================================================== */
(function loadMidiParser() {
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/tonejs-midi/2.0.28/Midi.min.js";
    document.head.appendChild(script);
})();
