# DextView User Guide

Usage guide for DextView. For installation, architecture, and CLI reference, see [README.md](README.md).

---

## 1. Before You Start

*One short paragraph orienting the reader: what DextView does in plain language (capture force + EMG around trigger events, with calibration and optional hardware feedback), and what a typical session looks like end-to-end.*

**You will need:**
- A computer with DextView installed (see README → Installation).
- One of: a Quattrocento device on the network, the OT BioLab+ application running locally, or the bundled simulator for dry runs.
- *(Optional)* A LabJack T4 connected via USB, if you plan to use closed-loop feedback hooks.
- A channel mapping file (`configs/channels_default.toml` is the starting point).

**Session at a glance:**
1. Pick a connection mode and launch DextView.
2. Verify the live signal looks right.
3. Calibrate (Rest → MVC → optionally Zero).
4. *(Optional)* Arm a feedback hook.
5. Run the task; captures are logged automatically if `--log-dir` is set.
6. Review captures in the visualizer; save calibration before closing.

---

## 2. Choosing a Connection Mode

DextView supports three connection modes:

| Situation | Mode |
| :--- | :--- |
| Lowest latency; manual hardware configuration acceptable | **Direct** |
| OT BioLab+ is already configured | **Rebroadcast** |
| OT BioLab+ records while DextView taps the stream for low-latency hooks | **Proxy** |
| No hardware available; testing the GUI | **Simulator + Rebroadcast** |

### 2.1 Direct (`--source real`)
*DextView connects straight to the Quattrocento hardware.*

- **Use when:** lowest latency is required and OT BioLab+ is not needed.
- **Setup:** *(launch command — `--host`/`--port` at the device, plus `--sample-rate`, `--n-channels`, and optionally `--conf2-config` for hardware input-block settings).*
- **Channel ordering:** the last 24 channels in the stream are always the 16 AUX channels followed by the 8 accessory channels — TOML channel indices must account for this.
- **Tradeoffs:** lowest latency; hardware must be configured manually.

### 2.2 Rebroadcast (`--source rebroadcast`)
*DextView listens to a stream that OT BioLab+ is broadcasting locally.*

- **Use when:** OT BioLab+ is already handling hardware configuration and DextView only needs to overlay its analysis.
- **Setup:** *(start OT BioLab+ → enable rebroadcast → launch DextView pointed at `127.0.0.1:31000`).*
- **Auto-detection:** set `--n-channels auto` and `--sample-rate auto` to read these from the OT BioLab+ stream header.
- **Channel ordering:** indices match the channels activated in OT BioLab+, with the last 8 channels always being the accessory channels. **Changing OT BioLab+'s channel selection requires updating the TOML mapping.**
- **Tradeoffs:** easiest setup; rebroadcast adds buffering delay — not ideal for time-sensitive hooks.

### 2.3 Proxy (`--source proxy`)
*DextView sits between the device and OT BioLab+, forwarding commands and tapping the data stream.*

- **Use when:** OT BioLab+ must keep recording while DextView taps the stream first for low-latency hooks.
- **Setup:**
  1. Launch DextView in proxy mode with `--host` pointing at the device and `--proxy-listen-host`/`--proxy-listen-port` set to a local address.
  2. In OT BioLab+, change the target device IP to the proxy address (typically `127.0.0.1`).
  3. Start acquisition in OT BioLab+. It connects to DextView, which forwards to the device.
- **Channel ordering:** same as Direct mode.
- **Tradeoffs:** OT BioLab+ records normally and DextView gets first-hand data; requires reconfiguring OT BioLab+'s target IP.

### 2.4 Simulator (dry run)
*A bundled fake device that emits dummy force, EMG, and trigger signals.*

- **Use when:** practicing the workflow, demoing, or testing without hardware.
- **Setup:** run `python run_simulator.py --trigger-interval 8.0` in one terminal, then launch DextView in `rebroadcast` mode pointed at `127.0.0.1:31000`.

---

## 3. Channel Configuration

*What the TOML file actually does, and when you need to touch it.*

- **Channel kinds:** `finger` (force, typically 10), `emg` (muscle activity), `trigger` (AUX channel that fires capture events).
- **`scale` field:** converts raw int16 samples into physical units. *(Brief note on units; full formula in README.)*
- **Edit the TOML when:** OT BioLab+'s active channels change (rebroadcast mode) or sensors are rewired.
- **Swapping configs without restarting:** *(if/how that's supported — TBD: confirm).*

---

## 4. Calibration

**Order matters: Rest → MVC → Zero (if needed).** Recalibrate at the start of every session and after any change to sensor placement.

### 4.1 Rest Calibration
*Establishes the baseline force for each finger at rest. Use the **Cal ▾ → Calibrate Rest** menu. The participant should be relaxed; DextView averages over the capture window.*

### 4.2 MVC Calibration
*Records the peak force per finger during a maximum voluntary contraction. Unlocks the **% MVC** toggle on the main plot.*

### 4.3 Zero Calibration
*Captures a true unloaded reference (e.g., sensors removed or detached). Used when baseline drift between sessions matters.*

### 4.4 Saving and Loading
*Save calibrations to a `.npz` file via the Cal menu. Load at the start of a session if reusing the same participant/setup.*

---

## 5. Running a Session

### 5.1 Live Monitors (`Live ▾`)

Open these *before* starting captures so you can confirm the signal looks right.

- **Trigger Channel Monitor** — confirms the trigger channel is wired correctly. Shows the adaptive threshold tracking the baseline.
- **Force Live Monitor** — rolling 10-second force traces; check sensors are responsive.
- **EMG Live Monitor** — rolling 10-second EMG traces. Toggle the bandpass (10–500 Hz) and 50/100/150 Hz powerline notch filters if you see line noise.

### 5.2 Capturing Trigger Events

*When the trigger channel crosses threshold, DextView automatically captures a fixed-length window.*

- **Window length and pre-trigger offset** are set on the command line (`--window-seconds`, `--window-offset`). *(Pointer to the table in README.)*
- The main visualizer plots all 10 finger profiles with P2P readouts and auto-detected onset markers.

### 5.3 Onset Markers

*Auto-detected dashed lines marking force onset per finger.*

- Drag a marker to override the auto-detection.
- Right-click a marker to reset it to the auto-detected position.

### 5.4 Browsing Capture History

*Use **< Prev** / **Next >** or the Left/Right arrow keys to navigate captures from the current session.*

---

## 6. Closed-Loop Feedback (Hooks)

*Hooks send a 5 ms TTL pulse on a LabJack T4 (pin FIO4) when a force condition is met — used to trigger external equipment (TMS, stimulators, recorders).*

**Requirements:** LabJack T4 connected via USB before launching DextView. Calibration (at minimum Rest + MVC) must be complete.

### 6.1 Any Finger Threshold
*Arms on force onset; fires when any finger crosses the % MVC threshold; re-arms when forces drop below the release threshold. HUD shows the running maximum.*

### 6.2 Hold In Target
*Tracks time spent inside a target % MVC band (e.g., 30% ± 20%). Fires once dwell time is reached and repeats periodically while held.*

---

## 7. Logging Captures

*If you launched with `--log-dir`, every captured event is written to disk as JSON.*

- **Where:** `<log-dir>/session_<timestamp>/event_NNNNN.json`.
- **What's in each file:** trigger timestamp/sample index, device config, channel metadata, calibration arrays, full timestamp + signal arrays.
- **Recommendation:** always pass `--log-dir` for real sessions, even if you don't think you'll need the data — it's cheap insurance.

---

## 8. Troubleshooting

*Short list, one-paragraph each — to be filled in based on actual common failures.*

- **DextView won't connect** — wrong mode, wrong host/port, OT BioLab+ not started yet, firewall.
- **No captures firing** — trigger channel wrong in TOML, threshold too high, no signal on AUX.
- **Force values look wrong / clipped** — channel ordering off in TOML (especially after changing OT BioLab+ config), wrong `scale`, wrong sensor wired to wrong input.
- **% MVC toggle is greyed out** — MVC calibration not yet completed this session.
- **Hooks not firing the TTL** — LabJack not detected (connect before launch), or threshold conditions never met.
- **`auto` sample-rate/n-channels fails** — only supported in rebroadcast mode; supply explicit values for direct/proxy.

---

## 9. Glossary

- **MVC** — Maximum Voluntary Contraction. The peak force a participant can produce.
- **% MVC** — force expressed as a percentage of MVC, used for thresholds and hooks.
- **P2P** — Peak-to-peak; max minus min within the capture window.
- **Baseline** — the resting force level subtracted before normalization.
- **Trigger** — an analog event (AUX channel crossing threshold) that initiates a capture.
- **AUX / Accessory channels** — the last channels in the device stream; AUX (16) come before accessory (8) in direct/proxy mode.
- **Hook** — a closed-loop rule that emits a TTL pulse when a force condition is met.
