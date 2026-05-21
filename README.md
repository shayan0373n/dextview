# DextView: Triggered Force & EMG Capture Application

Python/PyQt5 GUI for force and EMG capture from the **OT Bioelettronica Quattrocento**. Supports calibration, trigger-based event capture, and LabJack TTL feedback.

---

## Features

- **Connects to the Quattrocento** via three modes — direct, rebroadcast through OT BioLab+, or proxy. See [User Guide §2](USER_GUIDE.md#2-choosing-a-connection-mode).
- **Real-time visualization** of force, EMG, and trigger channels, with toggleable bandpass (10–500 Hz) and powerline-notch (50/100/150 Hz) filters on EMG.
- **Calibration system** — rest, MVC, and zero — with save/load to NumPy `.npz` files. Plots can be displayed in % MVC once calibrated.
- **Trigger-based capture** of fixed-length windows around analog trigger events, with per-finger force profiles, peak-to-peak readouts, and auto onset detection.
- **Closed-loop feedback hooks** that emit a 5 ms TTL pulse on a LabJack T4 (FIO4) based on force conditions (any-finger threshold or hold-in-target).
- **JSON event logging** of every captured window — see [Event Log Format](#event-log-format) below.

Channel mapping is configured via TOML. See [`configs/channels_default.toml`](configs/channels_default.toml) for the default 10-finger + EMG + trigger setup.

---

## Installation

### Prerequisites
Make sure you have [Conda](https://docs.conda.io/en/latest/) or a Python virtual environment ready. DextView requires Python 3.14 or later.

### Quick Setup
1.  **Clone the repository** and navigate to the project directory:
    ```bash
    git clone https://github.com/shayan0373n/pyquattrocento.git
    cd pyquattrocento
    ```

2.  **Activate your environment** (e.g., using `dexterity`):
    ```bash
    conda activate dexterity
    ```

3.  **Install DextView**:
    Regular install:
    ```bash
    pip install .
    ```
    Editable install (for development):
    ```bash
    pip install -e .
    ```
    Both install dependencies and register the `dextview` command.

---

## Usage

**Full user guide:** [USER_GUIDE.md](USER_GUIDE.md)

### Launch Examples

**Direct connection to the device:**
```bash
dextview --source real \
         --channels configs/channels_default.toml \
         --host 169.254.1.10 --port 23456 \
         --sample-rate 2048 --n-channels 16 \
         --conf2-config configs/quattrocento_conf2.toml
```

**Rebroadcast via OT BioLab+ (auto-detected parameters):**
```bash
dextview --source rebroadcast \
         --channels configs/channels_default.toml \
         --host 127.0.0.1 --port 31000 \
         --sample-rate auto --n-channels auto \
         --log-dir ./capture_logs
```

**Proxy mode (tap stream while OT BioLab+ records):**
```bash
dextview --source proxy \
         --channels configs/channels_default.toml \
         --host 169.254.1.10 --port 23456 \
         --proxy-listen-host 127.0.0.1 --proxy-listen-port 31001
```

**Local dry run with the bundled simulator** (terminal 1 runs the synthetic device, terminal 2 runs DextView):
```bash
python run_simulator.py --trigger-interval 8.0
```
```bash
dextview --source rebroadcast \
         --channels configs/channels_default.toml \
         --host 127.0.0.1 --port 31000
```

### Flags
Run `dextview --help` for the complete list:

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `str` | *Required* | Data source type: `real`, `rebroadcast`, or `proxy`. |
| `--channels` | `str` | *Required* | Path to the TOML channel mapping file. |
| `--host` | `str` | `None` | TCP hostname/IP for the device or rebroadcast server. |
| `--port` | `int` | `None` | TCP port for the device or rebroadcast server. |
| `--window-seconds` | `float` | `5.0` | Total length in seconds of the event-captured window. |
| `--window-offset` | `float` | `0.0` | Pre-trigger offset in seconds (e.g., `-1.0` to capture 1.0s before the trigger). |
| `--trigger-threshold`| `float` | `0.5` | Threshold for detecting manual analog trigger events (physical units). |
| `--sample-rate` | `int/str`| `None` | Sampling rate in Hz (e.g., `2048`, `10244`). `auto` for rebroadcast. |
| `--n-channels` | `int/str`| `None` | Number of channels per frame. `auto` for rebroadcast. |
| `--log-dir` | `str` | `None` | Directory path where captured events will be written as JSON. |
| `--conf2-config` | `str` | `None` | Optional hardware input-block settings file (real source only). |

---

## Signal Conversions & Scaling

### Physical Unit Scaling
Raw data is streamed as 16-bit signed integers. They are scaled into physical units by:
$$\text{Signal}_{\text{physical}} = \frac{\text{Raw}_{\text{int16}}}{32768} \times \text{scale}$$
Where `scale` is the channel-specific conversion factor defined in your `--channels` TOML.

### MVC Normalization
When rest and MVC calibrations are completed, the normalized force is calculated as:
$$\text{Force}_{(\%\text{ MVC})} = \frac{\text{Force}_{\text{physical}} - \text{Baseline}_{\text{physical}}}{\text{Peak}_{\text{physical}} - \text{Baseline}_{\text{physical}}} \times 100$$
Plot readouts and threshold hooks use this normalized value.

---

## Event Log Format

If launched with `--log-dir`, DextView writes one JSON file per captured trigger event to `<log-dir>/session_<timestamp>/event_NNNNN.json`. Each file contains:

- Trigger timestamp and sample index.
- Device configuration and channel metadata (labels, scales, kinds).
- Baseline, MVC, and zero calibration arrays at the time of capture.
- Full timestamp and signal arrays for the captured window.

---

## Repository Architecture

```
├── configs/                     # TOML configuration files
│   ├── channels_default.toml    # Default 10-finger + EMG + Trigger mapping
│   └── quattrocento_conf2.toml  # Input-block hardware configurations (HPF/LPF/modes)
├── dextview/                    # Core Python package
│   ├── hooks/                   # Low-latency feedback hooks and HUD widgets
│   │   ├── __init__.py
│   │   ├── compositors.py       # Hook coordination logic
│   │   ├── logic.py             # LabJack TTL pulse driver and state machines
│   │   └── ui.py                # Hook visual progress and target HUD dialogs
│   ├── stream/                  # Data access streams
│   │   ├── __init__.py
│   │   ├── _io.py               # Raw TCP socket abstractions
│   │   ├── direct.py            # Device direct interface
│   │   ├── parser.py            # int16 TCP frame decoding
│   │   ├── proxy.py             # Upstream client/device proxy tapping
│   │   └── rebroadcast.py       # Rebroadcast receiver
│   ├── app.py                   # Application entry point & CLI parsing
│   ├── capture_log.py           # JSON event recording hook
│   ├── channels.py              # Channels parser and mapping logic
│   ├── config.py                # Application state parameters
│   ├── controller.py            # Middleman orchestrator (wires model to view)
│   ├── models.py                # Shared data structures and protocol definitions
│   ├── processing.py            # Signal processing and trigger detection logic
│   ├── protocol.py              # Quattrocento network communication specifications
│   ├── rebroadcast_detect.py    # Auto-discovery of rebroadcast parameters
│   ├── settings.py              # Hardware command configuration builders
│   ├── simulator.py             # Standalone Quattrocento TCP server simulator
│   └── ui.py                    # Main PyQt5 layouts, charts, and monitors
├── tests/                       # Unit and integration tests
├── pyproject.toml               # Python package metadata and scripts
├── run_dextview.py              # Convenience script to run the GUI
└── run_simulator.py             # Convenience script to run the simulator
```
