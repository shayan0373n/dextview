# DextView: Triggered Force & EMG Capture Application

Python/PyQt5 GUI for force and EMG capture from the **OT Bioelettronica Quattrocento**. Supports calibration, trigger-based event capture, and LabJack TTL feedback.

---

## Features

- **Connects to the Quattrocento** in three modes: direct, rebroadcast through OT BioLab+, or proxy. See [User Guide §2](USER_GUIDE.md#2-choosing-a-connection-mode).
- **Real-time visualization** of force, EMG, and trigger channels, with toggleable bandpass (10–500 Hz) and powerline-notch (50/100/150 Hz) filters on EMG.
- **Calibration system** (rest, MVC, zero) with save/load to NumPy `.npz` files. Plots can be displayed in % MVC once calibrated.
- **Trigger-based capture** of fixed-length windows around analog trigger events, with per-finger force profiles, peak-to-peak readouts, and auto onset detection.
- **Closed-loop feedback hooks** that emit LabJack TTL pulses on configurable force conditions.
- **JSON event logging** of every captured window. See [User Guide §7](USER_GUIDE.md#7-logging-captures).

---

## Installation

### Prerequisites

- **Python 3.14** or later
- **Optional**: Python environment manager (e.g., Conda)

### Quick Setup
1.  **Clone the repository** and navigate to the project directory:
    ```bash
    git clone https://github.com/shayan0373n/dextview.git
    cd dextview
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

Run `dextview --help` for the full flag list.

---

## Testing

Run from the repository root:
```bash
python -m pytest
```

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
