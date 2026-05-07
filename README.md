Quattrocento Triggered Force Application

A modular GUI application for capturing and analyzing force sensor data from an OT Bioelettronica Quattrocento stream.

- `run_quattrocento.py`
- package: `quattrocento/`

## Architecture Layers

The application is structured logically from top to bottom:

- **App (`app.py`)**: Application entry point, CLI argument parsing, and dependency injection (wiring components together).
- **UI (`ui.py`)**: The presentation layer. PyQt5 user interface, real-time charting, and user controls.
- **Controller (`controller.py`)**: The middleman orchestrator. Coordinates the data stream, processing logic, and UI updates using Qt signals.
- **Processing (`processing.py`)**: The domain logic. Handles continuous data buffering, trigger edge detection, and extracting fixed-length capture windows.
- **Device (`device.py`, `protocol.py`)**: The data access layer. Manages the socket connection to the physical Quattrocento device (binary protocol) or the OT BioLab+ rebroadcast server (text + binary protocol).
- **Simulator (`simulator.py`)**: A standalone TCP server that mimics the BioLab+ rebroadcast stream for local development and demos.
- **Rebroadcast Detection (`rebroadcast_detect.py`)**: Automatic discovery of channel counts and sampling rates from rebroadcast streams.
- **Config (`config.py`, `settings.py`)**: Shared configuration data structures and TOML file loading used across all layers.

## Usage

Run the application with:

`python run_quattrocento.py --source real` (direct connection)
OR
`python run_quattrocento.py --source rebroadcast` (OT BioLab+ or simulator)

### Local Development with Simulator

1. Start the simulator in one terminal:
   `python -m quattrocento.simulator`
2. Start the main app in another:
   `python run_quattrocento.py --source rebroadcast --host 127.0.0.1 --port 31000`

The application features:

- 10 force sensors mapped one-to-one to 10 fingers.
- Analog AUX-in trigger detection for event-based capturing.
- Capture window after each trigger.
- Event history with navigation (Prev/Next buttons or Left/Right arrow keys).
- Real-time visualization of raw finger forces and peak force ranges.

## Real Quattrocento Source

To connect to a real device, use `--source real` and configure socket/channel settings in:

- `quattrocento/socket_stream_config.toml`

The TOML file exposes:

- `rec_on`, `fsamp`, `nch`, `decimation_enabled` (`ACQ_SETT` bits)
- `force_channel_indices`, `aux_in_channel_index`
- `conf2_defaults` and `conf2_overrides` for per-input `hpf` / `lpf` / `mode` (and `side`)

Run with:

`python run_quattrocento.py --source real`
