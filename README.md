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
- **Stream (`stream/`, `protocol.py`)**: The data access layer. Three concrete stream types: `DirectStream` (direct device connection), `RebroadcastStream` (OT BioLab+ / simulator), and `ProxyStream` (intercepts an upstream controller's session and taps the data locally).
- **Simulator (`simulator.py`)**: A standalone TCP server that mimics the BioLab+ rebroadcast stream for local development and demos.
- **Rebroadcast Detection (`rebroadcast_detect.py`)**: Automatic discovery of channel counts and sampling rates from rebroadcast streams.
- **Config (`config.py`, `settings.py`)**: Shared configuration data structures and TOML file loading used across all layers.

## Usage

`--channels` and `--trigger-channel` are required for all source types. See [Channel configuration](#channel-configuration) below.

Run the application with:

```
python run_quattrocento.py --source real \
    --channels quattrocento/channels_default.toml \
    --trigger-channel 10 \
    --host <device-ip> --port <device-port> \
    --sample-rate 2048 --n-channels 16
```
```
python run_quattrocento.py --source rebroadcast \
    --channels quattrocento/channels_default.toml \
    --trigger-channel 10 \
    --host <host> --port <port>
```
```
python run_quattrocento.py --source proxy \
    --channels quattrocento/channels_default.toml \
    --trigger-channel 10 \
    --host <device-ip> --port <device-port>
```

### Local Development with Simulator

1. Start the simulator in one terminal:
   `python -m quattrocento.simulator`
2. Start the main app in another:
   ```
   python run_quattrocento.py --source rebroadcast \
       --channels quattrocento/channels_default.toml \
       --trigger-channel 10 \
       --host 127.0.0.1 --port 31000
   ```

The application features:

- 10 force sensors mapped one-to-one to 10 fingers.
- Analog AUX-in trigger detection for event-based capturing.
- Capture window after each trigger.
- Event history with navigation (Prev/Next buttons or Left/Right arrow keys).
- Real-time visualization of raw finger forces and peak force ranges.

Run `python run_quattrocento.py --help` for the full flag reference.

## Channel configuration

The `--channels` file is a TOML file mapping human-readable labels to channel indices and physical-unit scale factors:

```toml
[labels]
"L Thumb"  = { index = 0, scale = 5 }
"L Index"  = { index = 1, scale = 5 }
# ... more channels ...
"trigger"  = { index = 10, scale = 5.0 }
```

`scale` converts raw int16 values to physical units: `signal_physical = raw_int16 / 32768 * scale`. The trigger threshold (`--trigger-threshold`, default `0.5`) is in these same physical units. `quattrocento/channels_default.toml` is a ready-to-use example for the bundled simulator.

## Real Quattrocento Source

To connect to a real device, use `--source real` with `--host`, `--port`, `--sample-rate`, and `--n-channels`. Hardware input-block settings (`hpf`, `lpf`, `mode`, etc.) can be supplied via `--conf2-config <file>` (TOML); omitting it applies firmware defaults.
