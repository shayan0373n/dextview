Quattrocento Triggered Force Application

A modular GUI application for capturing and analyzing force sensor data from an OT Bioelettronica Quattrocento stream.

- `run_quattrocento.py`
- package: `quattrocento/`

## Usage

Run the application with:

`python run_quattrocento.py`

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
