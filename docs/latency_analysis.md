# TMS Trigger Latency — Situation Summary

## The problem

Threshold-crossing → TMS-fire latency observed in `session_20260430_191420/event_00009.json` is **~140 ms** (range similar across events in that session). Target spec: **<10 ms median, <10 ms jitter**.

The log was produced via `--source rebroadcast`: data path was Quattrocento → OT BioLab+ → host. **OT BioLab+ adds its own software buffering (GUI-linked, typically 50–100 ms) on top of the amplifier's native packetisation**, which dominates the observed latency. The code was post-`a3ed826` (hooks) and pre-`76747b8` (window-semantics fix); subsequent commits don't touch the trigger path. The current `--source real` direct path through `stream/direct.py → controller.py → hooks.py` has not been measured.

## What we know (confirmed)

- **Log evidence**: R Index force crosses 10 % MVC at sample 30 in the captured window; trigger registers at sample 102. At 512 Hz that's `(102 − 30) / 512 ≈ 140 ms`. Plot confirms ~130–140 ms visually.
- **All Quattrocento channels share one ADC clock**: the 140 ms is real wall-clock delay between force crossing and TMS pulse, not a logging artefact.
- **Quattrocento direct-stream packet cadence (from OT Bioelettronica forum)** — samples per TCP packet depends on the NCH (channel-count) bit:
  - 120 ch → 12 samples/packet
  - 216 ch →  6 samples/packet
  - 312 ch →  4 samples/packet
  - 408 ch →  2 samples/packet
  - Packet period = samples-per-packet / sample-rate. At 512 Hz / 120 ch that's **23.4 ms**. At 2048 Hz / 120 ch, **5.9 ms**. At 5120 Hz / 408 ch, **~0.4 ms**.
- **Architecture in current code**:
  - `controller.py:_on_timer_tick` is driven by `QTimer(ui_refresh_ms=30 ms)`.
  - `stream/direct.py:read_batch` drains whatever is in the OS socket buffer per tick.
  - `hooks.py:PassedTenPercentRightIndex.__call__` runs on each batch; on first sample ≥ threshold it calls `_LabJackPulse.fire()`.
  - `_LabJackPulse.fire()` does `eWriteName(FIO4, 1)` → `time.sleep(0.005)` → `eWriteName(FIO4, 0)`. TMS fires on the rising edge; the sleep + low write are pulse-width shaping and block the UI thread for 5 ms (affects subsequent triggers, not this one).
- **Sample rate options** are `(512, 2048, 5120, 10240)` Hz per `protocol.py`.

## What we're inferring (high confidence)

- **Latency chain is one-way** from "force crossed the transducer" (acquisition stamp = sample 30) to "TTL rising edge at aux-in" (acquisition stamp = sample 102). Host receive time has no effect on either stamp. Components (in `--source real` direct mode):
  - Ingress: 0 to one Quattrocento packet period (23.4 ms at 512 Hz / 120 ch; smaller at higher rates / channel counts).
  - Polling: 0–30 ms — bytes sit in OS recv buffer until the next QTimer tick.
  - Hook + numpy: <1 ms.
  - LabJack USB write (FIO4 → high): ~1–4 ms.
  - Cable + TMS trigger response + aux-in ADC sample: ~1 ms.
- **Rebroadcast mode adds a BioLab buffering term in front of the chain.** BioLab aggregates packets and re-emits on its own cadence (GUI-linked, typically 50–100 ms). This term is the dominant contributor to the observed 140 ms.
- **Jitter floor = the ingress cadence** (Quattrocento packet period in direct mode; BioLab rebroadcast period in proxy mode).

## What we don't know (needs measurement)

- **Per-component latency breakdown in direct mode.** Current 140 ms is rebroadcast-mode total. We don't yet have a direct-mode number, nor a decomposition of either.
- **BioLab's actual rebroadcast cadence on this setup.** Inferred at 50–100 ms, but not measured.
- **LabJack T4 USB write latency distribution on this machine.** Manufacturer says ~1–4 ms; bench it before relying on it.

## What will work (confident)

1. **Switch to `--source real` (direct Quattrocento).** Single biggest win. Removes BioLab's buffering layer entirely; ingress floor drops from "BioLab cadence" (50–100 ms) to "Quattrocento packet cadence" (23.4 ms at 512 Hz / 120 ch, sub-10 ms at higher rates or channel counts). No code change — just CLI flag.
2. **Raise sample rate to 5120 Hz, keep decimation on.** Shrinks Quattrocento packet period ~10× from 512 Hz. Median latency and jitter drop with it. 10240 Hz buys another ~2× with diminishing returns vs storage cost.
3. **`winmm.timeBeginPeriod(1)`** at app startup. Drops Windows scheduler quantum from ~15.6 ms to 1 ms. Cheap, no downside.
4. **Dedicated socket reader thread** with blocking `recv`. Removes the 0–30 ms QTimer polling delay; polling jitter to sub-ms.
5. **Hardware-timed LabJack pulse.** FIO4 in DIO-EF pulse-out mode, arm with one write, no `time.sleep`. Doesn't reduce *this* trigger's latency (TMS fires on the rising edge), but unblocks the main thread for the next event.
6. **`Qt::PreciseTimer` flag** on remaining QTimers. Removes default ~5 % coarse-timer slop.
7. **Process priority `HIGH_PRIORITY_CLASS` + `THREAD_PRIORITY_TIME_CRITICAL` on the reader thread.** Tightens jitter under background load.

Realistic outcome:
- Direct mode at 512 Hz / 120 ch with #3–7: median ~15–25 ms, jitter ~25 ms (ingress-dominated).
- Direct mode at 2048 Hz / 120 ch with #3–7: median ~7–12 ms, jitter ~6 ms. **Within spec.**
- Direct mode at 5120 Hz with #3–7: median ~5–8 ms, jitter ~2–3 ms. **Comfortably within spec.**

## What may work (uncertain, worth trying if needed)

- **10240 Hz** — halves the packet floor again but pays in data volume (~2.5 MB/s at 120 ch, ~8 MB/s at 408 ch) and event-JSON size (~20× bigger than at 512 Hz).
- **Windows delayed-ACK tweak** (`TcpAckFrequency=1` registry). Likely sub-ms gain for a one-way stream; not the bottleneck but free.
- **Dedicated NIC / direct point-to-point cable** to the Quattrocento. Helps only if the lab LAN is congested.
- **Vectorising `_advance_trigger_dc`'s Python `for` loop** in `processing.py`. Fine at 512 Hz; at 5120 Hz the per-batch CPU rises 10×.

## What probably won't work (myths and dead ends)

- **`TCP_NODELAY` on the host receive socket.** Nagle is sender-side; the sender (Quattrocento firmware) is not reconfigurable from the host.
- **Shrinking `SO_RCVBUF`.** Not a latency knob — only changes backpressure threshold.
- **"1-sample DataBatches" / removing the `_byte_buffer`.** The host is downstream of the device's packet batching; reading in finer units doesn't make samples arrive earlier.
- **Rewriting in C, or switching to Linux PREEMPT_RT.** A few ms of jitter at best, and neither shrinks the device packet cadence nor the LabJack USB stack. Skip unless #1–7 land us still short.
- **Side-channel logging** (host-timestamping the LabJack fire instead of using aux-in). Doesn't reduce the physical delay; just hides it from the log.

## What changes things qualitatively (architectural options)

Only needed if direct mode + the above optimisations don't reach spec:

- **LabJack-as-detector.** Split the force signal to a LabJack analog input; run threshold detection in the LabJack's streaming path and fire the DIO from the LabJack itself. Quattrocento becomes a passive recorder. Expected latency ~1 ms, jitter sub-ms. Requires hardware (signal split, possibly a T7 instead of T4 for stream rate).
- **Predictive trigger.** Characterise typical latency L, fire at (10 − k) % so the TMS pulse lands near the 10 % crossing. Crude, doesn't fix jitter, acceptable as a stopgap.

## Suggested mode split

- **Experiment mode**: `--source real`, 5120 (or 10240) Hz, `timeBeginPeriod(1)`, HIGH priority, reader thread, hardware-timed LabJack pulse.
- **Review / Calibration mode**: `--source rebroadcast` if convenient, 512 Hz, normal priority, full-quality UI.

Mode set at session start; switching modes restarts the stream.

## Recommended next steps (in order)

1. **Re-run the session with `--source real`.** Same scene otherwise. Compare event-file trigger gaps. Expected: drop from ~140 ms to ~30–60 ms (512 Hz / 120 ch) without touching any code.
2. **Instrument and measure.** Add `perf_counter` taps at three points and dump into event JSON:
   - `t_packet_recv` — when `recv()` returns a batch containing the crossing sample.
   - `t_hook_seen` — when the hook detects pct ≥ threshold.
   - `t_fire` — immediately before `_hw.fire()`.
   - (`t_trigger_logged` = `time_buffer[trigger_sample]` is already in the log.)
   Decomposes the remaining latency into ingress / polling / USB.
3. **Raise sample rate to 2048 or 5120 Hz, decimation on.** Re-measure.
4. **Cheap wins**: `timeBeginPeriod(1)`, `Qt::PreciseTimer`, HIGH priority, hardware-timed LabJack pulse. Re-measure.
5. **If still short**: dedicated reader thread. Re-measure.
6. **If jitter still >10 ms and that's a hard spec**: commit to LabJack-as-detector. Scope and plan separately.

## Open questions to confirm with the user

- Is `<10 ms jitter` a hard requirement of the protocol, or a target you'd like to approach?
- Acceptable storage cost per session (controls how high we can push the sample rate)?
- Is splitting the force signal to a separate ADC (LabJack T7 + signal conditioning) on the table if needed?
