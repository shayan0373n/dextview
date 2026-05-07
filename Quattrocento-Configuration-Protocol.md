# Quattrocento Configuration Protocol

Source: *Quattrocento configuration protocol - v1.7*  
Date in source: *Nov 09ST 2022*  
Author metadata: *Enrico Merlo*

## Overview

To configure the **Quattrocento** bioelectrical signal amplifier, the host must send a **40-byte command string**.

This command string defines:

- acquisition parameters
- analog output routing
- per-input filter and detection settings
- metadata describing muscle, side, sensor, and adapter
- CRC

## 40-byte command layout

| Byte(s) | Name | Description |
|---|---|---|
| 1 | `ACQ_SETT` | Sampling frequency, number of channels, start/stop acquisition, start/stop recording |
| 2 | `AN_OUT_IN_SEL` | Select input source and gain for analog output |
| 3 | `AN_OUT_CH_SEL` | Select channel for analog output source |
| 4-6 | `IN1_CONF0/1/2` | Configuration for `IN1` |
| 7-9 | `IN2_CONF0/1/2` | Configuration for `IN2` |
| 10-12 | `IN3_CONF0/1/2` | Configuration for `IN3` |
| 13-15 | `IN4_CONF0/1/2` | Configuration for `IN4` |
| 16-18 | `IN5_CONF0/1/2` | Configuration for `IN5` |
| 19-21 | `IN6_CONF0/1/2` | Configuration for `IN6` |
| 22-24 | `IN7_CONF0/1/2` | Configuration for `IN7` |
| 25-27 | `IN8_CONF0/1/2` | Configuration for `IN8` |
| 28-30 | `MULTIPLE_IN1_CONF0/1/2` | Configuration for `MULTIPLE IN1` |
| 31-33 | `MULTIPLE_IN2_CONF0/1/2` | Configuration for `MULTIPLE IN2` |
| 34-36 | `MULTIPLE_IN3_CONF0/1/2` | Configuration for `MULTIPLE IN3` |
| 37-39 | `MULTIPLE_IN4_CONF0/1/2` | Configuration for `MULTIPLE IN4` |
| 40 | `CRC` | 8-bit CRC |

For each `INx` and `MULTIPLE_INx` block, the configuration covers:

- high-pass filter
- low-pass filter
- detection mode
- muscle
- side
- sensor
- adapter

---

## Byte 1: `ACQ_SETT`

Bit layout:

```text
bit: 7     6      5       4      3      2     1     0
     1   DECIM  REC_ON  FSAMP1 FSAMP0 NCH1  NCH0 ACQ_ON
```

### Bit definitions

- **bit 7**: fixed to `1`
- **bit 6 - `DECIM`**: decimator enable
  - `1`: decimator active. Signals are sampled at **10240 Hz** and one sample out of `2`, `5`, or `20` is sent to obtain the requested effective rate.
  - `0`: decimator inactive. Signals are sampled directly at the requested sampling frequency.
- **bit 5 - `REC_ON`**: recording/trigger-out acquisition mode
  - `1`: start acquisition associated with **Trigger OUT**
  - `0`: stop acquisition associated with **Trigger OUT**
- **bits 4:3 - `FSAMP<1:0>`**: sampling frequency selection
- **bits 2:1 - `NCH<1:0>`**: channel group selection
- **bit 0 - `ACQ_ON`**: acquisition enable
  - `1`: data sampling and transfer active
  - `0`: data sampling and transfer inactive

### `FSAMP<1:0>` values

| Bits | Sampling frequency |
|---|---|
| `11` | 10240 Hz |
| `10` | 5120 Hz |
| `01` | 2048 Hz |
| `00` | 512 Hz |

### `NCH<1:0>` values

| Bits | Active inputs | Total number of channels |
|---|---|---|
| `11` | All inputs active | 408 |
| `10` | `IN1`-`IN6` and `MULTIPLE IN1`-`MULTIPLE IN3` active | 312 |
| `01` | `IN1`-`IN4`, `MULTIPLE IN1`, and `MULTIPLE IN2` active | 216 |
| `00` | `IN1`, `IN2`, and `MULTIPLE IN1` active | 120 |

In every configuration, **eight additional accessory channels** are also transferred.

---

## Byte 2: `AN_OUT_IN_SEL`

Bit layout:

```text
bit: 7 6 5           4           3     2     1     0
     0 0 ANOUT_GAIN1 ANOUT_GAIN0 INSEL3 INSEL2 INSEL1 INSEL0
```

### Bit definitions

- **bit 7**: fixed to `0`
- **bit 6**: fixed to `0`
- **bits 5:4 - `ANOUT_GAIN<1:0>`**: analog output gain
- **bits 3:0 - `INSEL<3:0>`**: analog output source input selection

### `ANOUT_GAIN<1:0>` values

| Bits | Analog output gain |
|---|---|
| `11` | 16 |
| `10` | 4 |
| `01` | 2 |
| `00` | 1 |

### `INSEL<3:0>` values

| Bits | Analog output source |
|---|---|
| `1100` | `AUX IN` |
| `1011` | `MULTIPLE IN4` |
| `1010` | `MULTIPLE IN3` |
| `1001` | `MULTIPLE IN2` |
| `1000` | `MULTIPLE IN1` |
| `0111` | `IN8` |
| `0110` | `IN7` |
| `0101` | `IN6` |
| `0100` | `IN5` |
| `0011` | `IN4` |
| `0010` | `IN3` |
| `0001` | `IN2` |
| `0000` | `IN1` |

---

## Byte 3: `AN_OUT_CH_SEL`

Bit layout:

```text
bit: 7 6 5      4      3      2      1      0
     0 0 CHSEL5 CHSEL4 CHSEL3 CHSEL2 CHSEL1 CHSEL0
```

### Bit definitions

- **bit 7**: fixed to `0`
- **bit 6**: fixed to `0`
- **bits 5:0 - `CHSEL<5:0>`**: source channel index for analog output

The selected channel is interpreted relative to the input chosen by `AN_OUT_IN_SEL`.

- `0` = first channel
- `1` = second channel
- etc.

The selected channel is routed to the **ANALOG OUT BNC** on the rear panel.

---

## Bytes `INX_CONF0` and `MULTIPLE_INX_CONF0`

Bit layout:

```text
bit: 7 6    5    4    3    2    1    0
     0 MUS6 MUS5 MUS4 MUS3 MUS2 MUS1 MUS0
```

### Bit definitions

- **bit 7**: fixed to `0`
- **bits 6:0 - `MUS<6:0>`**: muscle index for `INX` or `MULTIPLE INX`

### Muscle index table

| Value | Muscle | Value | Muscle | Value | Muscle |
|---:|---|---:|---|---:|---|
| 0 | Not defined | 25 | Ext. Carpi Ulnaris | 50 | Soleus |
| 1 | Temporalis Anterior | 26 | Ext. Dig. Communis | 51 | Semitendinosus |
| 2 | Superfic. Masseter | 27 | Brachioradialis | 52 | Gluteus maximus |
| 3 | Splenius Capitis | 28 | Abd. Pollicis Brev. | 53 | Gluteus medius |
| 4 | Upper Trapezius | 29 | Abd. Pollicis Long. | 54 | Vastus lateralis |
| 5 | Middle Trapezius | 30 | Opponens Pollicis | 55 | Vastus medialis |
| 6 | Lower Trapezius | 31 | Adductor Pollicis | 56 | Rectus femoris |
| 7 | Rhomboideus Major | 32 | Flex. Poll. Brevis | 57 | Tibialis anterior |
| 8 | Rhomboideus Minor | 33 | Abd. Digiti Minimi | 58 | Peroneus longus |
| 9 | Anterior Deltoid | 34 | Flex. Digiti Minimi | 59 | Semimembranosus |
| 10 | Posterior Deltoid | 35 | Opp. Digiti Minimi | 60 | Gracilis |
| 11 | Lateral Deltoid | 36 | Dorsal Interossei | 61 | Ext. Anal Sphincter |
| 12 | Infraspinatus | 37 | Palmar Interossei | 62 | Puborectalis |
| 13 | Teres Major | 38 | Lumbrical | 63 | Urethral Sphincter |
| 14 | Erector Spinae | 39 | Rectus Abdominis | 64 | Not a Muscle |
| 15 | Latissimus Dorsi | 40 | Ext. Abdom. Obliq. |  |  |
| 16 | Bic. Br. Long Head | 41 | Serratus Anterior |  |  |
| 17 | Bic. Br. Short Head | 42 | Pectoralis Major |  |  |
| 18 | Tric. Br. Lat. Head | 43 | Sternoc. Ster. Head |  |  |
| 19 | Tric. Br. Med. Head | 44 | Sternoc. Clav. Head |  |  |
| 20 | Pronator Teres | 45 | Anterior Scalenus |  |  |
| 21 | Flex. Carpi Radial. | 46 | Tensor Fascia Latae |  |  |
| 22 | Flex. Carpi Ulnaris | 47 | Gastrocn. Lateralis |  |  |
| 23 | Palmaris Longus | 48 | Gastrocn. Medialis |  |  |
| 24 | Ext. Carpi Radialis | 49 | Biceps Femoris |  |  |

---

## Bytes `INX_CONF1` and `MULTIPLE_INX_CONF1`

Bit layout:

```text
bit: 7     6     5     4     3     2      1      0
     SENS4 SENS3 SENS2 SENS1 SENS0 ADAPT2 ADAPT1 ADAPT0
```

### Bit definitions

- **bits 7:3 - `SENS<4:0>`**: sensor index for `INX` or `MULTIPLE INX`
- **bits 2:0 - `ADAPT<2:0>`**: adapter index for `INX` or `MULTIPLE INX`

### Sensor index table (`SENS<4:0>`)

| Value | Sensor |
|---:|---|
| 0 | Not defined |
| 1 | 16 Monopolar EEG |
| 2 | Mon. intram. el. |
| 3 | Bip. el - CoDe |
| 4 | 8 Acceleromet. |
| 5 | Bipolar el. - DE1 |
| 6 | Bipolar el. - CDE |
| 7 | Bip. el. - other |
| 8 | 4 el. Array 10mm |
| 9 | 8 el. Array 5mm |
| 10 | 8 el. Array 10mm |
| 11 | 64el. Gr. 2.54mm |
| 12 | 64 el. Grid 8mm |
| 13 | 64 el. Grid 10mm |
| 14 | 64 el.Gr. 12.5mm |
| 15 | 16el.Array 2.5mm |
| 16 | 16 el. Array 5mm |
| 17 | 16 el. Array 10mm |
| 18 | 16 el. Array 10mm |
| 19 | 16 el. rectal pr. |
| 20 | 48 el. rectal pr. |
| 21 | 12 el. Armband |
| 22 | 16 el. Armband |
| 23 | Other sensor |

### Adapter index table (`ADAPT<2:0>`)

| Value | Adapter |
|---:|---|
| 0 | Not defined |
| 1 | 16ch AD1x16 |
| 2 | 8ch AD2x8 |
| 3 | 4ch AD4x4 |
| 4 | 64ch AD1x64 |
| 5 | 16ch AD8x2 |
| 6 | Other |
| 7 | Reserved / unspecified in source |

> Note: the source appears to include a stray `8` after value `7`; since `ADAPT<2:0>` is only 3 bits wide, valid values are `0-7`.

---

## Bytes `INX_CONF2` and `MULTIPLE_INX_CONF2`

Bit layout:

```text
bit: 7     6     5    4    3    2    1     0
     SIDE1 SIDE0 HPF1 HPF0 LPF1 LPF0 MODE1 MODE0
```

### Bit definitions

- **bits 7:6 - `SIDE<1:0>`**: side index
- **bits 5:4 - `HPF<1:0>`**: high-pass filter index
- **bits 3:2 - `LPF<1:0>`**: low-pass filter index
- **bits 1:0 - `MODE<1:0>`**: detection mode index

### `SIDE<1:0>` values

| Bits | Side |
|---|---|
| `11` | None |
| `10` | Right |
| `01` | Left |
| `00` | Not defined |

### `HPF<1:0>` values

| Bits | High-pass filter |
|---|---|
| `11` | 200 Hz |
| `10` | 100 Hz |
| `01` | 10 Hz |
| `00` | 0.7 Hz |

### `LPF<1:0>` values

| Bits | Low-pass filter |
|---|---|
| `11` | 4400 Hz |
| `10` | 900 Hz |
| `01` | 500 Hz |
| `00` | 130 Hz |

### `MODE<1:0>` values

| Bits | Detection mode |
|---|---|
| `10` | Bipolar |
| `01` | Differential |
| `00` | Monopolar |

> Note: `11` is not defined in the source.

---

## Byte 40: `CRC`

The final byte is an **8-bit CRC**.

The source document identifies this field but does **not** define:

- CRC polynomial
- initial value
- bit order / reflection
- final XOR

That means CRC generation cannot be implemented from this document alone without additional vendor information or reverse-engineering.

---

## Accessory channels

In addition to the signals acquired from the front-panel inputs and the auxiliary inputs on the rear panel, **eight accessory channels** are transferred from Quattrocento to the computer.

- Resolution: **16 bits** per channel

### Accessory channel definitions

#### Accessory channel 1: sample counter

An unsigned 16-bit sample counter incremented at every sample.

- range: `0-65535`
- resets when data transfer starts
- rolls over after reaching the maximum value
- useful for detecting lost samples during transfer or storage
- the first recorded value depends on the exact start instant and is not predictable

#### Accessory channel 2: trigger channel

A copy of the **TRIGGER BNC** state on the rear panel.

Use cases:

- synchronous start/stop with other devices
- alignment with stimulation pulses in electrically elicited acquisitions

Values:

- `0` = logical 0 on trigger BNC
- `31767` = logical 1 on trigger BNC

#### Accessory channel 3

Not used. Reserved for future implementations.

#### Accessory channel 4: buffer usage

Indicates the available byte space in the Quattrocento internal buffer.

- values close to zero indicate the buffer is nearly full
- if the PC does not read data soon enough, samples may be lost
- full-buffer conditions may be caused by:
  - slow computer
  - slow network adapter
  - busy network
  - slow network switch/router

#### Accessory channels 5-8

Not used. Reserved for future implementations.

---

## Implementation notes

### Per-input configuration pattern

Each configured input uses three bytes:

- `CONF0`: muscle index
- `CONF1`: sensor index + adapter index
- `CONF2`: side + filters + detection mode

This applies to:

- `IN1`-`IN8`
- `MULTIPLE IN1`-`MULTIPLE IN4`

### Practical parser/serializer structure

A clean software representation would look like:

```text
Command[40] = {
  ACQ_SETT,
  AN_OUT_IN_SEL,
  AN_OUT_CH_SEL,
  IN1_CONF0, IN1_CONF1, IN1_CONF2,
  ...
  IN8_CONF0, IN8_CONF1, IN8_CONF2,
  MULTIPLE_IN1_CONF0, MULTIPLE_IN1_CONF1, MULTIPLE_IN1_CONF2,
  ...
  MULTIPLE_IN4_CONF0, MULTIPLE_IN4_CONF1, MULTIPLE_IN4_CONF2,
  CRC,
}
```

### Undefined / ambiguous points in the source

The document leaves several implementation-critical points unspecified:

1. **CRC algorithm details are missing**.
2. **`MODE=11` is undefined**.
3. **`ADAPT=7` is not clearly described**, and the source contains formatting artifacts suggesting a typo.
4. **Sensor entry `17` and `18` appear identical** (`16 el. Array 10mm`), which may be intentional or a documentation issue.

---

## Raw bitfield summary

### `ACQ_SETT`

```text
bit 7      = 1 (fixed)
bit 6      = DECIM
bit 5      = REC_ON
bits 4:3   = FSAMP
bits 2:1   = NCH
bit 0      = ACQ_ON
```

### `AN_OUT_IN_SEL`

```text
bits 7:6   = 0 (fixed)
bits 5:4   = ANOUT_GAIN
bits 3:0   = INSEL
```

### `AN_OUT_CH_SEL`

```text
bits 7:6   = 0 (fixed)
bits 5:0   = CHSEL
```

### `INX_CONF0` / `MULTIPLE_INX_CONF0`

```text
bit 7      = 0 (fixed)
bits 6:0   = MUS
```

### `INX_CONF1` / `MULTIPLE_INX_CONF1`

```text
bits 7:3   = SENS
bits 2:0   = ADAPT
```

### `INX_CONF2` / `MULTIPLE_INX_CONF2`

```text
bits 7:6   = SIDE
bits 5:4   = HPF
bits 3:2   = LPF
bits 1:0   = MODE
```

---

## Status of this Markdown conversion

This Markdown is a structured extraction of the PDF content, with minor cleanup for readability:

- headings normalized
- tables reconstructed
- obvious formatting noise removed
- ambiguities preserved and explicitly noted

No protocol semantics were invented beyond what is inferable from the source formatting.
