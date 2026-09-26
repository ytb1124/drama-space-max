# Corrective IR Spatial Reconstruction

This repository documents an experimental live sound system that uses measured impulse responses to bring the early response and reverberant impression of a target hall into a different performance venue.

The system was developed for DIMA's Jiseong Hall, a roughly 200-seat live-band venue with a measured 500 Hz T30 of 0.39 s. The target was the University of York's Sir Jack Lyons Concert Hall, a roughly 350-seat classical concert hall with a published 500 Hz T30 of 1.89 s.

This project does **not** claim to reproduce the target hall's complete physical sound field. It combines per-channel correction of the initial response with a separately applied target-hall reverberation tail to reinforce selected acoustic characteristics in a live performance system.

## System overview

```text
stage-rigged AB microphone pair
        ↓
six measured loudspeaker-to-microphone system paths
        ↓
per-channel corrective FIR generation in Python
        ↓
real-time convolution in Max/MSP
        ↓
six-channel surround loudspeaker output
```

The main PA carried the direct sound and primary mix balance. The surround system added the corrected initial response and target-hall reverberant energy around it.

The Python pipeline:

1. Loads the current-venue and target-hall impulse responses.
2. Resamples the target IR when the sample rates differ.
3. Aligns the direct arrivals with GCC-PHAT delay estimation.
4. Gates the relevant response region.
5. Calculates a regularized inverse filter instead of an unconstrained spectral division.
6. Limits correction primarily to 125 Hz-4 kHz.
7. Applies complex-response smoothing, passband masking, gain clipping, and peak limiting to reduce excessive boost and ringing.
8. Exports a WAV FIR for each measured surround channel.

The live implementation used a hybrid structure:

- The corrective FIR focused on the initial response, approximately the first 40 ms.
- The target-hall IR from approximately 200 ms onward was applied separately as a reverberation tail through Convology XT.
- The 40-200 ms region functioned as a transition rather than a separately validated inverse-correction region.

## Evaluation

Evaluation compared the target IR with the result of convolving each measured current-system path with its corrective FIR. Within the initial 0-40 ms window and the 125 Hz-4 kHz correction band, the mean spectral error across the six surround channels decreased from **2.69 dB to 2.24 dB**. All six channels showed a reduction; the largest changes were measured on SRC (3.17 to 2.17 dB) and SLF (2.27 to 1.47 dB).

These results are a first-stage numerical validation at a defined measurement position, centered on the fourth audience row, using the same measurements involved in filter generation. They do not demonstrate equivalent responses throughout the audience area or a perceptual match to the target hall. No controlled listening test was conducted in this study.

Future evaluation should compare no correction, direct target-IR convolution, and the proposed method under matched conditions, then assess RT, EDT, C80, D50, energy-decay behavior, and listener responses.

## Requirements

- Max 9.0.7
- HISSTools, including `multiconvolve~`
- Convology XT VST3
- Python 3
- NumPy
- SciPy
- SoundFile

The Max patchers preserve paths from the original development machine. Before running the system on another computer, reselect the Python executable, scripts, input IRs, output locations, and media files in the Max interface.

## Python example

The principal FIR generator is `other/make_club_to_hall_fir_v31_plus3.py`. In the script interface, `--club` means the measured current-venue/system-path IR and `--hall` means the target-hall IR.

```bash
python3 other/make_club_to_hall_fir_v31_plus3.py \
  --club "/path/to/current-system-ir.wav" \
  --hall "/path/to/target-hall-ir.wav" \
  --out "/path/to/corrective-fir.wav" \
  --mode mono \
  --band_low 125 \
  --band_high 4000 \
  --reg_db -32 \
  --early_ms 40 \
  --late_ms 200 \
  --peak_limit_ir 0.95
```

The script writes the FIR as a WAV file and prints JSON-formatted diagnostic metrics. The repository also retains the original Max-driven workflow and experimental late-ratio code for documentation.

## Repository map

- `Drama space MAX.maxproj` - Max project entry point
- `patchers/` - FIR generation, downmix, compensation, and six-channel live system patchers
- `code/` - JavaScript and Node for Max helpers
- `other/` - Python FIR and IR conversion scripts
- `250930 hwacon2/IR/` - measured current-system-path IRs used for the six surround channels
- `250930 hwacon2/CF/` - generated channel-specific corrective FIRs
- `data/` - Max and Convology XT snapshots
- `media/` and `final backup/` - selected reference images and representative outputs retained from the working archive

The public repository is a curated subset of the original 6.5 GB project archive. Duplicate media, intermediate renders, macOS metadata, and the large external IR library are not included.

## Publications

- [Research paper in Korean](docs/corrective-ir-paper-ko.pdf) - *Generation of Corrective Impulse Responses and a Real-Time Convolution System for Spatial Reproduction*
- [Stage Sound Magazine Vol. 19 article in Korean](docs/corrective-ir-ssm-vol19-ko.pdf) - *Can We Make the Current Venue Sound Like Another Hall? Building a Spatial Reproduction System*

## Attribution and third-party material

The target-hall material is based on the Sir Jack Lyons Concert Hall dataset from the University of York AudioLab's OpenAIR library and remains subject to its original attribution and license terms. Max/MSP, HISSTools, Convology XT, and any other third-party software or assets remain subject to their respective owners' terms. Their inclusion or mention here does not relicense them.

## Rights

Copyright (c) Taebin Yoo. All rights reserved.

No open-source license is granted for the original code, patches, documentation, audio, or other project material in this repository. The repository is publicly available for viewing and academic reference; reuse, modification, or redistribution requires prior permission from the relevant rights holder.
