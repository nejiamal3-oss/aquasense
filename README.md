# AquaSense-ICS

**A Hybrid Real–Simulated Cyber-Physical Dataset and Leakage-Aware Benchmark for Smart Water Quality Monitoring and Attack Detection**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)

Authors: Amal Neji, Mohamed Fourati, Lamia Chaari Fourati
Digital Research Center of Sfax (CRNS), Laboratory SM@RTS, Sfax, Tunisia
Contact: lamia.chaari@isims.usf.tn

## Overview

AquaSense-ICS co-registers on a common elapsed-time axis:

1. a real 25.4 h recording window (8.0 h of recorded data, 26 % coverage, 23,688 records) from a Raspberry Pi 5 edge node (TDS probe and BME680 ambient sensor, 1–2 s sampling);
2. a looped network of 13 nodes (10 junctions, 2 reservoirs, 1 tank) simulated in EPANET-MSX for six water-quality species (Cl, pH, TURB, DO, COND, THM) plus one wall species (ClD);
3. a seeded, parameterised library of cyber-attacks (false data injection, stealthy FDI, replay, denial of service, drift) and sensor faults (white noise, spike, extreme value) with sample-level ground truth;
4. augmented and TimeGAN-generated sequences, flagged by the `origin` column.

The benchmark replaces random splits and feature-derived labels with a **leakage-aware protocol**: chronological splits, held-out attack types and forecasting tasks.

> **Important:** the real sensor did not measure the simulated network. The fusion is a *co-registration*, so cross-source relations are synthetic by design. The `tds_value` column is a scaled voltage, not a calibrated concentration (no unit).

## Repository structure

```
aquasense-ics/
├── README.md
├── LICENSE                  MIT (code)
├── LICENSE-DATA.md          CC BY 4.0 (data)
├── CITATION.cff
├── DATASHEET.md
├── dictionary.csv           column, unit, source, description
├── splits.json              train/val/test indices per task
├── requirements.txt
├── environment.yml
├── Makefile
├── aquasense/               perturbation generator, fusion
├── benchmark/               tasks T1, T3, T4
├── network/                 EPANET / EPANET-MSX files
└── data/                    CSV (small files only; full data on Zenodo)
```

## Files

| File | Rows | Cols | Content |
|---|---:|---:|---|
| `data/Real_data.csv` | 23,688 | 6 | `timestamp`, temperature (°C), humidity (%), gas resistance (Ω), `tds_value`, `quality` |
| `data/simulated_data.csv` | 24 | 79 | `time_h` (hourly, 24 h) and 78 variables (6 species × 13 nodes), wide format |
| `data/synthetic_data.csv` | 23,688 | 4 | TimeGAN channels: `tds_value`, `temperature`, `humidity`, `gas_resistance` |
| `data/attacked_data.csv` | 147,128 | 28 | Sensor channels, augmented variants, `label_fdi`, `label_attack` (**Zenodo only**) |
| `data/augmented_data.csv` | 23,688 | 27 | Sensor data with five augmented variants per channel (**Zenodo only**) |
| `data/fused_dataset.csv` | 47,376 | 114 | Real + TimeGAN rows, simulated variables, metadata (`origin`, `attack_id`, `attack_channel`, `fault_channel`), `attack_type`, `fault_type`, `quality_final` (**Zenodo only**) |
| `network/water_network.inp` | – | – | EPANET model: 10 junctions, 2 reservoirs, 1 tank, 17 pipes, 1 pump, 2 valves |
| `network/water_quality.msx` | – | – | EPANET-MSX model: 6 bulk species, 1 wall species, reaction coefficients |
| `network/injection_schedule.csv` | – | – | Injection schedule (if provided) |
| `splits.json` | – | – | Official train/validation/test indices |
| `dictionary.csv` | – | – | Machine-readable data dictionary |

## Download the full data

The large files (`fused_dataset.csv`, `attacked_data.csv`, `augmented_data.csv`, Parquet versions) are hosted on Zenodo:

**https://doi.org/10.5281/zenodo.xxxxxxx**

Place them in the `data/` folder after download.

## Installation

Requirements: Python 3.11, `make`.

```bash
git clone https://github.com/<votre-user>/aquasense-ics.git
cd aquasense-ics

# Option A: pip
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate aquasense
```

## Reproduction (`make all`)

```bash
make all
```

This single command:

1. regenerates the EPANET-MSX simulation from `network/*.inp` and `network/*.msx`;
2. applies the seeded perturbation generator (attacks and faults);
3. builds the fused tables;
4. runs the benchmark tasks T1, T3 and T4.

Individual steps:

```bash
make simulate     # EPANET-MSX simulation
make perturb      # attacks and faults (seeded)
make fuse         # co-registration and fused tables
make benchmark    # tasks T1, T3, T4
make clean        # remove generated files
```

Random seeds, software versions and hardware are recorded in `environment.yml`.

## Benchmark tasks

| Task | Target | Split |
|---|---|---|
| T1 Water-quality forecasting | `quality_final` at t+h | chronological 60/20/20; leave-one-node-out |
| T3 Attack and fault classification | `attack_type`, `fault_type` | chronological; leave-one-attack-type-out |
| T4 Synthetic-data utility | as T3 | train synthetic / test real (TSTR) |

Please report results on the official splits in `splits.json`.

## Intended and non-intended use

- **Intended:** benchmarking anomaly and attack detectors, forecasting, synthetic-data and domain-adaptation research.
- **Not intended:** drawing physical conclusions from relations between real and simulated channels, or certifying the potability of real water.

## Known limitations

- Short (25 h), single indoor deployment; 26 % temporal coverage with 23 gaps longer than 60 s.
- Only TDS is measured in the water; temperature, humidity and gas resistance describe ambient air.
- Water-quality labels are rule-based and not validated against laboratory samples.
- Attacks are injected at data level, without closed-loop physical consequences.
- The simulated network does not correspond to a real infrastructure.

## Citation

```bibtex
@article{neji2026aquasense,
  title   = {AquaSense-ICS: A Hybrid Real--Simulated Cyber-Physical Dataset and Leakage-Aware Benchmark for Smart Water Quality Monitoring and Attack Detection},
  author  = {Neji, Amal and Fourati, Mohamed and Chaari Fourati, Lamia},
  journal = {Internet of Things},
  year    = {2026},
  doi     = {xxxxxxx}
}
```

Dataset DOI: https://doi.org/10.5281/zenodo.xxxxxxx

## Licenses

- Data: [CC BY 4.0](LICENSE-DATA.md)
- Code: [MIT](LICENSE)