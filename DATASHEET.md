# Datasheet for AquaSense-ICS (v1.0)

Following the template of Gebru et al. (2021), "Datasheets for Datasets".

## 1. Motivation

**Why was the dataset created?**
Machine-learning research on water cyber-physical systems is limited by
datasets that are either real but chemically poor, or simulated but free of
sensing noise and attacks. AquaSense-ICS co-registers a real edge-sensing
stream with a multi-species EPANET-MSX network simulation, adds a seeded
library of attacks and faults with sample-level labels, and provides a
leakage-aware benchmark.

**Who created it?**
Amal Neji, Mohamed Fourati, Lamia Chaari Fourati, Digital Research Center of
Sfax (CRNS), Laboratory SM@RTS, Sfax, Tunisia.

**Who funded it?**
To be completed (funding sources).

## 2. Composition

**What do the instances represent?**
- Real rows: timestamped readings of one edge node (TDS probe, BME680 air
  temperature, relative humidity, gas resistance).
- Simulated rows: hourly states of 6 water-quality species at 13 nodes
  (EPANET-MSX).
- Synthetic rows: TimeGAN sequences (4 sensor channels, no common clock).

**How many instances?**

| Table | Rows |
|---|---:|
| Real_data.csv | 23,688 |
| simulated_data.csv | 24 |
| synthetic_data.csv | 23,688 |
| fused_dataset.csv | 47,376 (23,688 real + 23,688 TimeGAN) |
| attacked_data.csv | 147,128 |
| augmented_data.csv | 23,688 |

**What labels exist?**
- `attack_type`: none, FDI, FDI-stealth, replay, DoS, drift (from the seeded
  generator).
- `fault_type`: none, white noise, spike, extreme (from the generator).
- `quality_final`: Poor / Good / Very Good (rule-based, regulatory thresholds).
- The legacy label `qualite` is kept for backward compatibility only and is
  not used in the benchmark.

**Is any information missing?**
- The record covers 25.4 h but only 8.0 h contain samples (26 % coverage);
  23 gaps are longer than 60 s, the longest about 10.1 h.
- Simulated columns are missing for 2,893 real samples (outside the simulated
  24 h window) and for all TimeGAN rows.
- No event log of the physical disturbances (smoke, water cup, candle) is
  available.

**Does the dataset contain confidential or personal data?**
No.

## 3. Collection process

**How was the real data acquired?**
Raspberry Pi 5 with an ADS1115 ADC (TDS probe DAR-02-W40) and a Bosch BME680
over I2C. Sampling 1-2 s (median 1 s), from 2025-05-19 12:39:08 to
2025-05-20 14:01:00, timestamps in UTC synchronised by NTP.

**Physical disturbances applied (ambient, not in the water):**
cigarette smoke (gas resistance), cup of water (humidity), candle flame
(temperature). Start and end times were not logged.

**How was the simulation produced?**
EPANET-MSX, looped network of 10 junctions, 2 reservoirs, 1 tank, 17 pipes,
1 pump, 2 valves; 24 h, hourly reporting, 5-minute quality step. Two pulse
injections: chlorine at J3 (06:00-08:00) and turbidity at J7 (14:00-16:00).

**How were attacks and faults generated?**
By a seeded generator (`aquasense.perturb`): FDI, stealthy FDI, replay, DoS,
drift; white noise, spikes, extreme values. Data-level injection only.

## 4. Preprocessing

- Raw samples are kept in physical units with quality flags.
- No imputation of the four real sensor channels (no missing value).
- Normalisation is not applied to the released data; in the benchmark it is
  fitted on the training split only.
- Real and simulated data are joined by an as-of join on elapsed time
  (tolerance 3,600 s), not on minute-of-day.
- Augmented and TimeGAN data are released separately and must never enter
  validation or test splits.

## 5. Uses

**Intended uses**
Benchmarking anomaly and attack detectors, forecasting, synthetic-data and
domain-adaptation research.

**Not intended**
- Drawing physical conclusions from relations between real and simulated
  channels (they are co-registered, not coupled).
- Certifying the potability of real water.
- Using `tds_value` as a concentration: it is a scaled voltage, not
  calibrated, without unit.

**Known limitations**
- Short (25 h), single indoor deployment.
- Only TDS is measured in the water; temperature, humidity and gas resistance
  describe ambient air.
- Water-quality labels are rule-based, not validated by laboratory samples.
- Attacks are injected into recorded data, without closed-loop physical
  consequences.
- Under the leakage-aware protocol, some tasks are only partly feasible
  (for example, the chronological test split contains only normal and replay
  windows).
- The simulated network does not correspond to a real infrastructure.

## 6. Distribution

**How is it distributed?**
Zenodo (DOI https://doi.org/10.5281/zenodo.xxxxxxx) and GitHub
(https://github.com/<votre-user>/aquasense-ics).

**License**
Data: CC BY 4.0. Code: MIT.

## 7. Maintenance

**Who maintains it?**
The authors. Contact: lamia.chaari@isims.usf.tn

**Will it be updated?**
New acquisitions and simulations will be released as new versions under the
same Zenodo concept DOI, with an unchanged schema. Errata are listed in the
repository.

**Citation**
See `CITATION.cff`.