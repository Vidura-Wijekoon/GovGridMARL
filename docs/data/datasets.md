# Residential load datasets — access, licensing, and how they are used (task m2-t9)

Nothing under `data/raw/` is committed (git-ignored). The environment runs on the calibrated synthetic generator by default (`env.data.source: synthetic`, task m2-t10); the real sets plug in through the same `LoadDataset` interface once downloaded.

## 1. UK-DALE (UK Domestic Appliance-Level Electricity)

| | |
|---|---|
| Publisher | Jack Kelly & William Knottenbelt, Imperial College London; hosted by the UKERC Energy Data Centre |
| Content | 5 UK houses, whole-house mains at 1 s / 6 s (and 16 kHz for some) plus 5–54 appliance channels, 2012–2017 |
| Licence | Creative Commons Attribution 4.0 (CC-BY 4.0) — free to use with citation |
| Citation | Kelly, J. & Knottenbelt, W. (2015). *The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes.* Scientific Data 2, 150007. |
| Access | UKERC EDC catalogue → "UK-DALE" → download `ukdale.zip` (disaggregated `.dat`) or `ukdale.h5` (HDF5, NILMTK format). No registration needed for the CC-BY release. Size: ~4 GB (6 s) / >100 GB (16 kHz) |
| Layout expected by `load_ukdale` | `data/raw/ukdale/house_<n>/channel_1.dat` (unix_ts, W) — channel 1 is the house aggregate |

## 2. REDD (Reference Energy Disaggregation Data Set)

| | |
|---|---|
| Publisher | Kolter & Johnson, MIT CSAIL |
| Content | 6 US houses, two mains phases at 1 Hz plus 10–24 circuit channels at 1/3 Hz, 2011 |
| Licence | Research use only; **no redistribution**; access on request |
| Citation | Kolter, J. Z. & Johnson, M. J. (2011). *REDD: A public data set for energy disaggregation research.* SustKDD Workshop. |
| Access | E-mail the maintainers for the credentials to the low-frequency release; keep the files private |
| Layout expected by `load_redd` | `data/raw/redd/low_freq/house_<n>/channel_1.dat`, `channel_2.dat` (the two mains) |

## 3. Why the thesis uses a calibrated synthetic set by default

1. **Scale mismatch.** Both sets contain 5–6 houses; the environment needs 14 heterogeneous *clusters* with declared income groups, which neither dataset labels. Resampling five houses fourteen times with jitter (what `load_dataset` does for the real sets) adds little information over a generator calibrated to their published statistics.
2. **Reproducibility and licensing.** REDD cannot be redistributed and UK-DALE is multi-GB; a seeded generator keeps the repository self-contained and CI-testable (`tests/test_env.py::test_synthetic_profiles_calibrated`).
3. **Group heterogeneity is the object of study.** Income-group differences in daily energy, flexible share and comfort tolerance (proposal §6.2) must be *set*, not inferred from six houses.

## 4. Calibration anchors used by `generate_profiles`

| quantity | value | source |
|---|---|---|
| daily energy, low / mid / high | 4.9 / 7.4 / 11.2 kWh (≈ 1,800 / 2,700 / 4,100 kWh yr⁻¹) | Ofgem Typical Domestic Consumption Values (electricity profile class 1) |
| diurnal shape | night trough 0.25× mean, morning peak 07:45, evening peak 18:30 (dominant), midday shoulder | UK-DALE house aggregates; Elexon Profile Class 1 |
| between-house dispersion | log-normal σ = 0.35 on daily energy | order of magnitude of UK-DALE / REDD house spread |
| flexible share | 15 / 25 / 35 % of load (appliance-shifting, EV, heat pump increase with income) | energy-justice literature reviewed in Area 3 (`review_paper_source.md` §4) |
| comfort tolerance θ | 30 / 40 / 50 % of flexible energy per day | design choice, sensitivity in M11 |
| tariff | 0.12 / 0.26 / 0.40 per kWh (00–07 / day / 16–20) | indicative UK ToU (Economy-7-like + peak) |
| PV | 0.8 kW peak per home, clear-sky bell at 12:30 | indicative 3–4 kWp rooftop share |

Resulting dataset (seed 42, N = 14): mean 8.4 kWh/day (range 4.1–14.9), aggregate peak 18:00, load factor 0.42, flexible share 31 % — all inside the target bands (see `results/logs/validate_env_*.log`).

## 5. Switching to real data

```bash
python experiments/train.py env.data.source=ukdale env.data.path=data/raw/ukdale
python experiments/train.py env.data.source=redd   env.data.path=data/raw/redd/low_freq
```

Missing files raise `FileNotFoundError` with these instructions; there is no silent fallback to synthetic data.
