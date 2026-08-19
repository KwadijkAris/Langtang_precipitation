# README

Code repository for the data cleaning and analysis workflow used in Kwadijk et al. (2026). The scripts process precipitation, temperature, humidity, radiation, pressure, wind, lapse-rate/isotherm, and valley-geometry data for the Langtang Valley station network.

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.



## Repository Layout

This folder contains the Python code. The scripts expect the Zenodo-style directory layout:

```text
<zenodo_root>/
  code_repository/
  data/
    raw/
    Cleaned/
    Merged/
    Data_overview/
    Geometry/
    LapseRate/
    Moisture/
  results/
```

Most paths are resolved relative to the script location. In particular, `station_data.py` defines:

- `data` as `../data`
- `results` as `../results`
- station metadata as `../data/Data_overview/TEST2.txt`

## Requirements

Install the Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required packages are `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `geopandas`, `shapely`, and `rasterio`.

## Main Workflow

The processing chain is organized around individual cleaning scripts, merge scripts, and final analysis/plotting scripts.

### Shared Metadata and Readers

- `station_data.py` provides station metadata lookup, path resolution, season assignment, and raw/cleaned data readers.

### Cleaning and Derived Products

- `TB_clean_precip.py` cleans tipping-bucket precipitation records.
- `pluvio_cleaned_precip.py` cleans pluviometer and AWS precipitation records. By default, the current pluviometer cleaning run processes `Langshisha Pluvio` and `Morimoto Pluvio`; `Ganja La Pluvio` and `Yala Pluvio` are not included in the default run.
- `kochendorfer_correction.py` applies Kochendorfer wind-induced undercatch correction.
- `clean_temperature.py` loads and cleans temperature records from TB, pluviometer, AWS, and SNOWAMP stations.
- `clean_wind.py` cleans wind speed and wind direction records.
- `clean_RH.py` cleans relative humidity records.
- `clean_pressure.py` cleans atmospheric pressure records.
- `clean_SW_LW.py` cleans hourly shortwave and longwave radiation records for Kyangjin AWS and Yala BC AWS.
- `generate_humidity_timeseries.py` derives dew point, vapor pressure, mixing ratio, absolute humidity, specific humidity, and saturation specific humidity.
- `lapse_rate_isotherm.py` computes lapse rate and zero/one-degree isotherm elevation products.
- `valley_geometry.py` derives valley geometry products from the DEM.

### Merge Scripts (optional)

- `merge_temperature.py` writes per-station cleaned temperature files, `data/Merged/merged_temperature.csv`, and `data/Cleaned/Temperature/temp_merged_dfs.pkl`.
- `merge_precipitation.py` writes `data/Merged/merged_precipitation.csv`. The current default station list includes corrected/AWS precipitation for `Kyangjin AWS`, `Yala BC AWS`, `Langshisha Pluvio`, and `Morimoto Pluvio`, plus the tipping-bucket stations. `Ganja La Pluvio`, `Yala Pluvio`, and `snowAMP Ganja La` are no longer part of the default merged precipitation output.
- `merge_RH.py` writes `data/Merged/merged_RH.csv`.
- `merge_SW_LW.py` writes `data/Merged/merged_SW_LW.csv`.

### Plotting and Validation

- `plot_data_overview.py` creates the data-availability overview figure.
- `wrapperv3.py` contains the main analysis and figure functions. In the current version, `plot_percentage_below_zero(temp_merged_dfs)` is called at module level after the temperature pickle is loaded, so running/importing this script immediately computes that below-zero-temperature summary.
- `testing_function.py` runs the end-to-end processing chain in dependency order and saves generated figures to `../results_testing/`.

## Common Commands

Run the full validation/processing chain from inside `code_repository`:

```bash
python testing_function.py
```

Regenerate merged temperature products:

```bash
python merge_temperature.py
```

Regenerate pluvio/AWS precipitation cleaning outputs:

```bash
python pluvio_cleaned_precip.py
```
This regenerates the default cleaned pluviometer outputs for Langshisha and Morimoto.

Regenerate Kochendorfer-corrected precipitation:

```bash
python kochendorfer_correction.py
```

Regenerate cleaned radiation files:

```bash
python clean_SW_LW.py
```

Regenerate humidity and lapse-rate summary inputs:

```bash
python generate_humidity_timeseries.py
```

Regenerate merged single-CSV products:

```bash
python merge_precipitation.py
python merge_RH.py
python merge_SW_LW.py
python merge_temperature.py
```

## Notes on Reproducibility

`testing_function.py` is the most complete executable description of the dependency order. It backs up shipped derived data to `../testing_backup/` before overwriting selected derived products. It also notes two important reproducibility details:

The full run requires raw sensor data under `data/raw/`. See `data/raw/README.txt` in the Zenodo data package for raw-data availability notes.

## Generated Files

Depending on which scripts are run, outputs are written under:

- `data/Cleaned/`
- `data/Merged/`
- `data/Moisture/`
- `data/LapseRate/`
- `data/Geometry/`
- `results/`
- `results_testing/`

