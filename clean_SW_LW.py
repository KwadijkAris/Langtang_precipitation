"""Cleaning/loading of the shortwave and longwave radiation records of
PLU1 (Kyangjin AWS) and PLU2 (Yala BC AWS) — the only two stations with
radiation sensors used in the analysis.

Returns numeric KINC (SW in), KUPW (SW out) and LINC (LW in) series; the
analysis (wrapperv3.py) derives SW/LW ratios and albedo from these.

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import os
import numpy as np
import pandas as pd

from station_data import get_dir, read_AWS, _DATA_DIR


def get_aws_df_SW_LW(Station=None):
    """Hourly radiation data per station: DataFrame indexed by DATETIME with
    the available columns among KINC, KUPW, LINC (numeric, NA removed)."""
    if Station is None:
        station_names = ['Kyangjin AWS', 'Yala BC AWS']
    else:
        station_names = Station

    radiation_dict = {}
    for station in station_names:
        file_path = get_dir(station)
        df = read_AWS(file_path[0])
        df = df.replace('NA', np.nan)
        for col in ['KINC', 'KUPW', 'LINC']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df = df.set_index('DATETIME')
        cols = [c for c in ['KINC', 'KUPW', 'LINC'] if c in df.columns]
        radiation_dict[station] = df[cols]
    return radiation_dict


def save_cleaned_radiation():
    """Write the cleaned radiation series as hourly means to
    data/Cleaned/Radiation/<station>_radiation.csv (DATETIME + KINC/KUPW/LINC).

    These shipped files are what the analysis (wrapperv3.py) reads, so it
    runs without access to the raw data."""
    radiation_dict = get_aws_df_SW_LW()
    out_dir = _DATA_DIR / 'Cleaned' / 'Radiation'
    os.makedirs(out_dir, exist_ok=True)
    for station, df in radiation_dict.items():
        hourly = df.resample('h').mean()
        out = out_dir / f'{station}_radiation.csv'
        hourly.to_csv(out, index_label='DATETIME')
        print(f'Saved radiation data for {station} to {out}')
    return radiation_dict


if __name__ == '__main__':
    # Regenerates data/Cleaned/Radiation from the raw AWS records
    save_cleaned_radiation()
