"""Cleaning/loading of the shortwave and longwave radiation records of
PLU1 (Kyangjin AWS) and PLU2 (Yala BC AWS) — the only two stations with
radiation sensors used in the analysis.

Returns numeric KINC (SW in), KUPW (SW out) and LINC (LW in) series; the
analysis (wrapperv3.py) derives SW/LW ratios and albedo from these.

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import numpy as np
import pandas as pd

from station_data import get_dir, read_AWS


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
