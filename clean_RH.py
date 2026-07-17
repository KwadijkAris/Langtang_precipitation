"""Cleaning of the relative-humidity sensors of PLU1 (Kyangjin AWS),
PLU2 (Yala BC AWS) and PLU3 (Morimoto MicroMet).

load_rh() reads the raw RH series; apply_rh_quality_masks() removes the
periods in which the sensors are known to be unreliable.

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import numpy as np
import pandas as pd
from station_data import get_dir, read_AWS


def load_rh(station, file_path=None):
    """Raw hourly RH series for a station (MicroMet 15-min data is averaged
    to hourly means). Returns a DataFrame indexed by DATETIME with an 'RH'
    column."""
    if file_path is None:
        file_path = get_dir(station)
    df = read_AWS(file_path[0])
    df = df.replace('NA', np.nan)
    df['RH'] = pd.to_numeric(df.get('RH', np.nan), errors='coerce')
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.set_index('DATETIME')[['RH']]

    return df

def apply_rh_quality_masks(df, station):
    """Remove RH periods in which the sensor is known to be unreliable.
    Expects a DataFrame with a DATETIME column (as used in the humidity
    generation loop)."""
    if station == 'Kyangjin AWS':
        df.loc[df['DATETIME'] > pd.Timestamp('2017-10-01'), 'RH'] = np.nan
        df.loc[(df['DATETIME'] >= pd.Timestamp('2013-12-01')) & (df['DATETIME'] < pd.Timestamp('2014-06-01')), 'RH'] = np.nan
    elif station == 'Yala BC AWS':
        df.loc[((df['DATETIME'] >= pd.Timestamp('2014-12-01')) & (df['DATETIME'] < pd.Timestamp('2016-12-01'))) |
        ((df['DATETIME'] >= pd.Timestamp('2020-03-01')) & (df['DATETIME'] < pd.Timestamp('2021-07-01'))) |
        (df['DATETIME'] > pd.Timestamp('2019-11-30')), 'RH'] = np.nan
    elif station == 'Morimoto MM':
        df.loc[df['DATETIME'] > pd.Timestamp('2022-04-12'), 'RH'] = np.nan
    return df
