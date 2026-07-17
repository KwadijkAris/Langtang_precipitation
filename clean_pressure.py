"""Cleaning of the barometric-pressure records of PLU1 (Kyangjin AWS),
PLU2 (Yala BC AWS) and PLU3 (Morimoto MicroMet).

load_pressure() reads the raw pressure series (hourly means for the 15-min
MicroMet); apply_pressure_quality_masks() removes the periods in which the
sensor is known to be unreliable.

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import numpy as np
import pandas as pd

from station_data import get_dir, read_AWS


def load_pressure(station, file_path=None):
    """Raw barometric pressure series for a station (hourly means for the
    15-min MicroMet)."""
    if file_path is None:
        file_path = get_dir(station)
    pres_df = read_AWS(file_path[0])
    pres_df = pres_df.replace('NA', np.nan)
    pres_df['PRES'] = pd.to_numeric(pres_df.get('PRES', np.nan), errors='coerce')
    pres_df['DATETIME'] = pd.to_datetime(pres_df['DATETIME'])
    pres_df = pres_df.set_index('DATETIME')[['PRES']]
    if station == 'Morimoto MM':
        pres_df = pres_df.resample('h').mean()
    return pres_df


def apply_pressure_quality_masks(df, station):
    """Remove pressure periods in which the sensor is known to be unreliable.
    Expects a DataFrame indexed by DATETIME (as used in the humidity
    generation loop)."""
    if station == 'Yala BC AWS' and 'PRES' in df.columns:
        df['PRES'] = pd.to_numeric(df['PRES'], errors='coerce')
        df.loc[df.index.date == pd.Timestamp('2015-10-27').date(), 'PRES'] = np.nan
        df.loc[df.index.date == pd.Timestamp('2020-07-18').date(), 'PRES'] = np.nan
        df.loc[df.index.date == pd.Timestamp('2020-07-20').date(), 'PRES'] = np.nan
        df.loc[df.index > '2023-05-10', 'PRES'] = np.nan
    return df
