"""Optional: merge the relative-humidity series of PLU1, PLU2 and PLU3 from
the generated humidity timeseries into one CSV
(data/Merged/merged_RH.csv, one RH column per station).

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import os
import pandas as pd

from station_data import _DATA_DIR

MOISTURE_FILES = {
    'Kyangjin AWS': 'Kyangjin_AWS_humidity_timeseries.csv',
    'Yala BC AWS': 'Yala_BC_AWS_humidity_timeseries.csv',
    'Morimoto MM': 'Morimoto_MM_humidity_timeseries.csv',
}


def merge_RH():
    columns = {}
    for station, fname in MOISTURE_FILES.items():
        path = _DATA_DIR / 'Moisture' / fname
        df = pd.read_csv(path, na_values=['NA'])
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df = df.drop_duplicates(subset='DATETIME').set_index('DATETIME')
        columns[station] = pd.to_numeric(df['RH'], errors='coerce')
    merged = pd.concat(columns, axis=1)
    merged.index.name = 'DATETIME'
    return merged


if __name__ == '__main__':
    merged = merge_RH()
    out_dir = _DATA_DIR / 'Merged'
    os.makedirs(out_dir, exist_ok=True)
    out = out_dir / 'merged_RH.csv'
    merged.to_csv(out, na_rep='NA')
    print(f'Merged RH written to {out}')
