"""Optional: merge every cleaned precipitation record into one hourly CSV
(one Rainfall_1H column per station) at data/Merged/merged_precipitation.csv.

Kochendorfer-corrected precipitation is used for the four wind-corrected
stations (Kyangjin AWS, Yala BC AWS, Langshisha Pluvio, Morimoto Pluvio).

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import os
import pandas as pd

from station_data import read_pluvio_cleaned, _DATA_DIR

STATIONS = [
    # Pluviometers / AWS (Kochendorfer-corrected where available)
    'Kyangjin AWS', 'Yala BC AWS', 'Langshisha Pluvio', 'Morimoto Pluvio',
    'Ganja La Pluvio', 'Yala Pluvio', 'snowAMP Ganja La',
    # Tipping buckets
    'Syabru TB', 'Lama TB', 'Ganja La TB1', 'Ganja La TB2', 'Ganja La TB3',
    'Jathang TB', 'Numthang TB', 'Langshisha BC TB', 'Shalbachum TB',
    'Morimoto TB',
    # SNOWAMP tipping buckets
    'snowAMP Ganjala upper', 'SNOWAMP_lower',
]


def merge_precipitation(stations=None):
    if stations is None:
        stations = STATIONS
    columns = {}
    for station in stations:
        try:
            df = read_pluvio_cleaned([station], dt='1h')[station]
        except Exception as e:
            print(f"Skipping {station}: {e}")
            continue
        rain_col = next((c for c in ['Rainfall_1H', 'Rainfall_15min', 'CS725_Swek(mm)']
                         if c in df.columns), None)
        if rain_col is None:
            print(f"Skipping {station}: no rainfall column")
            continue
        columns[station] = df[rain_col]
    merged = pd.concat(columns, axis=1)
    merged.index.name = 'DATETIME'
    return merged


if __name__ == '__main__':
    merged = merge_precipitation()
    out_dir = _DATA_DIR / 'Merged'
    os.makedirs(out_dir, exist_ok=True)
    out = out_dir / 'merged_precipitation.csv'
    merged.to_csv(out, na_rep='NA')
    print(f'Merged precipitation written to {out}')
