"""Optional: merge every cleaned temperature sensor into one hourly dataset.

Writes per-station files to data/Cleaned/Temperature and a single combined
CSV (one column per station) to data/Merged/merged_temperature.csv.

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import os
import pickle
import pandas as pd
from station_data import _DATA_DIR
from clean_temperature import (get_TB_df_temp, get_Pluvio_temp,
                               get_aws_df_temp, get_SNOWAMP_df_temp)


def merge_datasets_hourly():

    # TB temperature sensors are deliberately excluded from the merged
    # temperature dataset (the isotherm/lapse-rate chain filters them out).
    all_merged_dfs = get_Pluvio_temp()

    # Get AWS data
    df_aws=get_aws_df_temp()
    all_merged_dfs.update(df_aws)

    include_snowamp = True
    if include_snowamp ==True:
        df_SNOWAMP = get_SNOWAMP_df_temp()
        all_merged_dfs.update(df_SNOWAMP)

    # Ensure that the 1H periods are aligned across all stations
    common_H = pd.date_range(start='2012-01-01', end='2024-12-01', freq='h')

    # Normalize every station to a single hourly-mean TEMP column on a
    # DATETIME index (drops helper/precipitation columns from the sources)
    for station, df in all_merged_dfs.items():
        if 'DATETIME' in df.columns:
            df = df.set_index('DATETIME')
        df.index = pd.to_datetime(df.index)
        hourly = df['TEMP'].resample('h').mean().to_frame('TEMP')
        hourly = hourly.reindex(common_H)
        hourly.index.name = 'DATETIME'
        all_merged_dfs[station] = hourly

    # Save temperature data to CSV files
    output_dir = str(_DATA_DIR / 'Cleaned' / 'Temperature')

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each station's data to a separate CSV file (DATETIME,TEMP)
    for station, df in all_merged_dfs.items():
        output_path = os.path.join(output_dir, f'{station}_temperature.csv')
        df.to_csv(output_path, index_label='DATETIME')
        print(f"Saved temperature data for {station} to {output_path}")

    return all_merged_dfs


def write_merged_csv(all_merged_dfs):
    merged = pd.concat({station: df['TEMP'] for station, df in all_merged_dfs.items()
                        if 'TEMP' in df.columns}, axis=1)
    merged.index.name = 'DATETIME'
    out_dir = _DATA_DIR / 'Merged'
    os.makedirs(out_dir, exist_ok=True)
    out = out_dir / 'merged_temperature.csv'
    merged.to_csv(out, na_rep='NA')
    print(f'Merged temperature written to {out}')
    return merged


def save_temp_merged_pickle(all_merged_dfs):
    """Recreate data/Cleaned/Temperature/temp_merged_dfs.pkl (input of
    lapse_rate_isotherm.py and the wrapperv3 figures)."""
    with open(_DATA_DIR / 'Cleaned' / 'Temperature' / 'temp_merged_dfs.pkl', 'wb') as f:
        pickle.dump(all_merged_dfs, f)


if __name__ == '__main__':
    # Also writes data/Cleaned/Temperature/<station>_temperature.csv per station
    merged_dfs = merge_datasets_hourly()
    write_merged_csv(merged_dfs)
    save_temp_merged_pickle(merged_dfs)