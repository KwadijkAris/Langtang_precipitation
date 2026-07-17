"""Optional: merge the shortwave/longwave radiation of PLU1 (Kyangjin AWS)
and PLU2 (Yala BC AWS) plus the SW/LW ratio into one CSV
(data/Merged/merged_SW_LW.csv).

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import os
import numpy as np
import pandas as pd

from station_data import _DATA_DIR
from clean_SW_LW import get_aws_df_SW_LW


def merge_SW_LW():
    radiation = get_aws_df_SW_LW()
    columns = {}
    for station, df in radiation.items():
        df = df[~df.index.duplicated(keep='first')]
        if 'KINC' in df.columns:
            columns[f'{station} SW_in'] = df['KINC']
        if 'LINC' in df.columns:
            columns[f'{station} LW_in'] = df['LINC']
        if 'KINC' in df.columns and 'LINC' in df.columns:
            ratio = df['KINC'] / df['LINC']
            # only physically meaningful (positive) ratios, as in the analysis
            columns[f'{station} SW_LW_ratio'] = ratio.where(ratio > 0, np.nan)
    merged = pd.concat(columns, axis=1)
    merged.index.name = 'DATETIME'
    return merged


if __name__ == '__main__':
    merged = merge_SW_LW()
    out_dir = _DATA_DIR / 'Merged'
    os.makedirs(out_dir, exist_ok=True)
    out = out_dir / 'merged_SW_LW.csv'
    merged.to_csv(out, na_rep='NA')
    print(f'Merged SW/LW written to {out}')
