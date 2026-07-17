import os
import numpy as np
import pandas as pd

"""
Apply Kochendorfer wind-induced undercatch correction for mixed and solid 
precipitation on hourly data for Pluvio and AWS stations.

Correction applied:
  - Snow  (T <= -2°C):  CE = a * exp(-b * U) + c   (Pluvio snow params)
  - Mixed (-2 < T <= 2°C): CE = a * exp(-b * U) + c (Pluvio mixed params)
  - Rain  (T > 2°C):  no correction applied

Output is saved per station to the specified output directory.
"""



# ---------- Kochendorfer correction parameters (from Kochendorfer et al.) ----------
PLUVIO_SNOW_PARAMS  = {'a': 0.728, 'b': 0.230, 'c': 0.336, 'u_cap': 7.2}
PLUVIO_MIXED_PARAMS = {'a': 0.668, 'b': 0.132, 'c': 0.339, 'u_cap': 7.2}

# ---------- Station configuration ----------
# Maps station name -> (pluvio CSV path, wind CSV path)
DATA_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CLEANED_PLUVIO_DIR = os.path.join(DATA_BASE, 'data', 'Cleaned', 'Pluvio')
CLEANED_WIND_DIR   = os.path.join(DATA_BASE, 'data', 'Cleaned', 'Wind')
KOCHENDORFER_CORRECTED_DIR = os.path.join(DATA_BASE, 'data', 'Cleaned', 'Kochendorfer_corrected')

# Raw AWS file for Kyangjin (contains pre-2016 wind speed that was filtered
# out of the cleaned wind file).  Wind direction before 2016 is unreliable
# and therefore not used, but wind speed is kept.
KYANGJIN_RAW_AWS = os.path.join(DATA_BASE, 'data', 'raw', 'AWS', 'Kyangjin_ICIMOD.csv')

STATIONS = {
    'Kyangjin AWS': {
        'pluvio': os.path.join(CLEANED_PLUVIO_DIR, 'Kyangjin AWS.csv'),
        'pluvio_uncorrected': os.path.join(CLEANED_PLUVIO_DIR, 'Kyangjin AWS.csv'),
        'pluvio_corrected': os.path.join(KOCHENDORFER_CORRECTED_DIR, 'Kyangjin AWS_kochendorfer_corrected.csv'),
        'wind':   os.path.join(CLEANED_WIND_DIR,   'Kyangjin_AWS_wind_data.csv'),
    },
    'Yala BC AWS': {
        'pluvio': os.path.join(CLEANED_PLUVIO_DIR, 'Yala BC AWS.csv'),
        'pluvio_uncorrected': os.path.join(CLEANED_PLUVIO_DIR, 'Yala BC AWS.csv'),
        'pluvio_corrected': os.path.join(KOCHENDORFER_CORRECTED_DIR, 'Yala BC AWS_kochendorfer_corrected.csv'),
        'wind':   os.path.join(CLEANED_WIND_DIR,   'Yala_BC_AWS_wind_data.csv'),
    },
    'Langshisha Pluvio': {
        'pluvio': os.path.join(CLEANED_PLUVIO_DIR, 'Langshisha Pluvio.csv'),
        'pluvio_uncorrected': os.path.join(CLEANED_PLUVIO_DIR, 'Langshisha Pluvio.csv'),
        'pluvio_corrected': os.path.join(KOCHENDORFER_CORRECTED_DIR, 'Langshisha Pluvio_kochendorfer_corrected.csv'),
        'wind':   os.path.join(CLEANED_WIND_DIR,   'Langshisha_Pluvio_wind_data.csv'),
    },
    'Morimoto Pluvio': {
        'pluvio': os.path.join(CLEANED_PLUVIO_DIR, 'Morimoto Pluvio.csv'),
        'pluvio_uncorrected': os.path.join(CLEANED_PLUVIO_DIR, 'Morimoto Pluvio.csv'),
        'pluvio_corrected': os.path.join(KOCHENDORFER_CORRECTED_DIR, 'Morimoto Pluvio_kochendorfer_corrected.csv'),
        'wind':   os.path.join(CLEANED_WIND_DIR,   'Morimoto_MM_wind_data.csv'),
    },
    'Ganja La Pluvio': {
        'pluvio': os.path.join(CLEANED_PLUVIO_DIR, 'Ganja La Pluvio.csv'),
        'pluvio_uncorrected': os.path.join(CLEANED_PLUVIO_DIR, 'Ganja La Pluvio.csv'),
        'pluvio_corrected': os.path.join(KOCHENDORFER_CORRECTED_DIR, 'Ganja La Pluvio_kochendorfer_corrected.csv'),
        'wind':   None,  # no wind data available
    },
    'Yala Pluvio': {
        'pluvio': os.path.join(CLEANED_PLUVIO_DIR, 'Yala Pluvio.csv'),
        'pluvio_uncorrected': os.path.join(CLEANED_PLUVIO_DIR, 'Yala Pluvio.csv'),
        'pluvio_corrected': os.path.join(KOCHENDORFER_CORRECTED_DIR, 'Yala Pluvio_kochendorfer_corrected.csv'),
        'wind':   os.path.join(CLEANED_WIND_DIR,   'Yala_BC_AWS_wind_data.csv'),
    },
}


# ---------- Helper functions ----------
def catch_efficiency(u, a, b, c):
    """CE = a * exp(-b * U) + c"""
    return a * np.exp(-b * u) + c




def load_pluvio(csv_path):
    """
    Load a cleaned pluvio CSV at its native resolution and return a
    DataFrame with columns 'Precipitation' and 'Temperature'.
    Works for both 15-min and 1-hour files without resampling.
    """
    df = pd.read_csv(csv_path)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.set_index('DATETIME', inplace=True)

    # Convert NAN strings to actual NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Standardise precipitation column name
    if 'Rainfall_1H' in df.columns:
        df = df.rename(columns={'Rainfall_1H': 'Precipitation'})
    elif 'Rainfall_15min' in df.columns:
        df = df.rename(columns={'Rainfall_15min': 'Precipitation'})
    else:
        raise ValueError(f"No rainfall column found in {csv_path}")

    # Standardise temperature column name
    if 'Temperature' not in df.columns:
        for alt in ['Temperature_1H', 'Temperature_15min']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'Temperature'})
                break

    return df[['Precipitation', 'Temperature']]


# Keep backward-compatible alias
load_pluvio_hourly = load_pluvio




def load_wind(csv_path, station=None):
    """
    Load a cleaned wind CSV at its native resolution and return a
    Series named 'Wind'.  No resampling is performed.

    For Kyangjin AWS the cleaned file only starts at 2016 because both
    WSPD and WINDDIR were set to NaN before that date.  The user wants
    to *keep* the original wind speeds before 2016 (only the wind
    direction correction should stay).  We therefore back-fill with
    the raw AWS file for the pre-2016 period.
    """
    df = pd.read_csv(csv_path)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.set_index('DATETIME', inplace=True)
    df['WSPD'] = pd.to_numeric(df['WSPD'], errors='coerce')

    wind = df['WSPD']

    # ------------------------------------------------------------------
    # Kyangjin AWS: restore pre-2016 wind speeds from the raw AWS file
    # ------------------------------------------------------------------
    if station == 'Kyangjin AWS' and os.path.isfile(KYANGJIN_RAW_AWS):
        raw = pd.read_csv(KYANGJIN_RAW_AWS)
        raw['DATETIME'] = pd.to_datetime(
            raw['DATE'] + ' ' + raw['TIME'], format='%Y-%m-%d %H:%M:%S'
        )
        raw.set_index('DATETIME', inplace=True)
        raw['WSPD'] = pd.to_numeric(raw['WSPD'], errors='coerce')

        # Keep only pre-2016 rows
        raw_pre2016 = raw.loc[raw.index < '2016-01-01', 'WSPD']

        # Combine: pre-2016 from raw, 2016+ from cleaned
        wind = pd.concat([raw_pre2016, wind]).sort_index()
        wind = wind[~wind.index.duplicated(keep='last')]  # prefer cleaned if overlap
        print(f'  Kyangjin AWS: restored {len(raw_pre2016)} pre-2016 wind speed records from raw AWS')

    wind.name = 'Wind'
    return wind


# Keep backward-compatible alias
load_wind_hourly = load_wind




def apply_kochendorfer(prec, u, t, snow_params, mixed_params):
    """
    Apply Kochendorfer correction for snow (T <= -2) and mixed (-2 < T <= 2).
    Rain (T > 2) is left unchanged.
    If wind speed is 0 or between 0 and 0.005, CE = 1.0 (no correction).
    Returns (corrected_precipitation, catch_efficiency, precip_type).
    """
    corrected = prec.copy()
    ce = np.full_like(prec, np.nan, dtype=float)
    ptype = np.full(len(prec), '', dtype=object)

    # Snow
    mask_snow = t <= -2
    if np.any(mask_snow):
        u_s = np.clip(u[mask_snow], 0, snow_params['u_cap'])
        ce_s = catch_efficiency(u_s, snow_params['a'], snow_params['b'], snow_params['c'])
        # Where wind speed is between 0 and 0.005, set CE = 1.0
        ce_s = np.where(u_s <= 0.005, 1.0, ce_s)
        # Cap CE at 1.0
        ce_s = np.minimum(ce_s, 1.0)
        corrected[mask_snow] = np.where(prec[mask_snow] == 0, 0, prec[mask_snow] / ce_s)
        ce[mask_snow] = ce_s
        ptype[mask_snow] = 'snow'

    # Mixed
    mask_mixed = (t > -2) & (t <= 2)
    if np.any(mask_mixed):
        u_m = np.clip(u[mask_mixed], 0, mixed_params['u_cap'])
        ce_m = catch_efficiency(u_m, mixed_params['a'], mixed_params['b'], mixed_params['c'])
        # Where wind speed is between 0 and 0.005, set CE = 1.0
        ce_m = np.where(u_m <= 0.005, 1.0, ce_m)
        # Cap CE at 1.0
        ce_m = np.minimum(ce_m, 1.0)
        corrected[mask_mixed] = np.where(prec[mask_mixed] == 0, 0, prec[mask_mixed] / ce_m)
        ce[mask_mixed] = ce_m
        ptype[mask_mixed] = 'mixed'

    # Rain – no correction
    mask_rain = t > 2
    ce[mask_rain] = 1.0
    ptype[mask_rain] = 'rain'

    # Where wind data is missing, CE stays NaN → set CE=1.0 (no correction)
    missing_wind = np.isnan(ce)
    ce[missing_wind] = 1.0
    corrected[missing_wind] = prec[missing_wind]

    # Where uncorrected precipitation is NaN, corrected must also be NaN
    prec_nan = np.isnan(prec)
    corrected[prec_nan] = np.nan

    return corrected, ce, ptype



def retrieve_uncorrected_precipitation():
    """Retrieve uncorrected precipitation timeseries from cleaned Pluvio directory."""
    uncorrected_data = {}
    for station, paths in STATIONS.items():
        pluvio_path = paths['pluvio_uncorrected']
        if not os.path.isfile(pluvio_path):
            print(f"Pluvio file not found for {station}: {pluvio_path}")
            continue
        df = load_pluvio(pluvio_path)
        uncorrected_data[station] = df
        print(f"Loaded {station}: {len(df)} records (native resolution)")
    return uncorrected_data


def identify_months_with_low_correction_coverage():
    """
    Identify months for each station where more than 90% of precipitation records
    lack wind data (i.e., cannot be corrected).
    Store results to CSV for later reference.
    """
    uncorrected_data = retrieve_uncorrected_precipitation()
    
    results = {}
    all_low_coverage = []
    
    for station in STATIONS:
        out_file = os.path.join(KOCHENDORFER_CORRECTED_DIR, f'{station}_kochendorfer_corrected.csv')
        if not os.path.isfile(out_file):
            continue
        
        corr_df = pd.read_csv(out_file, parse_dates=['DATETIME'], index_col='DATETIME')
        
        if station not in uncorrected_data:
            continue
        uncorr = uncorrected_data[station]['Precipitation']
        
        # Add year/month columns
        corr_df['Year'] = corr_df.index.year
        corr_df['Month'] = corr_df.index.month
        uncorr_df = uncorr.to_frame()
        uncorr_df['Year'] = uncorr_df.index.year
        uncorr_df['Month'] = uncorr_df.index.month
        
        # Group by year-month
        low_coverage_months = []
        for (year, month), group in corr_df.groupby(['Year', 'Month']):
            uncorr_group = uncorr_df[(uncorr_df['Year'] == year) & (uncorr_df['Month'] == month)]
            
            # Count records where precipitation exists but wind data is missing
            prec_available = uncorr_group['Precipitation'].notna().sum()
            wind_missing = group['Wind'].isna().sum()
            
            if prec_available > 0:
                missing_wind_ratio = wind_missing / prec_available
                if missing_wind_ratio > 0.9:
                    low_coverage_months.append({
                        'Station': station,
                        'Year': year, 
                        'Month': month, 
                        'Missing_wind_ratio': missing_wind_ratio,
                        'Records_no_wind': wind_missing, 
                        'Total_with_precip': prec_available
                    })
                    all_low_coverage.append({
                        'Station': station,
                        'Year': year,
                        'Month': month,
                        'Missing_wind_pct': round(missing_wind_ratio * 100, 1),
                        'Records_no_wind': wind_missing,
                        'Total_with_precip': prec_available
                    })
        
        results[station] = low_coverage_months
    
    # Print results
    print("\n" + "="*70)
    print("MONTHS WITH >90% MISSING WIND DATA (CANNOT CORRECT)")
    print("="*70)
    for station, months in results.items():
        if months:
            print(f"\n{station}:")
            for m in months:
                print(f"  {m['Year']}-{m['Month']:02d}: {m['Missing_wind_ratio']*100:.1f}% missing wind "
                        f"({m['Records_no_wind']}/{m['Total_with_precip']} records)")
        else:
            print(f"\n{station}: All months have wind data coverage ✓")
    
    # Save to CSV
    output_dir = os.path.join(DATA_BASE, 'data', 'Cleaned', 'Catch_efficiencies')
    os.makedirs(output_dir, exist_ok=True)
    
    if all_low_coverage:
        low_cov_df = pd.DataFrame(all_low_coverage)
        output_file = os.path.join(output_dir, 'months_with_missing_wind_data.csv')
        low_cov_df.to_csv(output_file, index=False)
        print(f"\n✓ Months with >90% missing wind data saved to: {output_file}")
    else:
        print("\n✓ No months with >90% missing wind data found!")
    
    return results


# ---------- Main processing ----------
def process_and_save():
    os.makedirs(KOCHENDORFER_CORRECTED_DIR, exist_ok=True)

    for station, paths in STATIONS.items():
        print(f'\n{"="*60}')
        print(f'Processing: {station}')
        print(f'{"="*60}')

        # --- Load precipitation & temperature (native resolution) ---
        if not os.path.isfile(paths['pluvio']):
            print(f'  Pluvio file not found: {paths["pluvio"]}  → skipping')
            continue
        df = load_pluvio(paths['pluvio'])

        # --- Load wind (native resolution) ---
        if paths['wind'] is not None and os.path.isfile(paths['wind']):
            wind = load_wind(paths['wind'], station=station)
        else:
            print(f'  No wind file for {station}  → all CE=1.0 (corrected = uncorrected)')
            wind = pd.Series(dtype=float, name='Wind')  # empty series, join will produce NaN

        # --- Merge on index ---
        if len(wind) > 0:
            wind_df = wind.to_frame().sort_index()
            df = df.sort_index()
            df = pd.merge_asof(
                df, wind_df,
                left_index=True, right_index=True,
                direction='nearest',
                tolerance=pd.Timedelta('1h')
            )
        else:
            df['Wind'] = np.nan

        # Keep ALL timestamps so corrected file has no gaps vs uncorrected
        n_before = len(df)
        valid = df  # no rows dropped
        n_with_precip = valid['Precipitation'].notna().sum()
        n_all3 = valid.dropna(subset=['Precipitation', 'Temperature', 'Wind']).shape[0]
        print(f'  Total rows: {n_before},  with precipitation: {n_with_precip},  with all 3 variables: {n_all3}')

        prec = valid['Precipitation'].values
        u    = valid['Wind'].values      # may contain NaN
        t    = valid['Temperature'].values  # may contain NaN

        # --- Apply Kochendorfer correction ---
        corrected, ce, ptype = apply_kochendorfer(
            prec, u, t, PLUVIO_SNOW_PARAMS, PLUVIO_MIXED_PARAMS
        )

        # --- Fill missing months using mean monthly correction ratio ---
        corrected = fill_missing_months_with_ratio(
            corrected, prec, ce, ptype, valid.index, station
        )

        # --- Build output DataFrame ---
        out = valid.copy()
        out['Precipitation_corrected'] = corrected
        out['Catch_efficiency'] = np.round(ce, 4)
        out['Precip_type'] = ptype

        # --- Summary statistics ---
        n_snow  = np.sum(ptype == 'snow')
        n_mixed = np.sum(ptype == 'mixed')
        n_rain  = np.sum(ptype == 'rain')
        n_nocorr = np.sum(ce == 1.0) - n_rain
        total_orig = np.nansum(prec)
        total_corr = np.nansum(corrected)
        ratio = total_corr / total_orig if total_orig > 0 else np.nan
        print(f'  Snow hours: {n_snow},  Mixed hours: {n_mixed},  Rain hours: {n_rain},  Uncorrected (no wind/temp): {n_nocorr}')
        print(f'  Total original precip:  {total_orig:.2f} mm')
        print(f'  Total corrected precip: {total_corr:.2f} mm')
        print(f'  Overall correction ratio: {ratio:.4f}')

        # --- Save ---
        out_file = os.path.join(KOCHENDORFER_CORRECTED_DIR, f'{station}_kochendorfer_corrected.csv')
        out.to_csv(out_file, index=True, index_label='DATETIME')
        print(f'  Saved → {out_file}')

    print(f'\n{"="*60}')
    print('Done. All corrected files saved to:')
    print(f'  {KOCHENDORFER_CORRECTED_DIR}')




def fill_missing_months_with_ratio(corrected, prec, ce, ptype, index, station):
    """
    For months with >90% missing wind data, apply the mean monthly correction ratio
    to hours that were not corrected (CE=1.0 due to missing wind).
    """
    low_coverage = identify_months_with_low_correction_coverage()
    
    if station not in low_coverage or not low_coverage[station]:
        return corrected
    
    # Load monthly correction ratios
    output_dir = os.path.join(DATA_BASE, 'data', 'Cleaned', 'Catch_efficiencies')
    ratio_file = os.path.join(output_dir, 'monthly_correction_ratio.csv')
    
    if not os.path.isfile(ratio_file):
        print(f"  Warning: {ratio_file} not found, skipping ratio-based fill for {station}")
        return corrected
    
    ratio_df = pd.read_csv(ratio_file)
    station_ratios = ratio_df[ratio_df['Station'] == station]
    
    corrected_copy = corrected.copy()
    
    for month_info in low_coverage[station]:
        year = month_info['Year']
        month = month_info['Month']
        
        # Get the mean ratio for this month (only filtered by month, not year)
        month_ratio = station_ratios[
            station_ratios['Month'] == month
        ]['Ratio'].values
        
        if len(month_ratio) == 0:
            # Fall back to average across all months for this station
            month_ratio = station_ratios['Ratio'].mean()
        else:
            month_ratio = month_ratio[0]
        
        # Find indices for this month
        month_mask = (index.year == year) & (index.month == month)
        
        # Apply ratio to records that were uncorrected (CE=1.0) and have precip data.
        # Exclude rain hours: rain needs no undercatch correction, so CE=1.0 there
        # is correct rather than a sign of missing wind data.
        uncorrected_mask = (ce == 1.0) & ~np.isnan(prec) & month_mask & (ptype != 'rain')
        corrected_copy[uncorrected_mask] = prec[uncorrected_mask] * month_ratio
        
        print(f"    {station} {year}-{month:02d}: applied ratio {month_ratio:.4f} to {np.sum(uncorrected_mask)} records")
    
    return corrected_copy


# # Call main processing and then plot results
process_and_save()
