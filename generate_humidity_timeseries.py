"""Generation of the humidity timeseries CSVs (data/Moisture/
<station>_humidity_timeseries.csv) for PLU1 (Kyangjin AWS), PLU2 (Yala BC
AWS) and PLU3 (Morimoto MicroMet).

Every quantity is computed by its own function:
  - thermodynamics: calculate_dew_point, saturation_vapor_pressure,
    mixing_ratio, absolute_humidity, specific_humidity
  - inputs: clean_RH.load_rh / apply_rh_quality_masks, clean_wind.
    get_aws_df_wind, load_temperature, load_pressure
  - generate_humidity_timeseries() combines them and writes the CSVs
  - compute_malr_and_rh_summaries() derives the moist adiabatic lapse rate
    and RH summary files in data/LapseRate
  - get_seasonal_RH_PRES_data() returns the seasonal packaging used by the
    analysis scripts

Running this file regenerates data/Moisture that I use in my thermodynamic analyses.

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import calendar
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from station_data import get_dir, read_AWS, read_pluvio_cleaned, get_season, _DATA_DIR
from clean_RH import load_rh, apply_rh_quality_masks
from clean_wind import get_aws_df_wind
from clean_pressure import load_pressure, apply_pressure_quality_masks


# --- Helper Functions for Calculations ---
def calculate_dew_point(T, RH):
    """Calculates dew point temperature using the Magnus formula."""
    T = pd.to_numeric(T, errors='coerce')
    RH = pd.to_numeric(RH, errors='coerce')

    valid_mask = ~np.isnan(T) & ~np.isnan(RH) & (RH > 0)
    Td = pd.Series(np.nan, index=T.index)

    if valid_mask.any():
        gamma = (17.27 * T[valid_mask]) / (237.7 + T[valid_mask]) + np.log(RH[valid_mask] / 100.0)
        Td.loc[valid_mask] = (237.7 * gamma) / (17.27 - gamma)
    return Td

def saturation_vapor_pressure(T):
    """Calculate saturation vapor pressure (hPa) using Tetens formula."""
    return 6.112 * np.exp((17.67 * T) / (T + 243.5))

def mixing_ratio(e, P):
    """Calculate mixing ratio (kg/kg) given vapor pressure and pressure."""
    return 0.622 * e / (P - e)

def absolute_humidity(e, T):
    """Calculate absolute humidity (g/m^3) given vapor pressure and temperature."""
    Rv = 461.5  # J/(kg·K)
    T_k = T + 273.15 # Convert temperature to Kelvin
    e_pa = e * 100 # Convert vapor pressure from hPa to Pa
    ah_kg_m3 = e_pa / (Rv * T_k)
    return ah_kg_m3 * 1000 # Convert to g/m^3

def specific_humidity(e, P):
    """Calculate specific humidity (kg/kg) given vapor pressure and pressure."""
    return (0.622 * e) / (P - 0.378 * e)

def load_temperature(station, file_path):
    """Hourly cleaned temperature for a station: from the cleaned pluvio /
    Kochendorfer files for the AWS stations, from the raw TAIR for the
    MicroMet (which has no cleaned pluvio file)."""
    if station == 'Morimoto MM':
        # No cleaned pluvio file exists for the MicroMet; use its own TAIR
        temp_df = read_AWS(file_path[0])
        temp_df = temp_df.replace('NA', np.nan)
        temp_df['TEMP'] = pd.to_numeric(temp_df['TAIR'], errors='coerce')
        temp_df['DATETIME'] = pd.to_datetime(temp_df['DATETIME'])
        temp_df = temp_df.set_index('DATETIME')[['TEMP']].resample('h').mean()
    else:
        cleaned_pluvio_data = read_pluvio_cleaned([station])
        temp_df = cleaned_pluvio_data[station]
        if 'Temperature_1H' in temp_df.columns:
            temp_df = temp_df.rename(columns={'Temperature_1H': 'TEMP'})
        if 'Temperature' in temp_df.columns:
            temp_df = temp_df.rename(columns={'Temperature': 'TEMP'})
    return temp_df[['TEMP']]


def generate_humidity_timeseries(save=True):
    """Build the humidity timeseries per station (RH + wind + temperature +
    pressure + derived humidity variables) and write them to
    data/Moisture/<station>_humidity_timeseries.csv."""
    station_names = ['Kyangjin AWS', 'Yala BC AWS', 'Morimoto MM']
    aws_data_dict = {}

    for station in station_names:

        # Get RH data
        file_path = get_dir(station)
        df = load_rh(station, file_path)

        # Get wind data
        wind_data_dict = get_aws_df_wind(Station=[station])
        wind_df = wind_data_dict[station]

        # Get temperature data from cleaned pluvio files
        temp_df = load_temperature(station, file_path)

        # Merge RH, wind, and temperature data
        df = df.join(wind_df[['WSPD', 'WINDDIR']], how='outer')
        df = df.join(temp_df[['TEMP']], how='outer')

        # Get original data again for pressure (clean_pressure.py)
        pres_df = load_pressure(station, file_path)

        # Merge pressure data
        df = df.join(pres_df, how='outer')
        # Remove unreliable pressure periods (clean_pressure.py)
        df = apply_pressure_quality_masks(df, station)

        df = df.reset_index()

        # Process wind speed (WSPD) and wind direction (WINDDIR)
        df['WSPD'] = pd.to_numeric(df['WSPD'], errors='coerce')
        df['WINDDIR'] = pd.to_numeric(df['WINDDIR'], errors='coerce')

        # Remove unreliable RH periods
        df = apply_rh_quality_masks(df, station)

        # --- Calculate humidity variables ---
        df['DEW_POINT'] = calculate_dew_point(df['TEMP'], df['RH'])
        e_s = saturation_vapor_pressure(df['TEMP'])
        e = (df['RH'] / 100.0) * e_s
        df['MIXING_RATIO'] = mixing_ratio(e, df['PRES']) * 1000 # g/kg
        df['ABSOLUTE_HUMIDITY'] = absolute_humidity(e, df['TEMP']) # g/m^3
        df['SPEC_HUM'] = specific_humidity(e, df['PRES']) * 1000 # g/kg
        df['SAT_SPEC_HUM'] = specific_humidity(e_s, df['PRES']) * 1000 # g/kg

        if save:
            # --- Prepare data for saving ---
            df_to_save = df.copy()

            # Rename columns to match the requested headers
            column_rename_map = {
                'WINDDIR': 'Wind_Dir',
                'WSPD': 'Wind_Speed',
                'PRES': 'Rel_Air_Press',
                'TEMP': 'TAIR',
                'RH': 'RH',
                'DATETIME': 'DATETIME',
                'DEW_POINT': 'DEW_POINT',
                'MIXING_RATIO': 'MIXING_RATIO',
                'ABSOLUTE_HUMIDITY': 'ABSOLUTE_HUMIDITY',
                'SPEC_HUM': 'SPEC_HUM',
                'SAT_SPEC_HUM': 'SAT_SPEC_HUM'
            }
            df_to_save.rename(columns=column_rename_map, inplace=True)

            # Select and order the columns as requested
            output_columns = [
                'DATETIME', 'Wind_Dir', 'Wind_Speed', 'Rel_Air_Press', 'TAIR', 'RH',
                'DEW_POINT', 'MIXING_RATIO', 'ABSOLUTE_HUMIDITY', 'SPEC_HUM', 'SAT_SPEC_HUM'
            ]
            # Ensure all requested columns exist, fill with NaN if not
            for col in output_columns:
                if col not in df_to_save.columns:
                    df_to_save[col] = np.nan
            df_to_save = df_to_save[output_columns]

            # --- Save the processed data to a CSV file ---
            output_dir = str(_DATA_DIR / "Moisture")
            os.makedirs(output_dir, exist_ok=True)

            sanitized_station_name = station.replace(" ", "_").replace("/", "_")
            output_filename = f"{sanitized_station_name}_humidity_timeseries.csv"
            output_path = os.path.join(output_dir, output_filename)

            df_to_save.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S', na_rep='NA')
            print(f"Saved humidity timeseries for {station} to {output_path}")

        aws_data_dict[station] = df

    return aws_data_dict

def compute_malr_and_rh_summaries(aws_data_dict):
    """Monthly RH climatology plots, moist adiabatic lapse rate per station
    (saved to data/LapseRate/monthly_avg_malr_<station>.csv) and the hourly /
    monthly RH summary CSVs in data/LapseRate."""
    df = list(aws_data_dict.values())[-1]

    # Plot humidity data for the last processed station
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['RH'], label='Relative Humidity (%)', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Relative Humidity (%)')
    plt.title('Relative Humidity Time Series')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot mean monthly RH data for a simulated year
    plt.figure(figsize=(12, 6))

    # Define month names for plotting
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]

    for station, df in aws_data_dict.items():
        if 'RH' in df.columns and not df['RH'].dropna().empty:
            # Ensure DATETIME is the index
            if 'DATETIME' in df.columns:
                df_plot = df.set_index('DATETIME')
            else:
                df_plot = df.copy()

            # Group by month and calculate the mean RH across all years
            monthly_avg_rh = df_plot.groupby(df_plot.index.month)['RH'].mean()

            # Plot the data
            plt.plot(monthly_avg_rh.index, monthly_avg_rh, label=f'{station} Mean Monthly RH', marker='.', linestyle='-')

    plt.xlabel('Month')
    plt.ylabel('Mean Relative Humidity (%)')
    plt.title('Average Monthly RH for AWS Stations (Simulated Year)')
    plt.xticks(ticks=range(1, 13), labels=month_names) # Set x-axis ticks to month names
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot mean monthly moist adiabatic lapse rate
    plt.figure(figsize=(12, 6))

    # Constants for MALR calculation
    g = 9.81  # m/s^2
    cp = 1005.7  # J/kg/K
    Rd = 287.058  # J/kg/K
    Rv = 461.5  # J/kg/K
    epsilon = Rd / Rv
    Lv = 2.501e6  # J/kg

    for station, df in aws_data_dict.items():
        if all(col in df.columns for col in ['TEMP', 'PRES']) and not df[['TEMP', 'PRES']].dropna().empty:
            # Ensure DATETIME is the index
            if 'DATETIME' in df.columns:
                df_plot = df.set_index('DATETIME')
            else:
                df_plot = df.copy()

            # Group by month and calculate mean T and P
            monthly_avg = df_plot.groupby(df_plot.index.month).agg({'TEMP': 'mean', 'PRES': 'mean'})

            # Calculate MALR for each month
            malr_values = []
            for month in monthly_avg.index:
                T_c = monthly_avg.loc[month, 'TEMP']
                P_hpa = monthly_avg.loc[month, 'PRES']

                if pd.notna(T_c) and pd.notna(P_hpa):
                    T_k = T_c + 273.15
                    # Saturation vapor pressure (hPa) using August-Roche-Magnus formula
                    es = 6.1094 * np.exp((17.625 * T_c) / (T_c + 243.04))
                    # Saturation mixing ratio (kg/kg)
                    rs = epsilon * es / (P_hpa - es)

                    # Moist Adiabatic Lapse Rate (K/km)
                    numerator = g * (1 + (Lv * rs) / (Rd * T_k))
                    denominator = cp + (Lv**2 * rs) / (Rv * T_k**2)
                    malr_k_per_m = numerator / denominator
                    malr_k_per_km = malr_k_per_m * 1000
                    malr_values.append(malr_k_per_km)
                else:
                    malr_values.append(np.nan)
            monthly_avg['MALR'] = malr_values
            # Plot the data
            plt.plot(monthly_avg.index, monthly_avg['MALR'], label=f'{station} MALR', marker='.', linestyle='-')
            # Define output directory and save the data
            output_dir = str(_DATA_DIR / "LapseRate")
            os.makedirs(output_dir, exist_ok=True)

            # Sanitize station name for filename
            sanitized_station_name = station.replace(' ', '_').replace('/', '_')
            output_filename = f"monthly_avg_malr_{sanitized_station_name}.csv"
            output_path = os.path.join(output_dir, output_filename)

            # Save the DataFrame to CSV
            monthly_avg.to_csv(output_path)
            print(f"Saved monthly average data for {station} to {output_path}")

    plt.xlabel('Month')
    plt.ylabel('Moist Adiabatic Lapse Rate (K/km)')
    plt.title('Average Monthly Moist Adiabatic Lapse Rate')
    plt.xticks(ticks=range(1, 13), labels=month_names)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Define output directory
    output_dir = str(_DATA_DIR / "LapseRate")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Save hourly RH data ---
    # Combine RH data from all stations into a single DataFrame
    all_rh_data = []
    for station, df in aws_data_dict.items():
        if 'RH' in df.columns:
        # Ensure DATETIME is the index
            if 'DATETIME' in df.columns:
                df_rh = df.drop_duplicates(subset='DATETIME').set_index('DATETIME')
            else:
                df_rh = df.copy()

            rh_series = df_rh['RH'].rename(station)
            all_rh_data.append(rh_series)

    if all_rh_data:
        combined_rh_df = pd.concat(all_rh_data, axis=1)
        hourly_rh_path = os.path.join(output_dir, 'hourly_rh_data.csv')
        combined_rh_df.to_csv(hourly_rh_path)
        print(f"Saved hourly RH data to {hourly_rh_path}")

    # --- 2. Save grouped mean monthly data ---
    monthly_avg_data = {}
    for station, df in aws_data_dict.items():
        if 'RH' in df.columns:
        # Ensure DATETIME is the index
            if 'DATETIME' in df.columns:
                df_rh = df.drop_duplicates(subset='DATETIME').set_index('DATETIME')
            else:
                df_rh = df.copy()

            monthly_avg_rh = df_rh.groupby(df_rh.index.month)['RH'].mean()
            monthly_avg_data[station] = monthly_avg_rh

    if monthly_avg_data:
        monthly_df = pd.DataFrame(monthly_avg_data)
        monthly_df.index.name = 'Month'
        monthly_rh_path = os.path.join(output_dir, 'mean_monthly_rh_data.csv')
        monthly_df.to_csv(monthly_rh_path)
        print(f"Saved mean monthly RH data to {monthly_rh_path}")


def get_seasonal_RH_PRES_data(save_summaries=True):
    """Full pipeline: generate the humidity timeseries, derive the MALR/RH
    summary files, and return the seasonal per-station packaging used by the
    analysis scripts."""
    aws_data_dict = generate_humidity_timeseries(save=True)

    if save_summaries:
        compute_malr_and_rh_summaries(aws_data_dict)

    # Make sure average temperature for every hour
    for station, df in aws_data_dict.items():
        if 'DATETIME' in df.columns:
            df.set_index('DATETIME', inplace=True)
        df.index = pd.to_datetime(df.index)  # Ensure the index is a DatetimeIndex
        df['h'] = df.index.floor('h')
        avg_temp_h = df.groupby('h')['TEMP'].mean().reset_index()
        avg_temp_h = avg_temp_h.rename(columns={'TEMP': 'Avg_Temp_H'})
        df = pd.merge(df, avg_temp_h, on='h', how='left')
        aws_data_dict[station] = df

    # Ensure that the 1H periods are aligned across all stations
    common_H = pd.date_range(start='2012-01-01', end='2026-01-01', freq='h')

    # Reindex each station's DataFrame to the common 1H periods, keeping missing values as NaN
    for station in aws_data_dict.keys():
        aws_data_dict[station] = aws_data_dict[station].drop_duplicates(subset='h').set_index('h').reindex(common_H)

    station_seasonal_data = {}
    for station, df in aws_data_dict.items():
        df = df.loc[:, ~df.columns.duplicated()]

        df = df[['RH', 'TEMP', 'PRES', 'WSPD', 'WINDDIR', 'DEW_POINT', 'MIXING_RATIO', 'ABSOLUTE_HUMIDITY', 'SPEC_HUM', 'SAT_SPEC_HUM']]
        df_clean = df.dropna(subset=['RH', 'TEMP', 'PRES', 'WSPD', 'WINDDIR'])
        df_clean['Season'] = df_clean.index.map(get_season)
        station_seasonal_data[station] = df_clean

    return station_seasonal_data


if __name__ == '__main__':
    # Regenerates data/Moisture and the data/LapseRate summary files
    # (overwrites the shipped humidity timeseries)
    get_seasonal_RH_PRES_data()
