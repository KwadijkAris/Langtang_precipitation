"""Shared station metadata access and raw-data readers.

All paths resolve relative to this file: raw and cleaned data live in
<zenodo_root>/data, results go to <zenodo_root>/results.

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(os.path.abspath(__file__)).parent   # always absolute
_DATA_DIR = _SCRIPT_DIR.parent / "data"                # <zenodo_root>/data
_RESULTS_DIR = _SCRIPT_DIR.parent / "results"          # <zenodo_root>/results
_DATA_OVERVIEW_TXT = str(_DATA_DIR / "Data_overview" / "TEST2.txt")


def resolve_path(file_path):
    """Resolve metadata paths against the Zenodo base directory."""
    path = Path(file_path)
    if path.is_absolute():
        return path

    parts = path.parts
    if parts and parts[0] == "..":
        return (_SCRIPT_DIR / path).resolve()
    if parts and parts[0] == "data":
        return (_SCRIPT_DIR.parent / path).resolve()
    return (_SCRIPT_DIR / path).resolve()


def read_csv(file_path):
    return pd.read_csv(resolve_path(file_path), delimiter='\t', header=0)


def get_dir(station_names):
    file_path = _DATA_OVERVIEW_TXT
    df = read_csv(file_path)

    if isinstance(station_names, str):
        station_names = [station_names]
    dirs = []
    for station_name in station_names:
        try:
            dir = df['dir'][df['name'] == station_name].values[0]
            dirs.append(dir)
        except IndexError:
            raise ValueError(f"Directory for station '{station_name}' not found.")
    return dirs


def get_elevation(station_names):
    file_path = _DATA_OVERVIEW_TXT

    df = read_csv(file_path)

    if isinstance(station_names, str):
        station_names = [station_names]
    elevs = []
    for station_name in station_names:
        try:
            elev = df['elev'][df['name'] == station_name].values[0]
            elevs.append(elev)
        except IndexError:
            raise ValueError(f"Elevation for station '{station_name}' not found.")
    return elevs


def get_measurement(station_names):
    file_path = _DATA_OVERVIEW_TXT

    df = read_csv(file_path)

    if isinstance(station_names, str):
        station_names = [station_names]
    elevs = []
    for station_name in station_names:
        try:
            elev = df['measurement'][df['name'] == station_name].values[0]
            elevs.append(elev)
        except IndexError:
            raise ValueError(f"measurement for station '{station_name}' not found.")
    return elevs


def get_station_coordinate(station_names):
    file_path = _DATA_OVERVIEW_TXT
    df = read_csv(file_path)

    if isinstance(station_names, str):
        station_names = [station_names]
    
    coordinates = []
    lon = []
    lat = []

    for station_name in station_names:
        try:
            station_row = df[df['name'] == station_name]
            if station_row.empty:
                raise IndexError(f"Station '{station_name}' not found.")
            lon.append(station_row['lon'].values[0])
            lat.append(station_row['lat'].values[0])
            coordinates.append((lon, lat))
        except (IndexError, KeyError):
            raise ValueError(f"Coordinates for station '{station_name}' not found.")
    
    return lon, lat


def read_pluvio_cleaned(station_names, dt=None, no_threshold=False):
    station_names2 = ['Syabru TB', 'Lama TB','Lirung Camp TB', 'Ganja La TB1', 'Ganja La TB2', 'Ganja La TB3', 'Morimoto TB','Langshisha Pluvio TB', 'Shalbachum TB', 'Langshisha BC TB', 'Numthang TB', 'Jathang TB','snowAMP Ganjala upper','SNOWAMP_lower', 'SNOWAMP_middle']
    kochendorfer_stations = ['Langshisha Pluvio', 'Morimoto Pluvio', 'Kyangjin AWS', 'Yala BC AWS']
    temp_aws = ['Yala Glacier AWS', 'Langtang Glacier AWS']
    cleaned_merged = {} 
    for station in station_names:
        if station in kochendorfer_stations:
            cleaned = str(_DATA_DIR / "Cleaned" / "Kochendorfer_corrected")
        elif station in temp_aws:
            cleaned = str(_DATA_DIR / "Cleaned" / "Temperature")
        elif no_threshold == False and station in station_names2:
            cleaned = str(_DATA_DIR / "Cleaned" / "TB")
        elif no_threshold == True and station in station_names2:
            cleaned = str(_DATA_DIR / "Cleaned" / "TB" / "no_threshold")
        else:
            cleaned = str(_DATA_DIR / "Cleaned" / "Pluvio")
        if station in kochendorfer_stations:
            # Exact file name: the corrected dir also holds unrelated files
            file_name = f'{station}_kochendorfer_corrected.csv'
            if not os.path.isfile(os.path.join(cleaned, file_name)):
                file_name = None
        else:
            file_name = next((f for f in os.listdir(cleaned) if station in f), None)
        if not file_name:
            raise FileNotFoundError(f"No file found for station {station} in {cleaned}")
        file_path = os.path.join(cleaned, file_name)

        df = pd.read_csv(file_path)
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df = df.set_index('DATETIME')

        # Handle Kochendorfer data: use Precipitation_corrected if available
        if 'kochendorfer' in file_path.lower() and 'Precipitation_corrected' in df.columns:
            # Corrected files keep their native resolution: name the rainfall
            # column after the actual timestep so the dt='1h' resampling
            # branch treats 15-min stations correctly.
            step = df.index.to_series().diff().median()
            rain_col = 'Rainfall_15min' if step <= pd.Timedelta('15min') else 'Rainfall_1H'
            df = df.rename(columns={'Precipitation_corrected': rain_col})
            # Drop the uncorrected series so downstream code cannot mix them up
            df = df.drop(columns=['Precipitation'], errors='ignore')
        elif station in kochendorfer_stations:
            print('No Kochendorfer correction was found for station:', station)

        # Cleaned/Temperature files store one hourly 'TEMP' column; expose it
        # under the reader's usual hourly name
        if 'TEMP' in df.columns:
            df['TEMP'] = pd.to_numeric(df['TEMP'].replace('NAN', np.nan), errors='coerce')
            df = df.rename(columns={'TEMP': 'Temperature_1H'})

        if 'Rainfall_15min' in df.columns:
            df['Rainfall_15min'] = df['Rainfall_15min'].replace('NAN', np.nan)
            df['Rainfall_15min'] = pd.to_numeric(df['Rainfall_15min'], errors='coerce')
        
        if 'Temperature_15min' in df.columns:
            df['Temperature_15min'] = df['Temperature_15min'].replace('NAN', np.nan)
            df['Temperature_15min'] = pd.to_numeric(df['Temperature_15min'], errors='coerce')

        if 'Rainfall_1H' in df.columns:
            df['Rainfall_1H'] = df['Rainfall_1H'].replace('NAN', np.nan)
            df['Rainfall_1H'] = pd.to_numeric(df['Rainfall_1H'], errors='coerce')

        if 'Temperature_1H' in df.columns:
            df['Temperature_1H'] = df['Temperature_1H'].replace('NAN', np.nan)
            df['Temperature_1H'] = pd.to_numeric(df['Temperature_1H'], errors='coerce')
        if 'Temperature' in df.columns:
            df['Temperature'] = df['Temperature'].replace('NAN', np.nan)
            df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')

        if 'CS725_Swek(mm)' in df.columns:
            df['CS725_Swek(mm)'] = df['CS725_Swek(mm)'].replace('NAN', np.nan)
            df['CS725_Swek(mm)'] = pd.to_numeric(df['CS725_Swek(mm)'], errors='coerce')

        if dt == '1h':
            if 'Rainfall_1H' in df.columns:
                df = df.rename(columns={'Temperature_15min': 'Temperature_1H', 'Temperature': 'Temperature_1H'})
                pass
            else:
                resampling_dict = {}
                if 'Rainfall_15min' in df.columns:
                    resampling_dict['Rainfall_15min'] = lambda x: x.sum(min_count=4)
                if 'Temperature_15min' in df.columns:
                    resampling_dict['Temperature_15min'] = 'mean'
                
                if 'Temperature_1H' in df.columns and 'Temperature_15min' not in df.columns:
                    resampling_dict['Temperature_1H'] = 'first'
                if 'Temperature' in df.columns and 'Temperature_15min' not in df.columns:
                    resampling_dict['Temperature'] = 'first'
                if 'CS725_Swek(mm)' in df.columns:
                    resampling_dict['CS725_Swek(mm)'] = 'mean'

                if resampling_dict:
                    df = df.resample('1h').agg(resampling_dict)
                    df = df.rename(columns={'Rainfall_15min': 'Rainfall_1H', 'Temperature_15min': 'Temperature_1H', 'Temperature': 'Temperature_1H'})

        cleaned_merged[station] = df

    return cleaned_merged


def read_temp_TB(file_path):
    try:
        df = pd.read_csv(resolve_path(file_path))
        df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y-%m-%d %H:%M:%S')
        df = df.drop(columns=['DATE', 'TIME'])
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None


def read_rainfall_TB(file_path):
    file_path = resolve_path(file_path)
    df = pd.read_csv(file_path)
    df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y-%m-%d %H:%M:%S')
    df = df.drop(columns=['DATE', 'TIME'])

    if file_path == resolve_path(r'../data/raw/TippingBuckets/TB_Numthang_34328_data.csv'):
        filepath2 = resolve_path(r'../data/raw/TippingBuckets/TB_Numthang_10271178_data.csv')
        df2 = pd.read_csv(filepath2)
        df2['DATETIME'] = pd.to_datetime(df2['DATE'] + ' ' + df2['TIME'], format='%Y-%m-%d %H:%M:%S')
        df2 = df2[df2['DATETIME'] >= '2014-10-10 09:22:58']
        df_combined = pd.concat([df, df2], ignore_index=True)
        df_combined['DATETIME'] = pd.to_datetime(df_combined['DATETIME'])
        df_combined = df_combined.sort_values(by='DATETIME').reset_index(drop=True)
        df = df_combined
    
    
    return df


def read_pluvio(file_path):
    df = pd.read_csv(resolve_path(file_path))
    df['DATETIME'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')
    df = df.drop(columns=['Date', 'Time'])

    # Check for Temp columns and create a new Temp column
    if 'Temp new' in df.columns and 'Temp old' in df.columns:
        df['Temp'] = df['Temp new'].combine_first(df['Temp old'])
    elif 'Temp new' in df.columns:
        df['Temp'] = df['Temp new']
    elif 'Temp old' in df.columns:
        df['Temp'] = df['Temp old']
    return df


def read_SNOWAMP(file_path):
    df = pd.read_csv(resolve_path(file_path))
    df = df.rename(columns={'DATE': 'Date', 'TIME': 'Time'})
    df['DATETIME'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')
    df = df.drop(columns=['Date', 'Time'])
    return df


def read_AWS(file_path):
    df = pd.read_csv(resolve_path(file_path))
    # MicroMet files (e.g. Morimoto MM) use different headers; normalize to AWS names
    df = df.rename(columns={'Wind_Dir': 'WINDDIR', 'Wind_Speed': 'WSPD', 'Rel_Air_Press': 'PRES'})
    try:
        df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y-%m-%d %H:%M')
    df = df.drop(columns=['DATE', 'TIME'])
    return df


def get_season(dt):
    m = dt.month
    d = dt.day
    if (m == 6 and d > 15) or m in [7, 8, 9]:
        return 'Monsoon'
    elif m == 10 or m == 11 or (m == 12 and d < 31):
        return 'Postmonsoon'
    elif m in [3, 4, 5] or (m == 6 and d <= 15):
        return 'Premonsoon'
    elif m == 1 or m == 2:
        return 'Winter'
    else:
        return 'Other'

