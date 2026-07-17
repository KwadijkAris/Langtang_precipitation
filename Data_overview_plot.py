import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

_SCRIPT_DIR        = Path(os.path.abspath(__file__)).parent   # always absolute
if getattr(sys, "frozen", False):
    _DATA_DIR = Path(sys.executable).parent / "data"
else:
    _local    = _SCRIPT_DIR.parent / "data"
    _local_ok = all([
        (_local / "Data_overview"   / "TEST2.txt").exists(),
        (_local / "LapseRate"       / "lapse_rate.csv").exists(),
        (_local / "Zero_isotherm"   / "zero_isotherm.csv").exists(),
        (_local / "Cleaned" / "Pluvio").exists(),
    ])
    _DATA_DIR = _local if _local_ok else Path(os.path.abspath(str(_SCRIPT_DIR.parent / "data")))
_DATA_OVERVIEW_TXT = str(_DATA_DIR / "Data_overview" / "TEST2.txt")





# # Define the file paths
file_path = _DATA_OVERVIEW_TXT
file_path2 = r"W:\field_data\langtang\Meteo\TippingBuckets\20231129_TB_LangshishaBC_10271177_data.csv"

# Create a dataframe from a csv file
def read_csv(file_path):
    return pd.read_csv(file_path, delimiter='\t', header=0)




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
        df = pd.read_csv(file_path)
        df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y-%m-%d %H:%M:%S')
        df = df.drop(columns=['DATE', 'TIME'])
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

# Create a dataframe from a CSV file
def read_rainfall_TB(file_path):
    df = pd.read_csv(file_path)
    df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y-%m-%d %H:%M:%S')
    df = df.drop(columns=['DATE', 'TIME'])

    if file_path == r'../data/raw/TippingBuckets/TB_Numthang_34328_data.csv':
        filepath2 = r'../data/raw/TippingBuckets/TB_Numthang_10271178_data.csv'
        df2 = pd.read_csv(filepath2)
        df2['DATETIME'] = pd.to_datetime(df2['DATE'] + ' ' + df2['TIME'], format='%Y-%m-%d %H:%M:%S')
        df2 = df2[df2['DATETIME'] >= '2014-10-10 09:22:58']
        df_combined = pd.concat([df, df2], ignore_index=True)
        df_combined['DATETIME'] = pd.to_datetime(df_combined['DATETIME'])
        df_combined = df_combined.sort_values(by='DATETIME').reset_index(drop=True)
        df = df_combined
    
    
    return df
