"""Cleaning of all wind sensors (Kyangjin AWS, Yala BC AWS, Morimoto
MicroMet, Langshisha Pluvio anemometer).

get_aws_df_wind() contains the wind cleaning and returns the gap-free
(dropna) seasonal series used by the humidity generation. The files in
data/Cleaned/Wind were saved from this cleaning chain; the exact save run
is not preserved in the code, so this script does not overwrite them.

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import calendar
from station_data import (get_dir, get_elevation, read_AWS, read_pluvio,
                          get_season, _DATA_DIR)


def get_aws_df_wind(Station=None):

    # This function plots and cleans the wind data
    # As the Kyangjin station has been rebuilt, the offset has been corrected for relative to the first measurements taken


    # AWS data only for Kyangjin and Yala BC as others do not measure wind.

    if Station == None:
        station_names = ['Kyangjin AWS', 'Yala BC AWS']
    else:
        station_names = Station


    elevation = get_elevation(station_names)
    aws_data_dict = {}
    
    #---------- DATA LOADING & PREPROCESSING ----------#
    
    # Read and preprocess wind data for each station
    for i in range(len(station_names)):
        file_path = get_dir(station_names[i])
        df = read_AWS(file_path[0])
        df['Elevation'] = elevation[i]
        # Extract and process wind data
        df = df.replace('NA', np.nan)
        df['WINDDIR'] = pd.to_numeric(df['WINDDIR'], errors='coerce')
        df['WSPD'] = pd.to_numeric(df['WSPD'], errors='coerce')
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])  # Ensure DATETIME is in datetime format

        # Sub-hourly stations (MicroMet logs at 15 min): clean out-of-range values,
        # then aggregate to hourly so they align with the hourly AWS stations.
        # WSPD uses a plain mean; WINDDIR needs a circular (vector) mean.
        step = df['DATETIME'].diff().median()
        if pd.notna(step) and step < pd.Timedelta('1h'):
            df.loc[df['WINDDIR'] > 400, 'WINDDIR'] = np.nan
            df.loc[df['WSPD'] > 40, 'WSPD'] = np.nan
            df = df.set_index('DATETIME')
            wd_rad = np.deg2rad(df['WINDDIR'])
            hourly = pd.DataFrame({
                'WSPD': df['WSPD'].resample('h').mean(),
                '_sin': np.sin(wd_rad).resample('h').mean(),
                '_cos': np.cos(wd_rad).resample('h').mean(),
            })
            hourly['WINDDIR'] = (np.rad2deg(np.arctan2(hourly['_sin'], hourly['_cos'])) + 360) % 360
            df = hourly.drop(columns=['_sin', '_cos']).reset_index()

        # Filter out data before 2017 for Kyangjin AWS
        if station_names[i] == 'Kyangjin AWS':
            mask = (df['DATETIME'].dt.year < 2016)
            df.loc[mask, ['WINDDIR', 'WSPD']] = np.nan

            # Remove the period ('2019-05-05', '2019-11-15') from the data
            remove_mask = (df['DATETIME'] >= '2019-05-04') & (df['DATETIME'] <= '2021-11-12')
            df.loc[remove_mask, ['WINDDIR', 'WSPD']] = np.nan
            aws_data_dict[station_names[i]] = df
        else:
            aws_data_dict[station_names[i]] = df

    
            
    # Process data for each station
    station_seasonal_data = {}  # To store seasonal data for plotting later


    
    for station, df in aws_data_dict.items():
        #---------- DATA CLEANING & TIME SERIES PREPARATION ----------#
        
        # Drop unnecessary columns
        df = df.loc[:, ~df.columns.duplicated()]
        # Check for gaps in the DATETIME column
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        expected_range = pd.date_range(start=df['DATETIME'].min(), end=df['DATETIME'].max(), freq='H')

        # Fill missing dates with NaN for all columns
        df = df.drop_duplicates(subset='DATETIME')
        df = df.set_index('DATETIME').reindex(expected_range).reset_index()
        df.rename(columns={'index': 'DATETIME'}, inplace=True)
        missing_dates = expected_range.difference(df['DATETIME'])

        if not missing_dates.empty:
            print(f"Missing dates in {station}:")
            print(missing_dates)
        else:
            print(f"No gaps in the DATETIME column for {station}.")
            
        if missing_dates.empty:
            start_date = df['DATETIME'].min()
            end_date = df['DATETIME'].max()
        if not missing_dates.empty:
            raise ValueError(f"Data gap detected in station {station} at the following times: {missing_dates}")
            
        df['DATETIME'] = pd.date_range(start=start_date, end=end_date, freq='H')[:len(df)]
        df.set_index('DATETIME', inplace=True)
        df = df[['WINDDIR', 'WSPD']]
        
         # Make a clean copy of data with datetime index for analysis
        df_clean_corrected = df.dropna(subset=['WINDDIR', 'WSPD'])
        df_clean = df.dropna(subset=['WINDDIR', 'WSPD'])
        station_seasonal_data[station] = df_clean_corrected
      
        df_clean = df_clean.copy()
        df_clean['Season'] = df_clean.index.map(get_season)
        station_seasonal_data[station] = df_clean


        debug_plotting = False
        if debug_plotting == True:
            # Plot original and cleaned wind direction for debugging
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            axes[0].scatter(df_clean.index, df_clean['WINDDIR'], color='red', s=0.5, alpha=0.5, label='Original WINDDIR')
            axes[0].set_ylabel('Original Wind Direction (deg)')
            axes[0].set_title(f'{station} Original Wind Direction')
            axes[0].legend()
            axes[0].grid(True)

            axes[1].scatter(df_clean_corrected.index, df_clean_corrected['WINDDIR'], color='blue', s=0.5, alpha=0.5, label='Cleaned WINDDIR')
            axes[1].set_ylabel('Cleaned Wind Direction (deg)')
            axes[1].set_title(f'{station} Cleaned Wind Direction')
            axes[1].legend()
            axes[1].grid(True)

            axes[1].set_xlabel('Date')
            plt.tight_layout()
            plt.show()

            # Plot wind strength for debugging
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.scatter(df_clean_corrected.index, df_clean_corrected['WSPD'], color='green', s=0.5, alpha=0.5, label='Wind Speed')
            ax.set_ylabel('Wind Speed (m/s)')
            ax.set_title(f'{station} Wind Speed')
            ax.legend()
            ax.grid(True)
            ax.set_xlabel('Date')
            plt.tight_layout()
            plt.show()

        # Store data for combined plotting after the loop
            

    plotting = False
    if plotting == True:
    # --- Create a single figure with two subplots for comparison ---
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        
        month_names = [calendar.month_abbr[i] for i in range(1, 13)]
        colors = {'Kyangjin AWS': ('blue', 'cyan'), 'Yala BC AWS': ('red', 'orange')}
        
        # --- Subplot 1: Wind Strength ---
        ax_strength = axes[0]
        # --- Subplot 2: Wind Direction ---
        ax_direction = axes[1]

        for station, df_clean_corrected in station_seasonal_data.items():
            # Define day and night masks
            df_copy = df_clean_corrected.copy()
            df_copy['hour'] = df_copy.index.hour
            
            # Nighttime: 18:00 to 06:00
            night_mask = (df_copy['hour'] >= 18) | (df_copy['hour'] <= 6)
            df_night = df_copy[night_mask]
            
            # Daytime: 07:00 to 17:00
            day_mask = (df_copy['hour'] > 6) & (df_copy['hour'] < 18)
            df_day = df_copy[day_mask]

            # --- Calculate Mean Monthly Wind Speed (Strength) for Day and Night ---
            monthly_avg_wspd_day = df_day.groupby(df_day.index.month)['WSPD'].mean()
            monthly_avg_wspd_night = df_night.groupby(df_night.index.month)['WSPD'].mean()

            # --- Calculate Prevalent Monthly Wind Direction for Day and Night ---
            def calculate_monthly_prevalent_direction(df):
                bins = np.arange(-11.25, 360, 22.5)
                labels = (bins[:-1] + bins[1:]) / 2
                labels[0] = 0
                df_temp = df.copy()
                df_temp['WINDDIR_binned'] = pd.cut(df_temp['WINDDIR'] % 360, bins=bins, labels=labels, right=False, include_lowest=True)
                monthly_prevalent = df_temp.groupby(df_temp.index.month)['WINDDIR_binned'].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                return monthly_prevalent.astype(float)

            monthly_avg_winddir_day = calculate_monthly_prevalent_direction(df_day)
            monthly_avg_winddir_night = calculate_monthly_prevalent_direction(df_night)

            # Plotting on the strength subplot
            ax_strength.plot(monthly_avg_wspd_day.index, monthly_avg_wspd_day, label=f'{station} Day', marker='o', linestyle='-', color=colors[station][0])
            ax_strength.plot(monthly_avg_wspd_night.index, monthly_avg_wspd_night, label=f'{station} Night', marker='x', linestyle='--', color=colors[station][1])
            
            # Plotting on the direction subplot
            ax_direction.plot(monthly_avg_winddir_day.index, monthly_avg_winddir_day, label=f'{station} Day', marker='o', linestyle='-', color=colors[station][0])
            ax_direction.plot(monthly_avg_winddir_night.index, monthly_avg_winddir_night, label=f'{station} Night', marker='x', linestyle='--', color=colors[station][1])

        # --- Finalize Strength Plot ---
        ax_strength.set_title('Mean Monthly Wind Speed (Day vs. Night)')
        ax_strength.set_ylabel('Wind Speed (m/s)')
        ax_strength.legend()
        ax_strength.grid(True)

        # --- Finalize Direction Plot ---
        ax_direction.set_title('Prevalent Monthly Wind Direction (Day vs. Night)')
        ax_direction.set_xlabel('Month')
        ax_direction.set_ylabel('Wind Direction (degrees)')
        ax_direction.set_ylim(0, 360)
        ax_direction.set_yticks(np.arange(0, 405, 45), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
        ax_direction.set_xticks(ticks=range(1, 13), labels=month_names)
        ax_direction.legend()
        ax_direction.grid(True)

        plt.tight_layout()
        plt.show()

        # ---------- SAVE CLEANED DATA ----------#
        save_data = False
        if save_data := True:
            output_dir = str(_DATA_DIR / "Cleaned" / "For_Quinten")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a filename for the station
            # Sanitize station name for filename
            sanitized_station_name = station.replace(' ', '_').replace('/', '_')
            output_filename = f"{sanitized_station_name}_wind_data.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # Select WSPD and WINDDIR columns from the corrected dataframe
            data_to_save = df_clean_corrected[['WSPD', 'WINDDIR']]
            
            # Save to CSV
            data_to_save.to_csv(output_path)
            
            print(f"Saved cleaned wind data for {station} to {output_path}")


    
        # ---------- MERGE AND SAVE CLEANED DATA FOR KYANGJIN AWS ----------#
        for station in station_names:
            if station == 'Kyangjin AWS':
                try:
                # Define the path to the target CSV file
                    target_csv_path = str(_DATA_DIR / "Cleaned" / "For_Quinten" / "Kyangjin AWS.csv")
                
                    # Read the existing CSV file
                    existing_df = pd.read_csv(target_csv_path)
                    
                    # Ensure the DATETIME column is in datetime format and set it as the index
                    existing_df['DATETIME'] = pd.to_datetime(existing_df['DATETIME'])
                    existing_df.set_index('DATETIME', inplace=True)
                    
                    # Select WSPD and WINDDIR columns from the corrected dataframe
                    data_to_add = df_clean_corrected[['WSPD', 'WINDDIR']]
                    
                    # Join the new data with the existing dataframe
                    # This will add WSPD and WINDDIR columns, aligning on the DATETIME index
                    merged_df = existing_df.join(data_to_add)
                    
                    # Save the merged dataframe back to the CSV file, overwriting it
                    merged_df.to_csv(target_csv_path)
                    
                    print(f"Successfully added wind data to {target_csv_path}")
                
                except FileNotFoundError:
                    print(f"Error: The file {target_csv_path} was not found. Cannot add wind data.")
                except Exception as e:
                    print(f"An error occurred while processing {station}: {e}")


    return station_seasonal_data
