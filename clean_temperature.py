"""Cleaning of all temperature sensors (TB temperature loggers, pluviometer
hydroclips, AWS, SNOWAMP stations). Each get_* function returns a dict of
per-station dataframes with a cleaned TEMP column.

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from station_data import (get_dir, get_elevation, get_measurement,
                          read_temp_TB, read_SNOWAMP, read_AWS, read_pluvio,
                          _DATA_DIR)


def get_TB_df_temp(dt='0.25h'):
    station_names = ['Shalbachum TB Temp','Morimoto TB Temp','Ganja La TB3 Temp', 'Ganja La TB2 Temp', 'Ganja La TB1 Temp', 'Langshisha BC TB Temp']


    dirs= get_dir(station_names)
    all_merged_dfs = {}

    # Loop again to process data and plot
    for (file_path, TB_name) in zip(dirs, station_names):  # TB_name is unchanged here

        # Use temp_file_path_map to get the correct temperature file path for the station
        df_TB_T = read_temp_TB(file_path)
        # Convert datetime columns
        df_TB_T['DATETIME'] = pd.to_datetime(df_TB_T['DATETIME'])
        df_TB_T = df_TB_T.resample(dt, on='DATETIME').mean().reset_index() 


#Clean the dataset from outliers/invalid data/low diurnal temperature fluctuations due to the insulating effects of snow     
        if TB_name == 'Ganja La TB1 Temp':
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2015-02-25') & (df_TB_T['DATETIME'] <= '2015-03-09'), 'TEMP'] = np.nan
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2015-02-25') & (df_TB_T['DATETIME'] <= '2015-06-07'), 'TEMP'] = np.nan
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2019-02-06') & (df_TB_T['DATETIME'] <= '2019-04-03'), 'TEMP'] = np.nan

        if TB_name == 'Ganja La TB3 Temp':
            df_TB_T.loc[df_TB_T['DATETIME'] >= '2021-11-10 07:00', 'TEMP'] = np.nan
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2019-02-06') & (df_TB_T['DATETIME'] <= '2019-04-28'), 'TEMP'] = np.nan
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2014-12-31') & (df_TB_T['DATETIME'] <= '2015-05-19'), 'TEMP'] = np.nan
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2021-11-10'), 'TEMP'] = np.nan
        
        if TB_name == 'Morimoto TB Temp':
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2015-02-28') & (df_TB_T['DATETIME'] <= '2015-03-12'), 'TEMP'] = np.nan
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2019-02-25') & (df_TB_T['DATETIME'] <= '2019-03-10'), 'TEMP'] = np.nan
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2024-11-02', 'TEMP')] = np.nan

        if TB_name == 'Shalbachum TB Temp':
            df_TB_T.loc[(df_TB_T['DATETIME'] < '2013-10-26 17:00', 'TEMP')] = np.nan
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2015-02-25') & (df_TB_T['DATETIME'] <= '2015-04-01'), 'TEMP'] = np.nan
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2019-02-06') & (df_TB_T['DATETIME'] <= '2019-04-03'), 'TEMP'] = np.nan
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2023-11-03', 'TEMP')] = np.nan

        if TB_name == 'Langshisha BC TB Temp':
            df_TB_T.loc[(df_TB_T['DATETIME'] >= '2015-02-25') & (df_TB_T['DATETIME'] <= '2015-03-06'), 'TEMP'] = np.nan


        # Append df_TB_T to a dictionary for later use
        all_merged_dfs[TB_name] = df_TB_T


        # Interpolate to fill gaps larger than the resampling period 'dt'
        df_TB_T = df_TB_T.set_index('DATETIME')
        # Calculate the time difference between consecutive rows
        time_diffs = df_TB_T.index.to_series().diff()
        # Identify gaps larger than the expected frequency
        resampling_freq = pd.to_timedelta(dt)
        gaps = time_diffs > resampling_freq
        
        # If there are gaps, interpolate them
        if gaps.any():
            # Reindex to create the missing timestamps and then interpolate
            # This ensures that interpolation happens only across the identified gaps
            df_TB_T = df_TB_T.reindex(pd.date_range(start=df_TB_T.index.min(), end=df_TB_T.index.max(), freq=dt)).interpolate(method='linear')
        
        df_TB_T = df_TB_T.reset_index().rename(columns={'index': 'DATETIME'})
        # Ensure that the data ranges from 2012-01 up to 2025-01
        common_range = pd.date_range(start='2012-01-01', end='2026-01-01', freq=dt)
        df_TB_T = df_TB_T.set_index('DATETIME').reindex(common_range).reset_index().rename(columns={'index': 'DATETIME'})

    return all_merged_dfs


def get_Pluvio_temp(station=None, agg_hourly=True):
    if station is None:
        station_names = ['Ganja La Pluvio', 'Yala Pluvio','Langshisha Pluvio', 'Morimoto Pluvio']
    else:
        station_names = [f'{station}']

    dirs = get_dir(station_names)

    measurement = get_measurement(station_names)
    dirs_Pluvio = [dirs[i] for i in range(len(dirs)) if measurement[i] == 'Pluviometer' and pd.notna(dirs[i])]
    Pluvio_name = [station_names[i] for i in range(len(station_names)) if measurement[i] == 'Pluviometer' and pd.notna(station_names[i])]
    elevation = get_elevation(station_names)
    # Initialize an empty dataframe to store merged data
    merged_df = {}

    # Iterate through each file and corresponding station name
    for file_path, name in zip(dirs_Pluvio, Pluvio_name):
        df_pluvio = read_pluvio(file_path)
        

                
        # Convert 'NA' strings to actual NaN values
        df_pluvio.replace("NA", pd.NA, inplace=True)
        
        # Apply mask to set specific values to NaN
        if 'Temp old' in df_pluvio.columns and 'Temp new' in df_pluvio.columns:
            df_pluvio['TEMP'] = df_pluvio.apply(lambda row: row['Temp new'] if pd.notna(row['Temp new']) else row['Temp old'], axis=1)
            df_pluvio.drop(columns=['Temp old', 'Temp new'], inplace=True)
        elif 'Temp old' in df_pluvio.columns:
            df_pluvio['TEMP'] = df_pluvio['Temp old']
            df_pluvio.drop(columns=['Temp old'], inplace=True)
        elif 'Temp new' in df_pluvio.columns:
            df_pluvio['TEMP'] = df_pluvio['Temp new']
            df_pluvio.drop(columns=['Temp new'], inplace=True)
        else:
            # If neither 'Temp_old' nor 'Temp new' are present, use 'Temp' if available
            if 'Temp' in df_pluvio.columns:
                df_pluvio['TEMP'] = df_pluvio['Temp']
            else:
                df_pluvio['TEMP'] = pd.NA
        
        # Add station name and elevation to the dataframe
        df_pluvio['Station'] = name
        df_pluvio['Elevation'] = elevation[station_names.index(name)]
        
        # Drop 'Temp_old' and 'Temp new' columns if they exist
        df_pluvio.drop(columns=['Temp old', 'Temp new'], inplace=True, errors='ignore')
        

        # Rename 'Temp' column to 'TEMP' if it exists
        if 'Temp' in df_pluvio.columns:
            df_pluvio.rename(columns={'Temp': 'TEMP'}, inplace=True)
        
        # Set data to NaN for specific dates for Langshisha Pluvio
        if name == 'Langshisha Pluvio':
            mask_dates1 = ((df_pluvio['DATETIME'] >= '2023-06-12') & (df_pluvio['DATETIME'] <= '2023-06-13')) | ((df_pluvio['DATETIME'] >= '2024-06-25') & (df_pluvio['DATETIME'] <= '2024-07-25'))
            mask_dates2 = ((df_pluvio['DATETIME'] >= '2023-07-23') & (df_pluvio['DATETIME'] <= '2023-07-30'))
            mask_dates3 = ((df_pluvio['DATETIME'] >= '2023-06-11') & (df_pluvio['DATETIME'] <= '2023-06-16'))
            mask_dates4 = ((df_pluvio['DATETIME'] >= '2023-11-06') & (df_pluvio['DATETIME'] <= '2023-11-11'))
            mask_dates6 = ((df_pluvio['DATETIME'] >= '2024-07-24') & (df_pluvio['DATETIME'] <= '2024-08-18'))
            mask_dates7 = ((df_pluvio['DATETIME'] >= '2021-05-23') & (df_pluvio['DATETIME'] <= '2021-11-02'))

            mask_dates = mask_dates1 | mask_dates2 | mask_dates3 | mask_dates4 | mask_dates6 | mask_dates7
            df_pluvio.loc[mask_dates, ['TEMP']] = np.nan

        if name == 'Yala Pluvio':
            df_pluvio.loc[df_pluvio['DATETIME'] < '2014-11-24', 'TEMP'] = np.nan

        if name == 'Ganja La Pluvio':
            df_pluvio.loc[(df_pluvio['DATETIME'] >= '2014-12-12') & (df_pluvio['DATETIME'] <= '2015-01-02'), 'TEMP'] = np.nan


        if name == 'Morimoto Pluvio':
            df_pluvio.loc[(df_pluvio['DATETIME'] >= '2023-12-31'), 'TEMP'] = np.nan
            
        merged_df[name] = df_pluvio



    # Resample to hourly data
    for station, df in merged_df.items():
        # Drop unnecessary columns
        df = df.loc[:, ~df.columns.duplicated()]
        df = df[['DATETIME', 'TEMP']]
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df['TEMP'] = pd.to_numeric(df['TEMP'], errors='coerce')
        
        
        if agg_hourly ==True:
                # Resample to hourly data
            df = df.resample('h', on='DATETIME').mean()
        df = df.reset_index()
        df.set_index('DATETIME', inplace=True)
        
        # Add to the dictionary
        merged_df[station] = df

    return merged_df


def get_aws_df_temp(Station=None):
    if Station == None:
        station_names = ['Kyangjin AWS', 'Yala BC AWS', 'Yala Glacier AWS']
    else:
        station_names = Station
    
    elevation = get_elevation(station_names)
    aws_data_dict = {}
    for station in station_names:
        file_path = get_dir(station)
        df = read_AWS(file_path[0])
        df['Elevation'] = elevation[station_names.index(station)]
        if 'TAIR' in df.columns:
            df = df.rename(columns={'TAIR': 'TEMP'})
        # Removing outliers for Yala BC AWS
        if station == 'Yala BC AWS':
            # Merge all mask dates from both code blocks
            mask_dates = [
                ('2022-06-04 19:00', '2023-06-25 02:00'),
                ('2016-06-13', '2016-06-14'),
                ('2023-06-26', '2023-06-28'),
                ('2023-07-30', '2023-08-01'),
                ('2023-08-06', '2023-08-08'),
                ('2023-10-10', '2023-10-12'),
                ('2023-11-08', None),
                ('2015-06-08', '2015-10-22'),
                ('2016-06-17', '2016-06-18'),
                ('2023-07-14', '2023-07-18'),
                ('2022-06-05', '2023-06-25'),
                ('2023-07-16', '2023-07-17'),
                ('2023-07-30', '2023-07-31'),
                ('2023-08-06', '2023-08-07'),
                ('2023-10-11 01:00', '2023-10-11 17:00'),
                ('2023-11-09', '2023-11-13'),
                ('2016-06-13', '2016-06-15'),
                ('2016-06-17 11:00', '2016-06-17 15:00'),
            ]
            mask = pd.Series(False, index=df.index)
            for start, end in mask_dates:
                if end:
                    mask |= (df['DATETIME'] >= start) & (df['DATETIME'] <= end)
                else:
                    mask |= (df['DATETIME'] >= start)
            df.loc[mask, 'TEMP'] = np.nan
            mask = pd.Series(False, index=df.index)
            for start, end in mask_dates:
                if end:
                    mask |= (df['DATETIME'] >= start) & (df['DATETIME'] <= end)
                else:
                    mask |= (df['DATETIME'] >= start)
            
            df.loc[mask, 'TEMP'] = np.nan



        if station == 'Yala Glacier AWS':
            mask = (df['DATETIME'] >= '2016-10-14 19:00') & (df['DATETIME'] <= '2016-10-15')
            df.loc[mask, 'TEMP'] = np.nan  

        if station == 'Kyangjin AWS':
            mask = (df['DATETIME'] >= '2015-01-04') & (df['DATETIME'] <= '2015-10-16')
            df.loc[mask, 'TEMP'] = np.nan
        
        aws_data_dict[station] = df

    for station, df in aws_data_dict.items():
        # Drop unnecessary columns
        df = df.loc[:, ~df.columns.duplicated()]
        df = df[['DATETIME', 'TEMP']]
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])

        # Resample to hourly data
        df['TEMP'] = pd.to_numeric(df['TEMP'], errors='coerce')
        df = df.resample('h', on='DATETIME').mean()
        df = df.reset_index()
        # Add to the dictionary
        aws_data_dict[station] = df

    return aws_data_dict


def get_SNOWAMP_df_temp(update_csv=False, plotting=False):
    
    station_names = ['snowAMP Ganja La','snowAMP Ganjala upper','SNOWAMP_lower','SNOWAMP_middle']

    measurement = get_measurement(station_names)
    elevation = get_elevation(station_names)
    SNOWAMP_data_dict = {}
    for i in range(len(station_names)):
        file_path = get_dir(station_names[i])
        df = read_SNOWAMP(file_path[0])
        df['Elevation'] = elevation[i]
        if 'Air Temperature(degC)' in df.columns:
            df = df.rename(columns={'Air Temperature(degC)': 'TEMP'})
        if station_names[i] == 'snowAMP Ganjala upper':
            df.loc[(df['DATETIME'] <= '2016-05-02'), 'TEMP'] = np.nan
            df.loc[(df['DATETIME'] >= '2019-02-26') & (df['DATETIME'] <= '2019-05-05'), 'TEMP'] = np.nan
            df.loc[(df['DATETIME'] >= '2019-02-14') & (df['DATETIME'] <= '2019-11-29'), 'TEMP'] = np.nan
            df.loc[(df['DATETIME'] >= '2020-05-01') & (df['DATETIME'] <= '2020-10-03'), 'TEMP'] = np.nan
            df.loc[(df['DATETIME'] >= '2020-11-16') & (df['DATETIME'] <= '2021-04-17'), 'TEMP'] = np.nan
            df.loc[(df['DATETIME'] <= '2016-04-19'), 'TEMP'] = np.nan
            df.loc[(df['DATETIME'] >= '2019-02-14') & (df['DATETIME'] <= '2019-11-05'), 'Precipitation TB(mm)'] = np.nan
            df.loc[(df['DATETIME'] >= '2021-05-28') & (df['DATETIME'] <= '2021-07-08'), 'Precipitation TB(mm)'] = np.nan


        if station_names[i] == 'SNOWAMP_middle':
            df.loc[(df['DATETIME'] >= '2016-09-20') & (df['DATETIME'] <= '2017-05-01'), 'TEMP'] = np.nan

        if station_names[i] == 'snowAMP Ganja La':
            df.loc[(df['DATETIME'] >= '2018-09-22') & (df['DATETIME'] <= '2018-09-24'), 'TEMP'] = np.nan
            df.loc[
                ((df['DATETIME'] >= '2015-10-31') & (df['DATETIME'] <= '2016-04-30')) |
                ((df['DATETIME'] >= '2018-06-15') & (df['DATETIME'] <= '2018-09-17')) |
                ((df['DATETIME'] >= '2018-12-12') & (df['DATETIME'] <= '2019-02-14')) |
                (df['DATETIME'] >= '2019-11-22'),
                'TEMP'
            ] = np.nan
            df.loc[
            ((df['DATETIME'] >= '2017-10-09') & (df['DATETIME'] <= '2017-11-20')) |
            ((df['DATETIME'] >= '2018-10-04') & (df['DATETIME'] <= '2018-10-28')) |
            ((df['DATETIME'] >= '2019-10-09') & (df['DATETIME'] <= '2019-10-27')) |
            ((df['DATETIME'] >= '2015-10-31') & (df['DATETIME'] <= '2016-04-30')) |
            ((df['DATETIME'] >= '2018-06-15') & (df['DATETIME'] <= '2018-09-17')) |
            ((df['DATETIME'] >= '2018-12-12') & (df['DATETIME'] <= '2019-02-14')) |
            ((df['DATETIME'] >= '2020-10-21') & (df['DATETIME'] <= '2020-11-15')) |
            ((df['DATETIME'] >= '2019-10-09') & (df['DATETIME'] <= '2019-10-27')) |
            ((df['DATETIME'] >= '2017-11-20') & (df['DATETIME'] < '2017-11-28')) |
            ((df['DATETIME'] >= '2018-10-03') & (df['DATETIME'] < '2018-10-28')) |


            (df['DATETIME'] >= '2021-10-18'),
            'Precipitation(mm)'
        ] = np.nan
            
            df.loc[df['DATETIME'] >= '2019-11-22', 'Precipitation TB(mm)'] = np.nan
        # Plot precipitation timeseries for Ganja La
        


        if station_names[i] == 'SNOWAMP_lower':
            df.loc[(df['DATETIME'] >= '2017-10-27') & (df['DATETIME'] <= '2018-03-18'), 'Precipitation TB(mm)'] = np.nan
    



       

        SNOWAMP_data_dict[station_names[i]] = df



        for station, df in SNOWAMP_data_dict.items(): 
             # Remove all values with hourly rain above 40mm
            if 'Precipitation TB(mm)' in df.columns:
                df.loc[df['Precipitation TB(mm)'] > 40, 'Precipitation TB(mm)'] = np.nan
            if 'Precipitation(mm)' in df.columns:
                df.loc[df['Precipitation(mm)'] > 40, 'Precipitation(mm)'] = np.nan
            
            if plotting == True:
            
                if station == 'snowAMP Ganja La':
                    plt.figure(figsize=(12, 4))
                    if 'Precipitation(mm)' in df.columns:
                        plt.plot(df['DATETIME'], df['Precipitation(mm)'], label='Precipitation(mm)')
                    
                    plt.xlabel('Date')
                    plt.ylabel('Precipitation (mm)')
                    plt.title('Precipitation Timeseries - Ganja La')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()
            
            # Drop unnecessary columns
            df = df.loc[:, ~df.columns.duplicated()]


            # Select only columns that exist in the DataFrame
            cols_to_use = ['DATETIME', 'TEMP']
            if 'Precipitation TB(mm)' in df.columns:
                cols_to_use.append('Precipitation TB(mm)')
            if 'Precipitation(mm)' in df.columns:
                cols_to_use.append('Precipitation(mm)')
            df = df[cols_to_use]
            df['DATETIME'] = pd.to_datetime(df['DATETIME'])

            # Resample to hourly data
            df['TEMP'] = pd.to_numeric(df['TEMP'], errors='coerce')
            df = df.resample('h', on='DATETIME').mean()
            df = df.reset_index()
            # Add to the dictionary
            SNOWAMP_data_dict[station] = df
    if plotting == True:

        fig, axs = plt.subplots(len(SNOWAMP_data_dict), 1, figsize=(12, 3 * len(SNOWAMP_data_dict)), sharex=True)
        if len(SNOWAMP_data_dict) == 1:
            axs = [axs]
        for ax, (station, df_plot) in zip(axs, SNOWAMP_data_dict.items()):
            ax.plot(df_plot['DATETIME'], df_plot['TEMP'], label=station)
            ax.set_ylabel('Temperature (°C)')
            ax.set_title(f'Hourly Temperature at {station}')
            ax.legend()
            ax.grid(True)
        axs[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        # Plot Precipitation TB(mm) if available for each station
        for station, df_plot in SNOWAMP_data_dict.items():
            if 'Precipitation TB(mm)' in df_plot.columns or 'Precipitation(mm)' in df_plot.columns:
                if station == 'snowAMP Ganja La':
                        plt.plot(df_plot['DATETIME'], df_plot['Precipitation TB(mm)'], label=f'Precipitation TB(mm) - {station}')
                    # Plot cumulative sum of Precipitation(mm)

                        plt.plot(df_plot['DATETIME'], df_plot['Precipitation(mm)'].cumsum(), label=f'Cumulative Precipitation (mm) - {station}')

                else:
                    plt.plot(df_plot['DATETIME'], df_plot['Precipitation TB(mm)'], label=f'Precipitation TB(mm) - {station}')
        plt.xlabel('Date')
        plt.ylabel('Precipitation TB (mm)')
        plt.title('Precipitation TB(mm) Timeseries')
        plt.legend()
        plt.grid(True)

        plt.show()

        plt.figure(figsize=(12, 6))
        for station, df_plot in SNOWAMP_data_dict.items():
            if 'Precipitation TB(mm)' in df_plot.columns:
                tb_diff = df_plot['Precipitation TB(mm)'].diff()
                tb_diff = tb_diff.where((tb_diff >= 0) & (tb_diff <= 40), np.nan)
                plt.plot(df_plot['DATETIME'], tb_diff, label=f'TB diff - {station}')

            if 'Precipitation(mm)' in df_plot.columns:
                mm_diff = df_plot['Precipitation(mm)'].diff()
                tb_diff = mm_diff.where((mm_diff >= 0) & (mm_diff <= 40), np.nan)
                plt.plot(df_plot['DATETIME'], tb_diff, label=f'pluvio diff - {station}')
        plt.xlabel('Date')
        plt.ylabel('TB Precipitation Difference (mm)')
        plt.title('Difference Between Timesteps for TB Precipitation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        

    # Calculate mm_diff and filter as requested
    for station, df_plot in SNOWAMP_data_dict.items():
        if 'Precipitation(mm)' in df_plot.columns:
            mm_diff = df_plot['Precipitation(mm)'].diff()
            tb_diff = mm_diff.where((mm_diff >= 0) & (mm_diff <= 40), np.nan)
            SNOWAMP_data_dict[station]['Hourly_Rain'] = tb_diff

    for station, df_plot in SNOWAMP_data_dict.items():
        if 'Precipitation TB(mm)' in df_plot.columns:
            tb_diff = df_plot['Precipitation TB(mm)'].diff()
            tb_diff = tb_diff.where((tb_diff >= 0) & (tb_diff <= 40), np.nan)
            SNOWAMP_data_dict[station]['Hourly_Rain'] = tb_diff  




    # Save as CSV in the same way as the TB example if update_csv is True
    if update_csv == True:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Cleaned', 'TB')
        for station, df in SNOWAMP_data_dict.items():
            if station == 'snowAMP Ganja La':
                continue  # Skip saving for 'snowAMP Ganja La'
            df_to_save = df.copy()
            # Ensure DATETIME is the index for saving
            if 'DATETIME' in df_to_save.columns:
                df_to_save = df_to_save.set_index('DATETIME')  
            print(df_to_save['TEMP'])
            # Save temperature and rainfall data for lower and upper stations
            if station in ['SNOWAMP_lower', 'snowAMP Ganjala upper']:
                cols_to_save = []
                if 'TEMP' in df_to_save.columns:
                    cols_to_save.append('TEMP')
                if 'Hourly_Rain' in df_to_save.columns:
                    cols_to_save.append('Hourly_Rain')
                if cols_to_save:
                    out_df = df_to_save[cols_to_save].copy()
                    # Rename columns as requested
                    rename_dict = {}
                    if 'TEMP' in out_df.columns:
                        rename_dict['TEMP'] = 'Temperature_1H'
                    if 'Hourly_Rain' in out_df.columns:
                        rename_dict['Hourly_Rain'] = 'Rainfall_1H'
                    out_df = out_df.rename(columns=rename_dict)
                    out_df = out_df.fillna('NAN')
                    file_name = f"04242025_{station}_TB.csv"
                    file_path = os.path.join(output_dir, file_name)
                    out_df.to_csv(
                        file_path,
                        index_label='DATETIME',
                        header=True
                    )

            # Save only temperature for middle station
            if station == 'SNOWAMP_middle':
                if 'TEMP' in df_to_save.columns:
                    out_df = df_to_save[['TEMP']].copy()
                    out_df = out_df.rename(columns={'TEMP': 'Temperature_1H'})
                    out_df = out_df.fillna('NAN')
                    file_name = f"04242025_{station}_TB.csv"
                    file_path = os.path.join(output_dir, file_name)
                    out_df.to_csv(
                        file_path,
                        index_label='DATETIME',
                        header=True
                    )

    return SNOWAMP_data_dict

