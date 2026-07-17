import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data_overview_plot import read_rainfall_TB, read_temp_TB, get_dir, get_measurement
from Data_overview_plot import get_elevation
from Corrections_pluvio import get_Pluvio_temp
from AWS import get_aws_df_temp
from Snowamp import get_SNOWAMP_df_temp
import os
import pandas as pd

# This script is used to process and analyze data from tipping bucket rain gauges and temperature sensors.
# It includes functions to read data, clean it, and plot the results. The script also includes a function to calculate the temperature gradient based on elevation.
# The data is resampled to specified (dt) intervals, and various corrections are applied to the data based on specific conditions.
# The script also includes functions to merge datasets, calculate rainfall events, and rank rainfall events based on their intensity and duration.
# Snow is removed from the data based on temperature thresholds.
# Cleaned data is saved to CSV files in ../data/Cleaned/TB
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



def get_TB_rain(dt='0.25h'):
    station_names = ['Ganja La TB1', 'Ganja La TB2', 'Ganja La TB3', 'Morimoto TB', 'Langshisha BC TB','Shalbachum TB'
                    ,'Shalbachum TB Temp','Morimoto TB Temp','Ganja La TB3 Temp', 'Ganja La TB2 Temp', 'Ganja La TB1 Temp', 'Langshisha BC TB Temp']

    all_merged_dfs = {}
    dirs_TB= get_dir(station_names)
    tipping_volume = 0.2

    all_temp_dfs = get_TB_df_temp(dt)



    # Loop again to process data and plot
    for file_path, TB_name in zip(dirs_TB, station_names):  # TB_name is unchanged here
        df_TB = read_rainfall_TB(file_path)

        # Convert datetime columns
        df_TB['DATETIME'] = pd.to_datetime(df_TB['DATETIME'])

        # Resample and calculate 15-minute rainfall
        df_TB = df_TB.resample(dt, on='DATETIME').size() * tipping_volume      
        df_TB = df_TB.reset_index(name='Hourly_Rain')

    
        # Only append ' Temp' if TB_name does not already end with 'Temp'
        temp_key = TB_name if TB_name.endswith('Temp') else TB_name + ' Temp'
        df_TB_T = all_temp_dfs[temp_key]
        df_TB_T = df_TB_T.resample(dt, on='DATETIME').mean().reset_index()

        # Rename columns for clarity before merging
        df_TB = df_TB.set_index('DATETIME')
        df_TB_T = df_TB_T.set_index('DATETIME')

        # Ensure both indices are datetime
        df_TB.index = pd.to_datetime(df_TB.index)
        df_TB_T.index = pd.to_datetime(df_TB_T.index)
      
       

        merged_df = pd.DataFrame(df_TB).merge(df_TB_T[['TEMP']], left_index=True, right_index=True)
    
        merged_df = merged_df.sort_index()
        merged_df.index = pd.to_datetime(merged_df.index)      
        merged_df.reset_index(inplace=True)

        # Add the nan values to the dataset
        merged_df.rename(columns={'index': 'DATETIME'}, inplace=True)
        all_merged_dfs[TB_name] = merged_df      

    # Ensure that the 1H periods are aligned across all stations
    common_H = pd.date_range(start='2012-01-01', end='2026-01-01', freq=dt)

    # Reindex each station's DataFrame to the common 1H periods, filling missing periods with NaN
    for station in all_merged_dfs.keys():
        all_merged_dfs[station] = all_merged_dfs[station].set_index('DATETIME').reindex(common_H).reset_index().rename(columns={'index': 'DATETIME'})

    

    
    return all_merged_dfs




def merge_datasets_hourly(dt='0.25h'):    

    all_merged_dfs = get_TB_df_temp(dt)
    
    #station_name_pluvio = ['Yala Pluvio']

    # Get pluvio data
    df_pluvio = get_Pluvio_temp()
    all_merged_dfs.update(df_pluvio)

    # Get AWS data
    df_aws=get_aws_df_temp()
    all_merged_dfs.update(df_aws)

    include_snowamp = True
    if include_snowamp ==True:
        df_SNOWAMP = get_SNOWAMP_df_temp()
        all_merged_dfs.update(df_SNOWAMP)


     # Make sure average temperature for every hours
    for station, df in all_merged_dfs.items():
        if df.index.name == 'DATETIME':             #Still need to fix DATETIME column and set it as index in all dataframes
            df = df.reset_index()
        df[dt] = df['DATETIME'].dt.floor(dt)
        avg_temp_h = df.groupby(dt)['TEMP'].mean().reset_index()
        avg_temp_h = avg_temp_h.rename(columns={'TEMP': 'Avg_Temp_H'})
        df = pd.merge(df, avg_temp_h, on=dt, how='left')
        all_merged_dfs[station] = df
    
    # Ensure that the 1H periods are aligned across all stations
    common_H = pd.date_range(start='2012-01-01', end='2026-12-01', freq=dt)

    # Reindex each station's DataFrame to the common 1H periods, filling missing periods with NaN
    for station in all_merged_dfs.keys():
        all_merged_dfs[station] = all_merged_dfs[station].drop_duplicates(subset=dt).set_index(dt).reindex(common_H).reset_index()



    #all_merged_dfs = all_merged_dfs.reindex(common_H)
    return all_merged_dfs


def TippingBucket_prec(update_csv=False, dt='0.25h'):

    all_merged_dfs = merge_datasets_hourly()

    station_ALL=['Kyangjin TB', 'Langtang TB', 'Lama TB', 'Syabru TB', 'Numthang TB', 'Lirung Camp TB', 'Jathang TB','Ganja La TB1', 'Ganja La TB2', 'Ganja La TB3', 'Morimoto TB', 'Shalbachum TB','Langshisha BC TB','Langshisha Pluvio TB']
    station_no_temp = ['Kyangjin TB', 'Langtang TB', 'Lama TB', 'Syabru TB', 'Numthang TB', 'Lirung Camp TB', 'Jathang TB', 'Langshisha Pluvio TB']
    elevation1 = get_elevation(station_no_temp)
    dirs= get_dir(station_no_temp)
    measurement = get_measurement(station_no_temp)
    tipping_volume = 0.2

    dirs_TB = [dirs[i] for i in range(len(dirs)) if measurement[i] == 'Tipping bucket' and pd.notna(dirs[i])]
    TB_name = [station_no_temp[i] for i in range(len(station_no_temp)) if measurement[i] == 'Tipping bucket' and pd.notna(station_no_temp[i])]
    tipping_buckets = {}

    def calculate_gradient(row):
        temps = []
        elevs = []
        for station in all_merged_dfs.keys():
            # Tipping bucket temperature sensors are not used for the lapse rate
            if 'TB' in station:
                continue
            temp = row[station]
            elev = row[f'{station}_Elevation']
            if not pd.isna(temp) and not pd.isna(elev):
                temps.append(temp)
                elevs.append(elev)
        if len(temps) > 6:
            gradient = np.polyfit(elevs, temps, 1)[0]
        else:
            gradient = np.nan
        return gradient
        

    # Create a DataFrame from all_merged_dfs with temperature and elevation data
    temp_elev_df = pd.DataFrame()
    for station, df in all_merged_dfs.items():


        if 'snowAMP' in station:


            
            if 'Temperature(deg C)' in df.columns:
                df = df.rename(columns={'Temperature(deg C)': 'TEMP'})


            if 'Temperature(degC)' in df.columns:
                df = df.rename(columns={'Temperature(degC)': 'TEMP'})



        temp_elev_df['DATETIME'] = df.index
        temp_elev_df[station] = df['TEMP']
        temp_elev_df[f'{station}_Elevation'] = get_elevation([station])[0]

    # Calculate the temperature gradient for each row in the DataFrame
    temp_elev_df['Gradient'] = temp_elev_df.apply(lambda row: calculate_gradient(row), axis=1)

    # Calculate the temperature time series at the elevations from elevation1 using Kyangjin AWS temperatures
    kyangjin_aws_df = all_merged_dfs['Kyangjin AWS']
    # Plot the temperature data of Kyangjin AWS

    # Create a DataFrame to store the temperature time series at different elevations
    temp_at_elevations_df = pd.DataFrame()
    temp_at_elevations_df['DATETIME'] = kyangjin_aws_df['index']

    # Calculate temperature at each elevation using the calculated gradient
    for station, elevation in zip(station_no_temp, elevation1):
        temp_at_elevations_df[station] = kyangjin_aws_df['Avg_Temp_H'] + temp_elev_df['Gradient'].values * (elevation - get_elevation(['Kyangjin AWS'])[0])

    # Fill NaN values with temperature estimates from the same date and hour of day but from other available years
    for station in station_no_temp:
        temp_at_elevations_df[station] = temp_at_elevations_df[station].fillna(
            temp_at_elevations_df.groupby(
                [temp_at_elevations_df['DATETIME'].dt.dayofyear,
                 temp_at_elevations_df['DATETIME'].dt.hour]
            )[station].transform('mean')
        )

    TB_all_temp = True

    if TB_all_temp == True:


        # Add these stations to the merged_df dataset
        for idx, (file_path, TB_name) in enumerate(zip(dirs_TB, TB_name)):

            # Define file paths for specific stations
      

           
            df_TB = read_rainfall_TB(file_path)
            df_TB['DATETIME'] = pd.to_datetime(df_TB['DATETIME'])
            # Ensure the timeframe for the rainfall data is between '2012-01-01' and '2024-12-01'
            df_TB = df_TB.resample(dt, on='DATETIME').size() * tipping_volume
            df_TB = df_TB.reset_index(name='Hourly_Rain')
            
            common_H = pd.date_range(start='2012-01-01', end='2026-01-01', freq=dt)
            df_TB = df_TB.set_index('DATETIME').reindex(common_H).reset_index().rename(columns={'index': 'DATETIME'})
            all_merged_dfs[TB_name] = df_TB

        tipping_buckets = get_TB_rain(dt)


        for station in station_no_temp:
            if station in temp_at_elevations_df.columns:
                temp_df = temp_at_elevations_df[['DATETIME', station]].rename(columns={station: 'TEMP'}).set_index('DATETIME')
                rain_df = all_merged_dfs[station][['DATETIME', 'Hourly_Rain']].set_index('DATETIME').reset_index()     


                
                
                merged_df = pd.merge_asof(rain_df.sort_values(by='DATETIME'), 
                                        temp_df.sort_values(by='DATETIME'), 
                                        on='DATETIME', 
                                        direction='nearest')
                
                
                merged_df = merged_df.set_index('DATETIME')


                

                # Add the merged_df to the tipping_buckets dictionary
                tipping_buckets[station] = merged_df

    for station in station_ALL:
        df = tipping_buckets.get(station)
        df.loc[(df['TEMP'] < 1), 'Hourly_Rain'] = np.nan
        if dt == 'h' or dt == 'H':
            df.loc[df['Hourly_Rain'] > 40, 'Hourly_Rain'] = np.nan
            # df.loc[(df['Hourly_Rain'] > 0) & (df['Hourly_Rain'] < 0.2), 'Hourly_Rain'] = np.nan
        if dt == '0.25h':
            df.loc[df['Hourly_Rain'] > 10, 'Hourly_Rain'] = np.nan

        # Remove outliers #Lama should not be used for any analysis
        df = tipping_buckets.get(station)
        if df is None:
            print(f"Warning: Station '{station}' not found in tipping_buckets.")
            continue
        if 'DATETIME' in df.columns:
            df['DATETIME'] = pd.to_datetime(df['DATETIME'])
            df = df.set_index('DATETIME')
        else:
            df.index = pd.to_datetime(df.index)     

        # Track if any removal was done
        removal_done = False

        if station == 'Lama TB':
            before = df.loc[df.index > '2014-08-02', 'Hourly_Rain'].notna().sum()
            df.loc[df.index > '2014-08-02', 'Hourly_Rain'] = np.nan
            after = df.loc[df.index > '2014-08-02', 'Hourly_Rain'].notna().sum()
            if before == after:
                print(f"Removal not done for {station} (2014-08-02)")
            else:
                removal_done = True

        if station == 'Langshisha Pluvio TB':
            mask = (
                (df.index > '2014-10-16') &
                (df.index < '2016-03-28')
            )
            before = df.loc[mask, 'Hourly_Rain'].notna().sum()
            df.loc[mask, 'Hourly_Rain'] = np.nan      
            after = df.loc[mask, 'Hourly_Rain'].notna().sum()
            if before == after:
                print(f"Removal not done for {station} (2014-10-16 to 2016-03-28)")
            else:
                removal_done = True

        if station == 'Langtang TB':
            mask = (
                (df.index > '2013-05-02') &
                (df.index < '2013-10-21')
            )
            before = df.loc[mask, 'Hourly_Rain'].notna().sum()
            df.loc[mask, 'Hourly_Rain'] = np.nan  
            after = df.loc[mask, 'Hourly_Rain'].notna().sum()
            if before == after:
                print(f"Removal not done for {station} (2013-05-02 to 2013-10-21)")
            else:
                removal_done = True

        if station == 'Kyangjin TB':
            mask = (
                (df.index > '2013-07-28') &
                (df.index < '2014-02-17')
            )
            before = df.loc[mask, 'Hourly_Rain'].notna().sum()
            df.loc[mask, 'Hourly_Rain'] = np.nan  
            after = df.loc[mask, 'Hourly_Rain'].notna().sum()
            if before == after:
                print(f"Removal not done for {station} (2013-07-28 to 2014-02-17)")
            else:
                removal_done = True

        if station == 'Morimoto TB':
            mask = (
                (df.index > '2015-02-28') &
                (df.index < '2015-06-12')
            )
            before = df.loc[mask, 'Hourly_Rain'].notna().sum()
            df.loc[mask, 'Hourly_Rain'] = np.nan  
            after = df.loc[mask, 'Hourly_Rain'].notna().sum()
            if before == after:
                print(f"Removal not done for {station} (2015-02-28 to 2015-06-12)")
            else:
                removal_done = True

            # Same windows as the TEMP removal in get_TB_df_temp (sensor snow-covered)
            mask_t = (df.index >= '2019-02-25') & (df.index <= '2019-03-10')
            df.loc[mask_t, 'Hourly_Rain'] = np.nan
            df.loc[df.index >= '2024-11-02', 'Hourly_Rain'] = np.nan

        # Same windows as the TEMP removal in get_TB_df_temp (sensor snow-covered,
        # so precipitation in these periods is likely snow)
        if station == 'Ganja La TB1':
            mask1 = (df.index >= '2015-02-25') & (df.index <= '2015-06-07')
            df.loc[mask1, 'Hourly_Rain'] = np.nan
            mask2 = (df.index >= '2019-02-06') & (df.index <= '2019-04-03')
            df.loc[mask2, 'Hourly_Rain'] = np.nan

        if station == 'Ganja La TB2':
            mask1 = (
                (df.index > '2021-06-19') &
                (df.index < '2021-07-17')
            )
            before1 = df.loc[mask1, 'Hourly_Rain'].notna().sum()
            df.loc[mask1, 'Hourly_Rain'] = np.nan  
            after1 = df.loc[mask1, 'Hourly_Rain'].notna().sum()
            if before1 == after1:
                print(f"Removal not done for {station} (2021-06-19 to 2021-07-17)")
            else:
                removal_done = True

            mask2 = (
                (df.index >= '2019-04-18') &
                (df.index <= '2019-10-09')
            )
            before2 = df.loc[mask2, 'Hourly_Rain'].notna().sum()
            df.loc[mask2, 'Hourly_Rain'] = np.nan        
            after2 = df.loc[mask2, 'Hourly_Rain'].notna().sum()
            if before2 == after2:
                print(f"Removal not done for {station} (2019-04-18 to 2019-10-09)")
            else:
                removal_done = True

            mask3 = (
                (df.index >= '2016-08-27') &
                (df.index <= '2016-10-08')
            )
            before3 = df.loc[mask3, 'Hourly_Rain'].notna().sum()
            df.loc[mask3, 'Hourly_Rain'] = np.nan        
            after3 = df.loc[mask3, 'Hourly_Rain'].notna().sum()
            if before3 == after3:
                print(f"Removal not done for {station} (2016-08-27 to 2016-10-08)")
            else:
                removal_done = True

        if station == 'Ganja La TB3':
            mask1 = (
                (df.index > '2017-07-04') &
                (df.index < '2017-10-29')
            )
            before1 = df.loc[mask1, 'Hourly_Rain'].notna().sum()
            df.loc[mask1, 'Hourly_Rain'] = np.nan  
            after1 = df.loc[mask1, 'Hourly_Rain'].notna().sum()
            if before1 == after1:
                print(f"Removal not done for {station} (2017-07-04 to 2017-10-29)")
            else:
                removal_done = True

            mask2 = (
                (df.index >= '2019-04-18') &
                (df.index <= '2019-10-09')
            )
            before2 = df.loc[mask2, 'Hourly_Rain'].notna().sum()
            df.loc[mask2, 'Hourly_Rain'] = np.nan
            after2 = df.loc[mask2, 'Hourly_Rain'].notna().sum()
            if before2 == after2:
                print(f"Removal not done for {station} (2019-04-18 to 2019-10-09)")
            else:
                removal_done = True

            # Same windows as the TEMP removal in get_TB_df_temp (sensor snow-covered)
            mask3 = (df.index >= '2014-12-31') & (df.index <= '2015-05-19')
            df.loc[mask3, 'Hourly_Rain'] = np.nan
            mask4 = (df.index >= '2019-02-06') & (df.index <= '2019-04-28')
            df.loc[mask4, 'Hourly_Rain'] = np.nan
            df.loc[df.index >= '2021-11-10', 'Hourly_Rain'] = np.nan

        # Lirung should not be used for any analysis
        if station == 'Lirung Camp TB':
            before = df.loc[df.index < '2013-05-10', 'Hourly_Rain'].notna().sum()
            df.loc[df.index < '2013-05-10', 'Hourly_Rain'] = np.nan
            after = df.loc[df.index < '2013-05-10', 'Hourly_Rain'].notna().sum()
            if before == after:
                print(f"Removal not done for {station} (before 2013-05-10)")
            else:
                removal_done = True

        if station == 'Numthang TB':
            before = df.loc[df.index > '2018-05-20', 'Hourly_Rain'].notna().sum()
            df.loc[df.index > '2018-05-20', 'Hourly_Rain'] = np.nan
            after = df.loc[df.index > '2018-05-20', 'Hourly_Rain'].notna().sum()
            if before == after:
                print(f"Removal not done for {station} (after 2018-05-20)")
            else:
                removal_done = True
            # Exclude 2017-07-04 till 2017-10-28
            mask = (
                (df.index >= '2017-07-04') &
                (df.index <= '2017-10-28')
            )
            before2 = df.loc[mask, 'Hourly_Rain'].notna().sum()
            df.loc[mask, 'Hourly_Rain'] = np.nan
            after2 = df.loc[mask, 'Hourly_Rain'].notna().sum()
            if before2 == after2:
                print(f"Removal not done for {station} (2017-07-04 to 2017-10-28)")
            else:
                removal_done = True

        # Remove 06-06-2020 from 6 to 7 o'clock
        if station == 'Langshisha BC TB': 
            mask1 = (
                    (df.index >= pd.to_datetime('2020-06-06 06:00')) &
                    (df.index < pd.to_datetime('2020-06-06 07:00'))
            )
            before1 = df.loc[mask1, 'Hourly_Rain'].notna().sum()
            df.loc[mask1, 'Hourly_Rain'] = np.nan
            after1 = df.loc[mask1, 'Hourly_Rain'].notna().sum()
            if before1 == after1:
                print(f"Removal not done for {station} (2020-06-06 06:00 to 07:00)")
            else:
                removal_done = True

            mask2 = (
                    (df.index >= pd.to_datetime('2014-10-18')) &
                    (df.index < pd.to_datetime('2016-04-14'))
            )
            before2 = df.loc[mask2, 'Hourly_Rain'].notna().sum()
            df.loc[mask2, 'Hourly_Rain'] = np.nan
            after2 = df.loc[mask2, 'Hourly_Rain'].notna().sum()
            if before2 == after2:
                print(f"Removal not done for {station} (2014-10-18 to 2016-04-14)")
            else:
                removal_done = True

            # Tipping bucket seems to be clogged
            mask3 = (
                    (df.index >= pd.to_datetime('2019-06-23')) &
                    (df.index < pd.to_datetime('2019-10-07'))
            )
            before3 = df.loc[mask3, 'Hourly_Rain'].notna().sum()
            df.loc[mask3, 'Hourly_Rain'] = np.nan
            after3 = df.loc[mask3, 'Hourly_Rain'].notna().sum()
            if before3 == after3:
                print(f"Removal not done for {station} (2019-06-23 to 2019-10-07)")
            else:
                removal_done = True

            # Tipping bucket seems to be clogged
            mask4 = (
                (df.index >= '2021-06-11') &
                (df.index < '2021-10-24')
            )
            before4 = df.loc[mask4, 'Hourly_Rain'].notna().sum()
            df.loc[mask4, 'Hourly_Rain'] = np.nan
            after4 = df.loc[mask4, 'Hourly_Rain'].notna().sum()
            if before4 == after4:
                print(f"Removal not done for {station} (2021-06-11 to 2021-10-24)")
            else:
                removal_done = True

        # I removed the data for Shalbachum from 2018 onward. When looking at the cumulative plot
        # A clear change in slope is visible from 2018 onward. To me this indicates that the station is not working
        # properly anymore
        if station == 'Shalbachum TB': 
            mask1 = (
                (df.index >= '2018-09-21') &
                (df.index < '2024-11-30')
            )
            before1 = df.loc[mask1, 'Hourly_Rain'].notna().sum()
            df.loc[mask1, 'Hourly_Rain'] = np.nan
            after1 = df.loc[mask1, 'Hourly_Rain'].notna().sum()
            if before1 == after1:
                print(f"Removal not done for {station} (2020-06-06 06:00 to 07:00)")
            else:
                removal_done = True

            # Same windows as the TEMP removal in get_TB_df_temp (sensor snow-covered)
            df.loc[df.index < '2013-10-26 17:00', 'Hourly_Rain'] = np.nan
            mask_t = (df.index >= '2015-02-25') & (df.index <= '2015-04-01')
            df.loc[mask_t, 'Hourly_Rain'] = np.nan
            df.loc[df.index >= '2023-11-03', 'Hourly_Rain'] = np.nan




        tipping_buckets[station] = df



    plot = True
    if plot == True:
        

        # Define the number of rows and columns for subplots
        n_cols = 2
        n_rows = (len(station_ALL) + n_cols - 1) // n_cols  # Calculate the number of rows needed

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True)

        # Flatten the axs array for easy iteration
        axs = axs.flatten()

        # Loop through each station and plot the data
        for idx, station in enumerate(station_ALL):
            df = tipping_buckets[station]

            if 'DATETIME' in tipping_buckets[station].columns:
                df.index= pd.to_datetime(df['DATETIME'])      
            # Resample the data to hourly data

            # Plotting on the corresponding subplot
            ax = axs[idx]  # Select the subplot for this station

            # Plot temperature on the primary y-axis
            ax.set_ylabel('Rainfall (m)')
            ax.plot(df.index, df['Hourly_Rain'], markersize=0.2, linestyle='-', linewidth=0.5, label= f'{station}')
            ax.legend(loc='upper right')
        

        # Tight layout to avoid overlap
        fig.tight_layout()



        # Create a new figure for the cumulative plot
        plt.figure(figsize=(15, 7))

        # Loop through each station to plot cumulative rainfall
        for station in station_ALL:
            df = tipping_buckets.get(station)
            if df is None:
                continue  # Skip if station data is not available

            # Ensure the index is datetime
            if 'DATETIME' in df.columns:
                df = df.set_index('DATETIME')
            
            # Calculate cumulative rainfall, filling NaNs with 0 so they don't stop the sum
            cumulative_rain = df['Hourly_Rain'].fillna(0).cumsum()
            
            # Plot the cumulative rainfall
            plt.plot(cumulative_rain.index, cumulative_rain.values, label=station, linewidth=0.8)

        # Add plot details
        plt.title('Cumulative Rainfall for All Stations')
        plt.xlabel('Datetime')
        plt.ylabel('Cumulative Rainfall (mm)')
        plt.legend(loc='upper left', ncol=2)
        plt.grid(True)
        plt.tight_layout()

        plt.show()





   
    if update_csv == True:
        # Save each tipping bucket data to a CSV file
        # Anchor to the script location so the output lands in Zenodo\data\Cleaned\TB
        # regardless of the working directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Cleaned', 'TB')
        os.makedirs(output_dir, exist_ok=True)
        for station, df in tipping_buckets.items():
            # Rain and temperature are already merged in every entry; the ' Temp'
            # entries are uncleaned duplicates, so only the plain station is saved
            if station.endswith(' Temp'):
                continue
            # Prepare the file name
            if 'index' in df.columns:
                df.rename(columns={'index': 'DATETIME'}, inplace=True)
            if 'DATETIME' in df.columns:
                df = df.set_index('DATETIME')

            # Extract the station name from the file name
            station_name = station

            # Read the TEST2.txt file and extract the 'Data gap' column
            test_file_path = r"../data/Data_overview/TEST2.txt"
            with open(test_file_path, 'r') as file:
                lines = file.readlines()
                data_gap_line = next((line for line in lines if station_name in line and 'Data gap' in line), None)

                # Extract the data gap periods
                if data_gap_line is not None:
                    data_gap_periods = data_gap_line.split('Data gap:')[1].strip()
                    if data_gap_periods != 'N/A':  # Check if data gap is not N/A
                        df_gap = pd.DataFrame({'Data gap': [data_gap_periods]})
                    else:
                        df_gap = pd.DataFrame({'Data gap': []})  # Handle the case where data gap is N/A
                else:
                    df_gap = pd.DataFrame({'Data gap': []})  # Handle the case where no data gap line is found      
                for _, row in df_gap.iterrows():
                    if pd.notna(row['Data gap']):
                        timeframes = row['Data gap'].split(':')
                        for timeframe in timeframes:
                            dates = timeframe.split(' ')
                            if len(dates) == 2:
                                start_date, end_date = dates
                                start_date = pd.to_datetime(start_date)
                                end_date = pd.to_datetime(end_date)
                                df.loc[(df.index >= start_date) & (df.index <= end_date), 'Hourly_Rain'] = np.nan      
            file_name = f"04242025_{station}_TB.csv"
            file_path = os.path.join(output_dir, file_name)

            # Save the data to CSV
            df = df.fillna('NAN')
            df['Hourly_Rain'] = pd.to_numeric(df['Hourly_Rain'], errors='coerce')
            df['TEMP'] = pd.to_numeric(df['TEMP'], errors='coerce')
            df[['Hourly_Rain', 'TEMP']].fillna(np.nan).to_csv(file_path, index_label='DATETIME', header=['Rainfall_15min', 'Temperature_15min'])
        

    return tipping_buckets
TippingBucket_prec(update_csv=True, dt='0.25h')