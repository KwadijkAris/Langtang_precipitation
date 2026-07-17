from Data_overview_plot import get_dir
from Data_overview_plot import get_elevation
from Data_overview_plot import read_pluvio_cleaned
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data_overview_plot import get_dir
import os
from pathlib import Path
import sys

_SCRIPT_DIR = Path(os.path.abspath(__file__)).parent   # always absolute
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
    _DATA_DIR = _local if _local_ok else Path(os.path.abspath(str(_SCRIPT_DIR / ".." / ".." / "Data")))
import calendar


#%%
# Define seasons
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



def read_AWS(file_path):
    df = pd.read_csv(file_path)
    # MicroMet files (e.g. Morimoto MM) use different headers; normalize to AWS names
    df = df.rename(columns={'Wind_Dir': 'WINDDIR', 'Wind_Speed': 'WSPD', 'Rel_Air_Press': 'PRES'})
    try:
        df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%Y-%m-%d %H:%M')
    df = df.drop(columns=['DATE', 'TIME'])
    return df





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




def get_aws_df_rain(Station=None, snow=False, apply_max_threshold=True,
                    apply_manual_removal=True):

    # AWS data only for Kyangjin and Yala BC as others do not measure rainfall. 
    # AWS data is measured at hourly intervals
    # After corrections performed in this script, this function is also used in Corrections_pluvio.py to merge all the pluvio meters into one dataset
    # This function was chosen to keep seperately in case other variables from the AWS station needed to be analysed. In case you want to merge all the pluvio
    # data from the Langtang valley, use correction_pluvio.py script!


    if Station == None:
        station_names = ['Kyangjin AWS', 'Yala BC AWS']     #These are the only usefull AWS stations for this research. Other stations have been removed from analysis
    else:
        station_names = Station

    station_names = ['Kyangjin AWS', 'Yala BC AWS']
    elevation = get_elevation(station_names)
    aws_data_dict = {}
    for i in range(len(station_names)):
        file_path = get_dir(station_names[i])
        df = read_AWS(file_path[0])
        df['Elevation'] = elevation[i]
        # Extract and process PVOL data
        df = df.replace('NA', np.nan)
        df['PVOL'] = pd.to_numeric(df['PVOL'], errors='coerce')
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])  # Ensure DATETIME is in datetime format

       
        
        aws_data_dict[station_names[i]] = df

    for station, df in aws_data_dict.items():
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
   

        # Pluvio was maintained during the following periods, so we set the values to NaN
        if station == 'Yala BC AWS':
            mask = (pd.Series(df.index).dt.date.values == pd.Timestamp('2023-10-13').date())
            mask |= (df.index >= '2022-10-07') & (df.index <= '2022-11-03')
            mask |= (df.index >= '2019-09-27') & (df.index <= '2020-03-27')
            mask |= (df.index >= '2018-09-13') & (df.index <= '2018-10-19')
            mask |= (df.index >= '2018-04-26 12:00') & (df.index <= '2018-04-26 16:00')
            mask |= (df.index >= '2014-09-30') & (df.index <= '2014-10-20') # I removed Hudhud because these values seem irrealistic, also comparing them to langshisha, shows that this is likely too high
            mask |= (df.index >= '2018-09-13') & (df.index <= '2018-10-09')
            mask |= (df.index >= '2019-09-25') & (df.index <= '2019-10-24')
            mask |= (df.index >= '2022-10-08') & (df.index <= '2022-11-15')
            mask |= (df.index >= '2020-09-26') & (df.index <= '2020-10-03')
            mask |= (df.index >= '2014-05-06') & (df.index <= '2014-05-07')
            mask |= (df.index >= '2016-09-19') & (df.index <= '2016-10-13')
            mask |= (df.index >= '2016-05-10') & (df.index <= '2017-04-01')
            if apply_manual_removal:
                df.loc[mask, ['BCON', 'PVOL']] = np.nan

        if station == 'Kyangjin AWS':
            mask = (df.index == '2022-11-18') | (df.index == '2022-11-19')
            mask |= (df.index >= '2016-10-18') & (df.index <= '2017-04-20')
            mask |= (df.index >= '2019-09-27') & (df.index <= '2020-03-27')
            mask |= (df.index >= '2016-03-01') & (df.index <= '2016-11-01')

            if apply_manual_removal:
                df.loc[mask, ['BCON', 'PVOL']] = np.nan

        # # Plot BCON for each station
        # plt.figure(figsize=(12, 4))
        # plt.plot(df.index, df['BCON'], label=f'{station} BCON', color='orange')
        # plt.xlabel('Date')
        # plt.ylabel('BCON')
        # plt.title(f'BCON Time Series for {station}')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()



        ## Correction 2) Set negative values to NaN
        df.loc[df['BCON'] < 0, 'BCON'] = np.nan
        df.loc[df['PVOL'] < 0, 'PVOL'] = np.nan

        # Remove BCON values above 850 (this is actually 750, the maximum range of the OTT Pluvio2)
        mask_bcon = df['BCON'] > 850
        df.loc[mask_bcon, 'BCON'] = np.nan
        df.loc[mask_bcon, 'PVOL'] = np.nan

        df['Bucket_diff'] = df['BCON'].diff().where(df['BCON'].notna()).apply(lambda x: x if x >= 0.2 else 0) # Set Bucket_diff to NaN where BCON is NaN
        df.loc[df['BCON'].isna(), 'Bucket_diff'] = np.nan

        # Set PVOL values less than 0.2 to zero
        # df.loc[(df['PVOL'] < 0.2) & (df['PVOL'].notna()), 'PVOL'] = 0

     

        
        # df['Hourly_Rain'] = df['PVOL']
        # Use Bucket_diff where available, otherwise use PVOL
        df['Hourly_Rain'] = df['Bucket_diff']
        df['Hourly_Rain'] = df['Bucket_diff'].where(df['Bucket_diff'].isna(), df['PVOL']) 
        
        # df.loc[df['Hourly_Rain'].isna(), 'Hourly_Rain'] = df['Bucket_diff']
        # Use Bucket_diff values where Hourly_Rain is between 0.2 and 0.5, Bucket_diff is not NaN, and Bucket_diff is larger than 0.2
        # df.loc[(df['Hourly_Rain'] >= 0.2) & (df['Hourly_Rain'] <= 0.5) & df['Bucket_diff'].notna() & (df['Bucket_diff'] > 0.2), 'Hourly_Rain'] = df['Bucket_diff']


        # Set values above 170 to NaN
        if apply_max_threshold:
            df['Hourly_Rain'] = df['Hourly_Rain'].apply(lambda x: np.nan if x > 170 else x)

        #df['Hourly_Rain'] = pd.to_numeric(df['Hourly_Rain'])

        #Remove false rain events based on moisture content
        #df.loc[(df['RH'] < 60) & (df['Hourly_Rain'] > 0), 'Hourly_Rain'] = 0

        # Plot RH (Relative Humidity) as a time series



        # Merge with temperature data for later use
        temp_df = get_aws_df_temp()[station]
        temp_df.set_index('DATETIME', inplace=True)
        # Drop any columns named 'Temp', 'temp', or 'TEMP' if they exist
        # df = df.drop(columns=[col for col in df.columns if col.lower() == 'temp'], errors='ignore')
        df = df[['Hourly_Rain']]
        # Merge temperature data based on the datetime index
        # Merge temperature data as 'TEMP' column
        df = df.merge(temp_df[['TEMP']], left_index=True, right_index=True, how='left')


        aws_data_dict[station] = df

                    
                
        if snow == True:

            temp_df = get_aws_df_temp()[station]
            df.loc[(df['TEMP'] > -2) & (df['Hourly_Rain'].notna()), 'Hourly_Rain'] = 0


        
    nan_count = df['Hourly_Rain'].isna().sum()
    print(f"Number of NaN values in {station}: {nan_count}")

   

    plotting = False
    if plotting == True:
        # Plotting precipitation time series for each station
        fig, axes = plt.subplots(len(aws_data_dict), 1, figsize=(12, 8), sharex=True)

        for i, (station, df) in enumerate(aws_data_dict.items()):
            # Filter data for the year 2016
            df_2016 = df[(df.index >= '2016-01-01') & (df.index < '2017-01-01')]
            axes[i].plot(df_2016.index, df_2016['Hourly_Rain'], label=f'{station} Hourly Rain (2016)', color='b')
            axes[i].set_ylabel('Rain (mm/h)')
            axes[i].set_title(f'{station} Precipitation Time Series (2016)')
            axes[i].legend(loc='upper right')
            axes[i].grid(True)

        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.show()
        # Plotting cumulative yearly data in separate subplots for each station

        yearly = False
        if yearly == True:
            fig, axes = plt.subplots(len(aws_data_dict), 1, figsize=(12, 8), sharex=True)

            for i, (station, df) in enumerate(aws_data_dict.items()):
                df.index = pd.to_datetime(df.index)  # Ensure index is a DatetimeIndex

                # Fill NaNs using the mean of the same date (month, day, hour) from other years
                dt_index = df.index
                mask_nan = df['Hourly_Rain'].isna()
                if mask_nan.any():
                    # Create a DataFrame with columns for month, day, hour
                    date_parts = pd.DataFrame({
                    'month': dt_index.month,
                    'day': dt_index.day,
                    'hour': dt_index.hour
                    }, index=dt_index)
                    # For each NaN, fill with mean of same month, day, hour from other years
                    for idx in df[mask_nan].index:
                        m, d, h = idx.month, idx.day, idx.hour
                        # Select all other years with same month, day, hour
                        mask_same_time = (
                            (date_parts['month'] == m) &
                            (date_parts['day'] == d) &
                            (date_parts['hour'] == h) &
                            (dt_index.year != idx.year)
                        )
                        mean_val = df.loc[mask_same_time, 'Hourly_Rain'].mean()
                        if not np.isnan(mean_val):
                            df.at[idx, 'Hourly_Rain'] = mean_val

                df['Year'] = df.index.year
                yearly_cumulative = df.groupby('Year')['Hourly_Rain'].sum()
                yearly_valid_counts = df.groupby('Year')['Hourly_Rain'].count()
                yearly_total_counts = df.groupby('Year')['Hourly_Rain'].size()
                valid_years = yearly_valid_counts / yearly_total_counts >= 0.9
                filtered_yearly_cumulative = yearly_cumulative[valid_years]

                axes[i].bar(filtered_yearly_cumulative.index, filtered_yearly_cumulative, label=f'{station} Yearly Cumulative Rain')
                # Calculate and annotate the mean
                mean_val = filtered_yearly_cumulative.mean()
                axes[i].axhline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f} mm')
                # Annotate mean value on the plot
                axes[i].text(
                    filtered_yearly_cumulative.index.min(), mean_val,
                    f'Mean: {mean_val:.1f} mm',
                    color='red', va='bottom', ha='left', fontsize=9, fontweight='bold'
                )
                axes[i].set_ylabel('Cumulative Rain (mm)')
                axes[i].set_title(f'{station} Yearly Cumulative Rainfall')
                axes[i].legend(loc='upper right')
                axes[i].grid(True)

            axes[-1].set_xlabel('Year')
            plt.tight_layout()
            plt.show()

    return aws_data_dict



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
        # # Define the periods as masks for the DataFrame index
        # periods = [
        #     ('2012-03-22', '2014-12-31'),
        #     ('2016-01-01', '2019-05-04'),
        #     ('2019-05-05', '2019-11-15'),
        #     # ('2020-03-27', '2021-11-12'),
        #     ('2021-11-13', '2023-11-06')
        # ]

        # # Create a mask for each period and combine them
        # period_mask = pd.Series(False, index=df_clean.index)
        # for start, end in periods:
        #     period_mask |= (df_clean.index >= pd.to_datetime(start)) & (df_clean.index <= pd.to_datetime(end))

        # Filter the DataFrame to include only the defined periods
        df_clean = df_clean.copy()
        df_clean['Season'] = df_clean.index.map(get_season)
        station_seasonal_data[station] = df_clean


        # # Calculate mean wind direction for each defined period
        # mean_wind_direction_periods = {}
        # for start, end in periods:
        #     period_data = df_clean[(df_clean.index >= pd.to_datetime(start)) & (df_clean.index <= pd.to_datetime(end))]
        #     if not period_data.empty:
        #         # Use circular mean for wind direction
        #         sin_sum = np.sum(np.sin(np.deg2rad(period_data['WINDDIR'].dropna())))
        #         cos_sum = np.sum(np.cos(np.deg2rad(period_data['WINDDIR'].dropna())))
        #         mean_dir_rad = np.arctan2(sin_sum, cos_sum)
        #         mean_dir_deg = (np.rad2deg(mean_dir_rad) + 360) % 360
        #         mean_wind_direction_periods[(start, end)] = mean_dir_deg
        #     else:
        #         mean_wind_direction_periods[(start, end)] = np.nan

        # print("Mean wind direction for each period:")
        # for period, mean_dir in mean_wind_direction_periods.items():
        #     print(f"Period {period[0]} to {period[1]}: {mean_dir:.2f} degrees")


        # # Calculate wind direction offset for each period compared to the reference period
        # reference_period = periods[0]
        # reference_mean_dir = mean_wind_direction_periods[reference_period]

        # # Calculate offsets for other periods
        # offsets = {}
        # for period in periods[1:]:
        #     mean_dir = mean_wind_direction_periods[period]
        #     # Calculate circular offset (difference, wrapped to [-180, 180])
        #     offset = ((mean_dir - reference_mean_dir + 180) % 360) - 180
        #     offsets[period] = offset
        #     print(f"Offset for period {period[0]} to {period[1]}: {offset:.2f} degrees")

        # # Apply offset correction to wind direction for each period except reference
        # df_clean_corrected = df_clean.copy()
        # for period in periods[1:]:
        #     offset = offsets[period]
        #     mask = (df_clean_corrected.index >= pd.to_datetime(period[0])) & (df_clean_corrected.index <= pd.to_datetime(period[1]))
        #     df_clean_corrected.loc[mask, 'WINDDIR'] = (df_clean_corrected.loc[mask, 'WINDDIR'] - offset) % 360

        # # Use df_clean_corrected for further analysis instead of df_clean
        # df_season = df_clean_corrected[period_mask].copy()
        # df_season['Season'] = df_season.index.map(get_season)
        # station_seasonal_data[station] = df_season

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



def get_seasonal_RH_PRES_data():

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


    station_names = ['Kyangjin AWS', 'Yala BC AWS', 'Morimoto MM']
    aws_data_dict = {}

    for station in station_names:

        # Get RH data
        file_path = get_dir(station)
        df = read_AWS(file_path[0])
        df = df.replace('NA', np.nan)
        df['RH'] = pd.to_numeric(df.get('RH', np.nan), errors='coerce')
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df = df.set_index('DATETIME')[['RH']]
        if station == 'Morimoto MM':
            # 15-min MicroMet data: use hourly means rather than on-the-hour samples
            df = df.resample('h').mean()

    # Get wind data
        wind_data_dict = get_aws_df_wind(Station=[station])
        wind_df = wind_data_dict[station]

    # Get temperature data from cleaned pluvio files
    # Assuming read_pluvio_cleaned is available and returns a dict of dataframes
    # from Data_overview_plot import read_pluvio_cleaned
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
        temp_df = temp_df[['TEMP']]
        # Merge RH, wind, and temperature data
        df = df.join(wind_df[['WSPD', 'WINDDIR']], how='outer')
        df = df.join(temp_df[['TEMP']], how='outer') 
        # Get original data again for pressure
        pres_df = read_AWS(file_path[0])
        pres_df = pres_df.replace('NA', np.nan)
        pres_df['PRES'] = pd.to_numeric(pres_df.get('PRES', np.nan), errors='coerce')
        pres_df['DATETIME'] = pd.to_datetime(pres_df['DATETIME'])
        pres_df = pres_df.set_index('DATETIME')[['PRES']]
        if station == 'Morimoto MM':
            pres_df = pres_df.resample('h').mean()

        # Merge pressure data
        df = df.join(pres_df, how='outer')
        # Add pressure data for Kyangjin AWS
        if station == 'Yala BC AWS' and 'PRES' in df.columns:
            df['PRES'] = pd.to_numeric(df['PRES'], errors='coerce')
            df.loc[df.index.date == pd.Timestamp('2015-10-27').date(), 'PRES'] = np.nan
            df.loc[df.index.date == pd.Timestamp('2020-07-18').date(), 'PRES'] = np.nan
            df.loc[df.index.date == pd.Timestamp('2020-07-20').date(), 'PRES'] = np.nan
            df.loc[df.index > '2023-05-10', 'PRES'] = np.nan

        df = df.reset_index()

   
        # Process wind speed (WSPD) and wind direction (WINDDIR)
        df['WSPD'] = pd.to_numeric(df['WSPD'], errors='coerce')

        df['WINDDIR'] = pd.to_numeric(df['WINDDIR'], errors='coerce')
    





        if station == 'Kyangjin AWS':
            df.loc[df['DATETIME'] > pd.Timestamp('2017-10-01'), 'RH'] = np.nan
            df.loc[(df['DATETIME'] >= pd.Timestamp('2013-12-01')) & (df['DATETIME'] < pd.Timestamp('2014-06-01')), 'RH'] = np.nan
        elif station == 'Yala BC AWS':
            df.loc[((df['DATETIME'] >= pd.Timestamp('2015-01-01')) & (df['DATETIME'] < pd.Timestamp('2016-12-01'))) |
            ((df['DATETIME'] >= pd.Timestamp('2020-03-01')) & (df['DATETIME'] < pd.Timestamp('2021-07-01'))) |
            (df['DATETIME'] > pd.Timestamp('2019-11-30')), 'RH'] = np.nan


        # --- Calculate humidity variables ---
        df['DEW_POINT'] = calculate_dew_point(df['TEMP'], df['RH'])
        e_s = saturation_vapor_pressure(df['TEMP'])
        e = (df['RH'] / 100.0) * e_s
        df['MIXING_RATIO'] = mixing_ratio(e, df['PRES']) * 1000 # g/kg
        df['ABSOLUTE_HUMIDITY'] = absolute_humidity(e, df['TEMP']) # g/m^3
        df['SPEC_HUM'] = specific_humidity(e, df['PRES']) * 1000 # g/kg
        df['SAT_SPEC_HUM'] = specific_humidity(e_s, df['PRES']) * 1000 # g/kg

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




    debug_plotting = True

    if debug_plotting == True:
        
      
        # aws_data_dict['snowAMP Ganja La'] = df
        # Plot humidity data for the Ganja La station
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['RH'], label='Relative Humidity (%)', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Relative Humidity (%)')
        plt.title('Relative Humidity Time Series for Ganja La')
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
    save_data = False
    if save_data == True:    
        # ---------- SAVE CLEANED DATA ----------#
        output_dir = str(_DATA_DIR / "Cleaned" / "AWS_pressure_RH")
        os.makedirs(output_dir, exist_ok=True)
        for station, df in station_seasonal_data.items():
            try:
                # Sanitize station name for filename
                sanitized_station_name = station.replace(' ', '_').replace('/', '_')
                target_csv_path = os.path.join(output_dir, f"{sanitized_station_name}.csv")

                # Select RH and PRES columns from the cleaned dataframe and include DATETIME header
                data_to_add = df[['RH', 'PRES']]
                data_to_add.index.name = 'DATETIME'

                if os.path.exists(target_csv_path):
                    # Read the existing CSV file
                    existing_df = pd.read_csv(target_csv_path)
                    
                    # Ensure the DATETIME column is in datetime format and set it as the index
                    if 'DATETIME' in existing_df.columns:
                        existing_df['DATETIME'] = pd.to_datetime(existing_df['DATETIME'])
                        existing_df.set_index('DATETIME', inplace=True)
                    else:
                        print(f"Warning: 'DATETIME' column not found in {target_csv_path}. Cannot merge.")
                        continue

                    # Join the new data with the existing dataframe
                    # This will add RH and PRES columns, aligning on the DATETIME index
                    merged_df = existing_df.join(data_to_add, how='outer')
                    
                    # Save the merged dataframe back to the CSV file, overwriting it
                    merged_df.to_csv(target_csv_path)
                    
                    print(f"Successfully added RH and PRES data to {target_csv_path}")
                else:
                    # If the file doesn't exist, save the new data directly
                    data_to_add.to_csv(target_csv_path)
                    print(f"Created new file and saved RH and PRES data for {station} to {target_csv_path}")

            except Exception as e:
                print(f"An error occurred while processing {station}: {e}")

    return station_seasonal_data
