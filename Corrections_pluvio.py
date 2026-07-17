import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data_overview_plot import get_dir
from Data_overview_plot import get_elevation
from Data_overview_plot import get_measurement
from AWS import get_aws_df_rain
import os


def read_pluvio(file_path):
    df = pd.read_csv(file_path)
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


  
def get_Pluvio_rain(station=None, snow=False, correction=True, update_csv=False,
                    apply_max_threshold=True, debug_plots=True,
                    apply_manual_removal=True, plot_bcon_cumulative=False):


    #This function merges the precipitation data of the pluvio meters.
    # It does the following corrections
    # 1. Where status warnings are indicated, set Prec and Bucket_content to NaN (all status warnings)
    # 2. Sets negative values to NaN for precipitation and bucket content
    # 3. Applies a minimum threshold of 0.2 for bucket content
    # 4. It only uses bucket content difference measurements, no instantaneous measurements

    if station is None:
        station_names = ['Ganja La Pluvio', 'Yala Pluvio', 'Langshisha Pluvio', 'Morimoto Pluvio']

    else:
        station_names = [station]

    dirs_Pluvio = get_dir(station_names)

    #As the aws have pluviometers, these are also appended. The corrections are done seperately in the aws script.
    AWS_rain = get_aws_df_rain(apply_max_threshold=apply_max_threshold,
                               apply_manual_removal=apply_manual_removal)
    merged_df = AWS_rain
    

    # Iterate through each pluvio station and do the corrections
    for file_path, name in zip(dirs_Pluvio, station_names):
        df_pluvio = read_pluvio(file_path)
        
        # Convert 'NA' strings to actual NaN values
        df_pluvio.replace("NA", pd.NA, inplace=True)
        
        # Ensure 'Status_pluvio' is interpreted as numeric
        df_pluvio['Status_pluvio'] = pd.to_numeric(df_pluvio['Status_pluvio'], errors='coerce')
        
        # Convert 'Prec', 'Status_pluvio', 'Bucket_content', 'TEMP', and 'Snowd' columns to floats for Morimoto dataset
        if name == 'Morimoto Pluvio':
            df_pluvio['Prec'] = df_pluvio['Prec'].astype(float)
            df_pluvio['Status_pluvio'] = df_pluvio['Status_pluvio'].astype(float)
            df_pluvio['Bucket_content'] = df_pluvio['Bucket_content'].astype(float)
            df_pluvio['Temp'] = df_pluvio['Temp'].astype(float)
            df_pluvio['SnowD'] = df_pluvio['SnowD'].astype(float)
        
        df_pluvio['DATETIME'] = pd.to_datetime(df_pluvio['DATETIME'])
        df_pluvio.set_index('DATETIME', inplace=True)

        if correction == True:

            #First check what dt the data has that was imported:
            if 'Hourly_Rain' in df_pluvio.columns:
                print(f"Station with 'Hourly_Rain' column: {name}")
                df_pluvio.rename(columns={'Hourly_Rain': 'Rainfall_1H'}, inplace=True)
            


            
            # #Correction 1) Where status warnings are indicated, set Prec and Bucket_content to NaN
            # Make an exception for status warning 1 (do not set to NaN if Status_pluvio == 1)
            mask = (pd.notna(df_pluvio['Status_pluvio'])) & (df_pluvio['Status_pluvio'] != 0) & (df_pluvio['Status_pluvio'] != 1)
            df_pluvio.loc[mask, ["Prec", "Bucket_content"]] = np.nan
            

            # Remove BCON values above 675: Bucket content = 750 * 0.9 
            mask_bcon = df_pluvio['Bucket_content'] > 675
            df_pluvio.loc[mask_bcon, 'Bucket_content'] = np.nan
            df_pluvio.loc[mask_bcon, 'Prec'] = np.nan



            ##Correction 2) Set negative values to NaN for bucket content and Prec
            df_pluvio.loc[df_pluvio['Bucket_content'] < 0, 'Bucket_content'] = np.nan
            df_pluvio.loc[df_pluvio['Prec'] < 0, 'Prec'] = np.nan
            

            #Correction 3) Set values minimum threshold to 0.2 (equal to TB) #Note that is equal for hourly data and 15min data....
            df_pluvio['Bucket_diff'] = df_pluvio['Bucket_content'].diff()
            df_pluvio.loc[df_pluvio['Bucket_diff'] < 0.2, 'Bucket_diff'] = 0

            
            #Correction 4) For the very fine measurements (0.2-0.5mm), use Bucket content instead of Prec, as this is more precise
            #df_pluvio['15min_rain'] = df_pluvio['Prec']

            df_pluvio['15min_rain'] = df_pluvio['Bucket_diff']
            df_pluvio['15min_rain'] = np.where(df_pluvio['Bucket_diff'].isna(), df_pluvio['Prec'], df_pluvio['15min_rain'])

            df_pluvio['15min_rain'] = np.where((df_pluvio['Bucket_diff'] == 0) & (df_pluvio['Prec'] > 0.2), df_pluvio['Prec'], df_pluvio['15min_rain'])
           
            df_pluvio['15min_rain'] = pd.to_numeric(df_pluvio['15min_rain'], errors='coerce')

            # #Correction 7 ) Remove rainfall events where hourly rainfall exceeds 170mm/h as data is per 15min, it is equivalent to 25mm/15min


            # Calculate the frequency of the dataframe
            freq = pd.infer_freq(df_pluvio.index)
            freq_delta = pd.to_timedelta(freq)

            # Apply threshold based on sampling frequency. This is only been applied for robustness, as all pluvio (NOT AWS) data is measured in 15min
            # NOTE: only the derived rainfall-rate column is thresholded here -
            # applying it to the whole dataframe also wiped out Bucket_content
            # (a cumulative reading that legitimately exceeds these thresholds),
            # which broke the BCON cumulative debug plot.
            if apply_max_threshold:
                if freq_delta == pd.Timedelta(hours=1) and 'Rainfall_1H' in df_pluvio.columns:
                    df_pluvio.loc[df_pluvio['Rainfall_1H'] > 100, 'Rainfall_1H'] = np.nan
                if '15min_rain' in df_pluvio.columns:
                    df_pluvio.loc[df_pluvio['15min_rain'] > 100/4, '15min_rain'] = np.nan

            merged_df[name] = df_pluvio


        # Merge with temperature data for later use
        temp_df = get_Pluvio_temp(station=name, agg_hourly=False)[name]
        df_pluvio.drop(columns=['TEMP', 'Temp'], inplace=True, errors='ignore')
        df_pluvio = pd.DataFrame(df_pluvio).merge(temp_df[['TEMP']], left_index=True, right_index=True)
        
        
        df_pluvio.reset_index(inplace=True)
        df_pluvio.set_index('DATETIME', inplace=True)


        #Update merged df with temperature data
        merged_df[name] = df_pluvio

    def data_gaps_removal(merged_df):

        for name, df_pluvio in merged_df.items():


            # In this function, data gaps are removed manually. These gaps have been described in the metafiles.
            # Also, periods with unrealistic values have been removed.
            if name == 'Ganja La Pluvio':
                        df_pluvio.loc[(df_pluvio.index >= '2014-08-03') & (df_pluvio.index <= '2014-10-13'), ['Hourly_Rain']] = np.nan


            if name == 'Yala Pluvio':
                df_pluvio.loc[(df_pluvio.index >= '2012-05-01') & (df_pluvio.index <= '2013-01-16'), ['15min_rain']] = np.nan
                df_pluvio.loc[(df_pluvio.index >= '2014-08-30') & (df_pluvio.index <= '2014-12-13'), ['15min_rain']] = np.nan
                df_pluvio.loc[(df_pluvio.index >= '2013-07-21') & (df_pluvio.index <= '2013-11-01'), ['15min_rain']] = np.nan

            # Still include ('2020-08-23', '2021-11-20'),
            if name == 'Morimoto Pluvio':
                nan_periods = [
                    ('2013-08-05', '2014-05-03'),
                    ('2014-10-10', '2015-06-07'),
                    ('2015-07-26', '2015-10-22'),
                    ('2021-02-26', '2021-11-13'),
                    ('2023-05-28', '2023-11-11'),
                    
                    ('2017-10-09', '2017-10-22'),
                    ('2017-03-12', '2017-03-15'),
                ]
                for start, end in nan_periods:
                    df_pluvio.loc[(df_pluvio.index >= start) & (df_pluvio.index <= end), ['15min_rain']] = np.nan
            if name == 'Langshisha Pluvio':
                df_pluvio.loc[(df_pluvio.index >= '2021-07-18') & (df_pluvio.index <= '2021-11-13'), ['15min_rain']] = np.nan
            # Add to the dictionary
            merged_df[name] = df_pluvio
        return merged_df


    #Update merge df with the data gaps removed

    if apply_manual_removal:
        merged_df = data_gaps_removal(merged_df)

    
#######################################Plotting#######################################
    def plot_results_for_debugging(merged_df, all_variables=False, yearly_totals=False, daily_values=False, Fill_gaps_check_cumulative=False):
        if all_variables == True:
            for name, df_pluvio in merged_df.items():
                # Plot all variables in the dataframe except 'DATETIME' in different subplots with shared x-axis
                num_vars = len(df_pluvio.columns) - 1  # Exclude 'DATETIME'
                fig, axes = plt.subplots(num_vars, 1, figsize=(12, 4 * num_vars), sharex=True)

                # If there is only one variable, axes will not be a list, so make it iterable
                if num_vars == 1:
                    axes = [axes]

                for ax, column in zip(axes, df_pluvio.columns):
                    if column != 'DATETIME':
                        ax.plot(df_pluvio.index, df_pluvio[column], label=column)
                        ax.set_title(f'{column} Timeseries')
                        ax.set_ylabel(column)
                        ax.legend()
                        ax.grid(True)

                # Set the x-axis label for the bottom subplot
                axes[-1].set_xlabel('Date')

                # Adjust layout to prevent overlap
                plt.tight_layout()
                plt.show()
        

        if daily_values == True:

            # Create a figure with subplots, one for each station
            num_stations = len(merged_df)
            fig, axes = plt.subplots(num_stations, 1, figsize=(12, 6 * num_stations), sharex=True)

            # If there is only one subplot, axes will not be a list, so make it iterable
            if num_stations == 1:
                axes = [axes]
        
            # Additionally plot uncorrected Morimoto Pluvio data if available
            for ax, (station, df) in zip(axes, merged_df.items()):
                if station == 'Morimoto Pluvio' and 'Prec' in df.columns:
                    # Resample uncorrected precipitation to hourly sums
                    uncorrected_hourly = df['Bucket_diff'].resample('h').sum(min_count=1)
                    ax.plot(uncorrected_hourly.index, uncorrected_hourly, label='Morimoto Pluvio - Uncorrected Hourly Prec', color='red', alpha=0.5)
                    ax.legend(loc='upper left')



            # Plot daily rainfall for each station in its own subplot with shared y-axis
            for ax, (station, df) in zip(axes, merged_df.items()):
                if '15min_rain' in df.columns:
                    daily_rain = df['15min_rain'].resample('h').sum()
                elif 'Hourly_Rain' in df.columns:
                    daily_rain = df['Hourly_Rain'].resample('h').sum()
                else:
                    print(f"The data is not available for {station}.")
                    continue

                ax.plot(daily_rain.index, daily_rain, label=f'{station} - Daily Rainfall', color='blue')
                ax.set_ylabel('Daily Rain (mm)', color='blue')
                ax.set_title(f'Daily Rainfall - {station}')
                ax.legend(loc='upper left')
                ax.grid()
            
            # Set a shared y-axis limit for all subplots
            all_daily_rain = []
            for df in merged_df.values():
                if '15min_rain' in df.columns:
                    daily_rain = df['15min_rain'].resample('h').sum(min_count=4)
                elif 'Hourly_Rain' in df.columns:
                    daily_rain = df['Hourly_Rain'].resample('h').sum(min_count=1)
                else:
                    continue
                all_daily_rain.append(daily_rain)
            shared_y_min = min(rain.min() for rain in all_daily_rain if not rain.empty)
            shared_y_max = max(rain.max() for rain in all_daily_rain if not rain.empty)
            for ax in axes:
                ax.set_ylim(shared_y_min, shared_y_max)

            # Set the x-axis label for the bottom subplot
            axes[-1].set_xlabel('Date')


            # Adjust layout to prevent overlap
            plt.tight_layout()
            plt.show()
        hourly_values = False
        if hourly_values == True:

            # Create a figure with subplots, one for each station
            num_stations = len(merged_df)
            fig, axes = plt.subplots(num_stations, 1, figsize=(12, 6 * num_stations), sharex=True)

            # If there is only one subplot, axes will not be a list, so make it iterable
            if num_stations == 1:
                axes = [axes]

            # Plot hourly rainfall for each station in its own subplot with shared y-axis
            for ax, (station, df) in zip(axes, merged_df.items()):
                if '15min_rain' in df.columns:
                    hourly_rain = df['15min_rain'].resample('H').sum(min_count=4)
                elif 'Hourly_Rain' in df.columns:
                    hourly_rain = df['Hourly_Rain'].resample('H').sum(min_count=1)
                else:
                    print(f"The data is not available for {station}.")
                    continue

            ax.plot(hourly_rain.index, hourly_rain, label=f'{station} - Hourly Rainfall', color='blue')
            ax.set_ylabel('Hourly Rain (mm)', color='blue')
            ax.set_title(f'Hourly Rainfall - {station}')
            ax.legend(loc='upper left')
            ax.grid()
            
            # Set a shared y-axis limit for all subplots
            all_hourly_rain = [
            (df['15min_rain'] if '15min_rain' in df.columns else df['Hourly_Rain']).resample('H').sum()
            for df in merged_df.values() if '15min_rain' in df.columns or 'Hourly_Rain' in df.columns
            ]
            shared_y_min = min(rain.min() for rain in all_hourly_rain if not rain.empty)
            shared_y_max = max(rain.max() for rain in all_hourly_rain if not rain.empty)
            for ax in axes:
                ax.set_ylim(shared_y_min, shared_y_max)

            # Set the x-axis label for the bottom subplot
            axes[-1].set_xlabel('Date')

            # Adjust layout to prevent overlap
            plt.tight_layout()
            plt.show()

            # Add another subplot for Morimoto Pluvio showing uncorrected hourly data
            for ax, (station, df) in zip(axes, merged_df.items()):
                if station == 'Morimoto Pluvio' and 'Prec' in df.columns:
                    # Resample uncorrected precipitation to hourly sums
                    uncorrected_hourly = df['Prec'].resample('H').sum(min_count=4)
                    # Create a new figure for uncorrected data
                    fig2, ax2 = plt.subplots(figsize=(12, 6), sharex=True)
                    ax2.plot(uncorrected_hourly.index, uncorrected_hourly, label='Morimoto Pluvio - Uncorrected Hourly Prec', color='red')
                    ax2.set_ylabel('Uncorrected Hourly Prec (mm)', color='red')
                    ax2.set_title('Morimoto Pluvio - Uncorrected Hourly Precipitation')
                    ax2.legend(loc='upper left')
                    ax2.grid()
                    ax2.set_xlabel('Date')
                    plt.tight_layout()
                    plt.show()


        if Fill_gaps_check_cumulative == True:

            # Plot cumulative precipitation for each pluviometer in one graph (hourly)
            plt.figure(figsize=(14, 7))
            cum_precip = {}
            base_station = 'Kyangjin AWS'
            # Get hourly rainfall for each station
            for station, df in merged_df.items():
                if '15min_rain' in df.columns:
                    rain_data = df['15min_rain']
                elif 'Hourly_Rain' in df.columns:
                    rain_data = df['Hourly_Rain']
                else:
                    raise ValueError(f"No valid rainfall data found for station: {station}")
                cum_precip[station] = rain_data.cumsum()

            # Find the earliest start date among all stations
            start_dates = {station: series.first_valid_index() for station, series in cum_precip.items()}
            base_start = start_dates.get(base_station)
            base_cum = cum_precip.get(base_station)

            for station, series in cum_precip.items():
                station_start = start_dates.get(station)
                if station == base_station or base_cum is None or station_start is None:
                    plt.plot(series.index, series, label=station)
                else:
                    # Find the value of base_cum at the start of this station
                    if station_start in base_cum.index:
                        offset = base_cum.loc[station_start]
                    else:
                        # If not found, use the closest previous value
                        offset = base_cum[:station_start].iloc[-1] if not base_cum[:station_start].empty else 0
                    plt.plot(series.index, series + offset, label=station)

            for station, df in merged_df.items():
                if '15min_rain' in df.columns:
                    rain_data = df['15min_rain']
                elif 'Hourly_Rain' in df.columns:
                    rain_data = df['Hourly_Rain']
                else:
                    continue # Skip station if no rain data found

                # Create a DataFrame with month, day, hour as columns for fast groupby
                dt_index = rain_data.index
                time_keys = pd.DataFrame({
                    'month': dt_index.month,
                    'day': dt_index.day,
                    'hour': dt_index.hour
                }, index=dt_index)
                # Group by month, day, hour and compute mean (excluding current value)
                group_means = rain_data.groupby([time_keys['month'], time_keys['day'], time_keys['hour']]).transform('mean')
                # Fill NaNs with group means
                filled = rain_data.copy()
                filled[filled.isna()] = group_means[filled.isna()]
                # Plot the interpolated values as dotted lines
                # Plot cumulative sum of interpolated values, aligned to base station
                cum_filled = filled.cumsum()
                if station == base_station or base_cum is None:
                    plt.plot(cum_filled.index, cum_filled, 'k:', label=f'{station} - Interpolated')
                else:
                    station_start = cum_filled.first_valid_index()
                    if station_start and station_start in base_cum.index:
                        offset = base_cum.loc[station_start]
                    else:
                        offset = base_cum[:station_start].iloc[-1] if not base_cum[:station_start].empty else 0
                    plt.plot(cum_filled.index, cum_filled + offset, 'k:', label=f'{station} - Interpolated')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Precipitation (mm)')
            plt.title('Cumulative Precipitation Comparison (Aligned to Kyangjin AWS)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        
        
        
        if yearly_totals == True:
            fig, axes = plt.subplots(len(merged_df), 1, figsize=(12, 8), sharex=True)

            for i, (station, df) in enumerate(merged_df.items()):
                df = df.copy()
                df.index = pd.to_datetime(df.index)  # Ensure index is a DatetimeIndex
                df['Year'] = df.index.year
                yearly_cumulative = df.groupby('Year')['15min_rain'].sum()
                yearly_valid_counts = df.groupby('Year')['15min_rain'].count()
                yearly_total_counts = df.groupby('Year')['15min_rain'].size()
                valid_years = yearly_valid_counts / yearly_total_counts >= 0.9
                filtered_yearly_cumulative = yearly_cumulative[valid_years]

                axes[i].bar(filtered_yearly_cumulative.index, filtered_yearly_cumulative, label=f'{station} Yearly Cumulative Rain')
                axes[i].set_ylabel('Cumulative Rain (mm)')
                axes[i].set_title(f'{station} Yearly Cumulative Rainfall')
                axes[i].legend(loc='upper right')
                axes[i].grid(True)

            axes[-1].set_xlabel('Year')
            plt.tight_layout()
            plt.show()
        
        
    if debug_plots:
        plot_results_for_debugging(merged_df, all_variables=False, yearly_totals=False, daily_values=False, Fill_gaps_check_cumulative=True)

    if plot_bcon_cumulative:
        plt.figure(figsize=(14, 7))
        for station_name, df_station in merged_df.items():
            if 'Bucket_content' in df_station.columns:
                bcon = pd.to_numeric(df_station['Bucket_content'], errors='coerce')
                bcon_cumulative = bcon - bcon.dropna().iloc[0] if not bcon.dropna().empty else bcon
                plt.plot(bcon_cumulative.index, bcon_cumulative, label=station_name)
                plotted = True

        if plotted:
            plt.xlabel('Date')
            plt.ylabel('BCON cumulative value (mm)')
            plt.title('Cumulative BCON values per station')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def sensitivity_to_max_threshold(merged_df):
        
            #This functions checks how sensitive the data is to changing the threshold for maximum hourly rainfall.


            # # Create a figure with subplots, one for each pluviometer
            num_pluviometers = len(merged_df)
            fig, axes = plt.subplots(num_pluviometers, 1, figsize=(12, 6 * num_pluviometers), sharex=True)

            # If there is only one subplot, axes will not be a list, so make it iterable
            if num_pluviometers == 1:
                axes = [axes]

            # Plot each pluviometer's data in its own subplot
            for ax, (station, df) in zip(axes, merged_df.items()):
                if correction == True:
                    # Plot data with different colors based on thresholds
                    df[df > 40] = np.nan

                    ax.plot(df.index, df['15min_rain'], label=f'{station} - Rainfall < 40', color='green')

                    df = df[(df > 40) & (df <= 50)]
                    ax.scatter(df.index, df['15min_rain'], label=f'{station} - Rainfall < 50', color='blue')

                    df = df[(df > 50) & (df <= 60)]
                    ax.scatter(df.index, df['15min_rain'], label=f'{station} - Rainfall < 60', color='orange', s=10)

                    df = df[(df > 60) & (df <= 70)]
                    ax.scatter(df.index, df['15min_rain'], label=f'{station} - Rainfall < 70', color='red')


                else:
                    ax.plot(df.index, df['Prec'], label=f'{station} - Rainfall', color='blue')
                if snow == True:
                    ax.set_ylabel('Hourly Snowfall (mm)', color='blue')
                    ax.set_title(f'Hourly Snowfall - {station}')
                else:
                    ax.set_ylabel('Hourly Rain (mm)', color='blue')
                ax.legend(loc='upper left')
                ax.grid()

                

                

            # Set the x-axis label for the bottom subplot
            axes[-1].set_xlabel('Date')

            # Adjust layout to prevent overlap
            plt.tight_layout()
            return plt.show()


        
        


        # Efficiently interpolate missing values using mean of same hour in other years
        

    def save_to_csv(merged_df):
        output_dir = r"../data/Cleaned/Pluvio"


        for station, df in merged_df.items():
            # Prepare the file name
            file_name = f"{station}.csv"
            file_path = os.path.join(output_dir, file_name)
            # Rename TAIR to TEMP
            df.rename(columns={'TAIR': 'TEMP'}, inplace=True)

            # Save the data to CSV
            df['DATETIME'] = df.index  # Set datetime as one of the columns 
            df['DATETIME'] = pd.to_datetime(df['DATETIME'])
            #df['DATETIME'] = df['DATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df = df.fillna('NAN')

            print(f"Saving station: {station}, columns: {df.columns.tolist()}")
            if '15min_rain' in df.columns:
                df_to_save = df[['DATETIME', '15min_rain', 'TEMP']]
                headers = ['DATETIME', 'Rainfall_15min', 'Temperature']
            elif 'Hourly_Rain' in df.columns:
                df_to_save = df[['DATETIME', 'Hourly_Rain', 'TEMP']]
                headers = ['DATETIME', 'Rainfall_1H', 'Temperature']
            else:
                # Fallback or error handling if neither rain column is present
                print(f"Warning: No rainfall data ('15min_rain' or 'Hourly_Rain') found for station {station}. Skipping save for this station.")
                continue

            df_to_save.fillna(np.nan).to_csv(file_path, index=False, header=headers)


    if update_csv == True:
        save_to_csv(merged_df)  



    return merged_df
# get_Pluvio_rain(station=None, snow=False, correction=True, update_csv=False, plot_bcon_cumulative=True, apply_max_threshold=True, debug_plots=True, apply_manual_removal=True)


















###### Different analysis #######

# ── Manual removal windows (single source for the plot functions below) ────
# Pluvio gaps: keep in sync with data_gaps_removal() inside get_Pluvio_rain.
MANUAL_GAPS_PLUVIO = {
    'Ganja La Pluvio':   [('2014-08-03', '2014-10-13')],
    'Yala Pluvio':       [('2012-05-01', '2013-01-16'),
                          ('2014-08-30', '2014-12-13'),
                          ('2013-07-21', '2013-11-01')],
    'Morimoto Pluvio':   [('2013-08-05', '2014-05-03'),
                          ('2014-10-10', '2015-06-07'),
                          ('2015-07-26', '2015-10-22'),
                          ('2021-02-26', '2021-11-13'),
                          ('2023-05-28', '2023-11-11'),
                          ('2017-10-09', '2017-10-22'),
                          ('2020-09-24', '2021-11-13'),
                          ('2017-03-12', '2017-03-15')],
    'Langshisha Pluvio': [('2021-07-18', '2021-11-13'),
                          ('2014-05-26', '2014-05-27')],
}
# AWS maintenance windows: keep in sync with get_aws_df_rain() in AWS.py.
AWS_MAINTENANCE = {
    'Kyangjin AWS': [('2022-11-18', '2022-11-19'),
                     ('2016-10-18', '2017-04-20'),
                     ('2019-09-27', '2020-03-27'),
                     ('2016-03-01', '2016-11-01')],
    'Yala BC AWS':  [('2023-10-13', '2023-10-14'),
                     ('2022-10-07', '2022-11-03'),
                     ('2019-09-27', '2020-03-27'),
                     ('2018-09-13', '2018-10-19'),
                     ('2018-04-26 12:00', '2018-04-26 16:00'),
                     ('2014-09-30', '2014-10-14'),
                     ('2018-09-13', '2018-10-09'),
                     ('2019-09-25', '2019-10-24'),
                     ('2022-10-08', '2022-11-15'),
                     ('2020-09-26', '2020-10-03'),
                     ('2014-05-06', '2014-05-07'),
                     ('2016-09-19', '2016-10-13'),
                     ('2016-05-10', '2017-04-01')],
}
