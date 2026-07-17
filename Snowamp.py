import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data_overview_plot import get_dir
from Data_overview_plot import get_elevation
from Data_overview_plot import get_measurement
import os
from datetime import datetime
station_names = ['snowAMP Ganja La','snowAMP Ganjala upper','SNOWAMP_lower','SNOWAMP_middle']



def read_SNOWAMP(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={'DATE': 'Date', 'TIME': 'Time'})
    df['DATETIME'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S')
    df = df.drop(columns=['Date', 'Time'])
    return df





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





def write_cleaned_csv_rain_temp_swe_pluvio():

    dir=get_dir('snowAMP Ganja La')
    df = read_SNOWAMP(dir[0])

    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.reset_index()


    df.loc[(df['DATETIME'] >= '2016-12-12') & (df['DATETIME'] <= '2017-01-20'), 'Air Temperature(degC)'] = np.nan

    # Filter out all values above 40 mm per hour
    df['Precipitation(mm)'] = df['Precipitation(mm)'].apply(lambda x: x if x <= 40 else np.nan)

    plt.figure(figsize=(10, 5))



    # Remove data for specified date ranges
   


   
    df.loc[(df['DATETIME'] >= '2018-09-22') & (df['DATETIME'] <= '2018-09-24'), 'Air Temperature(degC)'] = np.nan
    df.loc[
        ((df['DATETIME'] >= '2015-10-31') & (df['DATETIME'] <= '2016-04-30')) |
        ((df['DATETIME'] >= '2018-06-15') & (df['DATETIME'] <= '2018-09-17')) |
        ((df['DATETIME'] >= '2018-12-12') & (df['DATETIME'] <= '2019-02-14')) |
        (df['DATETIME'] >= '2019-11-22'),
        'Air Temperature(degC)'
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

    (df['DATETIME'] >= '2021-10-18'),'Precipitation(mm)'] = np.nan

    # Calculate cumulative rainfall
    df['Cumulative_Rainfall'] = df['Precipitation(mm)'].cumsum()



    # Plot cumulative rainfall
    plt.plot(df['DATETIME'], df['Cumulative_Rainfall'], label='Cumulative Rainfall (Ganja La)', color='orange')
    # Plot other pluvio meters from Cleaned/Pluvio directory
    pluvio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Cleaned', 'Pluvio')
    plt.figure(figsize=(10, 5))
    for fname in os.listdir(pluvio_dir):
        if fname.endswith('.csv') and 'Ganja La' not in fname:
            other_df = pd.read_csv(os.path.join(pluvio_dir, fname))
            if 'DATETIME' in other_df.columns and 'Rainfall_1H' in other_df.columns:
                other_df['DATETIME'] = pd.to_datetime(other_df['DATETIME'])
                other_df['Rainfall_1H'] = pd.to_numeric(other_df['Rainfall_1H'], errors='coerce')
                other_df['Cumulative_Rainfall'] = other_df['Rainfall_1H'].cumsum()
                label = fname.replace('.csv', '')
                plt.plot(other_df['DATETIME'], other_df['Cumulative_Rainfall'], label=f'Cumulative Rainfall ({label})')
    plt.show()
    # Second figure: daily rainfall
    plt.figure(figsize=(10, 5))
    # Plot daily rainfall for Ganja La
    df['Date'] = df['DATETIME'].dt.date
    daily_rain = df.groupby('Date')['Precipitation(mm)'].sum()
    plt.bar(daily_rain.index, daily_rain.values, label='Daily Rainfall (Ganja La)', color='orange', alpha=0.7)

    # Plot daily rainfall for other stations
    for fname in os.listdir(pluvio_dir):
        if fname.endswith('.csv') and 'Ganja La' not in fname:
            other_df = pd.read_csv(os.path.join(pluvio_dir, fname))
            if 'DATETIME' in other_df.columns and 'Rainfall_1H' in other_df.columns:
                other_df['DATETIME'] = pd.to_datetime(other_df['DATETIME'])
                other_df['Rainfall_1H'] = pd.to_numeric(other_df['Rainfall_1H'], errors='coerce')
                other_df['Date'] = other_df['DATETIME'].dt.date
                daily_rain_other = other_df.groupby('Date')['Rainfall_1H'].sum()
                label = fname.replace('.csv', '')
                plt.bar(daily_rain_other.index, daily_rain_other.values, label=f'Daily Rainfall ({label})', alpha=0.5)

    plt.xlabel('Date')
    plt.ylabel('Daily Rainfall (mm)')
    plt.title('Daily Rainfall Timeseries')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plt.plot(df['DATETIME'], df['CS725 SweK(mm)'], label='Pluvio Data', color='blue')
    # plt.xlabel('DateTime')
    # plt.ylabel('Precipitation (mm)')
    # plt.title('Timeseries of Pluvio Data')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    update_csv = False
    if update_csv == True:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Cleaned', 'Pluvio')
        # Prepare the file name
        file_name = datetime.now().strftime("%m%d%Y") + "_snowAMP Ganja La.csv"
        file_path = os.path.join(output_dir, file_name)

        # Rename TEMP column if necessary
        if 'Air Temperature(degC)' in df.columns:
            df.rename(columns={'Air Temperature(degC)': 'Temperature_1H'}, inplace=True)

        # Save the data to CSV
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df = df.fillna('NAN')
        df[['DATETIME', 'Precipitation(mm)','Temperature_1H', 'CS725 SweK(mm)']].fillna(np.nan).to_csv(file_path, index=False, header=['DATETIME', 'Rainfall_1H','Temperature_1H', 'CS725_Swek(mm)'])
