import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data_overview_plot import get_elevation
from Corrections_pluvio import get_Pluvio_temp
from AWS import get_aws_df_temp
from Snowamp import get_SNOWAMP_df_temp
from Snowamp import get_SNOWAMP_df_temp
from TippingBucket import get_TB_df_temp
import pickle
import os



def merge_datasets_hourly():    

    all_merged_dfs = get_TB_df_temp()
    
    #station_name_pluvio = ['Yala Pluvio']

    # Get pluvio data
    all_merged_dfs = get_Pluvio_temp()
    #all_merged_dfs.update(df_pluvio)

    # Get AWS data
    df_aws=get_aws_df_temp()
    all_merged_dfs.update(df_aws)

    include_snowamp = True
    if include_snowamp ==True:
        df_SNOWAMP = get_SNOWAMP_df_temp()
        all_merged_dfs.update(df_SNOWAMP)


    # Make sure average temperature for every hour
    for station, df in all_merged_dfs.items():
        if 'DATETIME' in df.columns:
            df.set_index('DATETIME', inplace=True)
        df['h'] = df.index.floor('h')
        avg_temp_h = df.groupby('h')['TEMP'].mean().reset_index()
        avg_temp_h = avg_temp_h.rename(columns={'TEMP': 'Avg_Temp_H'})
        df = pd.merge(df, avg_temp_h, on='h', how='left')
        all_merged_dfs[station] = df
    
    # Ensure that the 1H periods are aligned across all stations
    common_H = pd.date_range(start='2012-01-01', end='2024-12-01', freq='h')

    # Reindex each station's DataFrame to the common 1H periods, keeping missing values as NaN
    for station in all_merged_dfs.keys():
        all_merged_dfs[station] = all_merged_dfs[station].drop_duplicates(subset='h').set_index('h').reindex(common_H)

    # Save temperature data to CSV files
    output_dir = r'../data/Cleaned/Temperature'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each station's data to a separate CSV file
    for station, df in all_merged_dfs.items():
        output_path = os.path.join(output_dir, f'{station}_temperature.csv')
        df.to_csv(output_path)
        print(f"Saved temperature data for {station} to {output_path}")
    
    return all_merged_dfs
# merge_datasets_hourly()
# # # # Save temp_merged_dfs to a pickle file
# with open(r'../data/temp_merged_dfs.pkl', 'wb') as f:
#     pickle.dump(temp_merged_dfs, f)

# Load temp_merged_dfs from the pickle file
with open(r'../data/temp_merged_dfs.pkl', 'rb') as f:
    temp_merged_dfs = pickle.load(f)
# Temperature_Lapse_rate_season(temp_merged_dfs)
def zero_deg_altitude(all_merged_dfs):
    # Filter out stations with 'TB' in the name
    all_merged_dfs = {station: df for station, df in all_merged_dfs.items() if 'TB' not in station}
    
    # Create a DataFrame from all_merged_dfs with temperature and elevation data
    dfs_to_merge = []
    for station, df in all_merged_dfs.items():
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        # Create a df for the current station
        station_df = df[['TEMP']].copy()
        station_df.rename(columns={'TEMP': f'{station}_Temperature'}, inplace=True)
        station_df[f'{station}_Elevation'] = get_elevation([station])[0]
        dfs_to_merge.append(station_df)
        # remove temperature data between 2015-10 and 2016-02 as this is producing invalid isotherm values
        station_df.loc['2015-10':'2016-02', f'{station}_Temperature'] = np.nan

    # Merge all station dataframes on their datetime index
    temp_elev_df = pd.concat(dfs_to_merge, axis=1)
    temp_elev_df.index.name = 'DATETIME'

    # Create a DataFrame with elevations between the lowest and highest station
    min_elevation = min(get_elevation(all_merged_dfs.keys()))
    max_elevation = max(get_elevation(all_merged_dfs.keys()))
    elevation_range = np.arange(0, 8000, 40)  # Step size of 100m

    # Aggregate to hourly data, taking the mean over temperature
    hourly_temp_df = temp_elev_df.resample('h').mean()
    # Reset the index to include the datetime column
    hourly_temp_df.reset_index(inplace=True)
    hourly_temp_df.rename(columns={'DATETIME': 'DATE'}, inplace=True)
    # Set 'DATE' as the index
    hourly_temp_df.set_index('DATE', inplace=True)

    # Prepare for interpolation
    temp_cols = [col for col in hourly_temp_df.columns if '_Temperature' in col]
    elev_cols = [col for col in hourly_temp_df.columns if '_Elevation' in col]
    temps = hourly_temp_df[temp_cols].values
    elevs = hourly_temp_df[elev_cols].values
    valid_mask = ~np.isnan(temps)
    results = np.full((len(hourly_temp_df), len(elevation_range)), np.nan)
    lapse_rates = np.full(len(hourly_temp_df), np.nan)
    zero_deg_elevations = np.full(len(hourly_temp_df), np.nan)
    one_deg_elevations = np.full(len(hourly_temp_df), np.nan)

    # Iterate over each row (timestep) for 0°C and 1°C isotherms
    for i in range(len(hourly_temp_df)):
        row_mask = valid_mask[i, :]
        if np.sum(row_mask) >= 2:
            valid_elevs = elevs[i, row_mask]
            valid_temps = temps[i, row_mask]
            unique_elevs, inverse_indices = np.unique(valid_elevs, return_inverse=True)
            if len(unique_elevs) >= 6:
                unique_temps = np.bincount(inverse_indices, weights=valid_temps) / np.bincount(inverse_indices)
                sort_idx = np.argsort(unique_elevs)
                sorted_elevs = unique_elevs[sort_idx]
                sorted_temps = unique_temps[sort_idx]
                slope, intercept = np.polyfit(sorted_elevs, sorted_temps, 1)
                lapse_rates[i] = slope

                # 0°C isotherm
                zero_deg_elev = -intercept / slope
                if sorted_elevs.min() <= zero_deg_elev <= sorted_elevs.max():
                    above_zero = sorted_temps > 0
                    below_zero = sorted_temps < 0
                    if np.any(above_zero) and np.any(below_zero):
                        idx_above = np.where(above_zero)[0][-1]
                        idx_below = np.where(below_zero)[0][0]
                        elev_above = sorted_elevs[idx_above]
                        temp_above = sorted_temps[idx_above]
                        elev_below = sorted_elevs[idx_below]
                        temp_below = sorted_temps[idx_below]
                        zero_deg_elev = elev_above + (elev_below - elev_above) * (0 - temp_above) / (temp_below - temp_above)
                    else:
                        zero_deg_elev = -intercept / slope
                else:
                    zero_deg_elev = -intercept / slope
                zero_deg_elevations[i] = max(0, min(8000, zero_deg_elev))

                # 1°C isotherm
                one_deg_elev = (1 - intercept) / slope
                if sorted_elevs.min() <= one_deg_elev <= sorted_elevs.max():
                    above_one = sorted_temps > 1
                    below_one = sorted_temps < 1
                    if np.any(above_one) and np.any(below_one):
                        idx_above = np.where(above_one)[0][-1]
                        idx_below = np.where(below_one)[0][0]
                        elev_above = sorted_elevs[idx_above]
                        temp_above = sorted_temps[idx_above]
                        elev_below = sorted_elevs[idx_below]
                        temp_below = sorted_temps[idx_below]
                        one_deg_elev = elev_above + (elev_below - elev_above) * (1 - temp_above) / (temp_below - temp_above)
                    else:
                        one_deg_elev = (1 - intercept) / slope
                else:
                    one_deg_elev = (1 - intercept) / slope
                one_deg_elevations[i] = max(0, min(8000, one_deg_elev))

                # Interpolate and extrapolate the temperature profile using the linear model
                interpolated_values = slope * elevation_range + intercept
                results[i, :] = interpolated_values

    # Assign the calculated values back to the DataFrame
    interpolated_temp_df = pd.DataFrame(columns=elevation_range, index=hourly_temp_df.index)
    interpolated_temp_df.iloc[:, :] = results
    interpolated_temp_df['lapse_rate'] = lapse_rates
    interpolated_temp_df['Zero_Deg_Elevation'] = zero_deg_elevations
    interpolated_temp_df['One_Deg_Elevation'] = one_deg_elevations

    # Plot monthly means for both 0°C and 1°C isotherms
    monthly_zero = pd.Series(zero_deg_elevations, index=hourly_temp_df.index).groupby(hourly_temp_df.index.month).mean()
    monthly_one = pd.Series(one_deg_elevations, index=hourly_temp_df.index).groupby(hourly_temp_df.index.month).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_zero.index, monthly_zero.values, marker='o', label='0°C isotherm')
    plt.plot(monthly_one.index, monthly_one.values, marker='s', label='1°C isotherm')
    plt.xlabel('Month')
    plt.ylabel('Mean Isotherm Elevation (m)')
    plt.title('Monthly Mean Isotherm Elevations (0°C and 1°C)')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return interpolated_temp_df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')
