from plotting import get_elevation
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, BSpline






# Load temp_merged_dfs from the pickle file
with open(r'../data/temp_merged_dfs.pkl', 'rb') as f:
    temp_merged_dfs = pickle.load(f)

def zero_deg_altitude(all_merged_dfs):
    # Filter out stations with 'TB' in the name
    all_merged_dfs = {station: df for station, df in all_merged_dfs.items() if 'TB' not in station}
    
    # Create a DataFrame from all_merged_dfs with temperature and elevation data
    temp_elev_df = pd.DataFrame()
    for station, df in all_merged_dfs.items():
        temp_elev_df['DATETIME'] = df['index']
        temp_elev_df[f'{station}_Temperature'] = df['TEMP']
        temp_elev_df[f'{station}_Elevation'] = get_elevation([station])[0]

    # Create a DataFrame with elevations between the lowest and highest station
    min_elevation = min(get_elevation(all_merged_dfs.keys()))
    max_elevation = max(get_elevation(all_merged_dfs.keys()))
    elevation_range = np.arange(0, 8000, 40)  # Step size of 100m

    # Aggregate to daily data, taking the mean over temperature
    daily_temp_df = temp_elev_df.resample('h', on='DATETIME').mean()

    # Reset the index to include the datetime column
    daily_temp_df.reset_index(inplace=True)
    daily_temp_df.rename(columns={'DATETIME': 'DATE'}, inplace=True)
    # Set 'DATE' as the index
    daily_temp_df.set_index('DATE', inplace=True)
    

    # Find the maximum value of 'Yala BC AWS' in the dataset
    max_value = temp_elev_df['Yala BC AWS_Temperature'].quantile(0.9999)
    plt.figure(figsize=(10, 6))
    plt.scatter(temp_elev_df.index, temp_elev_df['Yala BC AWS_Temperature'], color='tab:blue', label='Yala BC AWS')
    plt.title('Temperature vs Elevation for Yala AWS')
    plt.xlabel('Elevation (m)')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"The maximum value of 'Yala BC AWS' is: {max_value}")

    # Calculate the elevation of 0 degrees using the lapse rate
    lapse_rate = 6.5 / 1000  # Convert lapse rate to °C/m
    zero_deg_elevation_max = max_value / lapse_rate + 5090
    min_two_deg_elevation_max = 5090 + (max_value-2) / lapse_rate 
    two_deg_elevation_max = 5090 + (max_value+2) / lapse_rate   
    print(f"The elevation of 0 degrees with the max yala value is: {zero_deg_elevation_max} meters")
    print(f"The elevation of 2 degrees with the max yala value is: {two_deg_elevation_max} meters")
    print(f"The elevation of -2 degrees with the max yala value is: {min_two_deg_elevation_max} meters")


     # Find the minimum value of 'Kyangjin AWS' in the dataset
    min_value = temp_elev_df['Kyangjin AWS_Temperature'].quantile(0.0001)
    # Plot temperature vs elevation for 'Kyangjin AWS'
    plt.figure(figsize=(10, 6))
    plt.scatter(temp_elev_df.index, temp_elev_df['Kyangjin AWS_Temperature'], color='tab:blue', label='Kyangjin AWS')
    plt.title('Temperature vs Elevation for Kyangjin AWS')
    plt.xlabel('Elevation (m)')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"The minimum value of 'Kyangjin AWS' is: {min_value}")

    # Calculate the elevation of 0 degrees using the lapse rate
    lapse_rate = 6.5 / 1000  # Convert lapse rate to °C/m
    zero_deg_elevation_min = 3862 + min_value / lapse_rate 
    min_two_deg_elevation_min = 3862 + (min_value-2) / lapse_rate 
    two_deg_elevation_min = 3862 + (min_value+2) / lapse_rate 

    print(f"The elevation of 0 degrees with the min Kyangjin value is: {zero_deg_elevation_min} meters")
    print(f"The elevation of 2 degrees with the min Kyangjin value is: {two_deg_elevation_min} meters")
    print(f"The elevation of -2 degrees with the min Kyangjin value is: {min_two_deg_elevation_min} meters")


    # Initialize a DataFrame to store interpolated temperature data
    interpolated_temp_df = pd.DataFrame(columns=elevation_range, index=daily_temp_df.index)


    # Perform piecewise linear interpolation for each timestep
    for _, row in daily_temp_df.iterrows():
        valid_stations = row.sort_index(key=lambda x: [
            -row[col] if '_Elevation' in col else -row[col.replace('_Elevation', '_Temperature')]
            for col in x if '_Elevation' in col or '_Temperature' in col
        ])

    # Perform piecewise linear interpolation for each timestep
    for index, row in daily_temp_df.iterrows():
        valid_stations = row
        valid_temps = valid_stations[valid_stations.index.str.endswith('_Temperature')]
        valid_elevs = valid_stations[valid_stations.index.str.endswith('_Elevation')]

        valid_temps = valid_stations[valid_stations.index.str.endswith('_Temperature')].dropna()
        valid_elevs = valid_stations[valid_stations.index.str.endswith('_Elevation')].dropna()
        # Track removed values and drop corresponding elevations
        removed_indices = valid_stations[valid_stations.index.str.endswith('_Temperature') & valid_stations.isna()].index
        valid_elevs = valid_elevs.drop(removed_indices.str.replace('_Temperature', '_Elevation'))
        # Count the number of valid measurements for each timestep
        if len(valid_temps) >= 4:

            # Ensure valid_elevs and valid_temps have the same length and remove duplicates
            unique_elevs, unique_indices = np.unique(valid_elevs.values, return_index=True)
            unique_temps = valid_temps.values[unique_indices]

            if len(unique_elevs) >= 2:  # Ensure there are at least two unique points for interpolation
                # Sort by elevation to avoid incorrect interpolation
                sorted_indices = np.argsort(unique_elevs)
                sorted_elevs = unique_elevs[sorted_indices]
                sorted_temps = unique_temps[sorted_indices]

                # Calculate the slope and intercept for extrapolation
                slope, intercept = np.polyfit(sorted_elevs, sorted_temps, 1)
                # Fit a linear regression model to all available data points
                slope, intercept = np.polyfit(sorted_elevs, sorted_temps, 1)

                # Use B-splines for interpolation
                t, c, k = splrep(sorted_elevs, sorted_temps, k=1)  # Fit a B-spline with degree 3
                spline = BSpline(t, c, k, extrapolate=False)  # Create the B-spline object without extrapolation

                # Evaluate the spline over the elevation range
                interpolated_values = spline(elevation_range)

                # Extrapolate using the derivative of all available measurements
                below_min = elevation_range < sorted_elevs[0]
                above_max = elevation_range > sorted_elevs[-1]

                interpolated_values[below_min] = slope * (elevation_range[below_min] - sorted_elevs[0]) + sorted_temps[0]
                interpolated_values[above_max] = slope * (elevation_range[above_max] - sorted_elevs[-1]) + sorted_temps[-1]

                interpolated_temp_df.loc[index] = interpolated_values


    

    
    
    


    # Find the elevation closest to 0°C for each timestep
    # Find the elevation closest to 0°C for each timestep
    plotting = True
    if plotting == True: 

            #Calculate the number of measurements for each timestep
        temp_elev_df['Measurement_Count'] = temp_elev_df.filter(like='_Temperature').notna().sum(axis=1)

        # Plot the number of measurements over time
        plt.figure(figsize=(10, 6))
        plt.plot(temp_elev_df.index, temp_elev_df['Measurement_Count'], color='tab:blue', label='Measurement Count')
        plt.title('Number of Measurements Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Measurements')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        zero_deg_elevation = interpolated_temp_df.apply(lambda row: elevation_range[np.where(row <= 0)[0][0]] if any(row <= 0) else np.nan, axis=1)
        min_two_deg_elevation = interpolated_temp_df.apply(lambda row: elevation_range[np.where(row <= -2)[0][0]] if any(row <= -2) else np.nan, axis=1)
        two_deg_elevation = interpolated_temp_df.apply(lambda row: elevation_range[np.where(row >= 2)[0][0]] if any(row >= 2) else np.nan, axis=1)

        # Add the zero-degree elevation as a new column in the interpolated_temp_df
        interpolated_temp_df['Zero_Deg_Elevation'] = zero_deg_elevation
        interpolated_temp_df['Min_Two_Deg_Elevation'] = min_two_deg_elevation
        interpolated_temp_df['Two_Deg_Elevation'] = two_deg_elevation

        
        # Add the daily zero-degree elevation to the interpolated_temp_df
        interpolated_temp_df['Daily_Zero_Deg_Elevation'] = interpolated_temp_df['Zero_Deg_Elevation'].groupby(interpolated_temp_df.index.date).transform('mean')
        interpolated_temp_df['Daily_Min_Two_Deg_Elevation'] = interpolated_temp_df['Min_Two_Deg_Elevation'].groupby(interpolated_temp_df.index.date).transform('mean')
        interpolated_temp_df['Daily_Two_Deg_Elevation'] = interpolated_temp_df['Two_Deg_Elevation'].groupby(interpolated_temp_df.index.date).transform('mean')


        # Plot the zero-degree elevation over time
        plt.figure(figsize=(10, 6))
        plt.plot(interpolated_temp_df.index, interpolated_temp_df['Daily_Zero_Deg_Elevation'], label='Zero-Degree Elevation', color='tab:blue', linewidth=1.5)
        
        plt.title('Zero-Degree isotherm Over Time')
        plt.xlabel('Date')
        plt.ylabel('Elevation (m)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.title('Zero-Degree isotherm Over Time hourly')
        plt.xlabel('Date')
        plt.ylabel('Elevation (m)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Set values below 2000m to NaN
        interpolated_temp_df.loc[:, :] = np.where(
            (interpolated_temp_df < zero_deg_elevation_min) | (interpolated_temp_df > zero_deg_elevation_max),
            np.nan,
            interpolated_temp_df
        )

        interpolated_temp_df.loc[:, :] = np.where(
            (interpolated_temp_df < two_deg_elevation_min) | (interpolated_temp_df > two_deg_elevation_max),
            np.nan, interpolated_temp_df)

        interpolated_temp_df.loc[:, :] = np.where(
            (interpolated_temp_df < min_two_deg_elevation_min) | (interpolated_temp_df > min_two_deg_elevation_max),
            np.nan, interpolated_temp_df)
        
        # Plot the zero-degree elevation over time
        plt.figure(figsize=(10, 6))
        plt.plot(interpolated_temp_df.index, interpolated_temp_df['Daily_Zero_Deg_Elevation'], label='0$^\circ$C', color='tab:blue', linewidth=1.5)
        plt.plot(interpolated_temp_df.index, interpolated_temp_df['Daily_Min_Two_Deg_Elevation'], label='-2$^\circ$C ', color='tab:orange', linewidth=1.5)
        #plt.plot(interpolated_temp_df.index, interpolated_temp_df['Daily_Two_Deg_Elevation'], label='2$^\circ$C ', color='tab:red', linewidth=1.5)

        plt.title('Daily Zero-Degree Elevation Over Time')
        plt.xlabel('Date')
        plt.ylabel('Elevation (m)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.title('Isotherms')
        plt.xlabel('Date')
        plt.ylabel('Elevation (m)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return interpolated_temp_df

# Call the zero_deg_altitude function and save the output
interpolated_temp_df = zero_deg_altitude(temp_merged_dfs)

# Save the output to a pickle file
output_pickle_path = r'../data/interpolated_temp_df.pkl'
with open(output_pickle_path, 'wb') as f:
    pickle.dump(interpolated_temp_df, f)

print(f"Interpolated temperature data saved to {output_pickle_path}")