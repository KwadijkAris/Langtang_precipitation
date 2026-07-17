"""Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from station_data import get_station_coordinate
from station_data import get_elevation
import os
import rasterio
from shapely.geometry import LineString, Point
import pandas as pd
from rasterio.plot import show
import traceback
import matplotlib.pyplot as plt

station_transverse = ['Kyangjin AWS', 'Jathang TB', 'Numthang TB', 'Langshisha BC TB', 'Morimoto TB', 'Morimoto Pluvio']
stations_perpendicular_entrance = ['Kyangjin AWS',  'Ganja La TB1', 'Ganja La TB2', 'Ganja La TB3']
stations_perpendicular2 = ['Shalbachum TB', 'Langshisha BC TB','Langshisha Pluvio']

# # Example Usage:
# stations_grad = ['Syabru TB', 'Lama TB', 'Kyangjin AWS', 'Langshisha BC TB',  'Jathang TB', 'Numthang TB', 'Morimoto TB', 'Morimoto Pluvio']
# profile_shp = r"../data/raw/ArcGIS/Profile4326.shp" # Adjust path as necessary
# calculate_daily_precipitation_gradient(stations_grad, profile_shp)
def analyze_valley_geometry(plot_map=True):
    """
    Calculates valley cross-sectional areas, top widths, and slopes along a profile.
    Generates plots and saves results to CSV and Shapefile.
    
    Args:
        plot_map (bool): If True, includes a map subplot with the DEM and vector overlays.
                         If False, only plots the profile characteristics charts.
    """
    # Station abbreviation mapping for display labels
    STATION_ABBREV = {
        'Syabru TB': 'TB1',
        'Lama TB': 'TB2',
        'Ganja La TB1': 'TB3',
        'Ganja La TB2': 'TB4',
        'Ganja La TB3': 'TB5',
        'Jathang TB': 'TB6',
        'Numthang TB': 'TB7',
        'Langshisha BC TB': 'TB8',
        'Shalbachum TB': 'TB9',
        'Langshisha Pluvio TB': 'TB10',
        'Morimoto TB': 'TB11',
        'snowAMP Ganja La': 'PLU1',
        'Kyangjin AWS': 'PLU2',
        'Yala BC AWS': 'PLU3',
        'Morimoto Pluvio': 'PLU4',
        'Langshisha Pluvio': 'PLU5',
        'snowAMP Ganja La lower': 'T1',
        'snowAMP Ganja La middle': 'T2',
        'snowAMP Ganja La upper': 'T3',
        'Yala Pluvio': 'T4',
        'Yala Glacier AWS': 'T5',
        'Langtang Glacier AWS': 'T6'
    }
    
    try:
        dem_path_map = r"../data/raw/ArcGIS/multidem4326.tif"
        profile_shp_map = r"../data/raw/ArcGIS/Profile4326.shp"
        
        with rasterio.open(dem_path_map) as dem:
            profile_gdf_map = gpd.read_file(profile_shp_map)
            
            # Ensure CRS match
            if profile_gdf_map.crs != dem.crs:
                profile_gdf_map = profile_gdf_map.to_crs(dem.crs)

            # Extract the LineString geometry
            profile_line_map = profile_gdf_map.geometry.iloc[0]
            if 'Multi' in profile_line_map.geom_type:
                profile_line_map = profile_line_map.geoms[0]

            # Create a high-resolution set of points along the line
            distances_m = np.arange(0, profile_line_map.length, 0.01)
            points_on_line = [profile_line_map.interpolate(d) for d in distances_m]
            coords = [(p.x, p.y) for p in points_on_line]
            
            # Sample DEM for profile elevations
            elevations = [z[0] for z in dem.sample(coords)]
            points_gdf = gpd.GeoDataFrame(geometry=points_on_line, crs=dem.crs)

            # Basic Parameters for Cross-section calculation
            dz = 100  # Vertical step size in meters
            max_rel_height = 1000  # Calculate up to 1000m above the valley floor
            
            # Search parameters for walls
            max_search_dist = 0.04 # degrees (approx 3km)
            search_step = 0.0001 # degrees 

            def get_dist_to_wall(angle, start_x, start_y, target_z, dem_dataset):
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                # Create array of sample points
                dists = np.arange(0, max_search_dist, search_step)
                for dist in dists:
                    px = start_x + dist * cos_a
                    py = start_y + dist * sin_a
                    try:
                        z_val = next(dem_dataset.sample([(px, py)]))[0]
                        if z_val >= target_z:
                            return dist
                    except (IndexError, StopIteration):
                        return dist
                return max_search_dist

            def deg_scalar_to_m(dist_deg, g_angle, m_lat, m_lon):
                """Convert scalar distance in degrees along a specific grid angle to meters."""
                dx_d = dist_deg * np.cos(g_angle)
                dy_d = dist_deg * np.sin(g_angle)
                return np.sqrt((dx_d * m_lon)**2 + (dy_d * m_lat)**2)

            cross_sectional_areas = []
            real_distances_m = [0] 

            # --- Loop 1: Calculate Cross-Sectional Areas ---
            print("Calculating cross-sectional areas...")
            for i, point in enumerate(points_on_line):
                
                # Local Metric Conversion Factors 
                lat_rad = np.radians(point.y)
                m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
                m_per_deg_lon = 111132.954 * np.cos(lat_rad)

                # Cumulative Distance 
                if i > 0:
                    prev = points_on_line[i-1]
                    dx_seg_deg = point.x - prev.x
                    dy_seg_deg = point.y - prev.y
                    step_dist_m = np.sqrt((dx_seg_deg * m_per_deg_lon)**2 + (dy_seg_deg * m_per_deg_lat)**2)
                    real_distances_m.append(real_distances_m[-1] + step_dist_m)

                # Geometry / Angle Calculations
                if i == 0:
                    p_next = points_on_line[i+1]
                    dx, dy = p_next.x - point.x, p_next.y - point.y
                elif i == len(points_on_line) - 1:
                    p_prev = points_on_line[i-1]
                    dx, dy = point.x - p_prev.x, point.y - p_prev.y
                else:
                    p_prev, p_next = points_on_line[i-1], points_on_line[i+1]
                    dx, dy = p_next.x - p_prev.x, p_next.y - p_prev.y

                dx_m = dx * m_per_deg_lon
                dy_m = dy * m_per_deg_lat
                angle_rad_metric = np.arctan2(dy_m, dx_m)
                
                perp_angle_left_metric = angle_rad_metric + np.pi / 2
                perp_angle_right_metric = angle_rad_metric - np.pi / 2
                
                def get_grid_angle(metric_angle, m_lat, m_lon):
                    u_m = np.cos(metric_angle)
                    v_m = np.sin(metric_angle)
                    u_deg = u_m / m_lon
                    v_deg = v_m / m_lat
                    return np.arctan2(v_deg, u_deg)

                grid_angle_left = get_grid_angle(perp_angle_left_metric, m_per_deg_lat, m_per_deg_lon)
                grid_angle_right = get_grid_angle(perp_angle_right_metric, m_per_deg_lat, m_per_deg_lon)

                floor_elevation = elevations[i]
                current_total_area = 0
                
                # Step upwards to calculate area
                for h_rel in np.arange(0, max_rel_height, dz):
                    target_elev = floor_elevation + h_rel + (dz/2)
                    dist_left_deg = get_dist_to_wall(grid_angle_left, point.x, point.y, target_elev, dem)
                    dist_right_deg = get_dist_to_wall(grid_angle_right, point.x, point.y, target_elev, dem)
                
                    width_left_m = deg_scalar_to_m(dist_left_deg, grid_angle_left, m_per_deg_lat, m_per_deg_lon)
                    width_right_m = deg_scalar_to_m(dist_right_deg, grid_angle_right, m_per_deg_lat, m_per_deg_lon)

                    width = width_left_m + width_right_m
                    current_total_area += width * dz
                
                cross_sectional_areas.append(current_total_area)

            distances_m = np.array(real_distances_m)

            # --- Loop 2: Calculate Visual Extents (Top Width) ---
            print("Calculating visual extents and valley widths...")
            cross_section_lines = []
            top_widths_m = []

            for i, point in enumerate(points_on_line):
                # Recalculate metrics for this loop context (could be optimized, but safer to re-derive)
                lat_rad = np.radians(point.y)
                m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
                m_per_deg_lon = 111132.954 * np.cos(lat_rad)
                
                if i < len(points_on_line) - 1:
                    dx = points_on_line[i+1].x - point.x
                    dy = points_on_line[i+1].y - point.y
                elif i > 0: 
                    dx = point.x - points_on_line[i-1].x
                    dy = point.y - points_on_line[i-1].y
                else:
                    dx, dy = 1, 0
                
                angle_rad_metric = np.arctan2(dy * m_per_deg_lat, dx * m_per_deg_lon)
                # Re-define local helper to ensure scope availability
                def get_grid_angle_vis(metric_angle, m_lat, m_lon):
                    u_m = np.cos(metric_angle)
                    v_m = np.sin(metric_angle)
                    return np.arctan2((v_m / m_lat), (u_m / m_lon))
                
                grid_angle_left = get_grid_angle_vis(angle_rad_metric + np.pi/2, m_per_deg_lat, m_per_deg_lon)
                grid_angle_right = get_grid_angle_vis(angle_rad_metric - np.pi/2, m_per_deg_lat, m_per_deg_lon)
                
                # Target height for width calculation (Visual Line)
                target_elev_top = elevations[i] + max_rel_height

                dist_left_top = get_dist_to_wall(grid_angle_left, point.x, point.y, target_elev_top, dem)
                dist_right_top = get_dist_to_wall(grid_angle_right, point.x, point.y, target_elev_top, dem)
                
                # Geometry for map
                p_left_x = point.x + dist_left_top * np.cos(grid_angle_left)
                p_left_y = point.y + dist_left_top * np.sin(grid_angle_left)
                p_right_x = point.x + dist_right_top * np.cos(grid_angle_right)
                p_right_y = point.y + dist_right_top * np.sin(grid_angle_right)
                
                cross_section_lines.append(LineString([(p_left_x, p_left_y), (p_right_x, p_right_y)]))
                
                # Calculate width in meters for plotting
                w_left = deg_scalar_to_m(dist_left_top, grid_angle_left, m_per_deg_lat, m_per_deg_lon)
                w_right = deg_scalar_to_m(dist_right_top, grid_angle_right, m_per_deg_lat, m_per_deg_lon)
                top_widths_m.append(w_left + w_right)

            cs_lines_gdf = gpd.GeoDataFrame(geometry=cross_section_lines, crs=dem.crs)

            # --- Export Width Lines as Shapefile ---
            output_shp_path = r"../data/raw/ArcGIS/calculated_width_lines.shp"
            try:
                cs_lines_gdf.to_file(output_shp_path)
                print(f"Exported width lines to {output_shp_path}")
            except Exception as e:
                print(f"Could not save shapefile: {e}")

            # --- Apply Plotting Limits (40.65 km) ---
            limit_km = 46
            dist_km = distances_m / 1000
            mask = dist_km <= limit_km
            
            # Filter all data arrays
            dist_km = dist_km[mask]
            distances_m = distances_m[mask]
            elevations = np.array(elevations)[mask]
            top_widths_m = np.array(top_widths_m)[mask]
            cross_sectional_areas = np.array(cross_sectional_areas)[mask]
            points_gdf = points_gdf[mask]

            # --- Plotting Phase ---
            print("Generating plots...")
            station_names_map = ['Syabru TB', 'Lama TB', 'Kyangjin AWS', 'Jathang TB', 'Numthang TB', 'Langshisha BC TB', 'Morimoto Pluvio', 'Morimoto TB']
            lon_map, lat_map = get_station_coordinate(station_names_map)
            lon_map, lat_map = np.array(lon_map)/100000, np.array(lat_map)/100000 
            
            stations_gdf_map = gpd.GeoDataFrame(
                {'Station': station_names_map},
                geometry=gpd.points_from_xy(lon_map, lat_map),
                crs="EPSG:4326"
            ).to_crs(dem.crs)
            
            station_elevations = get_elevation(station_names_map)
            
            station_indices = []
            for s_geom in stations_gdf_map.geometry:
                if points_gdf.empty: break
                dists_along_line = points_gdf.distance(s_geom)
                if dists_along_line.min() < 0.06:
                    closest_idx = dists_along_line.argmin()
                    station_indices.append((closest_idx, True))
                else:
                    station_indices.append((0, False))

            # Structure: Two subplots only (Elevation and Valley Width)
            if plot_map:
                fig = plt.figure(figsize=(16, 16))
                gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.3)
                
                ax_map = fig.add_subplot(gs[0])
                ax_profile = fig.add_subplot(gs[1])
                ax_width = fig.add_subplot(gs[2], sharex=ax_profile)

                # 1. Map Plot with grayscale DEM
                show(dem, ax=ax_map, cmap='gray', alpha=0.9)
                # Plot visual extent lines in dark gray
                cs_lines_gdf[mask].plot(ax=ax_map, color='#505050', linewidth=0.5, alpha=0.6, zorder=2, label=f'Valley Width (at +{max_rel_height}m)')
                # Plot filtered profile line in black
                points_gdf.plot(ax=ax_map, color='black', markersize=0.1, zorder=3, label='Profile Line')
                
                # Plot stations in red
                stations_gdf_map.plot(ax=ax_map, color='red', marker='o', markersize=100, zorder=10, edgecolor='darkred', linewidth=1.5, label='Weather Stations')
                for x, y, label in zip(stations_gdf_map.geometry.x, stations_gdf_map.geometry.y, stations_gdf_map.Station):
                    if points_gdf.distance(Point(x, y)).min() < 0.06:
                        abbrev = STATION_ABBREV.get(label, label)
                        ax_map.text(x, y, '  ' + abbrev, fontsize=31, color='darkred', fontweight='bold', va='center', zorder=11, 
                                    path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="white")])

                ax_map.spines['top'].set_visible(False)
                ax_map.spines['right'].set_visible(False)
                ax_map.spines['left'].set_visible(False)
                ax_map.spines['bottom'].set_visible(False)
                ax_map.legend(loc='upper right', frameon=False, fontsize=27)
                ax_map.grid(True, linestyle='--', alpha=0.3, color='gray')
            else:
                fig = plt.figure(figsize=(12, 12))
                gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.2], hspace=0.25)
                ax_profile = fig.add_subplot(gs[0])
                ax_width = fig.add_subplot(gs[1], sharex=ax_profile)

            # 2. Elevation Profile - Schematic style with black and gray
            ax_profile.plot(dist_km, elevations, 'k-', linewidth=3, zorder=5)
            ax_profile.fill_between(dist_km, elevations, min(elevations)-200, color='#b0b0b0', alpha=0.4, zorder=1)
            ax_profile.set_ylabel('Elevation (m)', fontsize=36, fontweight='bold')
            # ax_profile.grid(True, linestyle=':', alpha=0.4, color='gray')
            ax_profile.spines['top'].set_visible(False)
            ax_profile.spines['right'].set_visible(False)
            ax_profile.tick_params(labelsize=27)
            ax_profile.set_yticks([2000, 4000, 6000])
            
            plt.setp(ax_profile.get_xticklabels(), visible=False)
            
            for i, (idx, found) in enumerate(station_indices):
                if found:
                    s_name = station_names_map[i]
                    s_abbrev = STATION_ABBREV.get(s_name, s_name)
                    s_dist_km = dist_km[idx] 
                    s_elev = elevations[idx]
                    
                    # Plot station as red dot on profile
                    ax_profile.plot(s_dist_km, s_elev, 'o', color='red', markersize=24, zorder=10, 
                                   markeredgecolor='darkred', markeredgewidth=2)
                    # Station abbreviation label
                    if s_name == 'Syabru TB':
                        ax_profile.text(s_dist_km + 0.5, s_elev + 100, s_abbrev, rotation=90, ha='center', va='bottom', fontsize=27, color='darkred', fontweight='bold')
                    elif s_name == 'Morimoto TB':
                        ax_profile.text(s_dist_km - 0.5, s_elev - 200, s_abbrev, rotation=90, ha='center', va='top', fontsize=27, color='darkred', fontweight='bold')
                    else: 
                        ax_profile.text(s_dist_km, s_elev - 200, s_abbrev, rotation=90, ha='center', va='top', fontsize=27, color='darkred', fontweight='bold')
                    
                    # Plot red dots on width plot
                    ax_width.plot(s_dist_km, top_widths_m[idx], 'o', color='red', markersize=24, zorder=10, 
                                 markeredgecolor='darkred', markeredgewidth=2)

            # 3. Valley Width Plot - Schematic style
            ax_width.plot(dist_km, top_widths_m, color='#404040', linewidth=3, zorder=5)
            ax_width.fill_between(dist_km, top_widths_m, 0, color='#b0b0b0', alpha=0.4, zorder=1)
            ax_width.set_ylabel('Width (m)', fontsize=36, fontweight='bold')
            # ax_width.grid(True, linestyle=':', alpha=0.4, color='gray')
            ax_width.set_ylim(bottom=0)
            ax_width.spines['top'].set_visible(False)
            ax_width.spines['right'].set_visible(False)
            ax_width.set_xlabel('Distance (km)', fontsize=36, fontweight='bold')
            ax_width.tick_params(labelsize=27)
            ax_width.set_yticks([2000, 4000, 6000])

            # Set output directory
            output_dir_plot = r"../results/Limi_study_area"
            os.makedirs(output_dir_plot, exist_ok=True)
            
            # Save as SVG and PNG with transparent background
            output_svg_path = os.path.join(output_dir_plot, "valley_geometry_profile.svg")
            output_png_path = os.path.join(output_dir_plot, "valley_geometry_profile.png")
            
            fig.savefig(output_svg_path, format='svg', dpi=150, bbox_inches='tight', transparent=True)
            print(f"SVG saved to {output_svg_path}")
            
            fig.savefig(output_png_path, format='png', dpi=150, bbox_inches='tight', transparent=True)
            print(f"PNG saved to {output_png_path}")
            
            plt.tight_layout()
            plt.show()

            # Save Results as CSV
            output_dir_geom = r"../data/Geometry"
            os.makedirs(output_dir_geom, exist_ok=True)
            output_csv_path = os.path.join(output_dir_geom, "calculated_cross_sectional_areas_and_width.csv")

            output_area_df = pd.DataFrame({
                'distance_m': distances_m,
                'elevation_m': elevations,
                'valley_width_top_m': top_widths_m
            })
            
            output_area_df.to_csv(output_csv_path, index=False)
            print(f"Data saved to {output_csv_path}")

    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    # Regenerates data/Geometry/calculated_cross_sectional_areas_and_width.csv
    # (overwrites the shipped file, which was produced with an earlier code state)
    analyze_valley_geometry(plot_map=False)
