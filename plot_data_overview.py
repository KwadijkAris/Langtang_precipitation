"""Data-availability overview figure (data coverage timeline + station
photos). Saves results/Method/data_availability.svg/.pdf.

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
import os

from station_data import (get_dir, read_csv, get_elevation, read_pluvio_cleaned,
                          read_AWS, _DATA_DIR, _RESULTS_DIR, _DATA_OVERVIEW_TXT)


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
    'Morimoto TB': 'TB10',
    'Kyangjin AWS': 'PLU1',
    'Yala BC AWS': 'PLU2',
    'Morimoto Pluvio': 'PLU3',
    'Langshisha Pluvio': 'PLU4',
    'SNOWAMP_lower': 'T1',
    'SNOWAMP_middle': 'T2',
    'snowAMP Ganjala upper': 'T3',
    'Ganja La Pluvio': 'T4',
    'Yala Pluvio': 'T5',
    'Yala Glacier AWS': 'T6',
    'Langtang Glacier AWS': 'T7'
}


def get_data_overview(file_path, station_abbrev, font_scale=1.0):
    import matplotlib.dates as mdates

    df_meta = read_csv(file_path)  

    # Define station groups by variable type
    tb_stations = ['Ganja La TB1','Ganja La TB2','Ganja La TB3','Jathang TB','Kyangjin TB','Langshisha BC TB','Morimoto TB','Numthang TB','Shalbachum TB', 'Syabru TB', 'Lama TB']
    temp_stations = ['Ganja La Pluvio', 'Yala Pluvio','Langshisha Pluvio', 'Morimoto Pluvio','Kyangjin AWS', 'Yala BC AWS', 'Yala Glacier AWS','Langtang Glacier AWS','snowAMP Ganjala upper','SNOWAMP_lower','SNOWAMP_middle']
    wind_stations = ['Kyangjin AWS', 'Yala BC AWS', 'Morimoto Pluvio']
    precip_stations_all = ['Kyangjin AWS','Yala BC AWS','Langshisha Pluvio','Morimoto Pluvio', 'Ganja La Pluvio', 'Yala Pluvio']
    rh_stations = ['Kyangjin AWS', 'Yala BC AWS', 'Morimoto Pluvio']
    pressure_stations = ['Kyangjin AWS','Morimoto Pluvio']
    swnet_stations = ['Kyangjin AWS', 'Yala BC AWS']

    # Master list of stations to process via read_pluvio_cleaned.
    # NOTE: the glacier AWS stations are NOT in this list — they have no file in
    # Cleaned\Pluvio (read_pluvio_cleaned would raise FileNotFoundError); their
    # temperature is loaded from Cleaned\Temperature in a separate section below.
    station_names = [
        'Ganja La Pluvio','Yala Pluvio','Kyangjin AWS','Yala BC AWS',
        'Langshisha Pluvio','Morimoto Pluvio','snowAMP Ganja La','snowAMP Ganjala upper','SNOWAMP_lower','SNOWAMP_middle',
        'Ganja La TB1','Ganja La TB2','Ganja La TB3','Jathang TB','Langshisha BC TB','Morimoto TB','Numthang TB','Shalbachum TB', 'Syabru TB', 'Lama TB'
    ]
    
    # Filter metadata
    df_meta = df_meta[df_meta['name'].isin(station_names)].copy()
    
    # Store data segments here
    plot_data_overview = pd.DataFrame(columns=['elev', 'start_date', 'end_date', 'name', 'variable', 'type'])

    # Helper function to detect monthly coverage gaps (< 80%)
    def add_ranges_with_monthly_gaps(series, var_name, elev, station_name):
        if series.isnull().all():
            return []
        
        # 1. Resample to calculate monthly coverage
        coverage_check = series.notna().astype(int)
        monthly_coverage = coverage_check.resample('ME').mean()
        valid_months = monthly_coverage[monthly_coverage >= 0.8].index
        
        if valid_months.empty:
            return []

        valid_months_series = pd.Series(valid_months, index=valid_months)
        diffs = valid_months_series.diff()
        groups = (diffs > pd.Timedelta(days=32)).cumsum()
        
        local_plot_data = []

        for _, month_group in valid_months_series.groupby(groups):
            start_month = month_group.iloc[0]
            end_month = month_group.iloc[-1]
            blk_start = start_month - pd.offsets.MonthBegin(1)
            blk_end = end_month 
            
            local_plot_data.append({
                'elev': elev,
                'start_date': blk_start,
                'end_date': blk_end,
                'name': station_name,
                'variable': var_name,
                'type': 'Data'
            })
            
        return local_plot_data

    # 1. Process "cleaned data"
    cleaned_data = read_pluvio_cleaned(station_names)

    for station_name, station_df in cleaned_data.items():
        if station_df.empty: continue
        try:
            elev = get_elevation([station_name])[0]
        except: continue

        station_df = station_df.sort_index()
        rows_to_add = []

        # -- Rainfall / Precipitation --
        rain_col = None
        if 'Rainfall_15min' in station_df.columns: rain_col = 'Rainfall_15min'
        elif 'Rainfall_1H' in station_df.columns: rain_col = 'Rainfall_1H'
        elif 'CS725_Swek(mm)' in station_df.columns: rain_col = 'CS725_Swek(mm)'

        if rain_col:
            if station_name in tb_stations: var = 'Rain'
            elif station_name in precip_stations_all: var = 'Precipitation (rain + snow)'
            else: var = None
            
            if var:
                new_data = add_ranges_with_monthly_gaps(station_df[rain_col], var, elev, station_name)
                rows_to_add.extend(new_data)

        # -- Temperature --
        temp_col = None
        if 'Temperature_15min' in station_df.columns: temp_col = 'Temperature_15min'
        elif 'Temperature' in station_df.columns: temp_col = 'Temperature'
        elif 'Temperature_1H' in station_df.columns: temp_col = 'Temperature_1H'

        if temp_col and (station_name in temp_stations):
                new_data = add_ranges_with_monthly_gaps(station_df[temp_col], 'Temperature', elev, station_name)
                rows_to_add.extend(new_data)

        # -- SWnet (Incoming Shortwave Radiation) --
        swnet_col = None
        if 'KINC' in station_df.columns: swnet_col = 'KINC'
        elif 'Shortwave_In' in station_df.columns: swnet_col = 'Shortwave_In'

        if swnet_col and station_name in swnet_stations:
            new_data = add_ranges_with_monthly_gaps(station_df[swnet_col], 'SW', elev, station_name)
            rows_to_add.extend(new_data)
        
        if rows_to_add:
            plot_data_overview = pd.concat([plot_data_overview, pd.DataFrame(rows_to_add)], ignore_index=True)

    # 2. Process Extra Sensors (RH, Wind, Pressure)
    extra_sensor_paths = {
        'Kyangjin AWS': str(_DATA_DIR / "Moisture" / "Kyangjin_AWS_humidity_timeseries.csv"),
        'Yala BC AWS': str(_DATA_DIR / "Moisture" / "Yala_BC_AWS_humidity_timeseries.csv"),
        'Morimoto Pluvio': str(_DATA_DIR / "Moisture" / "Morimoto_MM_humidity_timeseries.csv")
    }

    for station_name, f_path in extra_sensor_paths.items():
        try:
            df_extra = pd.read_csv(f_path)
            df_extra['DATETIME'] = pd.to_datetime(df_extra['DATETIME'])
            df_extra = df_extra.set_index('DATETIME').sort_index()
            elev = get_elevation([station_name])[0]

            # RH
            if 'RH' in df_extra.columns and station_name in rh_stations:
                new_data = add_ranges_with_monthly_gaps(df_extra['RH'], 'RH', elev, station_name)
                if new_data:
                    plot_data_overview = pd.concat([plot_data_overview, pd.DataFrame(new_data)], ignore_index=True)

            # Wind
            if 'Wind_Speed' in df_extra.columns and station_name in wind_stations:
                new_data = add_ranges_with_monthly_gaps(df_extra['Wind_Speed'], 'Wind', elev, station_name)
                if new_data:
                    plot_data_overview = pd.concat([plot_data_overview, pd.DataFrame(new_data)], ignore_index=True)

            # Pressure
            if 'Rel_Air_Press' in df_extra.columns and station_name in pressure_stations:
                new_data = add_ranges_with_monthly_gaps(df_extra['Rel_Air_Press'], 'Pressure', elev, station_name)
                if new_data:
                    plot_data_overview = pd.concat([plot_data_overview, pd.DataFrame(new_data)], ignore_index=True)

        except Exception as e:
            print(f"Skipping extra sensors for {station_name}: {e}")

    # 2b. Glacier AWS temperature (no pluvio file; cleaned temperature CSV only)
    glacier_temp_paths = {
        'Yala Glacier AWS': str(_DATA_DIR / "Cleaned" / "Temperature" / "Yala Glacier AWS_temperature.csv"),
        # 'Langtang Glacier AWS': no cleaned temperature file available yet
    }
    for station_name, f_path in glacier_temp_paths.items():
        try:
            df_gl = pd.read_csv(f_path)
            df_gl['DATETIME'] = pd.to_datetime(df_gl['DATETIME'])
            df_gl = df_gl.set_index('DATETIME').sort_index()
            temp_col = next((c for c in ['TEMP', 'Avg_Temp_H', 'Temperature']
                             if c in df_gl.columns
                             and pd.to_numeric(df_gl[c], errors='coerce').notna().any()),
                            None)
            if temp_col is None:
                print(f"Skipping {station_name}: no valid temperature column")
                continue
            s_temp = pd.to_numeric(df_gl[temp_col], errors='coerce')
            elev = get_elevation([station_name])[0]
            new_data = add_ranges_with_monthly_gaps(s_temp, 'Temperature', elev, station_name)
            if new_data:
                plot_data_overview = pd.concat([plot_data_overview, pd.DataFrame(new_data)], ignore_index=True)
        except Exception as e:
            print(f"Skipping glacier AWS temperature for {station_name}: {e}")

    # 3. Process SWnet from AWS raw data
    aws_swnet_stations = ['Kyangjin AWS', 'Yala BC AWS']
    for station_name in aws_swnet_stations:
        try:
            file_paths = get_dir([station_name])
            if not file_paths or not file_paths[0]:
                continue
            aws_data = read_AWS(file_paths[0])
            if 'DATETIME' in aws_data.columns:
                aws_data['DATETIME'] = pd.to_datetime(aws_data['DATETIME'])
                aws_data = aws_data.set_index('DATETIME').sort_index()
            kinc_col = None
            if 'KINC' in aws_data.columns: kinc_col = 'KINC'
            elif 'Shortwave_In' in aws_data.columns: kinc_col = 'Shortwave_In'
            if kinc_col:
                elev = get_elevation([station_name])[0]
                new_data = add_ranges_with_monthly_gaps(aws_data[kinc_col], 'SW', elev, station_name)
                if new_data:
                    plot_data_overview = pd.concat([plot_data_overview, pd.DataFrame(new_data)], ignore_index=True)
        except Exception as e:
            print(f"Skipping SW for {station_name}: {e}")

    # De-duplicate: keep unique rows per station+variable+start_date
    plot_data_overview = plot_data_overview.drop_duplicates(subset=['name', 'variable', 'start_date'])

    # ── Zebra-band timeline, plot_seasonal_diurnal_compact styling ────────────
    from matplotlib.patches import Patch

    # Sized for copernicus.cls: \textwidth = 177 mm, so the figure is built at
    # its final print size and included with \includegraphics[width=\textwidth]
    # (no LaTeX rescaling). Font sizes are true points; captions in the class
    # are \small (~9 pt), so 8 pt figure text sits one step below the caption.
    TEXTWIDTH_IN = 177 / 25.4          # 6.97 in
    FS_TICK, FS_LABEL, FS_LEGEND = 8.0, 9.0, 8.0

    # Pastel palette (muted versions of the compact-figure hues); the dict order
    # is also the fixed slot order.
    VAR_ORDER = ['Rain', 'Precipitation (rain + snow)', 'Temperature', 'RH',
                 'Wind', 'Pressure', 'SW']
    variable_colors = {
        'Rain':                        '#7ba7d4',
        'Precipitation (rain + snow)': '#c3dcee',
        'Temperature':                 '#e89f7c',
        'RH':                          '#b5a3cc',
        'Wind':                        '#93c6a2',
        'Pressure':                    '#c4c4c4',
        'SW':                          '#e9d18d',
        'Other':                       '#bbbbbb',
    }
    VAR_LABEL = {
        'Rain': 'Rain', 'Precipitation (rain + snow)': 'Precipitation (rain + snow)',
        'Temperature': 'Temperature', 'RH': 'RH', 'Wind': 'Wind',
        'Pressure': 'Pressure', 'SW': 'SW',
    }

    # Sort stations: elevation ASCENDING (low elevation at bottom)
    stations_info = plot_data_overview[['name', 'elev']].drop_duplicates()
    stations_info = stations_info.sort_values(by=['elev', 'name'], ascending=[True, True])
    unique_stations = stations_info['name'].tolist()

    # Lane layout: one lane per station-variable, fixed slot order per station.
    LANE_H, GROUP_GAP = 1.0, 1.1
    lanes, groups = [], []
    y = 0.0
    for st in unique_stations:
        vars_here = [v for v in VAR_ORDER
                     if v in plot_data_overview.loc[plot_data_overview['name'] == st,
                                                    'variable'].unique()]
        vars_here = vars_here[::-1]          # first-in-order ends up on top
        y_start = y
        for v in vars_here:
            lanes.append({'station': st, 'var': v, 'y': y})
            y += LANE_H
        y_end = y
        elev = stations_info.loc[stations_info['name'] == st, 'elev'].values[0]
        groups.append({'station': st, 'y0': y_start - LANE_H / 2,
                       'y1': y_end - LANE_H / 2,
                       'yc': (y_start + y_end - LANE_H) / 2, 'elev': elev})
        y += GROUP_GAP
    n_lanes = len(lanes)
    y_min = -GROUP_GAP
    y_max = lanes[-1]['y'] + LANE_H / 2 + GROUP_GAP / 2

    # Row heights (inches): timeline scales with the lane count (~0.13 in per
    # lane), then a thin legend strip, then the two instrument photos. The
    # total stays below \textheight (54 baselines ≈ 9 in) so figure + caption
    # still fit on one page.
    timeline_h = min(max(4.0, n_lanes * 0.13 + 1.3), 5.4)
    LEGEND_H, PHOTO_H = 0.45, 2.3
    fig_h = timeline_h + LEGEND_H + PHOTO_H
    fig = plt.figure(figsize=(TEXTWIDTH_IN, fig_h), facecolor='white')
    gs = fig.add_gridspec(3, 2, height_ratios=[timeline_h, LEGEND_H, PHOTO_H],
                          hspace=0.15, wspace=0.05)
    ax1 = fig.add_subplot(gs[0, :])
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis('off')
    ax_photos = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

    # Alternating station background bands (kept faint for a minimal look)
    for i, g in enumerate(groups):
        if i % 2 == 0:
            pad = GROUP_GAP / 2
            ax1.axhspan(g['y0'] - pad, g['y1'] + pad, color='#f6f7f9', zorder=0)

    # Bars
    for lane in lanes:
        rows = plot_data_overview[(plot_data_overview['name'] == lane['station']) &
                                  (plot_data_overview['variable'] == lane['var'])]
        color = variable_colors.get(lane['var'], variable_colors['Other'])
        for _, row in rows.iterrows():
            duration_days = max((row['end_date'] - row['start_date']).days, 1)
            ax1.barh(lane['y'], duration_days, left=row['start_date'],
                     height=LANE_H * 0.72, color=color,
                     edgecolor='white', linewidth=0.4, align='center', zorder=3)

    # X axis: horizontal year labels, yearly minor ticks
    ax1.set_xlim(pd.Timestamp('2012-01-01'), pd.Timestamp('2025-01-01'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax1.tick_params(axis='x', labelsize=FS_TICK*font_scale, rotation=0)
    ax1.tick_params(axis='x', which='minor', length=2.5)
    ax1.grid(True, axis='x', which='major', linestyle=':', alpha=0.2, zorder=1)

    # Y axis: station abbreviations (left), elevations (right)
    ax1.set_ylim(y_min, y_max)
    ax1.set_yticks([g['yc'] for g in groups])
    ax1.set_yticklabels([STATION_ABBREV.get(g['station'], g['station']) for g in groups],
                        fontsize=FS_TICK*font_scale)
    ax1.tick_params(axis='y', length=0)
    # Full box around the timeline panel
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    ax1_elev = ax1.twinx()
    ax1_elev.set_ylim(ax1.get_ylim())
    ax1_elev.set_yticks([g['yc'] for g in groups])
    ax1_elev.set_yticklabels([f"{int(g['elev'])} m" for g in groups],
                             fontsize=FS_TICK*font_scale, color='#555555')
    ax1_elev.tick_params(axis='y', length=0)
    for spine in ax1_elev.spines.values():
        spine.set_visible(False)

    # Legend: horizontal strip between the timeline and the photo row — at
    # 177 mm total width a right-hand legend would cost ~17 % of the timeline.
    legend_handles = [Patch(facecolor=variable_colors[v], label=VAR_LABEL[v])
                      for v in VAR_ORDER]
    ax_leg.legend(handles=legend_handles, loc='center', ncol=4,
                  fontsize=FS_LEGEND*font_scale, frameon=False,
                  handlelength=1.4, handleheight=0.9, columnspacing=1.2)

    # ── Photo panels (b), (c) ─────────────────────────────────────────────
    import matplotlib.image as mpimg
    PHOTO_DIR = _DATA_DIR / "Results" / "Photos"
    photo_specs = [('Pluvio.png', '(b)'), ('Tipping_bucket.png', '(c)')]
    for axp, (fname, lab) in zip(ax_photos, photo_specs):
        try:
            img = mpimg.imread(str(PHOTO_DIR / fname))
            axp.imshow(img)
        except FileNotFoundError:
            axp.text(0.5, 0.5, f'{fname} not found', ha='center', va='center',
                     fontsize=FS_TICK*font_scale)
        axp.set_title(lab, loc='left', fontsize=FS_LABEL*font_scale,
                      fontweight='bold', pad=3)
        # Box around the photo: keep the axes frame, drop ticks/labels
        axp.set_xticks([])
        axp.set_yticks([])
        for spine in axp.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
    ax1.set_title('(a)', loc='left', fontsize=FS_LABEL*font_scale,
                  fontweight='bold', pad=4)

    # No bbox_inches='tight' at save time, so the file keeps exactly
    # \textwidth and fonts print at true size.
    plt.tight_layout()

    # Save as PDF for pdflatex/Overleaf (SVG kept for manual editing)
    output_dir = str(_RESULTS_DIR / "Method")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / "data_availability.svg"
    fig.savefig(str(output_file), format='svg')
    fig.savefig(str(output_file.with_suffix('.pdf')))
    print(f"Figure saved to {output_file} (+ .pdf)")

    plt.show()


if __name__ == '__main__':
    get_data_overview(_DATA_OVERVIEW_TXT, STATION_ABBREV, font_scale=1.5)
