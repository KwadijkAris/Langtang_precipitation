"""Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0."""
"This code was used for (Kwadijk, et al 2026),  "
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from scipy import stats

import pickle
import seaborn as sns
import calendar

from station_data import get_dir
from station_data import get_elevation
from station_data import read_pluvio_cleaned
from station_data import get_station_coordinate

from station_data import read_AWS
from generate_humidity_timeseries import get_seasonal_RH_PRES_data

import os
from pathlib import Path

# Portable paths: data lives in <repo_root>/data, outputs go to <repo_root>/results
_SCRIPT_DIR  = Path(os.path.abspath(__file__)).parent
_DATA_DIR    = _SCRIPT_DIR.parent / "data"
_RESULTS_DIR = _SCRIPT_DIR.parent / "results"

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

# Load temp_merged_dfs from the pickle file
with open(_DATA_DIR / 'Cleaned' / 'Temperature' / 'temp_merged_dfs.pkl', 'rb') as f:
    temp_merged_dfs = pickle.load(f)
file_path = str(_DATA_DIR / "Data_overview" / "TEST2.txt")


def plot_percentage_below_zero(all_merged_dfs):


    # Only keep stations from STATION_ABBREV that are PLU or TB (remove T stations)
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
        'snowAMP_lower': 'T1',
        'snowAMP_middle': 'T2',
        'snowAMP Ganjala upper': 'T3',
        'Yala Pluvio': 'T4',
        'Yala Glacier AWS': 'T5',
        'Langtang Glacier AWS': 'T6'
    }
    # Only keep stations whose abbreviation starts with 'PLU' or 'TB'
    station_names_all = [name for name, abbr in STATION_ABBREV.items() if abbr.startswith('PLU') or abbr.startswith('TB')]
    elevation1 = get_elevation(station_names_all)

    # Use read_pluvio_cleaned to get the data
    all_merged_dfs = read_pluvio_cleaned(station_names_all, dt='1h')

    # Create a new DataFrame to store temperature, elevation, and month
    temp_elev_month_df = pd.DataFrame()
    
    # Loop through each station and extract relevant data
    for station, df in all_merged_dfs.items():
        if df is not None and not df.empty and 'Temperature_1H' in df.columns:
            df_copy = df.copy()
            df_copy.rename(columns={'Temperature_1H': 'TEMP'}, inplace=True)
            df_copy['Month'] = df_copy.index.month
            df_copy['Elevation'] = get_elevation([station])[0]
            df_copy['Station'] = station
            temp_elev_month_df = pd.concat([temp_elev_month_df, df_copy[['Month', 'Elevation', 'Station', 'TEMP']]])

    # Drop rows with NaN temperature values to ensure all calculations are on valid data
    temp_elev_month_df.dropna(subset=['TEMP'], inplace=True)

    # Count the number of times the temperature is below zero for each station, for each month
    temp_below_zero = temp_elev_month_df[temp_elev_month_df['TEMP'] < 1].groupby(['Month', 'Elevation', 'Station']).size().reset_index(name='Count_Below_Zero')
    
    # Calculate the total number of valid observations for each station, for each month
    total_observations = temp_elev_month_df.groupby(['Month', 'Elevation', 'Station'])['TEMP'].count().reset_index(name='Total_Observations')
    
    # Merge the counts. Use a right merge to keep all month/elevation combinations from total_observations
    temp_below_zero_percentage = pd.merge(temp_below_zero, total_observations, on=['Month', 'Elevation', 'Station'], how='right')
    
    # If a group had no temperatures below zero, its count will be NaN after the merge. Fill these with 0.
    temp_below_zero_percentage['Count_Below_Zero'].fillna(0, inplace=True)

    # Calculate the percentage. Use np.divide for safe division (handles division by zero).
    temp_below_zero_percentage['Percentage_Below_Zero'] = np.divide(
        temp_below_zero_percentage['Count_Below_Zero'],
        temp_below_zero_percentage['Total_Observations'],
        out=np.zeros_like(temp_below_zero_percentage['Count_Below_Zero'], dtype=float),
        where=temp_below_zero_percentage['Total_Observations'] != 0
    ) * 100

    # Pivot the DataFrame to get months as columns and elevations as rows
    pivot_df = temp_below_zero_percentage.pivot_table(index='Elevation', columns='Month', values='Percentage_Below_Zero', aggfunc='mean')




    # Prepare mapping for abbreviations (use provided STATION_ABBREV)
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
        'snowAMP_lower': 'T1',
        'snowAMP_middle': 'T2',
        'snowAMP Ganjala upper': 'T3',
        'Yala Pluvio': 'T4',
        'Yala Glacier AWS': 'T5',
        'Langtang Glacier AWS': 'T6'
    }
    # Map elevation to station name
    elevation_to_station = {}
    for station in station_names_all:
        try:
            elev = get_elevation([station])[0]
            if elev not in elevation_to_station:
                elevation_to_station[elev] = station
        except:
            pass

    # ── Style settings matching other figures ────────────────────────────────
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 10})
    FS = {'title': 24.75, 'label': 22.5, 'tick': 20.25, 'annot': 18}
    MON_ABBREV = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def _abbrev_stn(elev):
        stn = elevation_to_station.get(elev, "Unknown")
        return STATION_ABBREV.get(stn, stn)

    pivot_sorted = pivot_df.sort_index(ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
    hm = sns.heatmap(
        pivot_sorted,
        cmap='coolwarm',
        annot=True,
        fmt='.1f',
        vmin=0,
        vmax=100,
        linewidths=0.3,
        linecolor='white',
        cbar_kws={'label': '% of time below 1$^\circ$C', 'pad': 0.08},
        annot_kws={'size': FS['annot']},
        ax=ax
    )

    # Left axis: station abbreviations (as in other panels)
    ax.set_yticklabels([_abbrev_stn(e) for e in pivot_sorted.index], rotation=0, fontsize=FS['tick'])

    # Right axis: elevation labels
    _ax_r = ax.twinx()
    _ax_r.set_ylim(ax.get_ylim())
    _ax_r.set_yticks([i + 0.5 for i in range(len(pivot_sorted.index))])
    _ax_r.set_yticklabels([f'{int(e)} m' for e in pivot_sorted.index], fontsize=FS['tick'])
    _ax_r.tick_params(axis='y', length=0)
    for _sp in _ax_r.spines.values():
        _sp.set_visible(False)

    ax.set_xlabel('Month', fontsize=FS['label'])
    ax.set_ylabel('')
    ax.set_xticklabels(MON_ABBREV, rotation=90, fontsize=FS['tick'])
    

    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FS['tick'])
    cbar.set_label('% of time below 1$^\circ$C', fontsize=FS['tick'])

    fig.subplots_adjust(hspace=0.38, wspace=0.60, left=0.06, right=0.96, top=0.95, bottom=0.11)
    plt.show()

    # Print summary statistics
    print("\n" + "="*100)
    print("PERCENTAGE OF TIME TEMPERATURE IS BELOW 1°C - SUMMARY BY STATION")
    print("="*100)
    
    summary_by_station = temp_below_zero_percentage.groupby('Station').agg({
        'Percentage_Below_Zero': ['mean', 'min', 'max'],
        'Total_Observations': 'sum'
    }).round(2)
    
    print(summary_by_station)
    print("="*100 + "\n")
    
    return temp_below_zero_percentage, pivot_df

# plot_percentage_below_zero(temp_merged_dfs)
def plot_seasonal_diurnal_compact(font_scale=2.0):
    """
    Single figure with 4 rows × 4 columns.
    Columns = seasons (Winter, Pre-monsoon, Monsoon, Post-monsoon).
    Rows:
      (0) Elevation × Hour precipitation heatmap
      (1) Inter-station Spearman r diurnal cycle ± SEM
    (2) Temperature diurnal cycle (3 stations) + LR (twin axis)
      (3) RH (dashed) + Mixing Ratio (solid, twin axis)
    All seasons shown side-by-side, one figure only.

    font_scale : float
        Multiplier applied to every font size in both figures (default 1.5).
    """
    plt.rcParams['font.family'] = 'Arial'

    tb_stations_all = [
        'Ganja La TB1', 'Ganja La TB2', 'Ganja La TB3', 'Morimoto TB',
        'Shalbachum TB', 'Langshisha BC TB', 'Numthang TB', 'Jathang TB',
        'Syabru TB', 'Lama TB',
    ]
    pluvio_aws   = ['Langshisha Pluvio', 'Morimoto Pluvio', 'Kyangjin AWS', 'Yala BC AWS']
    all_stations = pluvio_aws + tb_stations_all

    met_stations = ['Kyangjin AWS', 'Yala BC AWS', 'Morimoto Pluvio']
    temp_colors  = {'Kyangjin AWS': '#08519c', 'Yala BC AWS': '#d94701', 'Morimoto Pluvio': 'tab:olive'}
    temp_labels  = {'Kyangjin AWS': 'Kyangjin', 'Yala BC AWS': 'Yala', 'Morimoto Pluvio': 'Morimoto'}
    # Only Kyangjin for temperature in row 1
    temp_station = 'Kyangjin AWS'

    season_station_sets = {
        'Monsoon':      all_stations,
        'Pre-monsoon':  pluvio_aws,
        'Post-monsoon': pluvio_aws,
        'Winter':       pluvio_aws,
    }
    seasons_list = ['Post-monsoon', 'Winter', 'Pre-monsoon', 'Monsoon']
    # Monsoon is now col_i == 3 (rightmost)
    MONSOON_COL = 3  # col index of Monsoon in seasons_list

    def _smask(idx, season):
        m, d = idx.month, idx.day
        if season == 'Monsoon':
            return ((m == 6) & (d >= 15)) | m.isin([7, 8, 9])
        elif season == 'Pre-monsoon':
            return m.isin([3, 4, 5]) | ((m == 6) & (d < 15))
        elif season == 'Post-monsoon':
            return m.isin([10, 11])
        elif season == 'Winter':
            return m.isin([12, 1, 2])
        return pd.array([False] * len(idx))

    def assign_season(month, day):
        if month in [3, 4, 5] or (month == 6 and day <= 15):
            return 'Pre-monsoon'
        elif (month == 6 and day > 15) or month in [7, 8, 9]:
            return 'Monsoon'
        elif month in [10, 11] or (month == 12 and day < 31):
            return 'Post-monsoon'
        elif month in [1, 2]:
            return 'Winter'
        return 'Unknown'

    # ── Load precipitation ────────────────────────────────────────────────────
    rain_data = read_pluvio_cleaned(all_stations, dt='1h')
    merged = {}
    for station in all_stations:
        if station not in rain_data:
            continue
        df = rain_data[station].copy()
        col = next((c for c in ['Rainfall_1H', 'Hourly_Precip', 'Hourly_Rain']
                    if c in df.columns), df.columns[0])
        df = df[[col]].rename(columns={col: 'Hourly_Precip'})
        df['Hourly_Precip'] = pd.to_numeric(df['Hourly_Precip'], errors='coerce')
        merged[station] = df

    elev_map = dict(zip(all_stations, get_elevation(all_stations)))
    elev_to_abbrev = {int(round(v)): STATION_ABBREV.get(k, k) for k, v in elev_map.items()}

    dfs_heat = []
    for station in all_stations:
        if station not in merged:
            continue
        df = merged[station][['Hourly_Precip']].copy()
        df['Hour']      = df.index.hour
        df['Elevation'] = elev_map.get(station, np.nan)
        df['Station']   = station
        dfs_heat.append(df)
    rain_long = pd.concat(dfs_heat)

    heatmap_pivots = {}
    for season in seasons_list:
        msk  = _smask(rain_long.index, season)
        df_s = rain_long[msk]
        df_s = df_s[df_s['Station'].isin(season_station_sets[season])]
        pivot = df_s.pivot_table(index='Elevation', columns='Hour',
                                 values='Hourly_Precip', aggfunc='mean')
        heatmap_pivots[season] = pivot.reindex(columns=range(24)).sort_index(ascending=True)

    clims = {}
    for season, pivot in heatmap_pivots.items():
        vmax = pivot.values[np.isfinite(pivot.values)].max() if pivot.size > 0 else 1.0
        clims[season] = (0, vmax)
    clims['Monsoon'] = (0, 1.0)

    # ── Inter-station Spearman r diurnal correlations ─────────────────────────
    results_corr = {}
    results_grad = {}
    for season in seasons_list:
        _stations = season_station_sets[season]
        _station_series = {}
        for _st in _stations:
            if _st not in merged:
                continue
            msk = _smask(merged[_st].index, season)
            _s = merged[_st][msk]['Hourly_Precip']
            _station_series[_st] = _s
        _valid_stations = list(_station_series.keys())
        _pairs = [(a, b) for i, a in enumerate(_valid_stations)
                  for b in _valid_stations[i+1:]]
        _hourly_corrs = {h: [] for h in range(24)}
        for h in range(24):
            for _a, _b in _pairs:
                _sa = _station_series[_a]
                _sb = _station_series[_b]
                _common = _sa.index.intersection(_sb.index)
                _common_h = _common[_common.hour == h]
                if len(_common_h) >= 5:
                    _r, _ = stats.spearmanr(
                        _sa.loc[_common_h].fillna(0),
                        _sb.loc[_common_h].fillna(0)
                    )
                    if np.isfinite(_r):
                        _hourly_corrs[h].append(_r)
        _mean_arr = np.array([np.nanmean(_hourly_corrs[h]) if _hourly_corrs[h] else np.nan
                              for h in range(24)])
        _sem_arr  = np.array([stats.sem(_hourly_corrs[h]) if len(_hourly_corrs[h]) > 1 else 0.0
                              for h in range(24)])
        results_corr[season] = {'mean': _mean_arr, 'sem': _sem_arr}

        # Hourly precipitation-elevation gradient (%/km) and uncertainty.
        _grad, _grad_sem = [], []
        for h in range(24):
            _vals_h, _elevs_h = [], []
            for _st in _stations:
                if _st not in merged:
                    continue
                _elev = elev_map.get(_st, np.nan)
                if not np.isfinite(_elev):
                    continue
                _s = merged[_st].loc[_smask(merged[_st].index, season), 'Hourly_Precip']
                _s_h = _s[_s.index.hour == h].dropna()
                if len(_s_h) < 5:
                    continue
                _m = _s_h.mean()
                if np.isfinite(_m):
                    _vals_h.append(_m)
                    _elevs_h.append(_elev)

            if len(_vals_h) >= 3:
                _sl, _, _, _, _se = stats.linregress(_elevs_h, _vals_h)
                _mean_p = np.nanmean(_vals_h)
                if np.isfinite(_mean_p) and _mean_p > 0:
                    _grad.append((_sl * 1000.0 / _mean_p) * 100.0)
                    _grad_sem.append((_se * 1000.0 / _mean_p) * 100.0)
                else:
                    _grad.append(np.nan)
                    _grad_sem.append(np.nan)
            else:
                _grad.append(np.nan)
                _grad_sem.append(np.nan)

        results_grad[season] = {
            'mean': np.array(_grad, dtype=float),
            'sem': np.array(_grad_sem, dtype=float)
        }

    # ── Temperature ───────────────────────────────────────────────────────────
    temp_dfs = {}
    for st in met_stations:
        df = read_pluvio_cleaned([st], dt='1h')[st].copy()
        df['Hour']   = df.index.hour
        df['Season'] = [assign_season(m, d) for m, d in zip(df.index.month, df.index.day)]
        temp_dfs[st] = df

    # ── RH / Mixing Ratio ─────────────────────────────────────────────────────
    rh_mr_files = {
        'Kyangjin AWS':    str(_DATA_DIR / 'Moisture' / 'Kyangjin_AWS_humidity_timeseries.csv'),
        'Yala BC AWS':     str(_DATA_DIR / 'Moisture' / 'Yala_BC_AWS_humidity_timeseries.csv'),
        'Morimoto Pluvio': str(_DATA_DIR / 'Moisture' / 'Morimoto_MM_humidity_timeseries.csv'),
    }
    rh_mr_dfs = {}
    for st, path in rh_mr_files.items():
        try:
            df = pd.read_csv(path)
            df['DATETIME'] = pd.to_datetime(df['DATETIME'])
            df.set_index('DATETIME', inplace=True)
            df['Hour']   = df.index.hour
            df['Season'] = [assign_season(m, d) for m, d in zip(df.index.month, df.index.day)]
            rh_mr_dfs[st] = df
        except Exception:
            rh_mr_dfs[st] = None

    # ── Lapse rate ────────────────────────────────────────────────────────────
    lapse_rate_path = str(_DATA_DIR / 'LapseRate' / 'lapse_rate.csv')
    lapse_rate_df = pd.read_csv(lapse_rate_path, index_col=0, parse_dates=True)
    lapse_rate_df['Hour']   = lapse_rate_df.index.hour
    lapse_rate_df['Season'] = [assign_season(m, d)
                                for m, d in zip(lapse_rate_df.index.month, lapse_rate_df.index.day)]

    # ── Shared y-limits ───────────────────────────────────────────────────────
    temp_min, temp_max   = np.inf, -np.inf
    lapse_min, lapse_max = np.inf, -np.inf
    rh_min, rh_max       = np.inf, -np.inf
    mr_min, mr_max       = 0, 13

    for season in seasons_list:
        for st in [temp_station]:
            temp_col = next((c for c in ['Temperature_1H', 'Temperature', 'Temperature_15min']
                             if c in temp_dfs[st].columns), None)
            if temp_col:
                sub = temp_dfs[st][temp_dfs[st]['Season'] == season].groupby('Hour')[temp_col]
                if not sub.mean().empty:
                    temp_min = min(temp_min, sub.quantile(0.1).min())
                    temp_max = max(temp_max, sub.quantile(0.9).max())
            if rh_mr_dfs[st] is not None:
                rh_sub = rh_mr_dfs[st][rh_mr_dfs[st]['Season'] == season]
                if 'RH' in rh_sub.columns and not rh_sub.empty:
                    g = rh_sub.groupby('Hour')['RH']
                    rh_min = min(rh_min, g.quantile(0.1).min())
                    rh_max = max(rh_max, g.quantile(0.9).max())
        lr_sub = lapse_rate_df[lapse_rate_df['Season'] == season].groupby('Hour')['lapse_rate']
        if not lr_sub.mean().empty:
            lapse_min = min(lapse_min, (lr_sub.quantile(0.1) * 1000).min())
            lapse_max = max(lapse_max, (lr_sub.quantile(0.9) * 1000).max())

    if temp_min  == np.inf:  temp_min,  temp_max  = -10, 20
    if lapse_min == np.inf:  lapse_min, lapse_max = -10,  0
    if rh_min    == np.inf:  rh_min,    rh_max    =   0, 100

    hrs = np.arange(24)
    station_elevations = {st: get_elevation([st])[0] for st in met_stations}

    # Font scaling (set once here)
    FONT_SCALE = font_scale
    fs = lambda x: x * FONT_SCALE

    FS_TICK   = fs(7)
    FS_LABEL  = fs(8)
    FS_TITLE  = fs(10)
    FS_PANEL  = fs(9)
    FS_LEGEND = fs(6.5)
    FS_CBAR   = fs(7)
    FS_ELR    = fs(8)

    # ── Build figure: 3 rows × 4 cols ────────────────────────────────────────
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(22, 14),
        facecolor='white',
        gridspec_kw={'hspace': 0.45, 'wspace': 0.38,
                     'height_ratios': [1.2, 0.9, 0.9]}
    )

    # Panel labels: unique per panel (row x col)
    panel_labels = {
        (0, 0): '(a1)', (0, 1): '(a2)', (0, 2): '(a3)', (0, 3): '(a4)',
        (1, 0): '(b1)', (1, 1): '(b2)', (1, 2): '(b3)', (1, 3): '(b4)',
        (2, 0): '(c1)', (2, 1): '(c2)', (2, 2): '(c3)', (2, 3): '(c4)',
    }

    # Store twin axes for shared y-lim application later
    ax_elr_cols = {}
    ax_mr_cols  = {}

    for col_i, season in enumerate(seasons_list):
        pivot  = heatmap_pivots.get(season, pd.DataFrame())
        is_monsoon = (col_i == MONSOON_COL)

        # ── Row 0: Heatmap ────────────────────────────────────────────────────
        ax = axes[0, col_i]
        if not pivot.empty:
            vmin, vmax = clims[season]
            im = ax.pcolormesh(
                np.arange(-0.5, 24.5), np.arange(len(pivot) + 1),
                pivot.values,
                cmap='YlGnBu', vmin=vmin, vmax=vmax, shading='flat')
            cbar_pad = 0.25 if is_monsoon else 0.02
            cbar = fig.colorbar(im, ax=ax, pad=cbar_pad, fraction=0.046)
            cbar.set_label('mm/h', fontsize=FS_CBAR)
            cbar.ax.tick_params(labelsize=FS_CBAR)
            ytick_pos  = np.arange(len(pivot)) + 0.5
            ax.set_yticks(ytick_pos)
            if col_i == 0:
                # Left side: show station name + elevation
                ytick_lbls = [
                    f"{elev_to_abbrev.get(int(round(e)), '?')}\n{int(e)} m"
                    for e in pivot.index
                ]
                ax.set_yticklabels(ytick_lbls, fontsize=FS_TICK)
            elif is_monsoon:
                # Left side: station names only
                _left_lbls = [
                    elev_to_abbrev.get(int(round(_e)), '?')
                    for _e in pivot.index
                ]
                ax.set_yticklabels(_left_lbls, fontsize=FS_TICK)
                ax.tick_params(axis='y', length=0)
                # Right side: elevations only
                ax2 = ax.twinx()
                ax2.set_ylim(0, len(pivot))
                ax2.set_yticks(ytick_pos)
                ax2.set_yticklabels([f"{int(_e)} m" for _e in pivot.index],
                                    fontsize=FS_TICK)
                ax2.spines['top'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax2.tick_params(axis='y', length=0)
            else:
                ax.set_yticklabels([])
            ax.set_ylim(0, len(pivot))
        else:
            
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=FS_LABEL)
        ax.set_xticks(range(0, 24, 6))
        ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 6)], fontsize=FS_TICK)
        ax.set_xlim(-0.5, 23.5)
        ax.grid(False)
        ax.set_title(season, fontsize=FS_TITLE, fontweight='bold', color='black', pad=4)
        ax.text(0.01, 0.98, panel_labels[(0, col_i)], transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight='bold', va='top')
        _ht_marks = {'TB2': ' (\u2014)', 'PLU4': ' (- -)'}  # emdash / dashes

        # ── Row 1: Temperature + ELR ──────────────────────────────────────────
        ax = axes[1, col_i]
        ax_elr = ax.twinx()
        ax_elr_cols[col_i] = ax_elr
        for st in [temp_station]:
            c = temp_colors[st]
            temp_col = next((col for col in ['Temperature_1H', 'Temperature', 'Temperature_15min']
                             if col in temp_dfs[st].columns), None)
            if temp_col:
                sub = temp_dfs[st][temp_dfs[st]['Season'] == season]
                if not sub.empty:
                    grp  = sub.groupby('Hour')[temp_col]
                    mean = grp.mean()
                    p10  = grp.quantile(0.1)
                    p90  = grp.quantile(0.9)
                    ax.plot(mean.index, mean.values, color=c, linewidth=1.5,
                            label=f"{temp_labels[st]} ({int(station_elevations[st])} m)")
                    ax.fill_between(mean.index, p10, p90, color=c, alpha=0.10)

        lr_sub = lapse_rate_df[lapse_rate_df['Season'] == season]
        if not lr_sub.empty:
            lr_mean = lr_sub.groupby('Hour')['lapse_rate'].mean() * 1000
            ax_elr.plot(lr_mean.index, lr_mean.values, color='black',
                        linestyle='--', linewidth=1.4, label='LR')
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 6))
        ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 6)], fontsize=FS_TICK)
        ax.set_ylim(temp_min, temp_max)
        ax_elr.set_ylim(lapse_min, lapse_max)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax_elr.spines['top'].set_visible(False)
        ax_elr.spines['left'].set_visible(False)
        if col_i != 0:
            ax.set_yticklabels([])
        else:
            ax.tick_params(axis='y', labelsize=FS_TICK)
        # ELR label only on rightmost column (Monsoon)
        if col_i == n_cols - 1:
            ax_elr.set_ylabel('LR (°C/km)', fontsize=FS_ELR)
            ax_elr.tick_params(axis='y', labelsize=FS_TICK)
        else:
            ax_elr.set_yticklabels([])
        ax.text(0.01, 0.98, panel_labels[(1, col_i)], transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight='bold', va='top')
        # Legend in Monsoon column (b3 = col_i==3) lower right
        if is_monsoon:
            temp_handles = [
                Line2D([0], [0], color=temp_colors[temp_station], linewidth=1.5, label='Temperature'),
                Line2D([0], [0], color='black', linestyle='--', linewidth=1.4, label='LR (°C/km)')
            ]
            ax.legend(handles=temp_handles, fontsize=FS_LEGEND, loc='lower right',
                      framealpha=0.85, ncol=1)

        # ── Row 2: RH + Mixing Ratio ──────────────────────────────────────────
        ax = axes[2, col_i]
        ax_mr = ax.twinx()
        ax_mr_cols[col_i] = ax_mr
        for st in met_stations:
            c = temp_colors[st]
            df_rh = rh_mr_dfs.get(st)
            if df_rh is None:
                continue
            sub = df_rh[df_rh['Season'] == season]
            if sub.empty:
                continue
            if 'RH' in sub.columns:
                rh_mean = sub.groupby('Hour')['RH'].mean()
                ax.plot(rh_mean.index, rh_mean.values, color=c, linestyle='--', linewidth=1.5)
            if 'MIXING_RATIO' in sub.columns:
                mr_mean = sub.groupby('Hour')['MIXING_RATIO'].mean()
                ax_mr.plot(mr_mean.index, mr_mean.values, color=c, linestyle='-', linewidth=1.5)
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 6))
        ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 6)], fontsize=FS_TICK)
        ax.set_ylim(rh_min, rh_max)
        ax_mr.set_ylim(mr_min, mr_max)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax_mr.spines['top'].set_visible(False)
        ax_mr.spines['left'].set_visible(False)
        if col_i != 0:
            ax.set_yticklabels([])
        else:
            ax.tick_params(axis='y', labelsize=FS_TICK)
        if col_i == n_cols - 1:
            ax_mr.set_ylabel('Mixing Ratio (g/kg)', fontsize=FS_ELR)
            ax_mr.tick_params(axis='y', labelsize=FS_TICK)
        else:
            ax_mr.set_yticklabels([])
        ax.text(0.01, 0.98, panel_labels[(2, col_i)], transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight='bold', va='top')
        # Legend in Monsoon column: RH/MR style only (station colors are figure-level).
        if is_monsoon:
            rh_mr_handles = [
                Line2D([0], [0], color='black', linestyle='--', linewidth=1.6, label='RH (%)'),
                Line2D([0], [0], color='black', linestyle='-', linewidth=1.6, label='MR (g/kg)')
            ]
            ax.legend(handles=rh_mr_handles, fontsize=FS_LEGEND, loc='lower right',
                      framealpha=0.85, ncol=1)

    # ── Shared row y-labels (leftmost column only) ────────────────────────────
    axes[0, 0].set_ylabel('Elevation (m)', fontsize=FS_LABEL)
    axes[1, 0].set_ylabel('Temperature (°C)', fontsize=FS_LABEL)
    axes[2, 0].set_ylabel('RH (%)', fontsize=FS_LABEL)

    # ── Shared x-label ────────────────────────────────────────────────────────
    for col_i in range(n_cols):
        axes[2, col_i].set_xlabel('Hour of day', fontsize=FS_LABEL)

    # Use PLU1, PLU2, PLU3 abbreviations for the three main stations
    station_handles = [
        Line2D([0], [0], color=temp_colors[st], linewidth=2,
               label=f"{STATION_ABBREV.get(st, temp_labels[st])} ({int(station_elevations[st])} m)")
        for st in met_stations
    ]
    fig.legend(handles=station_handles, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.01), fontsize=FS_LEGEND * 1.5)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()

    # ── Station correlation figure: Monsoon only ──────────────────────────────
    fig_corr, (ax, ax_d) = plt.subplots(
        1, 2, figsize=(13, 4.5), facecolor='white',
        gridspec_kw={'wspace': 0.35}
    )
    season = 'Monsoon'
    color  = 'steelblue'
    corrs  = results_corr[season]['mean']
    sems   = results_corr[season]['sem']
    ax.fill_between(hrs, corrs - sems, corrs + sems, color=color, alpha=0.18, label='SEM')
    ax.plot(hrs, corrs, '-o', color=color, markersize=3, linewidth=1.6, label='Mean')
    ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 6))
    ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 6)], fontsize=fs(8))
    ax.set_xlabel('Hour of day', fontsize=fs(9))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=fs(8))
    ax.text(0.01, 0.98, '(a)', transform=ax.transAxes,
            fontsize=fs(10), fontweight='bold', va='top')
    ax.set_ylabel('Inter-station correlation (r) (Monsoon)', fontsize=fs(9))
    ax.legend(fontsize=fs(7.5), loc='lower right', framealpha=0.85)

    # ── Panel (b): schematic normalised diurnal – TB2 vs PLU4 ────────────────
    ax_d.axvspan(12, 16, color='#FFD700', alpha=0.22, zorder=0,
                 label='Daytime peak (12–16h)')
    ax_d.axvspan(21, 24, color='#6a51a3', alpha=0.18, zorder=0)
    ax_d.axvspan( 0,  1, color='#6a51a3', alpha=0.18, zorder=0,
                 label='Nighttime peak (21–01h)')
    ax_d.axhline(1.0, color='black', linewidth=0.8, linestyle=':', alpha=0.55, zorder=1)
    for _stn, _ls, _lbl in [
            ('Lama TB',         '-',  'TB2'),
            ('Morimoto Pluvio', '--', 'PLU4'),
    ]:
        if _stn in merged:
            _df = merged[_stn]
            _msk = _smask(_df.index, 'Monsoon')
            _p = _df.loc[_msk, 'Hourly_Precip'].dropna()
            if len(_p) >= 10 and _p.mean() > 0:
                _pn = _p / _p.mean()
                _hn = _pn.groupby(_pn.index.hour).mean().reindex(range(24))
                ax_d.plot(_hn.index, _hn.values, color='black',
                          linestyle=_ls, linewidth=2.2, label=_lbl)
    ax_d.set_xlim(-0.5, 23.5)
    ax_d.set_xticks(range(0, 24, 6))
    ax_d.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 6)], fontsize=fs(8))
    ax_d.set_xlabel('Hour of day', fontsize=fs(9))
    ax_d.set_ylabel('Normalised precipitation', fontsize=fs(9))
    ax_d.grid(True, linestyle='--', alpha=0.25)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.tick_params(axis='y', labelsize=fs(8))
    ax_d.text(0.01, 0.98, '(b)', transform=ax_d.transAxes,
              fontsize=fs(10), fontweight='bold', va='top')
    ax_d.legend(fontsize=fs(7.5), loc='upper center', bbox_to_anchor=(0.5, 1.0),
                framealpha=0.85, handlelength=2.5, ncol=1)

    plt.tight_layout()
    plt.show()
# plot_seasonal_diurnal_compact(font_scale=2.0)
def plot_monsoon_peak_strength_no_panel_b(months=None, month_label='Monsoon',
                                          exclude_stations=None,
                                          peak_mode='range',
                                          day_range=(13, 16),
                                          night_range=(21, 0),
                                          day_hour=14,
                                          night_hour=23):
    """
        Variant of plot_monsoon_peak_strength:
            (a) Normalised monsoon precipitation vs elevation at selected daytime
                    and nighttime peak windows/hours.
      (b) Polar clock of hourly OLS slope of normalised precipitation ~ elevation.
      (c) Daily day-max / night-max ratio vs elevation.

    The former inset panel is omitted. In the rose plot, positive z slopes use
    the daytime colour and negative z slopes use the nighttime colour.

    months : list of int, optional
        Calendar months to include (e.g. [8] for August only). Defaults to
        the full monsoon season, Jun 15 – Sep 30.
    month_label : str
        Shown as the figure title.
    exclude_stations : list of str, optional
        Stations to leave out of every panel (incl. the OLS slopes). Accepts
        full names ('Syabru TB') or abbreviations ('TB1', 'TB2').
    peak_mode : {'range', 'hour', 'slope'}, optional
        How panel (a) peak values are extracted:
          - 'range': average over day_range and night_range (default)
          - 'hour' : use fixed day_hour and night_hour
          - 'slope': use the strongest positive/negative hourly OLS slopes
    day_range, night_range : tuple(int, int), optional
        Inclusive hour windows for peak_mode='range'. Wrap-around is supported
        (e.g. night_range=(21, 0) -> [21, 22, 23, 0]).
    day_hour, night_hour : int, optional
        Hours used when peak_mode='hour'.
    """
    from scipy import stats
    from matplotlib.gridspec import GridSpec

    # ── Global style (same as combined snowfall/rainfall figure) ─────────
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 15})
    base_fs = float(plt.rcParams.get('font.size', 15))
    FS = {'title': 1.45 * base_fs, 'label': 1.30 * base_fs,
          'tick': 1.15 * base_fs, 'annot': 1.00 * base_fs}

    def _panel_outline(ax_, color='#333333', lw=1.0):
        for _sp in ax_.spines.values():
            _sp.set_visible(True)
            _sp.set_linewidth(lw)
            _sp.set_color(color)

    def _pl(ax_, letter, x=1.0):
        return ax_.text(x, -0.02, f'({letter})', transform=ax_.transAxes,
                        ha='right', va='top', fontsize=FS['title'],
                        fontweight='bold', zorder=10, clip_on=False)

    tb_stations_all = [
        'Ganja La TB1', 'Ganja La TB2', 'Ganja La TB3', 'Morimoto TB',
        'Shalbachum TB', 'Langshisha BC TB', 'Numthang TB', 'Jathang TB',
        'Syabru TB', 'Lama TB',
    ]
    pluvio_aws = ['Langshisha Pluvio', 'Morimoto Pluvio', 'Kyangjin AWS', 'Yala BC AWS']
    all_stations = pluvio_aws + tb_stations_all

    if exclude_stations:
        _excl = set(exclude_stations)
        all_stations = [s for s in all_stations
                        if s not in _excl
                        and STATION_ABBREV.get(s, '') not in _excl]
        print(f"Excluded stations: {sorted(_excl)} → "
              f"{len(all_stations)} stations remain")

    DAY_COLOR = '#FFD700'
    NIGHT_COLOR = '#6a51a3'

    rain_data = read_pluvio_cleaned(all_stations, dt='1h')
    elev_map = dict(zip(all_stations, get_elevation(all_stations)))

    def _monsoon(idx):
        if months is not None:
            return idx.month.isin(months)
        return ((idx.month == 6) & (idx.day >= 15)) | idx.month.isin([7, 8, 9])

    def _expand_hour_window(start_h, end_h):
        start_h = int(start_h) % 24
        end_h = int(end_h) % 24
        if start_h <= end_h:
            return list(range(start_h, end_h + 1))
        return list(range(start_h, 24)) + list(range(0, end_h + 1))

    def _hours_label(hours_):
        if len(hours_) == 1:
            return f"{hours_[0]:02d}h"
        return f"{hours_[0]:02d}-{hours_[-1]:02d}h"

    merged_pk = {}
    for st in all_stations:
        if st not in rain_data:
            continue
        df = rain_data[st].copy()
        col = next((c for c in ['Rainfall_1H', 'Hourly_Precip', 'Hourly_Rain']
                    if c in df.columns), df.columns[0])
        df = df[[col]].rename(columns={col: 'Hourly_Precip'})
        df['Hourly_Precip'] = pd.to_numeric(df['Hourly_Precip'], errors='coerce')
        merged_pk[st] = df

    # ── Hourly OLS slopes (norm. precip ~ elevation) ─────────────────────
    # Computed before the peak extraction so panel (a) can use the hours
    # with the strongest positive (afternoon) / negative (nocturnal) slope.
    station_profiles = {}
    for st in all_stations:
        if st not in merged_pk:
            continue
        prec = merged_pk[st]['Hourly_Precip']
        prec_s = prec[_monsoon(prec.index)].dropna()
        if len(prec_s) < 200 or prec_s.mean() == 0:
            continue
        norm = prec_s / prec_s.mean()
        station_profiles[st] = norm.groupby(norm.index.hour).mean()

    valid_stations = list(station_profiles.keys())
    elevations_arr = np.array([elev_map[s] for s in valid_stations])
    hours = list(range(24))

    ols_slope = []
    for h in hours:
        prec_h = np.array([
            station_profiles[s][h] if h in station_profiles[s].index else np.nan
            for s in valid_stations
        ])
        mask = ~np.isnan(prec_h)
        if mask.sum() >= 4:
            sl, _, _, _, _ = stats.linregress(elevations_arr[mask], prec_h[mask])
            ols_slope.append(sl)
        else:
            ols_slope.append(np.nan)

    # Panel (a) selection: range, fixed hour, or slope-selected hour
    mode = str(peak_mode).strip().lower()
    slopes_arr = np.array(ols_slope, dtype=float)
    if mode == 'range':
        DAY_HOURS = _expand_hour_window(*day_range)
        NIGHT_HOURS = _expand_hour_window(*night_range)
        print(f"Panel (a) selection for {month_label}: day { _hours_label(DAY_HOURS) }, "
              f"night { _hours_label(NIGHT_HOURS) } (range mode)")
    elif mode == 'hour':
        DAY_HOURS = [int(day_hour) % 24]
        NIGHT_HOURS = [int(night_hour) % 24]
        print(f"Panel (a) selection for {month_label}: day { _hours_label(DAY_HOURS) }, "
              f"night { _hours_label(NIGHT_HOURS) } (hour mode)")
    else:
        if np.all(np.isnan(slopes_arr)):
            DAY_HOURS, NIGHT_HOURS = [14], [23]   # fallback
        else:
            DAY_HOURS = [int(np.nanargmax(slopes_arr))]
            NIGHT_HOURS = [int(np.nanargmin(slopes_arr))]
        print(f"Panel (a) selection for {month_label}: strongest positive slope at "
              f"{DAY_HOURS[0]:02d}h, strongest negative slope at {NIGHT_HOURS[0]:02d}h "
              f"(slope mode)")

    DAY_LABEL = _hours_label(DAY_HOURS)
    NIGHT_LABEL = _hours_label(NIGHT_HOURS)

    rows = []
    for st in all_stations:
        if st not in merged_pk:
            continue
        prec = merged_pk[st]['Hourly_Precip']
        prec_s = prec[_monsoon(prec.index)].dropna()
        if len(prec_s) < 200 or prec_s.mean() == 0:
            continue

        pn_df = (prec_s / prec_s.mean()).to_frame('pnorm')
        pn_df['hour'] = pn_df.index.hour
        pn_df['date'] = pn_df.index.date
        daily = pn_df.pivot_table(
            index='date', columns='hour', values='pnorm', aggfunc='mean'
        ).dropna(thresh=20)
        n_days = len(daily)
        if n_days < 10:
            continue

        hn = daily.mean()

        def _window_vals(hours_sel):
            cols = [h for h in hours_sel if h in daily.columns]
            if not cols:
                return pd.Series(dtype=float)
            return daily[cols].mean(axis=1, skipna=True).dropna()

        def _window_value(hours_sel):
            vals = _window_vals(hours_sel)
            return vals.mean() if len(vals) > 0 else np.nan

        def _window_sem(hours_sel):
            vals = _window_vals(hours_sel)
            return vals.std() / np.sqrt(len(vals)) if len(vals) > 0 else np.nan

        def _window_n(hours_sel):
            return int(_window_vals(hours_sel).shape[0])

        # Per-day ratio of daytime max to nighttime max of the normalised
        # precipitation (day 07–18, night 19–06). Summarised per station by
        # the median with the IQR, because dry nights make the ratio skewed.
        day_cols   = [h for h in range(7, 19) if h in daily.columns]
        night_cols = [h for h in list(range(19, 24)) + list(range(0, 7))
                      if h in daily.columns]
        ratio_med = ratio_q1 = ratio_q3 = np.nan
        if day_cols and night_cols:
            day_max   = daily[day_cols].max(axis=1)
            night_max = daily[night_cols].max(axis=1)
            ok = day_max.notna() & night_max.notna() & (night_max > 0)
            ratios = day_max[ok] / night_max[ok]
            if len(ratios) >= 10:
                ratio_med = ratios.median()
                ratio_q1  = ratios.quantile(0.25)
                ratio_q3  = ratios.quantile(0.75)

        rows.append({
            'station': st,
            'elevation': elev_map[st],
            'day_peak': _window_value(DAY_HOURS),
            'night_peak': _window_value(NIGHT_HOURS),
            'day_sem': _window_sem(DAY_HOURS),
            'night_sem': _window_sem(NIGHT_HOURS),
            'day_n': _window_n(DAY_HOURS),
            'night_n': _window_n(NIGHT_HOURS),
            'n_days_total': int(n_days),
            'peak_ratio': ratio_med,
            'ratio_q1': ratio_q1,
            'ratio_q3': ratio_q3,
        })

    if not rows:
        print('No monsoon data for peak strength figure.')
        return

    df_pk = pd.DataFrame(rows).sort_values('elevation').reset_index(drop=True)
    elevs = df_pk['elevation'].values

    print(f"Sample sizes behind the peak estimates for {month_label}:")
    for _, row in df_pk.iterrows():
        st_label = STATION_ABBREV.get(row['station'], row['station'])
        print(
            f"  {st_label:>4s} ({row['station']}): "
            f"{int(row['n_days_total'])} daily profiles kept; "
            f"{int(row['day_n'])} days in {DAY_LABEL}, "
            f"{int(row['night_n'])} days in {NIGHT_LABEL}"
        )

    slopes_scaled = slopes_arr * 1e4
    max_abs = np.nanmax(np.abs(slopes_scaled))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    fig = plt.figure(figsize=(22, 7), facecolor='white')
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1], wspace=0.3)

    ax = fig.add_subplot(gs[0])
    ax_pol = fig.add_subplot(gs[1], projection='polar')

    for _, row in df_pk.iterrows():
        ax.errorbar(row['elevation'], row['day_peak'],
                    yerr=row['day_sem'], fmt='o', markersize=9,
                    color=DAY_COLOR, ecolor=DAY_COLOR, elinewidth=1.2, capsize=4,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=4)
        if pd.notna(row['night_peak']):
            ax.errorbar(row['elevation'], row['night_peak'],
                        yerr=row['night_sem'], fmt='s', markersize=9,
                        color=NIGHT_COLOR, ecolor=NIGHT_COLOR, elinewidth=1.2, capsize=4,
                        markeredgecolor='black', markeredgewidth=0.5, zorder=4)

    # for ann_stn, ann_lbl in [('Lama TB', 'TB2'), ('Morimoto Pluvio', 'PLU4')]:
    #     ann_row = df_pk[df_pk['station'] == ann_stn]
    #     if not ann_row.empty:
    #         ax.annotate(ann_lbl,
    #                     xy=(ann_row['elevation'].values[0], ann_row['day_peak'].values[0]),
    #                     xytext=(8, 4), textcoords='offset points',
    #                     fontsize=int(9 * FS), fontweight='bold', color='black')

    for col_name, base_c, ls, lbl in [
        ('day_peak', DAY_COLOR, '-', f'Afternoon {DAY_LABEL}'),
        ('night_peak', NIGHT_COLOR, '--', f'Nocturnal {NIGHT_LABEL}'),
    ]:
        valid = df_pk.dropna(subset=[col_name])
        if len(valid) >= 3:
            slope, intercept, r, _, _ = stats.linregress(
                valid['elevation'], valid[col_name])
            x_fit = np.linspace(elevs.min() - 50, 6500, 100)
            ax.plot(x_fit, slope * x_fit + intercept,
                    color=base_c, linestyle=ls, linewidth=1.8, alpha=0.8,
                    label=f'{lbl}  r={r:.2f}')

    ax.set_xlim(None, 6500)
    ax.set_xlabel('Elevation (m a.s.l.)', fontsize=int(FS['label']))
    ax.set_ylabel('Normalized precipitation', fontsize=int(FS['label']))
    ax.tick_params(axis='both', labelsize=int(FS['tick']))
    ax.legend(fontsize=int(FS['tick']), loc='lower right', frameon=True,
              facecolor='white', edgecolor='#d0d0d0')
    _panel_outline(ax)
    ax.grid(True, linestyle=':', alpha=0.45, color='gray')
    _pl(ax, 'a')
    theta = np.array([h * (2 * np.pi / 24) for h in hours])
    width = 2 * np.pi / 24 * 0.85

    for i, _ in enumerate(hours):
        if np.isnan(slopes_scaled[i]):
            continue
        s = slopes_scaled[i]
        col = DAY_COLOR if s >= 0 else NIGHT_COLOR
        ax_pol.bar(theta[i], abs(s) / max_abs, width=width, bottom=0,
                   color=col, alpha=0.95, edgecolor='white', linewidth=0.4)

    ax_pol.set_theta_zero_location('N')
    ax_pol.set_theta_direction(-1)
    ax_pol.set_xticks(theta[::3])
    ax_pol.set_xticklabels([f'{h:02d}h' for h in hours[::3]], fontsize=int(FS['tick']))
    ax_pol.set_ylim(0, 1)
    ax_pol.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_pol.set_yticklabels(
        [f'{v * max_abs:.2f}' for v in [0.25, 0.5, 0.75, 1.0]],
        fontsize=int(FS['annot']), color='grey')
    # Outline the bars used in panel (a), so the selection is visually
    # verifiable against the polar clock.
    for hh in sorted(set(DAY_HOURS + NIGHT_HOURS)):
        if not np.isnan(slopes_scaled[hh]):
            s_hh = slopes_scaled[hh]
            col_hh = DAY_COLOR if s_hh >= 0 else NIGHT_COLOR
            ax_pol.bar(theta[hh], abs(s_hh) / max_abs, width=width, bottom=0,
                       color=col_hh, edgecolor='black', linewidth=1.5, zorder=5)

    ax_pol.set_title('Norm. prec / elevation (x10${^4}$)', fontsize=int(FS['label']), pad=28)
    _panel_outline(ax_pol)
    ax_pol.grid(linestyle=':', alpha=0.45, color='gray')
    _pl(ax_pol, 'b', x=1.05)

    # leg_patches = [
    #     mpatches.Patch(color=DAY_COLOR, alpha=0.95, label='z > 0'),
    #     mpatches.Patch(color=NIGHT_COLOR, alpha=0.95, label='z < 0'),
    # ]
    # ax_pol.legend(handles=leg_patches, loc='lower right',
    #               bbox_to_anchor=(1.28, -0.18), fontsize=int(8 * FS), framealpha=0.9)

    # # ── (c) Daily day-max / night-max ratio vs elevation ─────────────────
    # valid_r = df_pk.dropna(subset=['peak_ratio'])
    # if not valid_r.empty:
    #     yerr_r = np.vstack([valid_r['peak_ratio'] - valid_r['ratio_q1'],
    #                         valid_r['ratio_q3'] - valid_r['peak_ratio']])
    #     ax_r.errorbar(valid_r['elevation'], valid_r['peak_ratio'], yerr=yerr_r,
    #                   fmt='o', markersize=9, color='#2a9d8f', ecolor='#2a9d8f',
    #                   elinewidth=1.2, capsize=4, linestyle='none',
    #                   markeredgecolor='black', markeredgewidth=0.5, zorder=4)
    # ax_r.axhline(1.0, color='grey', linestyle=':', linewidth=1.2, zorder=2)
    # if len(valid_r) >= 3:
    #     sl_r, ic_r, r_r, _, _ = stats.linregress(valid_r['elevation'],
    #                                              valid_r['peak_ratio'])
    #     x_fit_r = np.linspace(elevs.min() - 50, 6500, 100)
    #     ax_r.plot(x_fit_r, sl_r * x_fit_r + ic_r, color='#2a9d8f',
    #               linewidth=1.8, alpha=0.8, label=f'OLS  r={r_r:.2f}')
    #     ax_r.legend(fontsize=int(9 * FS), loc='upper right',
    #                 framealpha=0.95, edgecolor='grey')
    # ax_r.set_xlim(None, 6500)
    # ax_r.set_xlabel('Elevation (m a.s.l.)', fontsize=int(12 * FS))
    # ax_r.set_ylabel('Daily max ratio day / night (–)', fontsize=int(12 * FS))
    # ax_r.tick_params(axis='both', labelsize=int(10 * FS))
    # ax_r.spines['top'].set_visible(False)
    # ax_r.spines['right'].set_visible(False)
    # ax_r.grid(True, linestyle='--', alpha=0.25)

    # plt.tight_layout()
    # pos_a = ax.get_position()
    # pos_b = ax_pol.get_position()
    # pos_c = ax_r.get_position()
    # panel_y = pos_a.y1 - 0.02 * pos_a.height
    # fig.text(pos_a.x0 + 0.01 * pos_a.width, panel_y, '(a)',
    #          fontsize=int(11 * FS), fontweight='bold', va='top', ha='left')
    # fig.text(pos_b.x1 + 0.12 * pos_b.width, panel_y, '(b)',
    #          fontsize=int(11 * FS), fontweight='bold', va='top', ha='right')
    # fig.text(pos_c.x0 - 0.1 * pos_c.width, panel_y, '(c)',
    #          fontsize=int(11 * FS), fontweight='bold', va='top', ha='left')
    plt.show()

def plot_monthly_climate_overview():
    """
    Combined monthly overview figure with 6 panels (2 rows × 3 columns):
      (a) Daily-mean SW/LW ratio
      (b) Temperature lapse rate
      (c) Zero-degree isotherm elevation
      (d) Mixing ratio
      (e) Relative humidity
      (f) Mean monthly temperature (from cleaned data)
    All panels share the x-axis (months Jan–Dec).
    """
    MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    MONTHS = np.arange(1, 13)
    MIN_DAILY_HOURS = 20
    MONTHLY_COVERAGE_REQ = 0.50
    MIN_VALID_MONTHYEARS = 2
    def _hourly_full_index(series):
        s = pd.to_numeric(series, errors='coerce').sort_index()
        if s.empty:
            return s

        if getattr(s.index, 'has_duplicates', False):
            s = s.groupby(level=0).mean()

        s = s.resample("1h").mean()
        return s    
    def _daily_mean_complete(series, min_hours=MIN_DAILY_HOURS):
        s = _hourly_full_index(series)
        if s.empty:
            return s
        daily_mean = s.resample('D').mean()
        daily_count = s.resample('D').count()
        daily_mean[daily_count < min_hours] = np.nan
        return daily_mean

    def _monthyear_counts_and_coverage(series, required=MONTHLY_COVERAGE_REQ):
        s = _hourly_full_index(series)
        idx12 = range(1, 13)
        if s.empty:
            empty_n = pd.Series(0, index=idx12, dtype=int)
            empty_cov = pd.Series(np.nan, index=idx12, dtype=float)
            return empty_n, empty_cov

        monthly_mean = s.resample('MS').mean()
        monthly_valid = s.notna().resample('MS').sum()
        monthly_expected = s.resample('MS').size()
        monthly_cov = monthly_valid / monthly_expected
        ok = (monthly_cov >= required) & monthly_mean.notna()
       
        counts = ok.groupby(ok.index.month).sum().reindex(idx12, fill_value=0).astype(int)
        cov_pct = (monthly_cov.groupby(monthly_cov.index.month).mean().reindex(idx12)) * 100.0
        return counts, cov_pct
    def _monthly_climatology_from_valid_monthyears(
        series,
        required=MONTHLY_COVERAGE_REQ,
        min_samples=MIN_VALID_MONTHYEARS,
    ):
        s = _hourly_full_index(series)
        idx12 = range(1, 13)

        if s.empty:
            empty = pd.Series(np.nan, index=idx12)
            return (
                empty.copy(), empty.copy(), empty.copy(),
                pd.Series(0, index=idx12, dtype=int),
                pd.Series(np.nan, index=idx12)
            )

        # ---- Coverage check (unchanged) -----------------------------------
        monthly_mean = s.resample('MS').mean()
        monthly_valid = s.notna().resample('MS').sum()
        monthly_expected = s.resample('MS').size()
        monthly_cov = monthly_valid / monthly_expected


        ok = (monthly_cov >= required) & monthly_mean.notna()

        counts = ok.groupby(ok.index.month).sum().reindex(idx12, fill_value=0).astype(int)
        cov_pct = (monthly_cov.groupby(monthly_cov.index.month).mean().reindex(idx12)) * 100

        # Mean climatology (unchanged)
        clim_mean = monthly_mean[ok].groupby(monthly_mean[ok].index.month).mean().reindex(idx12)

        # -------------------------------------------------------------------
        # Compute P10/P90 from ALL DAILY VALUES in valid month-years
        # -------------------------------------------------------------------
        daily = _daily_mean_complete(series)

        valid_months = ok[ok].index

        mask = pd.Series(False, index=daily.index)
        for ts in valid_months:
            mask |= (
                (daily.index.year == ts.year) &
                (daily.index.month == ts.month)
            )

        daily_valid = daily[mask]

        grp = daily_valid.groupby(daily_valid.index.month)

        p10 = grp.quantile(0.10).reindex(idx12)
        p90 = grp.quantile(0.90).reindex(idx12)

        low_n = counts < min_samples
        clim_mean[low_n] = np.nan
        p10[low_n] = np.nan
        p90[low_n] = np.nan
       

        return clim_mean, p10, p90, counts, cov_pct

    coverage_report = []

    def _append_coverage(label, counts, cov_pct):
        coverage_report.append((label, counts, cov_pct))

    # ── Fonts: everything 1.5× the previous sizes, in Arial ─────────
    FSC = 2
    FS_LABEL  = 10 * FSC   # axis labels
    FS_LEGEND = 8 * FSC    # legends
    FS_PANEL  = 11 * FSC   # (a)–(f) panel letters
    FS_TICK   = 10 * FSC   # tick labels
    FS_XTICK  = 9 * FSC    # month labels
    _prev_rc = {k: plt.rcParams[k] for k in ('font.family', 'font.size')}
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size']   = 10 * FSC   # scales any text without explicit size

    # ── File paths ──────────────────────────────────────────────────
    LAPSE_RATE_PATH  = str(_DATA_DIR / 'LapseRate' / 'lapse_rate.csv')
    ISOTHERM_PATH    = str(_DATA_DIR / 'Zero_isotherm' / 'zero_isotherm.csv')
    MOISTURE_FILES   = {
        'Kyangjin AWS':  str(_DATA_DIR / 'Moisture' / 'Kyangjin_AWS_humidity_timeseries.csv'),
        'Yala BC AWS':   str(_DATA_DIR / 'Moisture' / 'Yala_BC_AWS_humidity_timeseries.csv'),
        'Morimoto':      str(_DATA_DIR / 'Moisture' / 'Morimoto_MM_humidity_timeseries.csv'),
    }
    AWS_STATIONS = ['Kyangjin AWS', 'Yala BC AWS', 'Morimoto']

    # ── Abbreviations and elevations for legend labels ───────────────
    STATION_LABEL = {
        'Kyangjin AWS': 'PLU1 (3862 m)',
        'Yala BC AWS':  'PLU2 (5090 m)',
        'Morimoto':     'PLU3 (4919 m)',
    }

    # ── Unified color/linestyle per station ─────────────────────────
    STATION_COLORS = {
        'Kyangjin AWS': '#08519c',   # dark blue
        'Yala BC AWS':  '#d94701',   # dark orange
        'Morimoto':     'tab:olive',
    }
    STATION_MARKERS = {'Kyangjin AWS': 'o', 'Yala BC AWS': 's', 'Morimoto': '^'}

    # ── Load KINC / LINC daily means and compute SW/LW ratio ────────
    swlw_monthly = {}; swlw_p10 = {}; swlw_p90 = {}
    swlw_hourly = {}  # store hourly ratio series per station
    try:
        for station in AWS_STATIONS:
            file_path = get_dir([station])
            if not file_path:
                file_path = get_dir(station)
            if not (isinstance(file_path, (list, tuple)) and file_path and file_path[0]):
                continue
            aws_data = read_AWS(file_path[0])
            if 'DATETIME' in aws_data.columns:
                aws_data['DATETIME'] = pd.to_datetime(aws_data['DATETIME'])
                aws_data.set_index('DATETIME', inplace=True)
            if 'KINC' in aws_data.columns and 'LINC' in aws_data.columns:
                kinc = pd.to_numeric(aws_data['KINC'], errors='coerce')
                linc = pd.to_numeric(aws_data['LINC'], errors='coerce')

               

                if station == 'Yala BC AWS':
                    linc = linc[(linc.index.year >= 2012) & (linc.index.year <= 2019)]
                    kinc = kinc[(kinc.index.year >= 2012) & (kinc.index.year <= 2019)]
                    

                ratio_h_raw = kinc / linc.reindex(kinc.index)
                ratio_h_raw = ratio_h_raw.where(ratio_h_raw > 0, np.nan)
                ratio_h = ratio_h_raw.dropna()
                
                

                swlw_hourly[station] = ratio_h
                # Ratio of plain daily means, pooled per calendar month.
                sw_daily = kinc.resample('D').mean()
                lw_daily = linc.resample('D').mean()
                ratio = (sw_daily / lw_daily.reindex(sw_daily.index)).dropna()
                ratio = ratio[ratio > 0]

                grp = ratio.groupby(ratio.index.month)
                swlw_monthly[station] = grp.mean()
                swlw_p10[station]     = grp.quantile(0.10)
                swlw_p90[station]     = grp.quantile(0.90)
                cnt, cov = _monthyear_counts_and_coverage(ratio_h_raw)
                _append_coverage(f"SW/LW - {station}", cnt, cov)

                

            # Plot daily timeseries for SW/LW for Yala BC AWS
           
        # ── Hourly SW/LW timeseries (all stations) ──────────────────
        if swlw_hourly:
            fig_h, ax_h = plt.subplots(figsize=(14, 4))
            for station, ratio_h in swlw_hourly.items():
                color = STATION_COLORS.get(station, None)
                label = STATION_LABEL.get(station, station)
                ax_h.plot(ratio_h.index, ratio_h.values, color=color,
                          linewidth=0.5, alpha=0.7, label=label)
            ax_h.set_title('SW/LW Ratio – Hourly Timeseries')
            ax_h.set_xlabel('Date')
            ax_h.set_ylabel('SW/LW (–)')
            ax_h.legend(fontsize=FS_LEGEND, frameon=False)
            ax_h.grid(True, linestyle='--', alpha=0.4)
            fig_h.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Could not load radiation data: {e}")

    # ── Load lapse rate ─────────────────────────────────────────────
    try:
        lr_df = pd.read_csv(LAPSE_RATE_PATH, index_col=0, parse_dates=True)
        monthly_lr, monthly_lr_p10, monthly_lr_p90, cnt, cov = \
            _monthly_climatology_from_valid_monthyears(lr_df['lapse_rate'])
        monthly_lr = monthly_lr * 1000
        monthly_lr_p10 = monthly_lr_p10 * 1000
        monthly_lr_p90 = monthly_lr_p90 * 1000
        _append_coverage("Lapse rate", cnt, cov)
        lr_ok = True
    except FileNotFoundError:
        lr_ok = False
        print("Lapse rate file not found.")

    # ── Load zero-degree isotherm ───────────────────────────────────
    try:
        iso_df = pd.read_csv(ISOTHERM_PATH, index_col=0, parse_dates=True)
        monthly_iso, monthly_iso_p10, monthly_iso_p90, cnt, cov = \
            _monthly_climatology_from_valid_monthyears(iso_df['Daily_Zero_Deg_Elevation'])
        _append_coverage("Zero-degree isotherm", cnt, cov)
        iso_ok = True
    except FileNotFoundError:
        iso_ok = False
        print("Isotherm file not found.")

    # ── Load moisture (RH + mixing ratio) ──────────────────────────
    moisture_monthly = {}
    for station, path in MOISTURE_FILES.items():
        try:
            df = pd.read_csv(path, index_col='DATETIME', parse_dates=True)
            df.columns = df.columns.str.strip()
            rh_mean = rh_p10 = rh_p90 = None
            mr_mean = mr_p10 = mr_p90 = None
            if 'RH' in df.columns:
                rh_mean, rh_p10, rh_p90, cnt_rh, cov_rh = _monthly_climatology_from_valid_monthyears(df['RH'])
                _append_coverage(f"RH - {station}", cnt_rh, cov_rh)
            if 'MIXING_RATIO' in df.columns:
                mr_mean, mr_p10, mr_p90, cnt_mr, cov_mr = _monthly_climatology_from_valid_monthyears(df['MIXING_RATIO'])
                _append_coverage(f"MR - {station}", cnt_mr, cov_mr)
            moisture_monthly[station] = {
                'RH_mean':  rh_mean,
                'MR_mean':  mr_mean,
                'RH_p10':   rh_p10,
                'RH_p90':   rh_p90,
                'MR_p10':   mr_p10,
                'MR_p90':   mr_p90,
            }
        except FileNotFoundError:
            print(f"Moisture file not found for {station}.")

    # ── Load mean monthly temperature from cleaned pluvio data ──────
    temp_monthly = {}; temp_p10 = {}; temp_p90 = {}
    for station in AWS_STATIONS:
        try:
            df = read_pluvio_cleaned([station], dt='1h')[station]
            t_col = next((c for c in ['Temperature_1H', 'Temperature_15min', 'Temperature']
                          if c in df.columns), None)
            if t_col:
                temp = pd.to_numeric(df[t_col], errors='coerce')
                temp_monthly[station], temp_p10[station], temp_p90[station], cnt, cov = \
                    _monthly_climatology_from_valid_monthyears(temp)
                _append_coverage(f"Temperature - {station}", cnt, cov)
        except Exception as e:
            print(f"Could not load temperature for {station}: {e}")

    print("\n" + "=" * 118)
    print("MONTHLY DATA COVERAGE OVERVIEW")
    print(f"Rules: daily means use >= {MIN_DAILY_HOURS} valid hours/day (where applicable), "
          f"month-year coverage >= {int(MONTHLY_COVERAGE_REQ * 100)}%, "
          f"monthly climatology needs >= {MIN_VALID_MONTHYEARS} valid month-years")
    print("=" * 118)
    for label, counts, cov in coverage_report:
        print(f"\n{label}")
        for m, mon in enumerate(MONTH_LABELS, start=1):
            c = int(counts.loc[m]) if m in counts.index else 0
            cv = cov.loc[m] if m in cov.index else np.nan
            cv_txt = f"{cv:5.1f}%" if pd.notna(cv) else "  n/a "
            used = "yes" if c >= MIN_VALID_MONTHYEARS else "no"
            print(f"  {mon:>3s}: mean_cov={cv_txt} | valid_month-years={c:2d} | used_in_mean={used}")
    print("=" * 118 + "\n")

    # ── Build 3×2 figure ────────────────────────────────────────────
    # Layout (row, col):
    #   (0,0) Temperature   (0,1) z0
    #   (1,0) SW/LW ratio   (1,1) Lapse rate
    #   (2,0) MR            (2,1) RH
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    fig.subplots_adjust(hspace=0.25, wspace=0.35, left=0.09, right=0.97, top=0.93, bottom=0.07)
    ax_t   = axes[0, 0]
    ax_iso = axes[0, 1]
    ax_sw  = axes[1, 0]
    ax_lr  = axes[1, 1]
    ax_mr  = axes[2, 0]
    ax_rh  = axes[2, 1]

    # ── (a) Daily-mean SW/LW ratio ──────────────────────────────────
    for i, station in enumerate(AWS_STATIONS):
        if station not in swlw_monthly:
            continue
        color  = STATION_COLORS[station]
        marker = STATION_MARKERS[station]
        ax_sw.fill_between(swlw_p10[station].index,
                           swlw_p10[station].values, swlw_p90[station].values,
                           color=color, alpha=0.15,
                           label='P10–P90' if i == 0 else '')
        ax_sw.plot(swlw_monthly[station].index, swlw_monthly[station].values,
                   marker=marker, color=color, linewidth=1.8, markersize=5, label=STATION_LABEL.get(station, station))
    ax_sw.set_ylabel('SW/LW (–)', fontsize=FS_LABEL)
    ax_sw.legend(fontsize=FS_LEGEND, frameon=False)
    ax_sw.grid(True, linestyle='--', alpha=0.4)


    # ── (b) Mean monthly temperature ────────────────────────────────
    for i, station in enumerate(AWS_STATIONS):
        if station not in temp_monthly:
            continue
        color  = STATION_COLORS[station]
        marker = STATION_MARKERS[station]
        ax_t.fill_between(temp_p10[station].index, temp_p10[station].values, temp_p90[station].values,
                           color=color, alpha=0.15,
                           label='P10–P90' if i == 0 else '')
        ax_t.plot(temp_monthly[station].index, temp_monthly[station].values,
                  marker=marker, color=color, linewidth=1.8, markersize=5, label=STATION_LABEL.get(station, station))
    ax_t.set_ylabel('T (°C)', fontsize=FS_LABEL)
    ax_t.legend(fontsize=FS_LEGEND, frameon=False)
    ax_t.grid(True, linestyle='--', alpha=0.4)

    # ── (c) Zero-degree isotherm ────────────────────────────────────
    if iso_ok:
        ax_iso.fill_between(monthly_iso_p10.index, monthly_iso_p10.values, monthly_iso_p90.values,
                            color='#0066CC', alpha=0.20, label='P10–P90')
        ax_iso.plot(monthly_iso.index, monthly_iso.values, marker='o', color='#0066CC',
                    linewidth=1.8, markersize=5, label='Mean')
    ax_iso.set_ylabel('$z_{0}$ (m a.s.l.)', fontsize=FS_LABEL)
    ax_iso.yaxis.set_label_position('right')
    ax_iso.yaxis.tick_right()
    ax_iso.legend(fontsize=FS_LEGEND, frameon=False)
    ax_iso.grid(True, linestyle='--', alpha=0.4)

    # ── (d) Lapse rate ──────────────────────────────────────────────
    if lr_ok:
        ax_lr.fill_between(monthly_lr_p10.index, monthly_lr_p10.values, monthly_lr_p90.values,
                            color='#FF8C00', alpha=0.20, label='P10–P90')
        ax_lr.plot(monthly_lr.index, monthly_lr.values, marker='o', color='#FF8C00',
                   linewidth=1.8, markersize=5, label='Mean')
    ax_lr.set_ylabel('LR (°C km$^{-1}$)', fontsize=FS_LABEL)
    ax_lr.yaxis.set_label_position('right')
    ax_lr.yaxis.tick_right()
    ax_lr.legend(fontsize=FS_LEGEND, frameon=False)
    ax_lr.grid(True, linestyle='--', alpha=0.4)

    # ── (e) Mixing ratio ────────────────────────────────────────────
    for station, data in moisture_monthly.items():
        if data['MR_mean'] is None:
            continue
        color  = STATION_COLORS.get(station, 'tab:gray')
        marker = STATION_MARKERS.get(station, 'o')
        ax_mr.fill_between(data['MR_p10'].index, data['MR_p10'].values, data['MR_p90'].values,
                            color=color, alpha=0.15)
        ax_mr.plot(data['MR_mean'].index, data['MR_mean'].values, marker=marker, color=color,
                   linewidth=1.8, markersize=5, label=STATION_LABEL.get(station, station))
    ax_mr.set_ylabel('MR (g kg$^{-1}$)', fontsize=FS_LABEL)
    ax_mr.legend(fontsize=FS_LEGEND, frameon=False)
    ax_mr.grid(True, linestyle='--', alpha=0.4)

    # ── (f) Relative humidity ───────────────────────────────────────
    for station, data in moisture_monthly.items():
        if data['RH_mean'] is None:
            continue
        color  = STATION_COLORS.get(station, 'tab:gray')
        marker = STATION_MARKERS.get(station, 'o')
        ax_rh.fill_between(data['RH_p10'].index, data['RH_p10'].values, data['RH_p90'].values,
                            color=color, alpha=0.15)
        ax_rh.plot(data['RH_mean'].index, data['RH_mean'].values, marker=marker, color=color,
                   linewidth=1.8, markersize=5, label=STATION_LABEL.get(station, station))
    ax_rh.set_ylabel('RH (%)', fontsize=FS_LABEL)
    ax_rh.yaxis.set_label_position('right')
    ax_rh.yaxis.tick_right()
    ax_rh.legend(fontsize=FS_LEGEND, frameon=False)
    ax_rh.grid(True, linestyle='--', alpha=0.4)

    # ── Panel labels ────────────────────────────────────────────────
    for ax, letter in zip([ax_t, ax_iso, ax_sw, ax_lr, ax_mr, ax_rh],
                          ['a', 'b', 'c', 'd', 'e', 'f']):
        ax.text(0.03, 0.97, f'({letter})', transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight='bold', va='top', ha='left')

    # ── Shared x-axis formatting (bottom row only) ──────────────────
    for ax in axes[2, :]:
        ax.set_xticks(MONTHS)
        ax.set_xticklabels(MONTH_LABELS, fontsize=FS_XTICK)
        ax.set_xlim(0.5, 12.5)

    for ax in axes.flatten():
        ax.tick_params(axis='both', labelsize=FS_TICK)

    plt.subplots_adjust(left=0.06, bottom=0.07, right=0.933, top=0.924,
                        wspace=0.036, hspace=0.25)
    plt.show()
    plt.rcParams.update(_prev_rc)   # don't leak Arial/size into other figures
    return fig
def plot_precip_august_may_comparison_with_slope_stations(STATION_ABBREV=STATION_ABBREV,
                                                          font_scale=1.5):
    """
    Extended version of plot_precip_august_may_comparison() that includes
    slope stations (TB3-TB5, PLU2/Yala BC AWS) as scatter points overlaid on the plot.
    These stations are not part of the main profile line but are shown with
    different markers and colors to distinguish them.

    Spatial visualization: how May (pre-monsoon) vs August (monsoon) daily
    precipitation is distributed along the valley profile.
    """
    plt.rcParams['font.family'] = 'Arial'
    from scipy import stats
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        _HAS_GPD = True
    except ImportError:
        _HAS_GPD = False

    GEOMETRY_CSV = str(_DATA_DIR / "Geometry"
                       / "calculated_cross_sectional_areas_and_width.csv")
    PROFILE_SHP  = str(_DATA_DIR / "ArcGIS" / "Study_area" / "profile_langtang.shp")

    # Profile stations only
    PROFILE_STATIONS = [
        'Syabru TB', 'Lama TB', 'Kyangjin AWS', 'Jathang TB', 'Numthang TB',
        'Langshisha BC TB', 'Morimoto TB', 'Morimoto Pluvio'
    ]

    # Slope stations (PLU2/Yala BC AWS + TB3, TB4, TB5)
    SLOPE_STATIONS = ['Yala BC AWS']
    ALL_STATIONS = PROFILE_STATIONS + SLOPE_STATIONS

    print(f"\n{'='*60}")
    print(f"Stations being loaded:")
    for stn in ALL_STATIONS:
        print(f"  {STATION_ABBREV.get(stn, stn[:4])}: {stn}")
    print(f"{'='*60}\n")

    
    # Monsoon = Jun 15 – Sep 30 (June before the 15th is filtered out below)
    MONTHS  = [('May', [5]), ('Monsoon', [6, 7, 8, 9])]
    PERIODS = [
        ('All hours', list(range(24))),
        ('Daytime',   list(range(7, 19))),                        # 07–19
        ('Nighttime', [19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]), # 19–07
        ('Daytime core',   [12, 13, 14, 15]),                     # 12–16
        ('Nighttime core', [21, 22, 23, 0]),                      # 21–01
    ]
    MONTH_STYLE = {
        'May':    dict(color='#e6a817', marker='D', ls='--', lw=2.0),
        'August': dict(color='#1f6eb5', marker='o', ls='-',  lw=2.5),
    }
    MIN_COV         = 0.70
    MIN_DAYS_PER_YR = 10

    # Font sizes, scalable via font_scale
    FS_LABEL  = 11  * font_scale   # y-axis labels
    FS_XLABEL = 12  * font_scale   # x-axis label
    FS_TICK   = 10  * font_scale   # tick labels
    FS_ANNOT  = 8.5 * font_scale   # station annotations
    FS_LEGEND = 9.5 * font_scale   # legend text and titles

    # ── Load geometry ─────────────────────────────────────────────────────
    try:
        df_geom = pd.read_csv(GEOMETRY_CSV)
    except FileNotFoundError:
        print(f"⚠ Geometry CSV not found:\n  {GEOMETRY_CSV}")
        return
    if not all(c in df_geom.columns for c in ['distance_m', 'elevation_m', 'valley_width_top_m']):
        print("⚠ Geometry CSV missing required columns. Re-run analyze_valley_geometry().")
        return
    # Compute slope from elevation/distance if not already stored
    if 'slope_m_per_km' not in df_geom.columns:
        dist_km_col = df_geom['distance_m'].values / 1000.0
        df_geom['slope_m_per_km'] = np.gradient(
            df_geom['elevation_m'].values, dist_km_col)
    geom_dist_km = df_geom['distance_m'].values / 1000.0

    # ── Station positions ─────────────────────────────────────────────────
    # Get station coordinates and elevations
    lon_raw, lat_raw = get_station_coordinate(ALL_STATIONS)
    lon_arr = np.array(lon_raw) / 100000.0
    lat_arr = np.array(lat_raw) / 100000.0
    stn_elev_actual = np.array(get_elevation(ALL_STATIONS))  # actual station elevation

    N = len(ALL_STATIONS)
    N_profile = len(PROFILE_STATIONS)
    station_dist_km = np.full(N, np.nan)

    # Project stations onto the profile line using proper UTM projection
    projection_success = True
    if _HAS_GPD:
        try:
            profile_gdf  = gpd.read_file(PROFILE_SHP)
            profile_utm  = profile_gdf.to_crs(epsg=32645)
            profile_line = profile_utm.geometry.union_all()

            print("\n  Station projection (EPSG:32645 UTM):")
            for i, (lon, lat) in enumerate(zip(lon_arr, lat_arr)):
                try:
                    pt = gpd.GeoSeries([Point(lon, lat)],
                                       crs='EPSG:4326').to_crs(epsg=32645).iloc[0]
                    dist = profile_line.project(pt) / 1000.0
                    station_dist_km[i] = dist
                    print(f"    {STATION_ABBREV.get(ALL_STATIONS[i], '?'):5s}: "
                          f"lon={lon:.5f} lat={lat:.5f} → d={dist:.2f} km")
                except Exception as e:
                    print(f"    {STATION_ABBREV.get(ALL_STATIONS[i], '?'):5s}: "
                          f"projection failed ({e})")
                    projection_success = False
        except Exception as e:
            print(f"⚠ Profile loading failed ({e}); will use elevation fallback for failed projections.")
            projection_success = False

    # Force Yala BC AWS to the same valley position as Jathang TB (TB6)
    if 'Yala BC AWS' in ALL_STATIONS and 'Jathang TB' in ALL_STATIONS:
        yala_idx = ALL_STATIONS.index('Yala BC AWS')
        tb6_idx  = ALL_STATIONS.index('Jathang TB')
        station_dist_km[yala_idx] = station_dist_km[tb6_idx]
        print(f"  → Forced Yala BC AWS to TB6 (Jathang TB) valley position: "
              f"{station_dist_km[tb6_idx]:.2f} km")

    # Valley floor elevation at each station's profile distance
    stn_floor_elev = np.interp(station_dist_km, geom_dist_km,
                               df_geom['elevation_m'].values)

    # Summary of station positions
    print("\n  Final station positions (sorted by distance along profile):")
    station_summary = [(i, ALL_STATIONS[i], station_dist_km[i], stn_elev_actual[i], stn_floor_elev[i])
                       for i in range(N) if np.isfinite(station_dist_km[i])]
    station_summary.sort(key=lambda x: x[2])
    for i, stn, dist, actual_elev, floor_elev in station_summary:
        is_profile = i < N_profile
        print(f"    {STATION_ABBREV.get(stn, stn[:4]):5s}: d={dist:6.2f} km  "
              f"actual_elev={actual_elev:5.0f} m  floor_elev={floor_elev:5.0f} m  "
              f"{'[profile]' if is_profile else '[slope]'}")

    # ── Load precipitation ────────────────────────────────────────────────
    try:
        rain_raw = read_pluvio_cleaned(ALL_STATIONS, dt='1h')
    except Exception as e:
        print(f"⚠ read_pluvio_cleaned failed: {e}")
        return

    # ── Compute mean daily precip ─────────────────────────────────────────
    precip_arr = {m: {p: np.full(N, np.nan) for p, _ in PERIODS}
                  for m, _ in MONTHS}
    n_days_used = {m: {} for m, _ in MONTHS}   # season → station → valid days
    for si, station in enumerate(ALL_STATIONS):
        df_r = rain_raw.get(station, pd.DataFrame())
        if df_r.empty:
            print(f"⚠ No data for {station} ({STATION_ABBREV.get(station, '?')})")
            continue
        if 'DATETIME' in df_r.columns:
            df_r = df_r.copy()
            df_r['DATETIME'] = pd.to_datetime(df_r['DATETIME'])
            df_r.set_index('DATETIME', inplace=True)
        rain_col = next((c for c in ['Rainfall_1H', 'Hourly_Rain']
                         if c in df_r.columns), None)
        if rain_col is None:
            continue
        rain = pd.to_numeric(df_r[rain_col], errors='coerce')

        # ── Daily validity mask (full 24 h day) ──────────────────────
        # Pluviometers (Pluvio / AWS): all 24 h must be present and valid.
        # Tipping buckets (TB): NaNs are allowed only in the evening/night
        # (after 19:00 and before 08:00); a day with more than 50 % NaN
        # values is removed entirely.
        _is_pluvio = ('Pluvio' in station) or ('AWS' in station)
        _rain_full = rain.resample('1h').asfreq()   # missing hours → NaN
        _valid     = _rain_full.notna()
        _day_idx   = _rain_full.index.normalize()
        _n_valid   = _valid.groupby(_day_idx).sum()
        if _is_pluvio:
            _day_ok = _n_valid >= 24
        else:
            _is_daytime  = ((_rain_full.index.hour >= 8) &
                            (_rain_full.index.hour < 19))
            _bad_daytime = (~_valid) & _is_daytime
            _n_bad_day   = _bad_daytime.groupby(_day_idx).sum()
            _day_ok = (_n_bad_day == 0) & (_n_valid >= 12)
        _ok_days = _day_ok.index[_day_ok]

        for month_name, month_nums in MONTHS:
            for period_name, hours in PERIODS:
                month_filter = rain.index.month.isin(month_nums)
                if month_name == 'Monsoon':
                    # Monsoon starts mid-June: drop Jun 1–14
                    month_filter = month_filter & ~(
                        (rain.index.month == 6) & (rain.index.day < 15))
                sub  = rain[month_filter & (rain.index.hour.isin(hours))]
                if len(sub) == 0:
                    continue
                year_means = []
                n_days_total = 0
                for yr in np.unique(sub.index.year):
                    yr_sub   = sub[sub.index.year == yr]
                    daily    = yr_sub.resample('D').sum()
                    daily_ok = daily[daily.index.isin(_ok_days)]
                    if len(daily_ok) >= MIN_DAYS_PER_YR:
                        year_means.append(daily_ok.mean())
                        n_days_total += len(daily_ok)
                if year_means:
                    precip_arr[month_name][period_name][si] = np.mean(year_means)
                if period_name == 'All hours':
                    n_days_used[month_name][station] = n_days_total

    # Number of valid days behind each daily mean (All hours)
    print("\nValid days used for the daily means:")
    print(f"  {'':5s} {'Station':22s} {'May':>5s} {'Monsoon':>8s}")
    for station in ALL_STATIONS:
        print(f"  {STATION_ABBREV.get(station, station[:4]):5s} "
              f"{station:22s} "
              f"{n_days_used['May'].get(station, 0):>5d} "
              f"{n_days_used['Monsoon'].get(station, 0):>8d}")

    # Debug: show which stations have monsoon data
    monsoon_data = precip_arr['Monsoon']['All hours']
    print("✓ Stations with monsoon data:")
    for si, station in enumerate(ALL_STATIONS):
        if np.isfinite(monsoon_data[si]):
            print(f"  {STATION_ABBREV.get(station, station[:4]):5} - {station}: {monsoon_data[si]:.2f} mm")
    print("\n✗ Stations missing monsoon data:")
    for si, station in enumerate(ALL_STATIONS):
        if not np.isfinite(monsoon_data[si]):
            print(f"  {STATION_ABBREV.get(station, station[:4]):5} - {station}")
    print()

    # ── Figure: single precipitation panel ──────────────────────────────
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, ax_all = plt.subplots(1, 1, figsize=(14, 7))

    # ── Check profile extent vs station distances ─────────────────────────
    geom_dist_max = geom_dist_km.max()
    for i, station in enumerate(ALL_STATIONS):
        if np.isfinite(station_dist_km[i]) and station_dist_km[i] > geom_dist_max:
            print(f"⚠  WARNING: '{station}' is at {station_dist_km[i]:.2f} km "
                  f"but the geometry/profile shapefile only extends to "
                  f"{geom_dist_max:.2f} km. "
                  f"Station will be plotted but falls outside the elevation profile.")

    # Fixed x range 0–44 km for the full figure
    x_min = 0.0
    x_max = 44.0

    # Elevation silhouette + secondary y-axis (right)
    ax_elev = ax_all.twinx()
    elev_min = df_geom['elevation_m'].min()
    elev_max = df_geom['elevation_m'].max()
    ax_elev.fill_between(geom_dist_km, df_geom['elevation_m'], elev_min,
                         color='#b0b0b0', alpha=0.45, zorder=0)
    ax_elev.set_ylim(elev_min - 200, elev_max + 600)
    ax_elev.set_xlim(x_min, x_max)
    ax_elev.set_ylabel('Elevation (m a.s.l.)', fontsize=FS_LABEL,
                       color='#555')
    ax_elev.tick_params(axis='y', labelcolor='#555', labelsize=FS_TICK)
    ax_elev.spines['top'].set_visible(False)

    # Station dots on elevation profile
    # On-profile stations: show at valley floor elevation (red circles)
    for i in range(N_profile):
        if np.isfinite(station_dist_km[i]):
            elev_at_stn = stn_floor_elev[i] if np.isfinite(stn_floor_elev[i]) else stn_elev_actual[i]
            ax_elev.plot(station_dist_km[i], elev_at_stn, 'o',
                         color='#c0392b', markersize=6, zorder=4,
                         markeredgecolor='darkred', markeredgewidth=0.8)

    # Off-profile slope stations: show at their actual elevation (floating above profile)
    # Plot both at floor elevation (connected to profile) and actual elevation (with marker)
    for i in range(N_profile, N):
        if np.isfinite(station_dist_km[i]) and np.isfinite(stn_elev_actual[i]):
            # Vertical line from floor to actual elevation (subtle connection)
            ax_elev.plot([station_dist_km[i], station_dist_km[i]],
                        [stn_floor_elev[i], stn_elev_actual[i]],
                        color='#e63946', linewidth=1, linestyle=':', alpha=0.6, zorder=3)
            # Marker at actual elevation
            ax_elev.scatter(station_dist_km[i], stn_elev_actual[i], marker='*',
                           s=300, color='#e63946', zorder=5,
                           edgecolors='black', linewidth=0.5)

    # Lines: (label, month_name, period_name, color, linestyle, marker, lw)
    # Monsoon sub-periods use tones of the monsoon blue:
    # light = daytime (12–16), dark = nighttime (21–01)
    LINES = [
        ('May daily mean',
         'May',      'All hours', '#e6a817', '-', 'D', 2.0),
        ('Monsoon daily mean',
         'Monsoon',  'All hours', '#1f6eb5', '-', 'o', 2.5),
        ('Monsoon daytime (12–16h)',
         'Monsoon',  'Daytime core',   '#6baed6', '-', '^', 2.0),
        ('Monsoon nighttime (21–01h)',
         'Monsoon',  'Nighttime core', '#08306b', '-', 's', 2.0),
    ]

    # Per-station y-offset (pts) for daily-mean labels.
    LABEL_OFFSET = {
        'Langshisha BC TB': -14,
        'Ganja La TB1': 11,
        'Ganja La TB2': 11,
        'Ganja La TB3': 11,
    }

    precip_handles = []
    for label, month_name, period_name, col, ls, mk, lw in LINES:
        y     = precip_arr[month_name][period_name]
        # For main lines, only use profile stations
        valid_profile = (np.isfinite(station_dist_km[:N_profile]) &
                        np.isfinite(y[:N_profile]))
        xv    = station_dist_km[:N_profile][valid_profile]
        yv    = y[:N_profile][valid_profile]
        nv    = [PROFILE_STATIONS[i] for i in range(N_profile) if valid_profile[i]]
        sort_i = np.argsort(xv)
        ln, = ax_all.plot(xv[sort_i], yv[sort_i],
                          color=col, linewidth=lw, linestyle=ls, marker=mk,
                          markersize=10, markeredgecolor='white',
                          markeredgewidth=0.8, zorder=5, label=label,
                          alpha=1.0 if period_name == 'All hours' else 0.6)
        precip_handles.append(ln)

        # Labels only on monsoon daily-mean line (profile stations only)
        if period_name == 'All hours' and month_name == 'Monsoon':
            for xi, yi, nm in zip(xv[sort_i], yv[sort_i],
                                   [nv[j] for j in sort_i]):
                yoff = LABEL_OFFSET.get(nm, 11)
                ax_all.annotate(STATION_ABBREV.get(nm, nm[:4]),
                                (xi, yi), textcoords='offset points',
                                xytext=(0, yoff), fontsize=FS_ANNOT,
                                ha='center',
                                va='top' if yoff < 0 else 'bottom',
                                color=col)

        # Overlay slope stations as scatter points (all-hours lines only)
        if period_name != 'All hours':
            continue
        valid_slope = (np.isfinite(station_dist_km[N_profile:]) &
                      np.isfinite(y[N_profile:]))
        if np.any(valid_slope):
            xv_slope = station_dist_km[N_profile:][valid_slope]
            yv_slope = y[N_profile:][valid_slope]
            nv_slope = [SLOPE_STATIONS[i] for i in range(len(SLOPE_STATIONS)) if valid_slope[i]]
            sort_i_slope = np.argsort(xv_slope)

            # One marker for all slope-station values: star, coloured like
            # the matching line (May yellow / monsoon blues)
            marker_slope = '*'
            size_slope = 300

            ax_all.scatter(xv_slope[sort_i_slope], yv_slope[sort_i_slope],
                          marker=marker_slope, s=size_slope, color=col,
                          edgecolors='black', linewidth=0.5, zorder=6, alpha=0.85)

            # Annotate slope stations on monsoon daily-mean line
            if period_name == 'All hours' and month_name == 'Monsoon':
                for xi, yi, nm in zip(xv_slope[sort_i_slope], yv_slope[sort_i_slope],
                                       [nv_slope[j] for j in sort_i_slope]):
                    yoff = LABEL_OFFSET.get(nm, 11)
                    ax_all.annotate(STATION_ABBREV.get(nm, nm[:4]),
                                    (xi, yi), textcoords='offset points',
                                    xytext=(0, yoff), fontsize=FS_ANNOT,
                                    ha='center',
                                    va='top' if yoff < 0 else 'bottom',
                                    color=col)

    ax_all.set_zorder(ax_elev.get_zorder() + 1)
    ax_all.patch.set_visible(False)
    ax_all.set_xlim(x_min-1, x_max)
    ax_all.set_ylabel('Mean precipitation (mm)', fontsize=FS_LABEL)
    ax_all.set_ylim(bottom=0)
    ax_all.set_xlabel('Distance along valley (km)', fontsize=FS_XLABEL)
    ax_all.tick_params(axis='both', labelsize=FS_TICK)
    ax_all.spines['top'].set_visible(False)
    ax_all.spines['right'].set_visible(False)
    ax_all.grid(True, linestyle='--', alpha=0.28, axis='y')
    ax_all.grid(True, linestyle=':', alpha=0.20, axis='x')

    # ── Legend: two categorised boxes ───────────────────────────────────
    from matplotlib.transforms import blended_transform_factory

    # Box 1 — precipitation series. Slope-station stars take the colour of
    # the matching line, so a single grey entry covers them all.
    
    elev_patch = Patch(facecolor='#b0b0b0', alpha=0.55, label='Elevation profile')
    stn_dot_h  = Line2D([0], [0], marker='o', color='none',
                         markerfacecolor='#c0392b', markeredgecolor='darkred',
                         markeredgewidth=0.8, markersize=6, label='Profile station')
    slope_dot_h = Line2D([0], [0], marker='*', color='none',
                         markerfacecolor='#e63946', markeredgecolor='black',
                         markeredgewidth=0.5, markersize=12, label='PLU2 location')

    trans_mixed = blended_transform_factory(ax_all.transData, ax_all.transAxes)
    leg_precip = ax_all.legend(handles=[*precip_handles],
                               fontsize=FS_LEGEND, title_fontsize=FS_LEGEND,
                               framealpha=0.92,
                               bbox_to_anchor=(17, 1.0), bbox_transform=trans_mixed,
                               loc='upper left', handlelength=1.8, labelspacing=0.3)
    ax_all.add_artist(leg_precip)
    ax_all.legend(handles=[elev_patch, stn_dot_h, slope_dot_h],
                  title='Valley profile',
                  fontsize=FS_LEGEND, title_fontsize=FS_LEGEND, framealpha=0.92,
                  loc='upper right', handlelength=1.8, labelspacing=0.3)

    plt.tight_layout()
    plt.savefig('precip_comparison_with_slope_stations.png', dpi=300, bbox_inches='tight')
    print("Figure saved as 'precip_comparison_with_slope_stations.png'")
    plt.show()

def plot_seasonal_undercatch_summary_from_corrected():
    """
    Seasonal undercatch summary based on the saved Kochendorfer correction:
    loads the *_kochendorfer_corrected.csv files (all stations combined) and
    plots (a) per-event undercatch box plots and (b) bulk seasonal undercatch
    bars. Also writes the seasonal statistics CSV.

    Reads the corrected files saved by process_and_save() in
    apply_kochendorfer_hourly.py.
    """
    # ── Global style (same as combined snowfall/rainfall figure) ─────────
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 15})
    base_fs = float(plt.rcParams.get('font.size', 15))
    FS = {'title': 1.45 * base_fs, 'label': 1.30 * base_fs,
          'tick': 1.15 * base_fs, 'annot': 1.00 * base_fs}

    def _panel_outline(ax_, color='#333333', lw=1.0):
        for _sp in ax_.spines.values():
            _sp.set_visible(True)
            _sp.set_linewidth(lw)
            _sp.set_color(color)

    def _pl(ax_, letter, x=1.0):
        return ax_.text(x, -0.02, f'({letter})', transform=ax_.transAxes,
                        ha='right', va='top', fontsize=FS['title'],
                        fontweight='bold', zorder=10, clip_on=False)

    SEASONS = {
        'annual': lambda idx: pd.Series(True, index=idx),
        'monsoon': lambda idx: ((idx.month == 6) & (idx.day >= 15)) | idx.month.isin([7, 8, 9]),
        'pre-monsoon': lambda idx: (idx.month >= 3) & ((idx.month < 6) | ((idx.month == 6) & (idx.day < 15))),
        'post-monsoon': lambda idx: idx.month.isin([10, 11, 12]),
        'winter': lambda idx: idx.month.isin([1, 2]),
    }
    season_colors = {
        'annual': '#555555',
        'pre-monsoon': '#e8a838',
        'monsoon': '#2ecc71',
        'post-monsoon': '#e67e22',
        'winter': '#3498db',
    }
    PLUVIO_SNOW_PARAMS  = {'a': 0.728, 'b': 0.230, 'c': 0.336, 'u_cap': 7.2}
    PLUVIO_MIXED_PARAMS = {'a': 0.668, 'b': 0.132, 'c': 0.339, 'u_cap': 7.2}

    # ── Load the corrected output files for all stations ─────────────────
    koch_dir = os.path.join(
        str(_DATA_DIR), 'Cleaned', 'Kochendorfer_corrected')
    stations = ['Kyangjin AWS', 'Yala BC AWS', 'Langshisha Pluvio',
                'Morimoto Pluvio', 'Ganja La Pluvio', 'Yala Pluvio']

    dfs = []
    for station in stations:
        out_file = os.path.join(koch_dir, f'{station}_kochendorfer_corrected.csv')
        if not os.path.isfile(out_file):
            print(f'Skipping {station}: corrected file not found.')
            continue
        df_st = pd.read_csv(out_file, parse_dates=['DATETIME'], index_col='DATETIME')
        required = {'Precipitation', 'Precipitation_corrected', 'Temperature', 'Catch_efficiency'}
        if not required.issubset(df_st.columns):
            print(f'Skipping {station}: required columns missing.')
            continue
        df_st['Station'] = station
        dfs.append(df_st)

    if not dfs:
        print('No corrected station files found — run process_and_save() first.')
        return

    df = pd.concat(dfs).sort_index()
    df = df.rename(columns={'Catch_efficiency': 'CE'})
    print(f'Seasonal undercatch summary over {len(dfs)} stations: '
          + ', '.join(sorted(df["Station"].unique())))

    prec      = df['Precipitation'].values
    corrected = df['Precipitation_corrected'].values
    t         = df['Temperature'].values

    # ── Seasonal statistics ───────────────────────────────────────────────
    season_labels, undercatch_pcts, undercatch_pcts_snowmixed = [], [], []
    median_ces, n_events, n_events_snowmixed = [], [], []
    box_data, box_data_snowmixed = [], []

    for season, condition in SEASONS.items():
        mask = np.asarray(condition(df.index))
        if not mask.any():
            continue

        p_s, c_s, t_s = prec[mask], corrected[mask], t[mask]

        # All events with precipitation > 0 and valid correction
        valid = (p_s > 0) & ~np.isnan(c_s) & ~np.isinf(c_s)
        # Only snow/mixed events
        snowmixed = valid & (t_s <= 2)

        if valid.sum() == 0:
            continue

        # Undercatch per event: (corrected - original) / corrected * 100
        undercatch = (c_s[valid] - p_s[valid]) / c_s[valid] * 100
        undercatch = undercatch[~np.isnan(undercatch) & ~np.isinf(undercatch)]
        box_data.append(undercatch)

        undercatch_sm = (c_s[snowmixed] - p_s[snowmixed]) / c_s[snowmixed] * 100
        undercatch_sm = undercatch_sm[~np.isnan(undercatch_sm) & ~np.isinf(undercatch_sm)]
        box_data_snowmixed.append(undercatch_sm)

        # Bulk undercatch (all)
        total_orig = np.nansum(p_s[valid])
        total_corr = np.nansum(c_s[valid])
        undercatch_pcts.append(
            (total_corr - total_orig) / total_corr * 100 if total_corr != 0 else 0)
        n_events.append(valid.sum())

        # Bulk undercatch (snow/mixed only)
        total_orig_sm = np.nansum(p_s[snowmixed])
        total_corr_sm = np.nansum(c_s[snowmixed])
        undercatch_pcts_snowmixed.append(
            (total_corr_sm - total_orig_sm) / total_corr_sm * 100 if total_corr_sm != 0 else 0)
        n_events_snowmixed.append(snowmixed.sum())

        ce_mask = mask & df['CE'].notna().values
        median_ces.append(np.nanmedian(df.loc[ce_mask, 'CE']) if ce_mask.any() else np.nan)

        season_labels.append(season)

    # ── Build figure: panel b only ────────────────────────────────────────
    fig, ax_bar = plt.subplots(1, 1, figsize=(9, 7), facecolor='white')

    # ── Panel b: bulk undercatch bars ─────────────────────────────────────
    width = 0.35
    x = np.arange(len(season_labels))
    colors = [season_colors.get(s, '#aaaaaa') for s in season_labels]
    bars_all = ax_bar.bar(x - width/2, undercatch_pcts, width=width, color=colors,
                          alpha=0.75, edgecolor='black', linewidth=0.6, label='All precipitation')
    bars_sm = ax_bar.bar(x + width/2, undercatch_pcts_snowmixed, width=width, color=colors,
                         alpha=0.4, edgecolor='black', linewidth=0.6, label='Snow/Mixed')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(season_labels, rotation=30, ha='right', fontsize=FS['tick'])
    ax_bar.set_ylabel('Total seasonal undercatch (%)', fontsize=FS['label'])
    ax_bar.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax_bar.tick_params(axis='y', labelsize=FS['tick'])
    # Add a small headroom so top annotations do not overlap the upper frame
    _max_bar_val = np.nanmax(np.r_[undercatch_pcts, undercatch_pcts_snowmixed]) if len(season_labels) else 0
    _ymin, _ymax = ax_bar.get_ylim()
    ax_bar.set_ylim(_ymin, max(_ymax, _max_bar_val + 3.0))
    _panel_outline(ax_bar)
    ax_bar.legend(fontsize=FS['tick'], frameon=True, facecolor='white', edgecolor='#d0d0d0')
    # Annotate bars with value
    for bar, val in zip(bars_all, undercatch_pcts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=FS['annot'], fontweight='bold')
    for bar, val in zip(bars_sm, undercatch_pcts_snowmixed):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=FS['annot'] - 1, fontweight='normal', color='gray')

    # ── Save seasonal statistics to CSV ───────────────────────────────────
    _csv_dir = str(_RESULTS_DIR / 'Kochendorfer')
    os.makedirs(_csv_dir, exist_ok=True)
    _stats_df = pd.DataFrame({
        'Season':                    season_labels,
        'Bulk_undercatch_all_pct':   [round(v, 2) for v in undercatch_pcts],
        'Bulk_undercatch_sm_pct':    [round(v, 2) for v in undercatch_pcts_snowmixed],
        'Median_CE':                 [round(v, 4) if not np.isnan(v) else np.nan for v in median_ces],
        'N_events_all':              n_events,
        'N_events_snowmixed':        n_events_snowmixed,
    })
    _params_df = pd.DataFrame([
        {'Precip_type': 'Snow (T<=-2C)',    'a': PLUVIO_SNOW_PARAMS['a'],  'b': PLUVIO_SNOW_PARAMS['b'],  'c': PLUVIO_SNOW_PARAMS['c'],  'u_cap': PLUVIO_SNOW_PARAMS['u_cap']},
        {'Precip_type': 'Mixed (-2<T<=2C)', 'a': PLUVIO_MIXED_PARAMS['a'], 'b': PLUVIO_MIXED_PARAMS['b'], 'c': PLUVIO_MIXED_PARAMS['c'], 'u_cap': PLUVIO_MIXED_PARAMS['u_cap']},
    ])
    _csv_path = os.path.join(_csv_dir, 'undercatch_seasonal_summary_hourly.csv')
    with open(_csv_path, 'w', newline='') as _f:
        _stats_df.to_csv(_f, index=False)
        _f.write('\n')
        _params_df.to_csv(_f, index=False)
    print(f'Seasonal undercatch summary saved to: {_csv_path}')

    plt.tight_layout()
    plt.show()


def snowfall_threshold_sensitivity():
    import os
    save_dir = str(_RESULTS_DIR / 'SNOW' / 'sensitivity')
    os.makedirs(save_dir, exist_ok=True)

    station_names_all = ['Langshisha Pluvio', 'Morimoto Pluvio', 'Kyangjin AWS', 'Yala BC AWS']

    elevation1 = get_elevation(station_names_all)

    # Build elevation -> "ABBREV (Full Name)" label map
    def _elev_label(elev):
        names = [n for n, e in zip(station_names_all, elevation1) if e == elev]
        if names:
            abbrev = STATION_ABBREV.get(names[0], names[0])
            return f'{abbrev}\n({names[0]})'
        return str(int(elev))

    # Create a new DataFrame to store rainfall, elevation, and month
    rain_elev_month_df = pd.DataFrame()

    merged_df = read_pluvio_cleaned(station_names_all)

    # Define temperature thresholds
    temp_thresholds = [-2, -1, 0, 1, 2]

    # Initialize a dictionary to store results for each threshold
    threshold_results = {}

    # Loop through each temperature threshold
    for threshold in temp_thresholds:
        rain_elev_month_df = pd.DataFrame()  # Reset for each threshold

        # Loop through each station and extract relevant data
        for station in station_names_all:
            df = merged_df[station].copy()
            print(f"Station: {station}, Columns: {df.columns.tolist()}")
            df.rename(columns={'Rainfall_1H': 'Hourly_Rain', 'Temperature_1H': 'TEMP'}, inplace=True)
            # Also handle alternative column names
            rain_col = next((c for c in df.columns if 'rain' in c.lower() or 'precip' in c.lower()), None)
            temp_col = next((c for c in df.columns if 'temp' in c.lower()), None)
            if 'Hourly_Rain' not in df.columns and rain_col:
                df.rename(columns={rain_col: 'Hourly_Rain'}, inplace=True)
            if 'TEMP' not in df.columns and temp_col:
                df.rename(columns={temp_col: 'TEMP'}, inplace=True)
            df.index = pd.to_datetime(df.index)

            # Rainfall
            df_rain = df.copy()
            df_rain['Hourly_Rain'] = df['Hourly_Rain'].where(df['TEMP'] >= threshold, np.nan)
            df_rain['Month'] = df_rain.index.month
            df_rain['Elevation'] = elevation1[station_names_all.index(station)]
            df_rain['Year'] = df_rain.index.year
            monthly_sum_rain = df_rain.groupby(['Year', 'Month'])['Hourly_Rain'].sum().reset_index()
            monthly_avg_rain = monthly_sum_rain.groupby('Month')['Hourly_Rain'].mean().reset_index()
            monthly_avg_rain['Elevation'] = elevation1[station_names_all.index(station)]

            # Snowfall
            df_snow = df.copy()
            df_snow['Hourly_Rain'] = df['Hourly_Rain'].where(df['TEMP'] < threshold, np.nan)
            df_snow['Month'] = df_snow.index.month
            df_snow['Elevation'] = elevation1[station_names_all.index(station)]
            df_snow['Year'] = df_snow.index.year
            monthly_sum_snow = df_snow.groupby(['Year', 'Month'])['Hourly_Rain'].sum().reset_index()
            monthly_avg_snow = monthly_sum_snow.groupby('Month')['Hourly_Rain'].mean().reset_index()
            monthly_avg_snow['Elevation'] = elevation1[station_names_all.index(station)]

            # Calculate the percentage of snowfall relative to total precipitation
            monthly_avg = monthly_avg_snow.copy()
            _total_mm = monthly_avg_snow['Hourly_Rain'] + monthly_avg_rain['Hourly_Rain']
            monthly_avg['Monthly_percentage'] = monthly_avg_snow['Hourly_Rain'] / _total_mm * 100
            monthly_avg['Snow_mm'] = monthly_avg_snow['Hourly_Rain'].values
            monthly_avg['Rain_mm'] = monthly_avg_rain['Hourly_Rain'].values
            rain_elev_month_df = pd.concat([rain_elev_month_df, monthly_avg[['Month', 'Elevation', 'Monthly_percentage', 'Snow_mm', 'Rain_mm']]])

        # Store results for the current threshold
        threshold_results[threshold] = rain_elev_month_df

    # Calculate relative change compared to 0°C threshold
    baseline = threshold_results[0].set_index(['Month', 'Elevation'])['Monthly_percentage']
    for threshold, df in threshold_results.items():
        if threshold != 0:
            df['Relative_change'] = (df.set_index(['Month', 'Elevation'])['Monthly_percentage'] - baseline).reset_index(drop=True)

    # ── Style settings matching plot_combined_snowfall_rainfall_analysis ──────
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 10})
    FS = {'title': 16.5, 'label': 15, 'tick': 13.5, 'annot': 12}
    MON_ABBREV = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    _CBAR_W, _CBAR_PAD = 0.012, 0.020

    def _abbrev_stn(elev):
        for i, e in enumerate(elevation1):
            if int(e) == int(elev):
                return STATION_ABBREV.get(station_names_all[i], station_names_all[i])
        return str(int(elev))

    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), facecolor='white')
    axes = axes.flatten()
    _panel_letters = list('abcdef')

    for idx, threshold in enumerate(temp_thresholds):
        ax = axes[idx]
        rain_elev_month_df = threshold_results[threshold]
        pivot_df = rain_elev_month_df.pivot_table(index='Elevation', columns='Month', values='Monthly_percentage', aggfunc='mean')
        pivot_sorted = pivot_df.sort_index(ascending=False)

        sns.heatmap(pivot_sorted, cmap='coolwarm', annot=True, fmt='.1f',
                    vmin=0, vmax=100, cbar=False,
                    ax=ax, annot_kws={'size': FS['annot']},
                    linewidths=0.3, linecolor='white')

        ax.set_yticklabels(
            [_abbrev_stn(e) for e in pivot_sorted.index],
            rotation=0, fontsize=FS['tick'])
        _ax_r = ax.twinx()
        _ax_r.set_ylim(ax.get_ylim())
        _ax_r.set_yticks([i + 0.5 for i in range(len(pivot_sorted.index))])
        _ax_r.set_yticklabels([f'{int(e)} m' for e in pivot_sorted.index], fontsize=FS['tick'])
        _ax_r.tick_params(axis='y', length=0)
        for _sp in _ax_r.spines.values():
            _sp.set_visible(False)

        ax.set_xticklabels(MON_ABBREV, fontsize=FS['tick'], rotation=90)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'Snowfall % – threshold {threshold:+d}°C', fontsize=FS['label'])
        ax.text(1.0, -0.02, f'({_panel_letters[idx]})', transform=ax.transAxes,
                ha='right', va='top', fontsize=FS['title'], fontweight='bold',
                zorder=10, clip_on=False)

    axes[5].set_visible(False)
    fig.subplots_adjust(hspace=0.38, wspace=0.60, left=0.08, right=0.92, top=0.96, bottom=0.09)
    fig.canvas.draw()

    _pos_top = axes[0].get_position()
    _pos_width = axes[1].get_position()
    _pos_bot = axes[4].get_position()
    _cax1 = fig.add_axes([_pos_width.x1 - _CBAR_PAD - _CBAR_W+0.1,
                          _pos_bot.y0, _CBAR_W, _pos_top.y1 - _pos_bot.y0])
    _sm1 = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=100))
    _sm1.set_array([])
    _cb1 = fig.colorbar(_sm1, cax=_cax1)
    _cb1.ax.yaxis.set_label_position('right')
    _cb1.ax.yaxis.tick_right()
    _cb1.set_label('Snowfall Percentage (%)', fontsize=FS['tick'])
    _cb1.ax.tick_params(labelsize=FS['tick'])

    plt.show()

    # Calculate percentage anomalies for each threshold compared to the 0°C threshold
    baseline = threshold_results[0].set_index(['Month', 'Elevation'])['Monthly_percentage']
    anomaly_results = {}

    for threshold, df in threshold_results.items():
        if threshold != 0:
            anomaly_df = df.copy()
            anomaly_df = anomaly_df.set_index(['Month', 'Elevation'])
            anomaly_df['Anomaly'] = anomaly_df['Monthly_percentage'] - baseline
            anomaly_df.reset_index(inplace=True)
            anomaly_results[threshold] = anomaly_df

    # Plot percentage anomalies as heatmaps
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10), facecolor='white')
    axes2 = axes2.flatten()
    _panel_letters2 = list('abcd')

    for idx, threshold in enumerate([t for t in temp_thresholds if t != 0]):
        ax = axes2[idx]
        anomaly_df = anomaly_results[threshold]
        pivot_df = anomaly_df.pivot_table(index='Elevation', columns='Month', values='Anomaly', aggfunc='mean')
        pivot_sorted2 = pivot_df.sort_index(ascending=False)

        sns.heatmap(pivot_sorted2, cmap='coolwarm', annot=True, fmt='.0f',
                    vmin=-50, vmax=50, cbar=False,
                    ax=ax, annot_kws={'size': FS['annot']},
                    linewidths=0.3, linecolor='white')

        ax.set_yticklabels(
            [_abbrev_stn(e) for e in pivot_sorted2.index],
            rotation=0, fontsize=FS['tick'])
        _ax_r2 = ax.twinx()
        _ax_r2.set_ylim(ax.get_ylim())
        _ax_r2.set_yticks([i + 0.5 for i in range(len(pivot_sorted2.index))])
        _ax_r2.set_yticklabels([f'{int(e)} m' for e in pivot_sorted2.index], fontsize=FS['tick'])
        _ax_r2.tick_params(axis='y', length=0)
        for _sp in _ax_r2.spines.values():
            _sp.set_visible(False)

        ax.set_xticklabels(MON_ABBREV, fontsize=FS['tick'], rotation=90)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'Snowfall % anomaly – threshold {threshold:+d}°C', fontsize=FS['label'])
        ax.text(1.05, -0.12, f'({_panel_letters2[idx]})', transform=ax.transAxes,
                ha='right', va='top', fontsize=FS['title'], fontweight='bold',
                zorder=10, clip_on=False)

    fig2.subplots_adjust(hspace=0.40, wspace=0.60, left=0.08, right=0.92, top=0.96, bottom=0.09)
    fig2.canvas.draw()

    _pos_top2 = axes2[0].get_position()
    _pos_bot2 = axes2[2].get_position()
    _pos_width2 = axes2[1].get_position()
    _cax2 = fig2.add_axes([_pos_width2.x1 - _CBAR_PAD - _CBAR_W +0.06,
                           _pos_bot2.y0, _CBAR_W, _pos_top2.y1 - _pos_bot2.y0])
    _sm2 = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-50, vmax=50))
    _sm2.set_array([])
    _cb2 = fig2.colorbar(_sm2, cax=_cax2)
    _cb2.ax.yaxis.set_label_position('right')
    _cb2.ax.yaxis.tick_right()
    _cb2.set_label('Percentage Anomaly (%)', fontsize=FS['tick'])
    _cb2.ax.tick_params(labelsize=FS['tick'])

    plt.show()

    # ── Figure 3: snow & rain amount anomalies (mm/month) ────────────────────
    # Snow+rain combined anomaly = 0 (total precipitation is conserved).
    # Rows show how many mm/month get reclassified between snow and rain.
    baseline_snow_mm = threshold_results[0].set_index(['Month', 'Elevation'])['Snow_mm']
    baseline_rain_mm = threshold_results[0].set_index(['Month', 'Elevation'])['Rain_mm']
    amount_anomaly_results = {}
    for threshold, df in threshold_results.items():
        if threshold != 0:
            _adf = df.copy().set_index(['Month', 'Elevation'])
            _adf['Snow_anomaly_mm'] = _adf['Snow_mm'] - baseline_snow_mm
            _adf['Rain_anomaly_mm'] = _adf['Rain_mm'] - baseline_rain_mm
            _adf.reset_index(inplace=True)
            amount_anomaly_results[threshold] = _adf

    _thresholds_no0 = [t for t in temp_thresholds if t != 0]
    # Compute symmetric colorbar limit from all anomaly values
    _all_snow_anom = np.concatenate([
        amount_anomaly_results[t]['Snow_anomaly_mm'].dropna().values
        for t in _thresholds_no0
    ])
    _vlim3 = max(np.abs(_all_snow_anom).max() * 1.1, 1)

    fig3, axes3 = plt.subplots(2, 4, figsize=(26, 10), facecolor='white')
    axes3 = axes3.flatten()
    _row_meta3 = [
        ('Snow_anomaly_mm', 'Snow amount anomaly', 'RdBu'),
        ('Rain_anomaly_mm', 'Rain amount anomaly', 'RdBu_r'),
    ]
    _panel_letters3 = list('abcdefgh')
    _idx3 = 0
    for col_key, col_label, cmap3 in _row_meta3:
        for threshold in _thresholds_no0:
            ax3 = axes3[_idx3]
            _adf = amount_anomaly_results[threshold]
            _piv3 = _adf.pivot_table(index='Elevation', columns='Month',
                                     values=col_key, aggfunc='mean')
            _piv3 = _piv3.sort_index(ascending=False)

            sns.heatmap(_piv3, cmap=cmap3, annot=True, fmt='.1f',
                        vmin=-_vlim3, vmax=_vlim3, cbar=False,
                        ax=ax3, annot_kws={'size': FS['annot']},
                        linewidths=0.3, linecolor='white')

            ax3.set_yticklabels([_abbrev_stn(e) for e in _piv3.index],
                                rotation=0, fontsize=FS['tick'])
            _ax_r3 = ax3.twinx()
            _ax_r3.set_ylim(ax3.get_ylim())
            _ax_r3.set_yticks([i + 0.5 for i in range(len(_piv3.index))])
            _ax_r3.set_yticklabels([f'{int(e)} m' for e in _piv3.index],
                                   fontsize=FS['tick'])
            _ax_r3.tick_params(axis='y', length=0)
            for _sp in _ax_r3.spines.values():
                _sp.set_visible(False)

            ax3.set_xticklabels(MON_ABBREV, fontsize=FS['tick'], rotation=90)
            ax3.set_xlabel('')
            ax3.set_ylabel('')
            ax3.set_title(f'{col_label} – threshold {threshold:+d}°C vs 0°C (mm/month)',
                          fontsize=FS['label'] - 1)
            ax3.text(1.0, -0.22, f'({_panel_letters3[_idx3]})', transform=ax3.transAxes,
                     ha='right', va='top', fontsize=FS['title'], fontweight='bold',
                     zorder=10, clip_on=False)
            _idx3 += 1

    fig3.suptitle(
        'Precipitation amount anomaly relative to 0°C threshold\n'
        '(top: snow, bottom: rain  |  snow + rain combined anomaly = 0 — total is conserved)',
        fontsize=FS['label'] - 1, fontweight='bold')
    fig3.subplots_adjust(hspace=0.55, wspace=0.55, left=0.06, right=0.90, top=0.88, bottom=0.18)
    fig3.canvas.draw()

    # One colorbar per row on the RIGHT side (shared symmetric scale)
    for _ri, (cmap3, label3) in enumerate([
        ('RdBu',   'Snow amount anomaly (mm/month)'),
        ('RdBu_r', 'Rain amount anomaly (mm/month)'),
    ]):
        _p_top3   = axes3[_ri * 4].get_position()       # leftmost axis of this row (y1 reference)
        _p_right3 = axes3[_ri * 4 + 3].get_position()  # rightmost axis of this row
        _cax3 = fig3.add_axes([_p_right3.x1 - _CBAR_PAD - _CBAR_W + 0.1,
                               _p_right3.y0, _CBAR_W, _p_top3.y1 - _p_right3.y0])
        _sm3 = plt.cm.ScalarMappable(cmap=cmap3,
                                     norm=plt.Normalize(vmin=-_vlim3, vmax=_vlim3))
        _sm3.set_array([])
        _cb3 = fig3.colorbar(_sm3, cax=_cax3)
        _cb3.ax.yaxis.set_label_position('right')
        _cb3.ax.yaxis.tick_right()
        _cb3.set_label(label3, fontsize=FS['tick'])
        _cb3.ax.tick_params(labelsize=FS['tick'])

    plt.show()



def plot_percentage_below_zero(all_merged_dfs):


    # Only keep stations from STATION_ABBREV that are PLU or TB (remove T stations)
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
        'snowAMP_lower': 'T1',
        'snowAMP_middle': 'T2',
        'snowAMP Ganjala upper': 'T3',
        'Yala Pluvio': 'T4',
        'Yala Glacier AWS': 'T5',
        'Langtang Glacier AWS': 'T6'
    }
    # Only keep stations whose abbreviation starts with 'PLU' or 'TB'
    station_names_all = [name for name, abbr in STATION_ABBREV.items() if abbr.startswith('PLU') or abbr.startswith('TB')]
    elevation1 = get_elevation(station_names_all)

    # Use read_pluvio_cleaned to get the data
    all_merged_dfs = read_pluvio_cleaned(station_names_all, dt='1h')

    # Create a new DataFrame to store temperature, elevation, and month
    temp_elev_month_df = pd.DataFrame()
    
    # Loop through each station and extract relevant data
    for station, df in all_merged_dfs.items():
        if df is not None and not df.empty and 'Temperature_1H' in df.columns:
            df_copy = df.copy()
            df_copy.rename(columns={'Temperature_1H': 'TEMP'}, inplace=True)
            df_copy['Month'] = df_copy.index.month
            df_copy['Elevation'] = get_elevation([station])[0]
            df_copy['Station'] = station
            temp_elev_month_df = pd.concat([temp_elev_month_df, df_copy[['Month', 'Elevation', 'Station', 'TEMP']]])

    # Drop rows with NaN temperature values to ensure all calculations are on valid data
    temp_elev_month_df.dropna(subset=['TEMP'], inplace=True)

    # Count the number of times the temperature is below zero for each station, for each month
    temp_below_zero = temp_elev_month_df[temp_elev_month_df['TEMP'] < 1].groupby(['Month', 'Elevation', 'Station']).size().reset_index(name='Count_Below_Zero')
    
    # Calculate the total number of valid observations for each station, for each month
    total_observations = temp_elev_month_df.groupby(['Month', 'Elevation', 'Station'])['TEMP'].count().reset_index(name='Total_Observations')
    
    # Merge the counts. Use a right merge to keep all month/elevation combinations from total_observations
    temp_below_zero_percentage = pd.merge(temp_below_zero, total_observations, on=['Month', 'Elevation', 'Station'], how='right')
    
    # If a group had no temperatures below zero, its count will be NaN after the merge. Fill these with 0.
    temp_below_zero_percentage['Count_Below_Zero'].fillna(0, inplace=True)

    # Calculate the percentage. Use np.divide for safe division (handles division by zero).
    temp_below_zero_percentage['Percentage_Below_Zero'] = np.divide(
        temp_below_zero_percentage['Count_Below_Zero'],
        temp_below_zero_percentage['Total_Observations'],
        out=np.zeros_like(temp_below_zero_percentage['Count_Below_Zero'], dtype=float),
        where=temp_below_zero_percentage['Total_Observations'] != 0
    ) * 100

    # Pivot the DataFrame to get months as columns and elevations as rows
    pivot_df = temp_below_zero_percentage.pivot_table(index='Elevation', columns='Month', values='Percentage_Below_Zero', aggfunc='mean')




    # Prepare mapping for abbreviations (use provided STATION_ABBREV)
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
        'snowAMP_lower': 'T1',
        'snowAMP_middle': 'T2',
        'snowAMP Ganjala upper': 'T3',
        'Yala Pluvio': 'T4',
        'Yala Glacier AWS': 'T5',
        'Langtang Glacier AWS': 'T6'
    }
    # Map elevation to station name
    elevation_to_station = {}
    for station in station_names_all:
        try:
            elev = get_elevation([station])[0]
            if elev not in elevation_to_station:
                elevation_to_station[elev] = station
        except:
            pass

    y_ticks = np.arange(len(pivot_df.index)) + 0.5
    y_tick_labels = [
        f'{int(elev)} m ({STATION_ABBREV.get(elevation_to_station.get(elev, "Unknown"), elevation_to_station.get(elev, "Unknown"))})'
        for elev in pivot_df.index
    ]

    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    sns.heatmap(
        pivot_df.sort_index(ascending=False),
        cmap='coolwarm',
        annot=True,
        fmt=".1f",
        cbar_kws={'label': '% of time below 1$^\circ$C'},
        annot_kws={"color": "black", "fontsize": 8},
        ax=ax
    )
    # Remove the title
    ax.set_title("")
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    ax.set_xticks(np.arange(0, 12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=90)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels[::-1], rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "="*100)
    print("PERCENTAGE OF TIME TEMPERATURE IS BELOW 1°C - SUMMARY BY STATION")
    print("="*100)
    
    summary_by_station = temp_below_zero_percentage.groupby('Station').agg({
        'Percentage_Below_Zero': ['mean', 'min', 'max'],
        'Total_Observations': 'sum'
    }).round(2)
    
    print(summary_by_station)
    print("="*100 + "\n")
    
    return temp_below_zero_percentage, pivot_df


def plot_snowcover_albedo(font_scale=2.25):
    plt.rcParams['font.family'] = 'Arial'

    # ── Font sizes (consistent with other figures) ────────────────────────
    fs = lambda x: x * font_scale
    FS_TICK   = fs(7)
    FS_LABEL  = fs(8)
    FS_PANEL  = fs(9)
    FS_LEGEND = fs(6.5)

    station_names = ['Yala BC AWS', 'Kyangjin AWS']
    aws_data_dict = {}
    for station in station_names:
        file_path = get_dir(station)
        aws_data_dict[station] = read_AWS(file_path[0])

    # Each station gets a row; timeseries left, bar plot right
    fig, axes = plt.subplots(len(station_names), 2,
                             figsize=(15, 6 * len(station_names)),
                             facecolor='white', sharex='col')

    if len(station_names) == 1:
        axes = np.array([axes])

    elevations = {
        'Yala BC AWS': '5050m',
        'Kyangjin AWS': '3850m'
    }
    abbrevs = {
        'Yala BC AWS': 'PLU2',
        'Kyangjin AWS': 'PLU1'
    }

    for i, station in enumerate(station_names):
        df = aws_data_dict[station]
        if 'DATETIME' in df.columns:
            df['DATETIME'] = pd.to_datetime(df['DATETIME'])
            df.set_index('DATETIME', inplace=True)

        df_day = df[df['KINC'] > 150].copy()
        df_day['albedo'] = np.where(df_day['KINC'] > 0, df_day['KUPW'] / df_day['KINC'], np.nan)
        df_day['albedo'] = df_day['albedo'].where((df_day['albedo'] >= 0) & (df_day['albedo'] <= 1))

        ax_timeseries = axes[i, 0]

        daily_albedo = df_day['albedo'].resample('D').mean()
        ax_timeseries.plot(daily_albedo.index, daily_albedo, label='Daily mean albedo',
                           color='red', linestyle='-', linewidth=0.3)
        ax_timeseries.set_ylabel('Albedo', fontsize=FS_LABEL)

        elevation = elevations.get(station, '')
        abbrev = abbrevs.get(station, '')
        ax_timeseries.text(0.02, 0.95, f'{abbrev} ({elevation})',
                           transform=ax_timeseries.transAxes,
                           fontsize=FS_LEGEND, va='top', ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # 0.3 threshold line
        ax_timeseries.axhline(0.3, color='steelblue', linewidth=0.8, linestyle='--')

        # Color shading for periods where albedo > 0.3
        valid_mask = daily_albedo.notna()
        high_albedo_mask = (daily_albedo > 0.3) & valid_mask
        idx = daily_albedo.index
        high_idxs = np.flatnonzero(high_albedo_mask.to_numpy())
        if len(high_idxs) > 0:
            for group in np.split(high_idxs, np.where(np.diff(high_idxs) != 1)[0] + 1):
                start = idx[group[0]]
                end = idx[group[-1]]
                ax_timeseries.axvspan(start, end, color='black', alpha=0.15)
        if i == 1:
            ax_timeseries.legend(loc='upper right', fontsize=FS_LEGEND, frameon=True,
                                     facecolor='white', edgecolor='#cccccc')

        ax_timeseries.set_ylim(0, 1)
        ax_timeseries.tick_params(axis='both', labelsize=FS_TICK)
        ax_timeseries.spines['top'].set_visible(False)
        ax_timeseries.spines['right'].set_visible(False)

        # Mean snowcover days per month (albedo > 0.3), averaged across years
        snowcover_flag = ((daily_albedo > 0.3) & daily_albedo.notna()).astype(int)
        valid_flag = daily_albedo.notna().astype(int)
        monthly_summary = pd.DataFrame({'snow_days': snowcover_flag, 'valid_days': valid_flag})
        monthly_summary['year'] = monthly_summary.index.year
        monthly_summary['month'] = monthly_summary.index.month
        ym = monthly_summary.groupby(['year', 'month'])[['snow_days', 'valid_days']].sum()
        ym.loc[ym['valid_days'] == 0, 'snow_days'] = np.nan
        mean_snow_days = ym['snow_days'].groupby(level='month').mean().reindex(range(1, 13), fill_value=0)
        mean_snow_days.index = [calendar.month_abbr[m] for m in mean_snow_days.index]

        ax_bar = axes[i, 1]
        sns.barplot(x=mean_snow_days.index, y=mean_snow_days.values, ax=ax_bar,
                    palette="viridis", linewidth=0.5)
        ax_bar.text(0.02, 0.95, f'{abbrev} ({elevation})', transform=ax_bar.transAxes,
                    fontsize=FS_LEGEND, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        ax_bar.set_ylabel('Mean # of days snowcover', fontsize=FS_LABEL)
        ax_bar.set_xlabel('')
        ax_bar.tick_params(axis='both', labelsize=FS_TICK)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)

    # ── Panel labels (a), (b), (c), (d) ──────────────────────────────────
    for ax, letter in zip([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
                          ['a', 'b', 'c', 'd']):
        ax.text(0.97, 0.03, f'({letter})', transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight='bold', va='bottom', ha='right')

    fig.tight_layout()
    plt.show()


def analyze_precipitation_sensitivity_to_temperature_change_all_seasons(font_scale=1.5, only_panel_a=False):
    """
    Analyzes how precipitation amount varies with temperature change (dT/dt)
    using hourly data, for all seasons, and plots all seasons together in one figure.

    Uses merged data from AWS and pluviometer stations:
    - Kyangjin AWS, Yala BC AWS
    - Morimoto Pluvio (precip/temp)

    Monsoon is defined as June 15 onward (not June 1).

    Parameters
    ----------
    font_scale : float
        Font scaling factor.
    only_panel_a : bool
        If True, only plot panel a (dT vs precipitation) and suppress panel annotations.
    """
    year_round_stations = ['Kyangjin AWS', 'Yala BC AWS']
    pluvio_temp_stations = ['Morimoto Pluvio']

    season_names = ['Post-monsoon', 'Winter', 'Pre-monsoon', 'Monsoon']
    season_colors = {
        'Post-monsoon': '#2aaa5c',
        'Winter': '#9b45c7',
        'Pre-monsoon': '#e6a817',
        'Monsoon': '#1a6fba',
    }
    season_markers = {
        'Post-monsoon': 'D',
        'Winter': 's',
        'Pre-monsoon': '^',
        'Monsoon': 'o',
    }

    # Use Arial and allow simple font scaling
    plt.rcParams['font.family'] = 'Arial'

    # Load AWS data (time index and temperature)
    aws_data_dict = {}
    for station in year_round_stations:
        file_path = get_dir(station)
        df = read_AWS(file_path[0])
        if 'DATETIME' in df.columns:
            df['DATETIME'] = pd.to_datetime(df['DATETIME'])
            df.set_index('DATETIME', inplace=True)
        aws_data_dict[station] = df

    # Load precipitation data
    all_stations_pluvio = list(set(year_round_stations + pluvio_temp_stations))
    try:
        rain_data = read_pluvio_cleaned(all_stations_pluvio, dt='1h')
        precip_merged_df = {}
        for station, df in rain_data.items():
            if 'Rainfall_1H' in df.columns:
                df.rename(columns={'Rainfall_1H': 'Hourly_Rain'}, inplace=True)
            if 'Temperature_1H' in df.columns:
                df.rename(columns={'Temperature_1H': 'TEMP'}, inplace=True)
            if 'DATETIME' in df.columns:
                df['DATETIME'] = pd.to_datetime(df['DATETIME'])
                df.set_index('DATETIME', inplace=True)
            precip_merged_df[station] = df
    except Exception as e:
        print(f"Error loading precipitation data: {e}")
        return

    dtemp_bin_width = 0.25
    dtemp_bins = np.arange(-4, 4 + dtemp_bin_width, dtemp_bin_width)

    # Helper for correct season assignment
    def get_season_custom(dt):
        m = dt.month
        d = dt.day
        if (m == 6 and d >= 15) or m in [7, 8, 9]:
            return 'Monsoon'
        elif m == 10 or m == 11 or (m == 12 and d < 31):
            return 'Post-monsoon'
        elif m in [3, 4, 5] or (m == 6 and d < 15):
            return 'Pre-monsoon'
        elif m == 1 or m == 2:
            return 'Winter'
        else:
            return 'Other'

    def calculate_lcl_absolute(T, RH):
        """Compute absolute LCL relative to ground observation from temperature (°C) and RH (%)."""
        T0, Rv, L, eo = 273.15, 461.5, 2.5e6, 0.6113
        e_s = eo * np.exp(17.2694 * T / (T + 243.5))
        e = RH / 100.0 * e_s
        e = np.where(e > 0, e, np.nan)
        Td_K = 1 / (1 / T0 - (Rv / L) * np.log(e / eo))
        Td = Td_K - 273.15
        lcl_rel = 125 * (T - Td)
        return lcl_rel

    def compute_monthly_surface_lcl_by_station():
        """Monthly mean surface LCL, grouped by month across all years."""
        monthly_lcl = {}
        try:
            station_seasonal_data = get_seasonal_RH_PRES_data()
        except Exception as e:
            print(f"Error loading RH/TEMP data for LCL subplot: {e}")
            return monthly_lcl

        for station, df_station in station_seasonal_data.items():
            if station != 'Kyangjin AWS':
                continue
            if 'TEMP' not in df_station.columns or 'RH' not in df_station.columns:
                continue

            T = pd.to_numeric(df_station['TEMP'], errors='coerce')
            RH = pd.to_numeric(df_station['RH'], errors='coerce')
            RH = RH.where((RH > 0) & (RH <= 100), other=np.nan)
            lcl_abs = calculate_lcl_absolute(T.values, RH.values)

            s_lcl_abs = pd.Series(lcl_abs, index=df_station.index)
            s_lcl_abs = s_lcl_abs.where(s_lcl_abs > 0, other=np.nan)

            df_lcl = pd.DataFrame(index=s_lcl_abs.index)
            df_lcl['Surface_LCL'] = s_lcl_abs
            df_lcl['Month'] = df_lcl.index.month

            monthly_mean_surface_lcl = df_lcl['Surface_LCL'].groupby(df_lcl['Month']).mean()
            if not monthly_mean_surface_lcl.dropna().empty:
                monthly_lcl[station] = monthly_mean_surface_lcl

        return monthly_lcl

    # Collect and merge all data for all stations
    all_data_list = []
    for station in year_round_stations:
        if station not in aws_data_dict or station not in precip_merged_df:
            continue
        df_aws = aws_data_dict[station].copy()
        df_rain = precip_merged_df[station].copy()
        # Temperature
        if 'TAIR' in df_aws.columns:
            df_aws['TAIR'] = pd.to_numeric(df_aws['TAIR'], errors='coerce')
            df_aws['dT_1h'] = df_aws['TAIR'].diff(periods=1)
        elif 'TA' in df_aws.columns:
            df_aws['TA'] = pd.to_numeric(df_aws['TA'], errors='coerce')
            df_aws['dT_1h'] = df_aws['TA'].diff(periods=1)
        else:
            df_aws['dT_1h'] = np.nan
        # Rain
        if 'Hourly_Rain' in df_rain.columns:
            df_rain['Hourly_Rain'] = pd.to_numeric(df_rain['Hourly_Rain'], errors='coerce')
        # Merge
        df_combined = pd.DataFrame(index=df_aws.index)
        df_combined['dT_1h'] = df_aws['dT_1h']
        df_combined['Hourly_Rain'] = df_rain['Hourly_Rain'].reindex(df_combined.index)
        df_combined['season'] = df_combined.index.map(get_season_custom)
        all_data_list.append(df_combined)
    # Pluvio stations: temperature and precipitation from cleaned pluvio data.
    for pluvio_station in pluvio_temp_stations:
        if pluvio_station not in precip_merged_df:
            continue
        df_precip = precip_merged_df[pluvio_station].copy()
        if 'TEMP' in df_precip.columns:
            df_precip['TEMP'] = pd.to_numeric(df_precip['TEMP'], errors='coerce')
            df_precip['dT_1h'] = df_precip['TEMP'].diff(periods=1)
        else:
            df_precip['dT_1h'] = np.nan
        if 'Hourly_Rain' in df_precip.columns:
            df_precip['Hourly_Rain'] = pd.to_numeric(df_precip['Hourly_Rain'], errors='coerce')
        df_combined = pd.DataFrame(index=df_precip.index)
        df_combined['dT_1h'] = df_precip['dT_1h']
        df_combined['Hourly_Rain'] = df_precip['Hourly_Rain']
        df_combined['season'] = df_combined.index.map(get_season_custom)
        all_data_list.append(df_combined)
    # Merge all stations into one DataFrame
    if not all_data_list:
        print("No station data available for temperature-change sensitivity.")
        return
    df_all = pd.concat(all_data_list, axis=0)
    df_all = df_all.dropna(subset=['Hourly_Rain'])

    # Plot all seasons together on one compact axis.
    FS_TICK   = 12.6  * font_scale
    FS_LABEL  = 14.4  * font_scale
    FS_PANEL  = 16.2  * font_scale
    FS_LEGEND = 11.7  * font_scale

    if only_panel_a:
        _fig, ax = plt.subplots(1, 1, figsize=(8.5, 8), facecolor='white')
        ax_lcl = None
    else:
        _fig, (ax, ax_lcl) = plt.subplots(1, 2, figsize=(17, 8), facecolor='white')
    season_binned = {}
    corr_summary = {}
    for season_name in season_names:
        df_season = df_all[df_all['season'] == season_name]
        df_temp = df_season.dropna(subset=['dT_1h']).copy()
        if len(df_temp) == 0:
            continue
        df_temp = df_temp[
            (df_temp['dT_1h'] >= dtemp_bins[0]) &
            (df_temp['dT_1h'] <= dtemp_bins[-1])
        ].copy()
        if len(df_temp) == 0:
            continue
        df_temp['dtemp_bin'] = pd.cut(df_temp['dT_1h'], bins=dtemp_bins)
        binned_precip = (
            df_temp.groupby('dtemp_bin', observed=True)['Hourly_Rain']
            .agg(['mean', 'count', 'sem'])
            .dropna(subset=['mean'])
        )
        if binned_precip.empty:
            continue

        bin_centers = np.array([interval.mid for interval in binned_precip.index], dtype=float)
        corr = df_temp['dT_1h'].corr(df_temp['Hourly_Rain'])
        corr_summary[season_name] = corr
        season_binned[season_name] = binned_precip

        ax.plot(
            bin_centers, binned_precip['mean'].values,
            marker=season_markers[season_name], markersize=5.5,
            linewidth=1.7, color=season_colors[season_name], alpha=0.92,
            markeredgecolor='black', markeredgewidth=0.45,
            label=season_name, zorder=3
        )
        _sem = binned_precip['sem'].fillna(0).values
        ax.fill_between(bin_centers,
                        binned_precip['mean'].values - _sem,
                        binned_precip['mean'].values + _sem,
                        color=season_colors[season_name], alpha=0.15, zorder=2)

    ax.axvline(0, color='black', linewidth=0.9, linestyle=':', alpha=0.65, zorder=1)
    ax.set_xlabel(r'$\mathrm{d}T\,\mathrm{d}t^{-1}$ ($^\circ$C h$^{-1}$)',
                  fontsize=FS_LABEL)
    ax.set_ylabel(r'Precipitation intensity (mm h$^{-1}$)', fontsize=FS_LABEL)
    x_min, x_max = -1, 1
    ax.set_xlim(x_min, x_max)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    if not only_panel_a:
        ax.text(0.01, 0.98, 'a)', transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight='bold', va='top')
    if season_binned:
        y_max = max(v['mean'].max() for v in season_binned.values())
        if np.isfinite(y_max) and y_max > 0:
            ax.set_ylim(0, 0.6)
            label_y = 0.55
            ax.text(x_min + 0.10, label_y,
                    r'Cooling ($\mathrm{d}T\,\mathrm{d}t^{-1} < 0$)',
                    fontsize=FS_LEGEND, ha='left', va='top', color='dimgray')
            ax.text(x_max - 0.10, label_y,
                    r'Warming ($\mathrm{d}T\,\mathrm{d}t^{-1} > 0$)',
                    fontsize=FS_LEGEND, ha='right', va='top', color='dimgray')
        ax.legend(fontsize=FS_LEGEND, loc='center right', ncol=2,
                  framealpha=0.90, edgecolor='grey')
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=FS_LABEL)

    if not only_panel_a:
        # Subplot B: Monthly mean surface LCL (all years, all stations).
        monthly_lcl_by_station = compute_monthly_surface_lcl_by_station()
        month_ticks = np.arange(1, 13)
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        if monthly_lcl_by_station:
            for station, monthly_series in monthly_lcl_by_station.items():
                ax_lcl.plot(
                    monthly_series.index, monthly_series.values,
                    marker='o', markersize=4.5, linewidth=1.5, alpha=0.90,
                    label='Surface LCL PLU1', zorder=3
                )
            ax_lcl.legend(fontsize=FS_LEGEND - 1.0, loc='best', framealpha=0.90, edgecolor='grey')
        else:
            ax_lcl.text(0.5, 0.5, 'No data', transform=ax_lcl.transAxes,
                        ha='center', va='center', fontsize=FS_LABEL)

        ax_lcl.axhline(0, color='black', linewidth=0.9, linestyle=':', alpha=0.65, zorder=1)
        ax_lcl.set_xlabel('Month', fontsize=FS_LABEL)
        ax_lcl.set_ylabel('Surface LCL (m)', fontsize=FS_LABEL)
        ax_lcl.set_xticks(month_ticks)
        ax_lcl.set_xticklabels(month_labels)
        ax_lcl.tick_params(axis='both', labelsize=FS_TICK)
        ax_lcl.spines['top'].set_visible(False)
        ax_lcl.spines['right'].set_visible(False)
        ax_lcl.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax_lcl.text(0.01, 0.98, 'b)', transform=ax_lcl.transAxes,
                    fontsize=FS_PANEL, fontweight='bold', va='top')

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"{'='*90}\nCORRELATION SUMMARY (dT vs Precipitation, All Stations Merged)\n{'='*90}")
    print(f"{'Season':<20} {'dT-Precip Corr':<20}")
    print(f"{'-'*40}")
    for season_name in season_names:
        if season_name not in corr_summary:
            continue
        corr = corr_summary[season_name]
        corr_str = f"{corr:.4f}" if not np.isnan(corr) else "N/A"
        print(f"{season_name:<20} {corr_str:<20}")
    return

def plot_combined_snowfall_rainfall_analysis():
    """
    Combined precipitation analysis figure:
      a) Mean monthly precipitation by elevation (heatmap, all stations)
      b) Daily precipitation intensity distribution – Kyangjin AWS (stacked bars, % of days)
      c) Mean monthly snowfall by elevation (heatmap, Pluvio/AWS stations)
      d) Top-20 daily extreme events per station (cyclonic events shown as markers)
    """
    import matplotlib.dates as mdates

    # ── Global style ──────────────────────────────────────────────────────────
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 15})
    base_fs = float(plt.rcParams.get('font.size', 15))
    FS = {'title': 1.45 * base_fs, 'label': 1.30 * base_fs,
        'tick': 1.15 * base_fs, 'annot': 1.00 * base_fs}

    #I am distinguishing snow and rain station as pluviometers and tipping buckets. Pluvio measure both rain+snow, tipping buckets only rain
    # ── Station setup ─────────────────────────────────────────────────────────
    snow_stations    = ['Langshisha Pluvio', 'Kyangjin AWS', 'Yala BC AWS', 'Morimoto Pluvio']
    rain_stations    = ['Numthang TB', 'Jathang TB', 'Ganja La TB1', 'Ganja La TB2', 'Ganja La TB3',
                        'Morimoto TB', 'Shalbachum TB', 'Langshisha BC TB', 'Langshisha Pluvio',
                        'Morimoto Pluvio', 'Kyangjin AWS', 'Yala BC AWS', 'Lama TB', 'Syabru TB']
    extreme_stations = ['Kyangjin AWS', 'Yala BC AWS', 'Langshisha Pluvio']
    all_stations     = list(set(snow_stations + rain_stations + extreme_stations))

    def _abbrev(elev, emap):
        names = [n for n, e in emap.items() if e == elev]
        return STATION_ABBREV.get(names[0], names[0]) if names else ''

    # ── Load data (15 min preferred, fallback 1 h) ────────────────────────────
    all_merged_dfs = {}
    for station in all_stations:
        df = read_pluvio_cleaned([station], dt='15min').get(station)
        if df is None or df.empty:
            df = read_pluvio_cleaned([station], dt='1h').get(station)
        if df is not None and not df.empty:
            all_merged_dfs[station] = df

    elevation_map = {s: get_elevation([s])[0] for s in all_stations}

    COV_THR, T_THR = 1.0, 1.0

    # ── 2. Snowfall processing ────────────────────────────────────────────────
    snowfall_df = pd.DataFrame()
    for station in snow_stations:
        df = all_merged_dfs.get(station)
        if df is None or df.empty:
            continue
        rc = next((c for c in ['Rainfall_1H', 'Rainfall_15min'] if c in df.columns), None)
        tc = next((c for c in ['Temperature_1H', 'Temperature'] if c in df.columns), None)
        if not rc or not tc:
            continue
        # Rename the temperature and precipitation columns as T, R
        d = df[[rc, tc]].copy()
        d.columns = ['R', 'T']
        # Make year and month indices
        d['Y'], d['M'] = d.index.year, d.index.month
        cov = d.groupby(['Y', 'M'])['R'].apply(
            lambda x: x.notna().sum() / x.size if x.size else 0)
        # Require full monthly coverage (1.0)
        ym = cov[cov >= COV_THR]
        if ym.empty:
            continue
        # Require at least 3 qualifying year-months before monthly climatology
        if len(ym) < 3:
            continue
        d_ok = d.join(ym.rename('ok'), on=['Y', 'M'], how='inner').drop(columns='ok')
        d_ok['S'] = d_ok['R'].where(d_ok['T'] <= T_THR, other=0).fillna(0)
        ms = d_ok.groupby(['Y', 'M'])['S'].sum().loc[ym.index].groupby('M').mean()
        r = ms.reset_index(); r.columns = ['Month', 'Snowfall']; r['Elevation'] = elevation_map[station]
        snowfall_df = pd.concat([snowfall_df, r])

    # ── 3. Rainfall processing (tipping buckets) ────────────────────────────────────────────────
    rain_df = pd.DataFrame()
    for station in rain_stations:
        df = all_merged_dfs.get(station)
        if df is None or df.empty:
            continue
        rc = next((c for c in ['Rainfall_1H', 'Rainfall_15min'] if c in df.columns), None)
        tc = next((c for c in ['Temperature_1H', 'Temperature'] if c in df.columns), None)
        if not rc:
            continue
        d = df[[rc]].copy(); d.columns = ['R']
        if tc and tc in df.columns:
            d['T'] = df[tc]
        if 'TB' in station and 'T' in d.columns:
            d.loc[(d['T'] < 1) & (d['R'] == 0), 'R'] = np.nan
        d['Y'], d['M'] = d.index.year, d.index.month
        thr = 0.5 if 'TB' in station else 1.0
        cov = d.groupby(['Y', 'M'])['R'].apply(lambda x: x.notna().sum() / x.size if x.size else 0)
        ym = cov[cov >= thr]
        if ym.empty:
            continue
        vy = ym.index.get_level_values(0).unique()
        sums = d[d['Y'].isin(vy)].groupby(['Y', 'M'])['R'].sum()
        means = sums.loc[ym.index].groupby('M').mean().reset_index()
        means.columns = ['Month', 'R']; means['Elevation'] = elevation_map[station]
        rain_df = pd.concat([rain_df, means])

    # ── 4. Intensity distribution (Kyangjin) — % of days ─────────────────────
    _ic = plt.cm.YlGn
    INT_COLORS = {
        'Dry':      '#f0f0f0',
        '<2 mm':    _ic(1/7), '2–5 mm':    _ic(2/7),
        '5–10 mm':  _ic(3/7), '10–20 mm':  _ic(4/7),
        '20–50 mm': _ic(5/7), '>50 mm':    _ic(6/7),
    }
    INT_BINS   = [-1, 0.001, 2, 5, 10, 20, 50, 200]
    INT_LABELS = list(INT_COLORS.keys())
    avg_cat = None
    df_kj = all_merged_dfs.get('Kyangjin AWS')
    if df_kj is not None:
        dk = df_kj.copy()
        for o, n in [('Rainfall_1H', 'R'), ('Rainfall_15min', 'R')]:
            if o in dk.columns:
                dk = dk.rename(columns={o: n})
        if 'R' in dk.columns:
            dk['R'] = pd.to_numeric(dk['R'], errors='coerce')
            daily = dk.resample('D')['R'].sum().to_frame('R')
            daily['Cat'] = pd.cut(daily['R'], bins=INT_BINS, labels=INT_LABELS, right=False)
            daily['Y'], daily['M'] = daily.index.year, daily.index.month
            monthly_cat = daily.groupby(['Y', 'M', 'Cat'], observed=False).size().unstack(fill_value=0)
            # Convert counts to temperature belowtotal days in that year-month
            days_in_month = daily.groupby(['Y', 'M']).size()
            monthly_cat_pct = monthly_cat.div(days_in_month, axis=0) * 100
            avg_cat = monthly_cat_pct.groupby('M').mean()

    # ── 5. Extreme events ─────────────────────────────────────────────────────
    EXT_COLORS = {
        'Kyangjin AWS':      '#1f78b4',
        'Yala BC AWS':       '#e6821e',
        'Langshisha Pluvio': '#2ca02c',
    }

    # Cyclonic events: (marker, color, date)
    CYCLONE_MARKERS = {
        'Phailin': ('*', '#d62728', pd.Timestamp('2013-10-14')),
        'Hudhud':  ('^', '#e6550d', pd.Timestamp('2014-10-14')),
        'Yaas':    ('P', '#756bb1', pd.Timestamp('2021-05-27')),
        'Hamoon':  ('X', '#31a354', pd.Timestamp('2021-10-19')),
    }

    data_dict, all_extremes, meas_periods = {}, [], {}
    for station in extreme_stations:
        df = read_pluvio_cleaned([station], dt='15min').get(station)
        if df is None or df.empty:
            df = read_pluvio_cleaned([station], dt='1h').get(station)
        if df is not None and not df.empty:
            data_dict[station] = df

    for station in extreme_stations:
        if station not in data_dict:
            continue
        df = data_dict[station].copy()
        for o, n in [('Rainfall_1H', 'R'), ('Rainfall_15min', 'R'),
                     ('Temperature_1H', 'T'), ('Temperature', 'T')]:
            if o in df.columns:
                df = df.rename(columns={o: n})
        df['R'] = pd.to_numeric(df['R'] if 'R' in df.columns else pd.Series(dtype=float), errors='coerce')
        df['T'] = pd.to_numeric(df['T'] if 'T' in df.columns else pd.Series(dtype=float), errors='coerce')
        df['S'] = np.where((df['T'] <= 1.0) & (df['R'] > 0), df['R'], 0)
        dv = df['R'].notna().resample('D').max().astype(bool)
        grps = (dv != dv.shift()).cumsum()
        vg = dv[dv].groupby(grps[dv])
        if vg.ngroups > 0:
            meas_periods[station] = (vg.apply(lambda x: x.index[0]),
                                     vg.apply(lambda x: x.index[-1]))
        dagg = df.resample('D').agg({'R': 'sum', 'S': 'sum'})
        for date, val in dagg['R'].nlargest(20).items():
            snow_val = df.loc[date:date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), 'S'].sum()
            all_extremes.append({'Date': date, 'Precip': val, 'Snow': snow_val, 'Station': station})

    # ── Calculate annual totals for pluviometers ──────────────────────────────
    pluvio_stations = [s for s in rain_stations if 'Pluvio' in s or 'AWS' in s]
    annual_totals = {}
    for station in pluvio_stations:
        station_data = rain_df[rain_df['Elevation'] == elevation_map[station]]
        if not station_data.empty:
            annual_total = station_data['R'].sum()
            annual_totals[station] = annual_total

    print("\n" + "="*70)
    print("ANNUAL PRECIPITATION TOTALS (mm)")
    print("="*70)
    for station in sorted(annual_totals.keys(), key=lambda s: annual_totals[s], reverse=True):
        elev = elevation_map[station]
        total = annual_totals[station]
        print(f"{station:30s} ({int(elev):4d} m): {total:7.1f} mm")

    kyangjin_total = annual_totals.get('Kyangjin AWS', 0)
    yala_total     = annual_totals.get('Yala BC AWS', 0)
    if kyangjin_total > 0 and yala_total > 0:
        diff     = kyangjin_total - yala_total
        pct_diff = (diff / yala_total) * 100
        print("-"*70)
        print(f"Kyangjin AWS receives {diff:7.1f} mm ({pct_diff:+.1f}%) MORE than Yala BC AWS")
        print("="*70 + "\n")

    # ── 6. Build figure ───────────────────────────────────────────────────────

    # ── Prepare pivot tables ──────────────────────────────────────────────────
    rain_df.loc[
        rain_df['Elevation'].isin([e for s, e in elevation_map.items() if 'TB' in s]) &
        rain_df['Month'].isin([1, 2, 12]), 'R'
    ] = np.nan
    piv_rain = rain_df.pivot_table(
        index='Elevation', columns='Month', values='R').sort_index(ascending=False)
    plu_stations = ['Langshisha Pluvio', 'Morimoto Pluvio', 'Kyangjin AWS', 'Yala BC AWS']
    plu_elevs    = [elevation_map[s] for s in plu_stations if s in elevation_map]
    piv_rain[13] = piv_rain.sum(axis=1)
    piv_rain.loc[~piv_rain.index.isin(plu_elevs), 13] = np.nan

    piv_snow = snowfall_df.pivot_table(
        index='Elevation', columns='Month', values='Snowfall').sort_index(ascending=False)
    piv_snow[13] = piv_snow.sum(axis=1)
    piv_snow.loc[~piv_snow.index.isin(plu_elevs), 13] = np.nan
    vmin_snow, vmax_snow = piv_snow[range(1, 13)].min().min(), piv_snow[range(1, 13)].max().max()

    ext_df = pd.DataFrame(all_extremes).sort_values('Date') if all_extremes else pd.DataFrame()

    # ── Build figure ──────────────────────────────────────────────────────────
    # Slightly taller layout so panel boxes are vertically larger
    fig2 = plt.figure(figsize=(17.5, 12.0), facecolor='white', dpi=100)
    gs2  = GridSpec(2, 4, figure=fig2,
                    height_ratios=[1.50, 1.00],
                    hspace=0.36, wspace=0.60)

    ax2_rain = fig2.add_subplot(gs2[0, :2])
    ax2_int  = fig2.add_subplot(gs2[0, 2:4])
    ax2_snow = fig2.add_subplot(gs2[1, :2])
    ax2_ext  = fig2.add_subplot(gs2[1, 2:4])

    def _pl2(ax, letter, x=1.0):
        return ax.text(x, -0.02, f'({letter})', transform=ax.transAxes,
                       ha='right', va='top', fontsize=FS['title'], fontweight='bold',
                       zorder=10, clip_on=False)

    def _panel_outline(ax, color='#333333', lw=1.0):
        for _sp in ax.spines.values():
            _sp.set_visible(True)
            _sp.set_linewidth(lw)
            _sp.set_color(color)

    # ── Panel A: mean monthly precipitation heatmap ───────────────────────────
    sns.heatmap(piv_rain, cmap='YlGnBu', annot=True, fmt='.0f', cbar=False,
                ax=ax2_rain, vmin=0, vmax=350,
                annot_kws={'size': FS['annot']}, linewidths=0.3, linecolor='white')
    month_abbrev = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    ax2_rain.set_xticklabels(month_abbrev, fontsize=FS['tick'], rotation=90)
    ax2_rain.set_yticklabels([_abbrev(e, elevation_map) for e in piv_rain.index],
                              rotation=0, fontsize=FS['tick'])
    ax2_rain.tick_params(axis='y', labelsize=FS['tick'])
    ax2r_r = ax2_rain.twinx()
    ax2r_r.set_ylim(ax2_rain.get_ylim())
    ax2r_r.set_yticks([i + 0.5 for i in range(len(piv_rain.index))])
    ax2r_r.set_yticklabels([f'{int(e)} m' for e in piv_rain.index], fontsize=FS['tick'])
    ax2r_r.tick_params(axis='y', length=0)
    for _sp in ax2r_r.spines.values(): _sp.set_visible(False)
    ax2_rain.set_xlabel(''); ax2_rain.set_ylabel('')
    _panel_outline(ax2_rain)
    ann_a = _pl2(ax2_rain, 'a', x=1.15)

    # ── Panel B: daily total distribution as % of days ──────────────────────────
    if avg_cat is not None:
        bot2 = np.zeros(12)
        for lbl in INT_LABELS:
            if lbl in avg_cat.columns:
                vals = avg_cat[lbl].reindex(range(1, 13), fill_value=0).values
                ax2_int.bar(range(1, 13), vals, bottom=bot2, color=INT_COLORS[lbl],
                            width=0.85, label=lbl, edgecolor='white', linewidth=0.4)
                bot2 += vals
    ax2_int.set_xticks(range(1, 13))
    month_abbrev_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax2_int.set_xticklabels(month_abbrev_short, rotation=90, fontsize=FS['tick'])
    ax2_int.set_ylabel('Daily total distribution (% of days)', fontsize=FS['tick'])
    ax2_int.yaxis.set_label_position('right'); ax2_int.yaxis.tick_right()
    ax2_int.tick_params(axis='y', labelsize=FS['tick'])
    ax2_int.set_ylim(0, 105)   # percentages sum to ~100
    _panel_outline(ax2_int)
    ax2_int.grid(axis='y', linestyle=':', alpha=0.45, color='gray')
    leg_int = ax2_int.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INT_COLORS[l]) for l in INT_LABELS],
                             labels=INT_LABELS, loc='lower left', ncol=4, fontsize=FS['tick'], frameon=True,
                             facecolor='white', edgecolor='#d0d0d0',
                             handlelength=1, handleheight=0.9)
    ann_b = _pl2(ax2_int, 'b')

    # ── Panel C: mean monthly snowfall heatmap ────────────────────────────────
    sns.heatmap(piv_snow, cmap='Blues', annot=True, fmt='.0f', cbar=False,
                vmin=vmin_snow, vmax=vmax_snow,
                ax=ax2_snow, annot_kws={'size': FS['annot']},
                linewidths=0.4, linecolor='white')
    ax2_snow.set_xticklabels(month_abbrev, fontsize=FS['tick'], rotation=90)
    ax2_snow.set_yticklabels([_abbrev(e, elevation_map) for e in piv_snow.index],
                              rotation=0, fontsize=FS['tick'])
    ax2_snow_r = ax2_snow.twinx()
    ax2_snow_r.set_ylim(ax2_snow.get_ylim())
    ax2_snow_r.set_yticks([i + 0.5 for i in range(len(piv_snow.index))])
    ax2_snow_r.set_yticklabels([f'{int(e)} m' for e in piv_snow.index], fontsize=FS['tick'])
    ax2_snow_r.tick_params(axis='y', length=0)
    for _sp in ax2_snow_r.spines.values(): _sp.set_visible(False)
    ax2_snow.set_xlabel(''); ax2_snow.set_ylabel('')
    _panel_outline(ax2_snow)
    ann_c = _pl2(ax2_snow, 'c', x=1.15)

    # ── Panel D: extreme events + cyclone markers ─────────────────────────────
    leg_ext_main, leg_ext_cyc, ann_d = None, None, None
    if all_extremes:
        # Station bars and dots
        for _, row in ext_df.iterrows():
            c = EXT_COLORS[row['Station']]
            ax2_ext.vlines(row['Date'], 0, row['Precip'], color=c, alpha=0.75, linewidth=1.8, zorder=2)
            ax2_ext.scatter(row['Date'], row['Precip'], color=c, s=28, zorder=3,
                            edgecolors='white', linewidth=0.4)
            if row['Snow'] > 0:
                ax2_ext.scatter(row['Date'], row['Snow'], color='#a8d8ea', edgecolors='#1a6896',
                                marker='D', s=22, zorder=4)

        # Cyclone markers: one distinct shape per event plotted just above the bar top
        cyclone_legend_handles = []
        for name, (mkr, col, date) in CYCLONE_MARKERS.items():
            match = ext_df[ext_df['Date'] == date]
            if name in ('Hamoon', 'Yaas'):
                y_pos = 100
            else:
                y_pos = match['Precip'].max() + 4 if not match.empty else 108
            ax2_ext.scatter(date, y_pos, marker=mkr, color=col, s=320,
                            zorder=6, edgecolors='white', linewidth=0.6)
            cyclone_legend_handles.append(
                Line2D([0], [0], marker=mkr, color='w', markerfacecolor=col,
                       markeredgecolor='white', markersize=17, label=name)
            )

        # Station + snow legend above panel (use PLU1, PLU2, PLU3 abbreviations)
        hdl_st2 = [Line2D([0], [0], color=c, lw=2,
                          label=f'{STATION_ABBREV.get(s, s)}  ({int(elevation_map[s])} m)')
                   for s, c in EXT_COLORS.items()]
        # Overwrite labels for the three main stations
        for i, s in enumerate(EXT_COLORS.keys()):
            abbrev = STATION_ABBREV.get(s, s)
            hdl_st2[i].set_label(f'{abbrev}  ({int(elevation_map[s])} m)')

        hdl_snow2 = Line2D([0], [0], marker='D', color='w', markerfacecolor='#a8d8ea',
                           markeredgecolor='#1a6896', markersize=11, label='Snow event')
        # Station legend below panel D (one row)
        leg_ext_main = ax2_ext.legend(
            handles=hdl_st2,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.23),
            ncol=3,
            fontsize=FS['tick'],
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.5)
        ax2_ext.add_artist(leg_ext_main)

        # Extremes legend at top of panel D: cyclones + snow event, single row
        leg_ext_cyc = ax2_ext.legend(
            handles=cyclone_legend_handles + [hdl_snow2],
            loc='lower center',
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(cyclone_legend_handles) + 1,
            fontsize=FS['tick'],
            frameon=False,
            columnspacing=0.8,
            handletextpad=0.3,
            handlelength=1.2,
            borderaxespad=0.1)
        ax2_ext.add_artist(leg_ext_cyc)

        ax2_ext.set_ylim(0, 320)
        ax2_ext.set_ylabel('Extreme daily precipitation (mm)', fontsize=FS['tick'])
        ax2_ext.yaxis.set_label_position('right'); ax2_ext.yaxis.tick_right()
        _panel_outline(ax2_ext)
        ax2_ext.grid(axis='y', ls=':', alpha=0.45, color='gray')
        ax2_ext.tick_params(labelsize=FS['tick'])
        ax2_ext.xaxis.set_major_locator(mdates.YearLocator())
        ax2_ext.set_xlim(right=pd.Timestamp('2026-12-31'))
        from matplotlib.ticker import FuncFormatter as _FF2
        ax2_ext.xaxis.set_major_formatter(
            _FF2(lambda x, _: '' if mdates.num2date(x).year >= 2027
                 else mdates.num2date(x).strftime('%Y')))
        ax2_ext.tick_params(axis='x', labelrotation=90)
        ann_d = _pl2(ax2_ext, 'd')


    # Layout: more generous margins, keep everything inside 1920x1200px
    fig2.subplots_adjust(hspace=0.32, wspace=0.60,
                         left=0.10, right=0.92, top=0.96, bottom=0.15)
    fig2.canvas.draw()

    # Colorbar width and padding
    CBAR_W, CBAR_PAD, CBAR_FS = 0.012, 0.040, FS['tick']
    pos_a2 = ax2_rain.get_position()
    pos_c2 = ax2_snow.get_position()
    # Place colorbars just inside the left edge, not outside
    left_x2 = max(0.01, min(pos_a2.x0, pos_c2.x0) - CBAR_PAD - CBAR_W)

    cax_a2 = fig2.add_axes([left_x2, pos_a2.y0, CBAR_W, pos_a2.height])
    cb_a2  = fig2.colorbar(ax2_rain.collections[0], cax=cax_a2)
    cb_a2.ax.yaxis.set_label_position('left'); cb_a2.ax.yaxis.tick_left()
    cb_a2.set_label('Precipitation (mm)', fontsize=CBAR_FS)
    cb_a2.ax.tick_params(labelsize=CBAR_FS)
    cb_a2.outline.set_visible(False)
    for _sp in cax_a2.spines.values():
        _sp.set_visible(False)

    cax_c2 = fig2.add_axes([left_x2, pos_c2.y0, CBAR_W, pos_c2.height])
    cb_c2  = fig2.colorbar(ax2_snow.collections[0], cax=cax_c2)
    cb_c2.ax.yaxis.set_label_position('left'); cb_c2.ax.yaxis.tick_left()
    cb_c2.set_label('Snowfall (mm w.e.)', fontsize=CBAR_FS)
    cb_c2.ax.tick_params(labelsize=CBAR_FS)
    cb_c2.outline.set_visible(False)
    for _sp in cax_c2.spines.values():
        _sp.set_visible(False)

    plt.show()
    return fig2

# plot_precip_august_may_comparison_with_slope_stations(STATION_ABBREV=STATION_ABBREV,font_scale=1.5)