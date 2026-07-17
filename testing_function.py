"""End-to-end test of the full processing chain.

Runs every cleaning and derivation step of this repository in dependency
order and finally produces the analysis figures of wrapperv3.py:

    1. tipping-bucket precipitation cleaning        (TB_clean_precip)
    2. pluvio + AWS precipitation cleaning          (pluvio_cleaned_precip)
    3. Kochendorfer undercatch correction           (pluvio_cleaned_precip)
    4. wind cleaning                                (clean_wind)
    5. SW/LW radiation loading                      (clean_SW_LW)
    6. humidity timeseries generation (incl. RH)    (generate_humidity_timeseries)
    7. temperature merge (all sensors) + pickle     (merge_temperature)
    8. lapse rate / isotherm computation            (lapse_rate_isotherm)
    9. valley geometry                              (valley_geometry)
   10. optional merged single-CSV products          (merge_*)
   11. data-availability overview figure            (plot_data_overview)
   12. wrapperv3 analysis figures

NOTE — this run REGENERATES the derived data in data/ (Cleaned/Pluvio,
Kochendorfer_corrected, Cleaned/Temperature incl. temp_merged_dfs.pkl,
Moisture, LapseRate summaries, Geometry). The previously shipped versions
are copied to <zenodo_root>/testing_backup/ before anything is overwritten.
Regenerated files can differ slightly from the shipped (frozen) versions —
see the README. The TB and wind cleaning are executed but not written: their
exact save-runs are not preserved in the code, so the shipped data/Cleaned/TB
and data/Cleaned/Wind files remain in place and are used by the figures.

Every figure produced during the run is additionally saved to
<zenodo_root>/results_testing/ as PNG + EPS, so the output of a run can be
compared figure-by-figure against a reference run.

Requires the raw sensor data in data/raw/ (see data/raw/README.txt — the raw
data is available on request from a.kwadijk@uu.nl or w.w.immerzeel@uu.nl).

Run from inside the code folder:  python testing_function.py

Copyright (c) 2026 A. Kwadijk, Utrecht University. Licensed under CC BY 4.0.
"""
import matplotlib
matplotlib.use('Agg')  # run unattended: figures are saved, not shown
import matplotlib.pyplot as plt

import os
import re
import shutil
import sys
import time
import traceback

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')  # some figure functions print unicode

from station_data import _DATA_DIR


def _backup_shipped_data():
    """Copy every file this test will overwrite to <zenodo_root>/testing_backup."""
    root = _DATA_DIR.parent / 'testing_backup'
    targets = [
        'Cleaned/Pluvio', 'Cleaned/Kochendorfer_corrected', 'Cleaned/Temperature',
        'Moisture', 'LapseRate', 'Geometry',
    ]
    os.makedirs(root, exist_ok=True)
    for rel in targets:
        src = _DATA_DIR / rel
        dst = root / rel.replace('/', '_')
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if not os.path.exists(dst / f):
                shutil.copy2(src / f, dst / f)
    pkl = _DATA_DIR / 'Cleaned' / 'Temperature' / 'temp_merged_dfs.pkl'
    if os.path.exists(pkl) and not os.path.exists(root / 'temp_merged_dfs.pkl'):
        shutil.copy2(pkl, root / 'temp_merged_dfs.pkl')
    print(f'Shipped derived data backed up to {root}')


def _save_open_figures(stage_name):
    """Save every open matplotlib figure of a stage to
    <zenodo_root>/results_testing as PNG + EPS, so the output of a full test
    run can be compared figure-by-figure against earlier runs."""
    out = _DATA_DIR.parent / 'results_testing'
    os.makedirs(out, exist_ok=True)
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', stage_name).strip('_')[:70]
    for i, num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(num)
        base = out / f'{safe}_fig{i:02d}'
        try:
            fig.savefig(f'{base}.png', dpi=200)
            fig.savefig(f'{base}.eps')
        except Exception as e:
            print(f'  (could not save figure {i} of stage "{stage_name}": {e})')


def testing_function(backup_first=True):
    results = []

    def stage(name, fn):
        print(f"\n{'='*70}\nSTAGE: {name}\n{'='*70}")
        t0 = time.time()
        try:
            fn()
            results.append((name, 'PASS', time.time() - t0))
        except Exception as e:
            traceback.print_exc()
            results.append((name, f'FAIL: {type(e).__name__}: {e}', time.time() - t0))
        _save_open_figures(name)
        plt.close('all')

    if backup_first:
        _backup_shipped_data()

    # ---- 1. TB precipitation cleaning (executed, not written: the shipped
    #         data/Cleaned/TB files are authoritative) ----
    def tb():
        from TB_clean_precip import TippingBucket_prec
        tipping_buckets = TippingBucket_prec(update_csv=False, dt='0.25h')
        print(f'TB cleaning produced {len(tipping_buckets)} station series '
              f'(not written; figures use the shipped data/Cleaned/TB)')
    stage('Tipping-bucket precipitation cleaning', tb)

    # ---- 2. Pluvio + AWS precipitation cleaning (writes data/Cleaned/Pluvio) ----
    def pluvio():
        from pluvio_cleaned_precip import get_Pluvio_rain
        get_Pluvio_rain(update_csv=True)
    stage('Pluvio + AWS precipitation cleaning', pluvio)

    # ---- 3. Kochendorfer correction (writes data/Cleaned/Kochendorfer_corrected) ----
    def koch():
        from kochendorfer_correction import process_and_save
        process_and_save()
    stage('Kochendorfer undercatch correction', koch)

    # ---- 4. Wind cleaning (executed, not written: the exact save-run of the
    #         shipped data/Cleaned/Wind files is not preserved in the code) ----
    def wind():
        from clean_wind import get_aws_df_wind
        wind_data = get_aws_df_wind()
        for st, df in wind_data.items():
            print(f'  {st}: {len(df)} cleaned wind records')
    stage('Wind cleaning', wind)

    # ---- 5. SW/LW radiation loading (PLU1 + PLU2) ----
    def swlw():
        from clean_SW_LW import get_aws_df_SW_LW
        rad = get_aws_df_SW_LW()
        for st, df in rad.items():
            print(f'  {st}: {len(df)} records, columns {list(df.columns)}')
    stage('SW/LW radiation loading', swlw)

    # ---- 6. Humidity timeseries generation, incl. RH cleaning
    #         (writes data/Moisture + LapseRate summaries) ----
    def humidity():
        from generate_humidity_timeseries import get_seasonal_RH_PRES_data
        get_seasonal_RH_PRES_data()
    stage('Humidity timeseries generation (RH cleaning + thermodynamics)', humidity)

    # ---- 7. Temperature merge of all sensors
    #         (writes data/Cleaned/Temperature, data/Merged/merged_temperature.csv
    #          and refreshes data/temp_merged_dfs.pkl used by the figures) ----
    def temperature():
        import merge_temperature as mt
        merged = mt.merge_datasets_hourly()
        mt.write_merged_csv(merged)
        mt.save_temp_merged_pickle(merged)
        print('temp_merged_dfs.pkl refreshed')
    stage('Temperature cleaning + merge (all sensors)', temperature)

    # ---- 8. Lapse rate / isotherm computation (compute only: the shipped
    #         lapse_rate.csv / zero_isotherm.csv are authoritative) ----
    def isotherm():
        from lapse_rate_isotherm import load_temp_merged, zero_deg_altitude
        interpolated = zero_deg_altitude(load_temp_merged())
        print(interpolated[['lapse_rate', 'Zero_Deg_Elevation', 'One_Deg_Elevation']].describe())
    stage('Lapse rate / zero-degree isotherm computation', isotherm)

    # ---- 9. Valley geometry (writes data/Geometry CSV) ----
    def geometry():
        from valley_geometry import analyze_valley_geometry
        analyze_valley_geometry(plot_map=False)
    stage('Valley geometry from DEM', geometry)

    # ---- 10. Optional merged single-CSV products ----
    def merges():
        os.makedirs(_DATA_DIR / 'Merged', exist_ok=True)
        from merge_precipitation import merge_precipitation
        m = merge_precipitation()
        m.to_csv(_DATA_DIR / 'Merged' / 'merged_precipitation.csv', na_rep='NA')
        print(f'  merged_precipitation: {m.shape}')
        from merge_RH import merge_RH
        m = merge_RH()
        m.to_csv(_DATA_DIR / 'Merged' / 'merged_RH.csv', na_rep='NA')
        print(f'  merged_RH: {m.shape}')
        from merge_SW_LW import merge_SW_LW
        m = merge_SW_LW()
        m.to_csv(_DATA_DIR / 'Merged' / 'merged_SW_LW.csv', na_rep='NA')
        print(f'  merged_SW_LW: {m.shape}')
    stage('Merged single-CSV products', merges)

    # ---- 11. Data-availability overview figure ----
    def overview():
        import plot_data_overview as pdo
        pdo.get_data_overview(pdo._DATA_OVERVIEW_TXT, pdo.STATION_ABBREV, font_scale=1.5)
    stage('Data overview figure', overview)

    # ---- 12. wrapperv3 analysis figures ----
    # Importing wrapperv3 executes the figure calls that are enabled at module
    # level; the remaining figure functions are then called individually.
    def wrapper_import():
        global _w
        import wrapperv3 as _w
    stage('wrapperv3 module-level figures', wrapper_import)

    extra_figures = [
        'plot_percentage_below_zero',
        'plot_seasonal_diurnal_compact',
        'plot_monthly_climate_overview',
        'plot_precip_august_may_comparison_with_slope_stations',
        'plot_seasonal_undercatch_summary_from_corrected',
        'snowfall_threshold_sensitivity',
        'plot_snowcover_albedo',
        'analyze_precipitation_sensitivity_to_temperature_change_all_seasons',
    ]
    import wrapperv3 as w
    for fname in extra_figures:
        fn = getattr(w, fname, None)
        if fn is None:
            results.append((f'wrapperv3.{fname}', 'FAIL: not found', 0.0))
            continue
        if fname == 'plot_percentage_below_zero':
            stage(f'wrapperv3.{fname}', lambda fn=fn: fn(w.temp_merged_dfs))
        else:
            stage(f'wrapperv3.{fname}', fn)

    # ---- summary ----
    print(f"\n\n{'='*70}\nTESTING SUMMARY\n{'='*70}")
    n_fail = 0
    for name, res, dt in results:
        flag = 'PASS' if res == 'PASS' else 'FAIL'
        if flag == 'FAIL':
            n_fail += 1
        print(f"{flag}  {name}  ({dt:.0f}s)" + ('' if res == 'PASS' else f"\n      {res}"))
    print(f"\n{len(results) - n_fail}/{len(results)} stages passed")
    return results


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    testing_function()


   