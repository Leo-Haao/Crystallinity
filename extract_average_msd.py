# -*- coding: utf-8 -*-
"""
extract_average_msd.py

This script loads all existing MSD data for 500K, 600K, and 700K simulations,
calculates the average MSD curve for each temperature, and saves the combined
results to a single summary CSV file.

It assumes that the MSD data for 600K and 700K has already been calculated
and is available in the respective CSV files, while the 500K data exists as
pre-calculated XCD files.
"""

import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.interpolate import interp1d

# ==============================================================================
# --- 1. Global Configuration ---
# ==============================================================================
ROOT_FOLDER = r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\Tri_comb\Crystal"
OUTPUT_DIR = os.path.join(ROOT_FOLDER, "combined_analysis_output")
AVERAGE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "All_Temperatures_Average_MSD_extracted.csv")
INTERP_STEP = 1.0  # ps, for interpolation step in averaging

# ==============================================================================
# --- 2. Data Source Configurations ---
# ==============================================================================
CONFIG_700K = {
    'TEMP': 700,
    'CUTOFF_TIME': 27.0,
    'CSV_PATH': os.path.join(ROOT_FOLDER, "MSD_from_27ps_700K.csv"),
    'XCD_PATHS': [
        r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\PEO_RUN\crystal\10ns\700K\supercell_17_700K_run_2\PEO_Li_final_supercell Forcite MSD.xcd",
    ]
}

CONFIG_600K = {
    'TEMP': 600,
    'CUTOFF_TIME': 22.5,
    'CSV_PATH': os.path.join(ROOT_FOLDER, "MSD_from_22_5ps_600K.csv"),
    'XCD_PATHS': [
        r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\PEO_RUN\crystal\10ns\600K\supercell_17_600K_run_3\PEO_Li_final_supercell Forcite MSD.xcd",
        r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\PEO_RUN\crystal\10ns\600K\supercell_17_600K_run_7\PEO_Li_final_supercell Forcite MSD.xcd"
    ]
}

CONFIG_500K = {
    'TEMP': 500,
    'CUTOFF_TIME': 47.5,
    'DATA_ROOT': r'E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\PEO_RUN\crystal\10ns\500K'
}

# ==============================================================================
# --- 3. Data Loading and Processing Functions ---
# ==============================================================================

def extract_msd_from_xcd(file_path: str, component: str = None) -> Dict[float, float]:
    """Extracts a pre-computed MSD data series from an XCD file."""
    msd_results = {}
    target_line = 'Total MSD' if component is None else component
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        series_match = re.search(f'Name="{target_line}".*?>(.*?)</SERIES_2D>', content, re.DOTALL)
        if series_match:
            points = re.findall(r'<POINT_2D XY="([^"]+)"', series_match.group(1))
            for p_str in points:
                time, msd = map(float, p_str.split(',')[:2])
                msd_results[time] = msd
    except Exception as e:
        print(f"[Error] Failed to extract {target_line} from {os.path.basename(file_path)}: {e}")
    return msd_results

def load_data_from_csv_and_xcd(config: Dict) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """Loads and combines data from a CSV and additional XCD files."""
    csv_path = config['CSV_PATH']
    xcd_paths = config.get('XCD_PATHS', [])
    cutoff_time = config['CUTOFF_TIME']
    all_data = {k: [] for k in ['total', 'xx component', 'yy component', 'zz component']}

    # Load data from the main CSV file
    try:
        df = pd.read_csv(csv_path)
        # Ensure 'Relative_Time (ps)' exists, otherwise calculate it
        if 'Relative_Time (ps)' not in df.columns:
             df['Relative_Time (ps)'] = df['Sample_Time (ps)'] - cutoff_time

        for name in sorted(df['Sample_Name'].unique()):
            group = df[df['Sample_Name'] == name]
            all_data['total'].append((group['Relative_Time (ps)'].values, group['Total_MSD (Å²)'].values))
            all_data['xx component'].append((group['Relative_Time (ps)'].values, group['X_MSD (Å²)'].values))
            all_data['yy component'].append((group['Relative_Time (ps)'].values, group['Y_MSD (Å²)'].values))
            all_data['zz component'].append((group['Relative_Time (ps)'].values, group['Z_MSD (Å²)'].values))
    except FileNotFoundError:
        print(f"[Error] CSV file not found for {config['TEMP']}K: {csv_path}")
    except Exception as e:
        print(f"[Error] Failed to load CSV data for {config['TEMP']}K: {e}")

    # Load data from supplementary XCD files
    for path in xcd_paths:
        if not os.path.exists(path):
            print(f"[Warning] Supplementary XCD file not found: {path}")
            continue
        for comp_key in all_data.keys():
            component_name = comp_key if comp_key != 'total' else None
            msd_dict = extract_msd_from_xcd(path, component_name)
            if msd_dict:
                times, msds = np.array(sorted(msd_dict.keys())), np.array([msd_dict[t] for t in sorted(msd_dict.keys())])
                mask = times >= cutoff_time
                if np.any(mask):
                    all_data[comp_key].append((times[mask] - cutoff_time, msds[mask]))
    return all_data

def load_data_from_existing_xcds(config: Dict) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """Scans a directory for MSD.xcd files and loads data from them."""
    data_root = config['DATA_ROOT']
    cutoff_time = config['CUTOFF_TIME']
    all_data = {k: [] for k in ['total', 'xx component', 'yy component', 'zz component']}

    if not os.path.isdir(data_root):
        print(f"[Error] Data directory not found for {config['TEMP']}K: {data_root}")
        return all_data

    folder_pattern = rf'supercell_17_{config["TEMP"]}K_run_\d+'
    subfolders = sorted(
        [d for d in os.listdir(data_root) if re.match(folder_pattern, d) and os.path.isdir(os.path.join(data_root, d))],
        key=lambda x: int(re.search(r'run_(\d+)', x).group(1))
    )

    for folder in subfolders:
        folder_path = os.path.join(data_root, folder)
        found_xcd = False
        for dirpath, _, files in os.walk(folder_path):
            for f in files:
                if f.endswith('MSD.xcd'):
                    xcd_path = os.path.join(dirpath, f)
                    for comp_key in all_data.keys():
                        component_name = comp_key if comp_key != 'total' else None
                        msd_dict = extract_msd_from_xcd(xcd_path, component_name)
                        if msd_dict:
                            times, msds = np.array(sorted(msd_dict.keys())), np.array([msd_dict[t] for t in sorted(msd_dict.keys())])
                            mask = times >= cutoff_time
                            if np.any(mask):
                                all_data[comp_key].append((times[mask] - cutoff_time, msds[mask]))
                    found_xcd = True
                    break # Assume one MSD file per run folder
            if found_xcd:
                break
    return all_data


def calculate_average_msd(all_data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]) -> Dict[str, np.ndarray]:
    """Calculates the average MSD curves from a list of runs."""
    all_max_times = [np.max(t) for t, _ in all_data['total'] if len(t) > 0]
    if not all_max_times:
        return {}
    global_time_max = min(all_max_times)

    average_data = {}

    plot_configs = [
        ('total', 'Total_MSD_Avg'),
        ('xx component', 'X_MSD_Avg'),
        ('yy component', 'Y_MSD_Avg'),
        ('zz component', 'Z_MSD_Avg')
    ]

    for comp_key, avg_col_name in plot_configs:
        truncated_data = [(x[x <= global_time_max], y[x <= global_time_max]) for x, y in all_data[comp_key] if len(x) > 0]
        if not truncated_data:
            continue

        unified_t = np.arange(0, global_time_max, INTERP_STEP)
        interpolated_msds = [interp1d(t, m, bounds_error=False, fill_value="extrapolate")(unified_t) for t, m in truncated_data if np.all(np.diff(t) >= 0)]

        if interpolated_msds:
            avg_msd = np.mean(interpolated_msds, axis=0)
            if not average_data:  # Store time axis only once
                average_data["Time (ps)"] = unified_t
            average_data[f"{avg_col_name} (Å²)"] = avg_msd

    return average_data

# ==============================================================================
# --- 4. Main Execution ---
# ==============================================================================
def main():
    """Main function to run the data extraction and averaging workflow."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_configs = [CONFIG_700K, CONFIG_600K, CONFIG_500K]
    all_average_dataframes = []

    print("Starting extraction of average MSD data...")

    for config in all_configs:
        temp = config['TEMP']
        print(f"\n" + "="*70)
        print(f"Processing data for {temp}K...")

        # Step 1: Load data based on the configuration type
        if 'CSV_PATH' in config:
            all_data = load_data_from_csv_and_xcd(config)
        elif 'DATA_ROOT' in config:
            all_data = load_data_from_existing_xcds(config)
        else:
            print(f"[Error] Invalid configuration for {temp}K. Skipping.")
            continue

        if not any(all_data.values()):
            print(f"[Warning] No data was loaded for {temp}K. Cannot calculate average.")
            continue

        # Step 2: Calculate the average MSD
        average_data = calculate_average_msd(all_data)

        if average_data:
            # Step 3: Convert to DataFrame and add temperature suffixes to columns
            df = pd.DataFrame(average_data)
            df.columns = [f"{col.split(' ')[0]}_{temp}K {col.split(' ')[1]}" if 'Time' not in col else f"Time_{temp}K (ps)" for col in df.columns]
            all_average_dataframes.append(df)
            print(f"[Success] Averaged data for {temp}K calculated.")
        else:
            print(f"[Warning] Could not calculate average for {temp}K.")

    # Step 4: Combine all DataFrames and save to a single CSV
    if all_average_dataframes:
        print("\n" + "="*70)
        print("Combining and saving all average MSD data...")
        final_avg_df = pd.concat(all_average_dataframes, axis=1)

        # Reorder columns to group by component
        time_cols = sorted([col for col in final_avg_df.columns if 'Time' in col], reverse=True)
        total_cols = sorted([col for col in final_avg_df.columns if 'Total' in col], reverse=True)
        x_cols = sorted([col for col in final_avg_df.columns if 'X_MSD' in col], reverse=True)
        y_cols = sorted([col for col in final_avg_df.columns if 'Y_MSD' in col], reverse=True)
        z_cols = sorted([col for col in final_avg_df.columns if 'Z_MSD' in col], reverse=True)

        final_avg_df = final_avg_df[time_cols + total_cols + x_cols + y_cols + z_cols]

        final_avg_df.to_csv(AVERAGE_OUTPUT_PATH, index=False, float_format='%.6f', encoding='utf-8-sig')
        print(f"[Success] All average MSD data has been saved to:\n{AVERAGE_OUTPUT_PATH}")
    else:
        print("\n[Complete] No average data was generated to save.")

    print("\n" + "="*70)
    print("Workflow complete.")

if __name__ == "__main__":
    main()
