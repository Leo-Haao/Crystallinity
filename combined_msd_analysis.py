# -*- coding: utf-8 -*-
"""
A unified script for calculating and plotting Mean Squared Displacement (MSD) data
from molecular dynamics simulations for multiple temperatures (700K, 600K, 500K).

This script integrates robust data analysis with specific plotting formats.

It supports two distinct workflows based on a 'TASK_TYPE' configuration:
1.  **'calculate_and_plot' (for 700K, 600K):**
    - Calculates MSD from raw XYZ/XCD trajectory files using an improved PBC unwrapping algorithm.
    - Processes additional pre-calculated MSD data from specified XCD files.
    - Saves all combined results into a single, unified, temperature-specific CSV file.
    - Loads the unified CSV data to generate plots.
2.  **'plot_from_existing' (for 500K):**
    - Directly scans a directory for pre-calculated MSD.xcd files.
    - Extracts data from these files to generate plots without performing new calculations.

It generates two styles of plots for each temperature:
a)  **Style 1 (Individual Runs):** Each run is plotted as a distinct colored line.
b)  **Style 2 (Average Trend):** All runs are plotted in grey with a prominent red average line.

Finally, all generated plots are displayed on screen.
"""

import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import gc
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ==============================================================================
# --- 1. General Configuration ---
# ==============================================================================
TARGET_ION = "Li"
# --- Path Placeholders ---
# Define a base path to make the script more portable.
# All other paths will be constructed relative to this one.
# IMPORTANT: Update this path to match your local directory structure.
BASE_PATH = r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents"

# --- Main Data Folders ---
ROOT_FOLDER = os.path.join(BASE_PATH, "Tri_comb", "Crystal")
XCD_DATA_ROOT = os.path.join(BASE_PATH, "PEO_RUN", "crystal", "10ns")

INTERP_STEP = 1.0   # ps, for interpolation step in averaging

# Shared plotting style configurations (with improved labels from script 2)
Y_LIMITS_CONFIG = {
    700: {
        'total': {'ylim': (0, 10000), 'yticks': np.arange(0, 10001, 1800)},
        'xx component': {'ylim': (0, 3000), 'yticks': np.arange(0, 3001, 600), 'label': r'$X\ [100]$'},
        'yy component': {'ylim': (0, 3000), 'yticks': np.arange(0, 3001, 600), 'label': r'$Y\ [010]$'},
        'zz component': {'ylim': (0, 6000), 'yticks': np.arange(0, 6001, 1000), 'label': r'$Z\ [001]$'},
    },
    600: {
        'total': {'ylim': (0, 3000), 'yticks': np.arange(0, 3001, 600)},
        'xx component': {'ylim': (0, 1000), 'yticks': np.arange(0, 1001, 200), 'label': r'$X\ [100]$'},
        'yy component': {'ylim': (0, 1000), 'yticks': np.arange(0, 1001, 200), 'label': r'$Y\ [010]$'},
        'zz component': {'ylim': (0, 2500), 'yticks': np.arange(0, 2501, 500), 'label': r'$Z\ [001]$'},
    },
    500: {
        'total': {'ylim': (0, 1000), 'yticks': np.arange(0, 1001, 200)},
        'xx component': {'ylim': (0, 200), 'yticks': np.arange(0, 201, 50), 'label': r'$X\ [100]$'},
        'yy component': {'ylim': (0, 100), 'yticks': np.arange(0, 101, 20), 'label': r'$Y\ [010]$'},
        'zz component': {'ylim': (0, 700), 'yticks': np.arange(0, 701, 140), 'label': r'$Z\ [001]$'},
    }
}
SETOFF_TIMES = {
    700: 27.0,
    600: 22.5,
    500: 47.5
}

# ==============================================================================
# --- 2. Temperature-Specific Configurations ---
# ==============================================================================
CONFIG_700K = {
    'TASK_TYPE': 'calculate_and_plot',
    'TEMP': 700,
    'CUTOFF_TIME': 27.0,
    'RUN_FOLDERS': [
        "supercell_17_700K_run_1", "supercell_17_700K_run_4", "supercell_17_700K_run_5",
        "supercell_17_700K_run_6", "supercell_17_700K_run_7", "supercell_17_700K_run_10"
    ],
    'XCD_PATHS': [
        os.path.join(XCD_DATA_ROOT, '700K', 'supercell_17_700K_run_2', 'PEO_Li_final_supercell Forcite MSD.xcd'),
    ],
    'OUTPUT_CSV_PATH': os.path.join(ROOT_FOLDER, "MSD_from_27ps_700K.csv"),
}

CONFIG_600K = {
    'TASK_TYPE': 'calculate_and_plot',
    'TEMP': 600,
    'CUTOFF_TIME': 22.5,
    'RUN_FOLDERS': [
        "supercell_17_600K_run_11", "supercell_17_600K_run_12", "supercell_17_600K_run_13",
        "supercell_17_600K_run_14", "supercell_17_600K_run_15"
    ],
    'XCD_PATHS': [
        os.path.join(XCD_DATA_ROOT, '600K', 'supercell_17_600K_run_3', 'PEO_Li_final_supercell Forcite MSD.xcd'),
        os.path.join(XCD_DATA_ROOT, '600K', 'supercell_17_600K_run_7', 'PEO_Li_final_supercell Forcite MSD.xcd')
    ],
    'OUTPUT_CSV_PATH': os.path.join(ROOT_FOLDER, "MSD_from_22_5ps_600K.csv")
}

CONFIG_500K = {
    'TASK_TYPE': 'plot_from_existing',
    'TEMP': 500,
    'CUTOFF_TIME': 47.5,
    'DATA_ROOT': os.path.join(XCD_DATA_ROOT, '500K')
}

# ==============================================================================
# --- 3. Core Data Parsing and MSD Calculation (Analysis logic from Script 2) ---
# ==============================================================================
def parse_xyz_file(file_path: str) -> List[Dict]:
    """Parses an XYZ file and returns a list of frames."""
    frames = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        i = 0
        natoms = None
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if natoms is None:
                natoms = int(line)
            i += 1
            comment_line = lines[i].strip()
            time_match = re.search(r'Time: (\d+\.?\d*)', comment_line)
            if not time_match:
                i += natoms + 1
                continue
            frame_time = float(time_match.group(1))
            i += 1
            atom_lines = [lines[j].strip() for j in range(i, i + natoms)]
            frames.append({'time': frame_time, 'atoms': atom_lines})
            i += natoms
    except Exception as e:
        print(f"[Error] Failed to parse XYZ file {os.path.basename(file_path)}: {e}")
    return frames

def parse_xcd_file_for_lattice(file_path: str) -> List[List[float]]:
    """Parses an XCD file to extract lattice parameters for MSD calculation."""
    lattice_params = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        axis_data = {'A': [], 'B': [], 'C': []}
        for series in root.findall('.//SERIES_2D'):
            series_name = series.get('Name', '')
            for axis in ['A', 'B', 'C']:
                if f'Length {axis}' in series_name:
                    points = series.findall('POINT_2D')
                    axis_data[axis] = [float(p.get('XY').split(',')[1]) for p in points if ',' in p.get('XY')]
        min_len = min(len(d) for d in axis_data.values())
        for i in range(min_len):
            lattice_params.append([axis_data['A'][i], axis_data['B'][i], axis_data['C'][i]])
    except Exception as e:
        print(f"[Error] Failed to parse XCD for lattice {os.path.basename(file_path)}: {e}")
    return lattice_params

def extract_msd_from_xcd(file_path, component=None) -> Dict[float, float]:
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

def process_single_sample_for_msd(folder_name: str, output_file: str, cutoff_time: float):
    """Processes a single sample, calculates MSD with improved PBC, and appends the result to the output file."""
    folder_path = os.path.join(ROOT_FOLDER, folder_name)

    traj1_path = os.path.join(folder_path, "supercell_17.xyz")
    traj2_path = os.path.join(folder_path, "3D Atomistic.xyz")
    xcd1_path = os.path.join(folder_path, "PEO_Li_final_supercell Forcite Cell parameters.xcd")
    xcd2_path = os.path.join(folder_path, "3D Atomistic Forcite Cell parameters.xcd")

    missing_files = [f for f in [traj1_path, traj2_path, xcd1_path, xcd2_path] if not os.path.exists(f)]
    if missing_files:
        print(f"[Warning] Sample {folder_name} is missing files: {', '.join(map(os.path.basename, missing_files))}, skipping.")
        return

    traj1_frames = parse_xyz_file(traj1_path)
    traj2_frames = parse_xyz_file(traj2_path)
    lattice1 = parse_xcd_file_for_lattice(xcd1_path)
    lattice2 = parse_xcd_file_for_lattice(xcd2_path)

    if not traj1_frames or not traj2_frames or len(traj1_frames) != len(lattice1) or len(traj2_frames) != len(lattice2):
        print(f"[Warning] Sample {folder_name} has inconsistent frame or lattice data, skipping.")
        return

    t1_end = traj1_frames[-1]['time']
    for frame in traj2_frames:
        frame['time'] += t1_end
    total_frames = traj1_frames + traj2_frames
    total_lattice = lattice1 + lattice2

    try:
        cutoff_index = next(i for i, frame in enumerate(total_frames) if frame['time'] >= cutoff_time)
    except StopIteration:
        print(f"[Warning] No data found after {cutoff_time} ps for {folder_name}, skipping.")
        return

    frames_truncated = total_frames[cutoff_index:]
    lattice_truncated = total_lattice[cutoff_index:]
    timestamps_truncated = [frame['time'] for frame in frames_truncated]

    target_indices = [i for i, atom in enumerate(frames_truncated[0]['atoms']) if atom.split()[0] == TARGET_ION]
    if not target_indices:
        print(f"[Warning] Target ion {TARGET_ION} not found in {folder_name}, skipping.")
        return

    num_ions, num_frames = len(target_indices), len(frames_truncated)
    unwrapped_coords = np.zeros((num_ions, num_frames, 3))

    for i, ion_idx in enumerate(target_indices):
        unwrapped_coords[i, 0, :] = np.array(list(map(float, frames_truncated[0]['atoms'][ion_idx].split()[1:4])))

    # Correct PBC unwrapping logic from script 2
    for frame_idx in range(1, num_frames):
        box_dims = (np.array(lattice_truncated[frame_idx-1]) + np.array(lattice_truncated[frame_idx])) / 2
        for i, ion_idx in enumerate(target_indices):
            prev_coords_unwrapped = unwrapped_coords[i, frame_idx-1, :]
            curr_coords_wrapped = np.array(list(map(float, frames_truncated[frame_idx]['atoms'][ion_idx].split()[1:4])))
            prev_coords_wrapped = np.array(list(map(float, frames_truncated[frame_idx-1]['atoms'][ion_idx].split()[1:4])))

            displacement_wrapped = curr_coords_wrapped - prev_coords_wrapped

            correction = box_dims * np.round(displacement_wrapped / box_dims)
            unwrapped_displacement = displacement_wrapped - correction

            unwrapped_coords[i, frame_idx, :] = prev_coords_unwrapped + unwrapped_displacement

    displacements = unwrapped_coords - unwrapped_coords[:, 0:1, :]
    msd_df = pd.DataFrame({
        'Sample_Name': folder_name,
        'Sample_Time (ps)': timestamps_truncated,
        'Relative_Time (ps)': np.array(timestamps_truncated) - timestamps_truncated[0],
        'Total_MSD (Å²)': np.mean(np.sum(displacements**2, axis=2), axis=0),
        'X_MSD (Å²)': np.mean(displacements[:, :, 0]**2, axis=0),
        'Y_MSD (Å²)': np.mean(displacements[:, :, 1]**2, axis=0),
        'Z_MSD (Å²)': np.mean(displacements[:, :, 2]**2, axis=0)
    })

    msd_df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file), float_format='%.6f', encoding='utf-8-sig')
    print(f"[Success] Sample {folder_name} processed.")
    gc.collect()

def process_single_xcd_for_msd(xcd_path: str, output_file: str, cutoff_time: float):
    """
    Reads MSD data directly from an XCD file, applies cutoff time,
    and appends it to the main MSD CSV file for unified plotting.
    """
    comp_keys = ['total', 'xx component', 'yy component', 'zz component']
    # Use the existing `extract_msd_from_xcd` helper
    msd_data = {k: extract_msd_from_xcd(xcd_path, k if k != 'total' else None) for k in comp_keys}

    if not msd_data['total']:
        return

    times = np.array(sorted(msd_data['total'].keys()))
    mask = times >= cutoff_time

    if not np.any(mask):
        return

    times_truncated = times[mask]
    relative_time = times_truncated - cutoff_time

    data_dict = {
        'Sample_Name': os.path.basename(os.path.dirname(xcd_path)),
        'Sample_Time (ps)': times_truncated,
        'Relative_Time (ps)': relative_time,
        'Total_MSD (Å²)': np.array([msd_data['total'].get(t, np.nan) for t in times_truncated]),
        'X_MSD (Å²)': np.array([msd_data['xx component'].get(t, np.nan) for t in times_truncated]),
        'Y_MSD (Å²)': np.array([msd_data['yy component'].get(t, np.nan) for t in times_truncated]),
        'Z_MSD (Å²)': np.array([msd_data['zz component'].get(t, np.nan) for t in times_truncated]),
    }

    msd_df = pd.DataFrame(data_dict)
    msd_df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file), float_format='%.6f', encoding='utf-8-sig')
    print(f"[Processing] Sample (XCD): {os.path.basename(os.path.dirname(xcd_path))} added.")
    gc.collect()

def run_msd_calculation(config: Dict):
    """Orchestrates the MSD calculation for a given temperature configuration."""
    temp, run_folders, xcd_paths, output_csv, cutoff_time = (
        config['TEMP'], config['RUN_FOLDERS'], config['XCD_PATHS'], config['OUTPUT_CSV_PATH'], config['CUTOFF_TIME']
    )
    print("=" * 70)
    print(f"Starting MSD Calculation for {temp}K ({len(run_folders)} samples + {len(xcd_paths)} XCDs)")
    print(f"Output CSV: {os.path.basename(output_csv)}")
    print("=" * 70)
    if os.path.exists(output_csv):
        os.remove(output_csv)

    for folder_name in run_folders:
        process_single_sample_for_msd(folder_name, output_csv, cutoff_time)

    for xcd_path in xcd_paths:
        process_single_xcd_for_msd(xcd_path, output_csv, cutoff_time)

    print(f"\n[Complete] MSD calculation for {temp}K finished.")

# ==============================================================================
# --- 4. Data Loading and Plotting (Plotting logic from Script 1) ---
# ==============================================================================
class PlotProperties:
    def __init__(self, font_type="Times New Roman", font_size=26, axis_ticks_font_size=24, label_x=r'Time / ps', label_y=r'MSD / $\AA^2$', fig_size=(8, 6), legend_size=20):
        self.font_type, self.font_size, self.axis_ticks_font_size = font_type, font_size, axis_ticks_font_size
        self.label_x, self.label_y, self.fig_size, self.legend_size = label_x, label_y, fig_size, legend_size

    def apply_style(self):
        plt.figure(figsize=self.fig_size)
        axis_font = {'fontname': self.font_type, 'size': self.font_size, 'weight': 'bold'}
        plt.rc('font', **{'family': self.font_type, 'size': self.legend_size, 'weight': 'bold'})
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = self.font_type
        plt.xlabel(self.label_x, **axis_font)
        plt.ylabel(self.label_y, **axis_font)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(axis='both', direction='in', width=2, length=6, top=True, right=True, labelsize=self.axis_ticks_font_size)
        plt.tight_layout()
        return plt

def load_data_from_csv_and_xcd(config: Dict) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Simplified: Loads all data from the pre-populated CSV.
    The XCD files were already merged into the CSV by run_msd_calculation.
    """
    csv_path = config['OUTPUT_CSV_PATH']
    all_data = {k: [] for k in ['total', 'xx component', 'yy component', 'zz component']}
    try:
        df = pd.read_csv(csv_path)
        time_col = 'Relative_Time (ps)'
        for name in sorted(df['Sample_Name'].unique()):
            group = df[df['Sample_Name'] == name].sort_values(time_col).drop_duplicates(subset=[time_col], keep='last')
            all_data['total'].append((group[time_col].values, group['Total_MSD (Å²)'].values))
            all_data['xx component'].append((group[time_col].values, group['X_MSD (Å²)'].values))
            all_data['yy component'].append((group[time_col].values, group['Y_MSD (Å²)'].values))
            all_data['zz component'].append((group[time_col].values, group['Z_MSD (Å²)'].values))
    except Exception as e:
        print(f"[Error] Failed to load CSV data: {e}")
    return all_data

def load_data_from_existing_xcds(config: Dict) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """Scans a directory for MSD.xcd files and loads data from them."""
    data_root, cutoff_time = config['DATA_ROOT'], config['CUTOFF_TIME']
    all_data = {k: [] for k in ['total', 'xx component', 'yy component', 'zz component']}

    folder_pattern = rf'supercell_17_{config["TEMP"]}K_run_\d+'
    subfolders = sorted([d for d in os.listdir(data_root) if re.match(folder_pattern, d)],
                        key=lambda x: int(re.search(r'run_(\d+)', x).group(1)))

    for folder in subfolders:
        for dirpath, _, files in os.walk(os.path.join(data_root, folder)):
            for f in files:
                if f.endswith('MSD.xcd'):
                    xcd_path = os.path.join(dirpath, f)
                    for comp in all_data.keys():
                        msd_dict = extract_msd_from_xcd(xcd_path, comp if comp != 'total' else None)
                        if msd_dict:
                            times, msds = np.array(sorted(msd_dict.keys())), np.array([msd_dict[t] for t in sorted(msd_dict.keys())])
                            mask = times >= cutoff_time
                            if np.any(mask):
                                all_data[comp].append((times[mask] - cutoff_time, msds[mask]))
                    break
    return all_data

def plot_style_1_individual_runs(all_data, temperature):
    """Plots each run with a distinct color."""
    plot_style, total_runs = PlotProperties(), len(all_data['total'])
    colors = plt.get_cmap('plasma')(np.linspace(0, 0.8, total_runs))
    legend_title = f'{temperature} K Simulation\n' + r'$\mathrm{setoff}: ' + f'{SETOFF_TIMES[temperature]:.1f}' + r'\ \mathrm{ps}$'
    plot_configs = Y_LIMITS_CONFIG[temperature]

    for comp_key, config in plot_configs.items():
        plt_comp = plot_style.apply_style()
        for i, (x, y) in enumerate(all_data[comp_key]):
            if len(x) > 0 and len(y) > 0:
                plt_comp.plot(x, y, color=colors[i], linestyle='--', lw=2, alpha=0.8, marker=".", ms=10, label=f"Run {i+1}")

        plt_comp.ylim(config['ylim'])
        plt_comp.yticks(config['yticks'])
        plt_comp.legend(title=legend_title, loc='upper left', ncol=2, framealpha=1, edgecolor='k', fontsize=12, title_fontsize=14)
        title = f"Total MSD – {temperature} K" if comp_key == 'total' else f"{config['label']} – {temperature} K"
        plt_comp.title(title, fontsize=plot_style.font_size, fontweight='bold', y=1.03)

def plot_style_2_average_trend(all_data, temperature):
    """Plots all runs in grey and their average in red."""
    style = PlotProperties(fig_size=(12, 8))
    all_max_times = [np.max(t) for t, _ in all_data['total'] if len(t) > 0]
    if not all_max_times: return
    global_time_max = min(all_max_times)

    plot_configs = Y_LIMITS_CONFIG[temperature]
    for comp_key, config in plot_configs.items():
        plt_comp = style.apply_style()

        # Plot individual runs in grey
        truncated_data = [(x[x <= global_time_max], y[x <= global_time_max]) for x, y in all_data[comp_key] if len(x) > 0]
        for x, y in truncated_data:
            plt_comp.plot(x, y, 'grey', linestyle='-', linewidth=2, alpha=0.3)

        # Calculate and plot the average
        unified_t = np.arange(0, global_time_max, INTERP_STEP)
        # Filter out non-monotonic time series before interpolation
        valid_data = [d for d in truncated_data if np.all(np.diff(d[0]) > 0)]
        if valid_data:
            interpolated_msds = [interp1d(t, m, bounds_error=False, fill_value="extrapolate")(unified_t) for t, m in valid_data]
            if interpolated_msds:
                plt_comp.plot(unified_t, np.mean(interpolated_msds, axis=0), 'red', linewidth=4, label="Average")

        # Apply axis limits and ticks from config
        plt_comp.xlim(0, global_time_max)
        plt_comp.xticks(np.linspace(0, global_time_max, 6))
        plt_comp.ylim(config['ylim'])
        plt_comp.yticks(config['yticks'])

        plt_comp.legend(loc='upper left', framealpha=1, edgecolor='k', fontsize=style.legend_size)
        title = f"Total MSD – {temperature} K" if comp_key == 'total' else f"{config['label']} – {temperature} K"
        plt_comp.title(title, fontsize=style.font_size, fontweight='bold', y=1.03)

def run_plotting(config: Dict, all_data: Dict):
    """Orchestrates the plotting of MSD data for a given temperature."""
    temp = config['TEMP']
    print("\n" + "=" * 70)
    print(f"Starting Plotting for {temp}K")
    print("=" * 70)
    plot_style_1_individual_runs(all_data, temp)
    plot_style_2_average_trend(all_data, temp)
    print(f"[Complete] Plotting for {temp}K finished.")

# ==============================================================================
# --- 5. Main Execution ---
# ==============================================================================
def main():
    """Main function to run the entire analysis workflow."""
    all_configs_to_run = [CONFIG_700K, CONFIG_600K, CONFIG_500K]

    for config in all_configs_to_run:
        task_type = config['TASK_TYPE']

        if task_type == 'calculate_and_plot':
            run_msd_calculation(config)
            if not os.path.exists(config['OUTPUT_CSV_PATH']):
                print(f"[Error] MSD calculation for {config['TEMP']}K failed to produce an output file. Skipping plotting.")
                continue
            all_data = load_data_from_csv_and_xcd(config)

        elif task_type == 'plot_from_existing':
            all_data = load_data_from_existing_xcds(config)

        else:
            print(f"[Error] Unknown TASK_TYPE '{task_type}' for {config['TEMP']}K. Skipping.")
            continue

        if not any(all_data.values()):
            print(f"[Warning] No data was loaded for {config['TEMP']}K. Skipping plotting.")
            continue

        run_plotting(config, all_data)

    print("\n" + "="*70)
    print("All analyses complete. Displaying all generated plots...")
    print("="*70)
    plt.show()

if __name__ == "__main__":
    main()
