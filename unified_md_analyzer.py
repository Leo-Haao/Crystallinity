# -*- coding: utf-8 -*-
"""
A unified script for calculating and plotting Mean Squared Displacement (MSD) data
from molecular dynamics simulations for multiple temperatures.

This script performs the following steps for each configured temperature (e.g., 700K, 600K):
1.  Calculates MSD from raw XYZ and XCD trajectory files for a list of specified simulation runs.
    - It handles interrupted simulations by merging trajectory segments and correcting timestamps.
    - The calculated MSD data (total and per-component) is saved to a temperature-specific CSV file.
2.  Loads the newly created CSV data along with data from additional, pre-existing XCD files.
3.  Generates two styles of plots from the combined dataset:
    a)  **Style 1 (Individual Runs):** Each simulation run is plotted as a distinct colored line
        with markers, allowing for individual comparison.
    b)  **Style 2 (Average Trend):** All individual runs are plotted as thin, grey lines,
        with a prominent, thick red line representing the calculated average MSD. This style
        emphasizes the overall trend.
4.  All plots are styled using 'Times New Roman' font and formatted for publication quality.
5.  Finally, all generated plots are displayed on screen.
"""

import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import gc
from typing import List, Dict
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ==============================================================================
# --- 1. General Configuration ---
# ==============================================================================
TARGET_ION = "Li"
ROOT_FOLDER = r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\Tri_comb\Crystal"
INTERP_STEP = 1.0   # ps, for interpolation step in averaging

# Plotting style configurations (used as lookups)
Y_LIMITS_CONFIG = {
    700: {
        'total': {'ylim': (0, 10000), 'yticks': np.arange(0, 10001, 1800)},
        'xx component': {'ylim': (0, 3000), 'yticks': np.arange(0, 3001, 600), 'label': 'X[100]'},
        'yy component': {'ylim': (0, 3000), 'yticks': np.arange(0, 3001, 600), 'label': 'Y[010]'},
        'zz component': {'ylim': (0, 6000), 'yticks': np.arange(0, 6001, 1000), 'label': 'Z[001]'},
    },
    600: { # Assuming 600K might have smaller MSD values, adjust if necessary
        'total': {'ylim': (0, 8000), 'yticks': np.arange(0, 8001, 1600)},
        'xx component': {'ylim': (0, 2500), 'yticks': np.arange(0, 2501, 500), 'label': 'X[100]'},
        'yy component': {'ylim': (0, 2500), 'yticks': np.arange(0, 2501, 500), 'label': 'Y[010]'},
        'zz component': {'ylim': (0, 4000), 'yticks': np.arange(0, 4001, 800), 'label': 'Z[001]'},
    }
}
SETOFF_TIMES = {
    500: 47.5,
    600: 22.5,
    700: 27.0
}

# ==============================================================================
# --- 2. Temperature-Specific Configurations ---
# ==============================================================================
CONFIG_700K = {
    'TEMP': 700,
    'CUTOFF_TIME': 27.0,
    'RUN_FOLDERS': [
        "supercell_17_700K_run_1", "supercell_17_700K_run_4", "supercell_17_700K_run_5",
        "supercell_17_700K_run_6", "supercell_17_700K_run_7", "supercell_17_700K_run_10"
    ],
    'XCD_PATHS': [
        r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\PEO_RUN\crystal\10ns\700K\supercell_17_700K_run_2\PEO_Li_final_supercell Forcite MSD.xcd",
    ],
    'OUTPUT_CSV_PATH': os.path.join(ROOT_FOLDER, "MSD_from_27ps_700K_final.csv"),
}

CONFIG_600K = {
    'TEMP': 600,
    'CUTOFF_TIME': 22.5,
    'RUN_FOLDERS': [
        "supercell_17_600K_run_11", "supercell_17_600K_run_12", "supercell_17_600K_run_13",
        "supercell_17_600K_run_14", "supercell_17_600K_run_15"
    ],
    'XCD_PATHS': [
        r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\PEO_RUN\crystal\10ns\600K\supercell_17_600K_run_3\PEO_Li_final_supercell Forcite MSD.xcd",
        r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\PEO_RUN\crystal\10ns\600K\supercell_17_600K_run_7\PEO_Li_final_supercell Forcite MSD.xcd"
    ],
    'OUTPUT_CSV_PATH': os.path.join(ROOT_FOLDER, "MSD_from_22_5ps_600K.csv")
}


# ==============================================================================
# --- 3. Core Data Parsing and MSD Calculation ---
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

def process_single_sample_for_msd(folder_name: str, output_file: str, cutoff_time: float):
    """Processes a single sample, calculates MSD, and appends the result to the output file."""
    folder_path = os.path.join(ROOT_FOLDER, folder_name)
    print(f"\n[Processing] Sample: {folder_name}")

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

    for frame_idx in range(1, num_frames):
        box_dims = (np.array(lattice_truncated[frame_idx-1]) + np.array(lattice_truncated[frame_idx])) / 2
        for i, ion_idx in enumerate(target_indices):
            prev_coords = unwrapped_coords[i, frame_idx-1, :]
            curr_coords = np.array(list(map(float, frames_truncated[frame_idx]['atoms'][ion_idx].split()[1:4])))
            displacement = curr_coords - np.array(list(map(float, frames_truncated[frame_idx-1]['atoms'][ion_idx].split()[1:4])))
            correction = box_dims * np.round(displacement / box_dims)
            unwrapped_coords[i, frame_idx, :] = prev_coords + (displacement - correction)

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

def run_msd_calculation(config: Dict):
    """Orchestrates the MSD calculation for a given temperature configuration."""
    temp, run_folders, output_csv, cutoff_time = config['TEMP'], config['RUN_FOLDERS'], config['OUTPUT_CSV_PATH'], config['CUTOFF_TIME']
    print("=" * 70)
    print(f"Starting MSD Calculation for {temp}K ({len(run_folders)} samples)")
    print(f"Output CSV: {os.path.basename(output_csv)}")
    print("=" * 70)
    if os.path.exists(output_csv):
        os.remove(output_csv)
    for folder_name in run_folders:
        process_single_sample_for_msd(folder_name, output_csv, cutoff_time)
    print(f"\n[Complete] MSD calculation for {temp}K finished.")

# ==============================================================================
# --- 4. Plotting ---
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

def extract_msd_from_xcd(file_path, component=None):
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

def load_data_for_plotting(config: Dict):
    """Loads and combines data from the CSV and additional XCD files for plotting."""
    csv_path, xcd_paths, cutoff_time = config['OUTPUT_CSV_PATH'], config['XCD_PATHS'], config['CUTOFF_TIME']
    all_data = {k: [] for k in ['total', 'xx component', 'yy component', 'zz component']}
    try:
        df = pd.read_csv(csv_path)
        df['Relative_Time'] = df['Sample_Time (ps)'] - cutoff_time
        for name in sorted(df['Sample_Name'].unique()):
            group = df[df['Sample_Name'] == name]
            all_data['total'].append((group['Relative_Time'].values, group['Total_MSD (Å²)'].values))
            all_data['xx component'].append((group['Relative_Time'].values, group['X_MSD (Å²)'].values))
            all_data['yy component'].append((group['Relative_Time'].values, group['Y_MSD (Å²)'].values))
            all_data['zz component'].append((group['Relative_Time'].values, group['Z_MSD (Å²)'].values))
    except Exception as e:
        print(f"[Error] Failed to load CSV data for plotting: {e}")

    for path in xcd_paths:
        if not os.path.exists(path):
            print(f"[Warning] XCD file not found: {path}")
            continue
        for comp, msd_dict in [('total', extract_msd_from_xcd(path)),
                               ('xx component', extract_msd_from_xcd(path, 'xx component')),
                               ('yy component', extract_msd_from_xcd(path, 'yy component')),
                               ('zz component', extract_msd_from_xcd(path, 'zz component'))]:
            if msd_dict:
                times, msds = np.array(sorted(msd_dict.keys())), np.array([msd_dict[t] for t in sorted(msd_dict.keys())])
                mask = times >= cutoff_time
                if np.any(mask):
                    all_data[comp].append((times[mask] - cutoff_time, msds[mask]))
    return all_data

def plot_style_1_individual_runs(all_data, temperature):
    """Plots each run with a distinct color."""
    plot_style, total_runs = PlotProperties(), len(all_data['total'])
    colors = plt.get_cmap('plasma')(np.linspace(0, 0.8, total_runs))
    legend_title = f'{temperature} K Simulation\n' + r'$\mathrm{setoff}: ' + f'{SETOFF_TIMES[temperature]:.1f}' + r'\ \mathrm{ps}$'
    plot_configs = Y_LIMITS_CONFIG[temperature]

    for comp_key in plot_configs:
        plt_comp = plot_style.apply_style()
        for i, (x, y) in enumerate(all_data[comp_key]):
            if len(x) > 0 and len(y) > 0:
                plt_comp.plot(x, y, color=colors[i], linestyle='--', lw=2, alpha=0.8, marker=".", ms=10, label=f"Run {i+1}")
        config = plot_configs[comp_key]
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

    plot_configs = [('total', 'Total MSD'), ('xx component', 'X[100]'), ('yy component', 'Y[010]'), ('zz component', 'Z[001]')]
    total_ylim = None
    for comp_key, comp_label in plot_configs:
        plt_comp = style.apply_style()
        truncated_data = [(x[x <= global_time_max], y[x <= global_time_max]) for x, y in all_data[comp_key] if len(x) > 0]
        for x, y in truncated_data:
            plt_comp.plot(x, y, 'grey', linestyle='-', linewidth=2, alpha=0.3)

        unified_t = np.arange(0, global_time_max, INTERP_STEP)
        interpolated_msds = [interp1d(t, m, bounds_error=False, fill_value="extrapolate")(unified_t) for t, m in truncated_data if np.all(np.diff(t) > 0)]
        if interpolated_msds:
            plt_comp.plot(unified_t, np.mean(interpolated_msds, axis=0), 'red', linewidth=4, label="Average")

        plt_comp.xlim(0, global_time_max)
        plt_comp.xticks(np.linspace(0, global_time_max, 6))
        if comp_key == 'total':
            total_ylim = (0, max(y.max() for _, y in truncated_data) * 1.1 if truncated_data else 1)
        if total_ylim:
            plt_comp.ylim(total_ylim)
            plt.yticks(np.linspace(total_ylim[0], total_ylim[1], 6))

        plt_comp.legend(loc='upper left', framealpha=1, edgecolor='k', fontsize=style.legend_size)
        plt_comp.title(f"{comp_label} - {temperature}K", fontsize=26, fontweight='bold', y=1.03)

def run_plotting(config: Dict):
    """Orchestrates the plotting of MSD data for a given temperature."""
    temp = config['TEMP']
    print("\n" + "=" * 70)
    print(f"Starting Plotting for {temp}K")
    print("=" * 70)
    if not os.path.exists(config['OUTPUT_CSV_PATH']):
        print(f"[Error] Cannot plot. Data file not found: {config['OUTPUT_CSV_PATH']}")
        return
    all_plot_data = load_data_for_plotting(config)
    plot_style_1_individual_runs(all_plot_data, temp)
    plot_style_2_average_trend(all_plot_data, temp)
    print(f"[Complete] Plotting for {temp}K finished.")

# ==============================================================================
# --- 5. Main Execution ---
# ==============================================================================
def main():
    """Main function to run the entire analysis workflow."""
    all_configs_to_run = [CONFIG_700K, CONFIG_600K]

    for config in all_configs_to_run:
        run_msd_calculation(config)
        run_plotting(config)

    print("\n" + "="*70)
    print("All analyses complete. Displaying all generated plots...")
    print("="*70)
    plt.show()

if __name__ == "__main__":
    main()
