# -*- coding: utf-8 -*-
"""
A unified script for calculating and plotting Mean Squared Displacement (MSD) data
from molecular dynamics simulations.

This script performs the following steps:
1.  Calculates MSD from raw XYZ and XCD trajectory files for a list of specified simulation runs.
    - It handles interrupted simulations by merging trajectory segments and correcting timestamps.
    - The calculated MSD data (total and per-component) is saved to a single CSV file.
2.  Loads the newly created CSV data along with data from additional, pre-existing XCD files.
3.  Generates two styles of plots from the combined dataset:
    a)  **Style 1 (Individual Runs):** Each simulation run is plotted as a distinct colored line
        with markers, allowing for individual comparison.
    b)  **Style 2 (Average Trend):** All individual runs are plotted as thin, grey lines,
        with a prominent, thick red line representing the calculated average MSD. This style
        emphasizes the overall trend.
4.  All plots are styled using 'Times New Roman' font and formatted for publication quality.
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
# --- 1. Configuration ---
# ==============================================================================
TEMPERATURE = 700
CUTOFF_TIME = 27.0
TARGET_ION = "Li"
ROOT_FOLDER = r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\Tri_comb\Crystal"
RUN_FOLDERS = [
    "supercell_17_700K_run_1",
    "supercell_17_700K_run_4",
    "supercell_17_700K_run_5",
    "supercell_17_700K_run_6",
    "supercell_17_700K_run_7",
    "supercell_17_700K_run_10"
]
XCD_FILE_PATHS = [
    r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\PEO_RUN\crystal\10ns\700K\supercell_17_700K_run_2\PEO_Li_final_supercell Forcite MSD.xcd",
]
OUTPUT_CSV_PATH = os.path.join(ROOT_FOLDER, "MSD_from_27ps_final.csv")
OUTPUT_DIR_PLOTS = os.path.join(ROOT_FOLDER, "combined_plots_final_format")
INTERP_STEP = 1.0   # ps, for interpolation step in averaging

# Y-axis limits configuration for plotting
Y_LIMITS_CONFIG = {
    700: {
        'total': {'ylim': (0, 10000), 'yticks': np.arange(0, 10001, 1800)},
        'xx component': {'ylim': (0, 3000), 'yticks': np.arange(0, 3001, 600), 'label': 'X[100]'},
        'yy component': {'ylim': (0, 3000), 'yticks': np.arange(0, 3001, 600), 'label': 'Y[010]'},
        'zz component': {'ylim': (0, 6000), 'yticks': np.arange(0, 6001, 1000), 'label': 'Z[001]'},
    }
}
SETOFF_TIMES = {
    500: 47.5,
    600: 22.5,
    700: 27.0
}

# ==============================================================================
# --- 2. Core Data Parsing and MSD Calculation ---
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
            i += 1 # Move to the comment line
            comment_line = lines[i].strip()
            time_match = re.search(r'Time: (\d+\.?\d*)', comment_line)
            if not time_match:
                i += natoms + 1 # Skip this frame if no time is found
                continue
            frame_time = float(time_match.group(1))
            i += 1 # Move to atom coordinates
            atom_lines = []
            for _ in range(natoms):
                if i < len(lines):
                    atom_lines.append(lines[i].strip())
                i += 1
            frames.append({'time': frame_time, 'atoms': atom_lines})
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
                    for point in series.findall('POINT_2D'):
                        xy_str = point.get('XY', '')
                        if ',' in xy_str:
                            try:
                                length = float(xy_str.split(',')[1])
                                axis_data[axis].append(length)
                            except (ValueError, IndexError):
                                continue
        min_len = min(len(axis_data['A']), len(axis_data['B']), len(axis_data['C']))
        for i in range(min_len):
            lattice_params.append([axis_data['A'][i], axis_data['B'][i], axis_data['C'][i]])
    except Exception as e:
        print(f"[Error] Failed to parse XCD for lattice {os.path.basename(file_path)}: {e}")
    return lattice_params

def process_single_sample_for_msd(folder_name: str, output_file: str):
    """Processes a single sample, calculates MSD, and appends the result to the output file."""
    folder_path = os.path.join(ROOT_FOLDER, folder_name)
    print(f"\n[Processing] Sample: {folder_name}")

    # Define file paths
    traj1_path = os.path.join(folder_path, "supercell_17.xyz")
    traj2_path = os.path.join(folder_path, "3D Atomistic.xyz")
    xcd1_path = os.path.join(folder_path, "PEO_Li_final_supercell Forcite Cell parameters.xcd")
    xcd2_path = os.path.join(folder_path, "3D Atomistic Forcite Cell parameters.xcd")

    # Check for file existence
    missing_files = [f for f in [traj1_path, traj2_path, xcd1_path, xcd2_path] if not os.path.exists(f)]
    if missing_files:
        print(f"[Warning] Sample {folder_name} is missing files: {', '.join(map(os.path.basename, missing_files))}, skipping.")
        return

    # Parse trajectory and lattice files
    traj1_frames = parse_xyz_file(traj1_path)
    traj2_frames = parse_xyz_file(traj2_path)
    lattice1 = parse_xcd_file_for_lattice(xcd1_path)
    lattice2 = parse_xcd_file_for_lattice(xcd2_path)

    if not traj1_frames or not traj2_frames or len(traj1_frames) != len(lattice1) or len(traj2_frames) != len(lattice2):
        print(f"[Warning] Sample {folder_name} has inconsistent frame or lattice data, skipping.")
        return

    # Merge trajectories and correct timestamps
    t1_end = traj1_frames[-1]['time']
    for frame in traj2_frames:
        frame['time'] += t1_end

    total_frames = traj1_frames + traj2_frames
    total_lattice = lattice1 + lattice2

    # Truncate data after CUTOFF_TIME
    try:
        cutoff_index = next(i for i, frame in enumerate(total_frames) if frame['time'] >= CUTOFF_TIME)
    except StopIteration:
        print(f"[Warning] No data found after {CUTOFF_TIME} ps for {folder_name}, skipping.")
        return

    frames_truncated = total_frames[cutoff_index:]
    lattice_truncated = total_lattice[cutoff_index:]
    timestamps_truncated = [frame['time'] for frame in frames_truncated]

    print(f"[Info] Total trajectory time: {total_frames[-1]['time']:.2f} ps")
    print(f"[Info] Truncated to {len(frames_truncated)} data points from {timestamps_truncated[0]:.2f} ps to {timestamps_truncated[-1]:.2f} ps")

    # Calculate MSD
    target_indices = [i for i, atom in enumerate(frames_truncated[0]['atoms']) if atom.split()[0] == TARGET_ION]
    if not target_indices:
        print(f"[Warning] Target ion {TARGET_ION} not found in {folder_name}, skipping.")
        return

    num_ions = len(target_indices)
    num_frames = len(frames_truncated)

    # Unwrap coordinates to handle periodic boundary conditions
    unwrapped_coords = np.zeros((num_ions, num_frames, 3))
    for i, ion_idx in enumerate(target_indices):
        coords = np.array(list(map(float, frames_truncated[0]['atoms'][ion_idx].split()[1:4])))
        unwrapped_coords[i, 0, :] = coords

    for frame_idx in range(1, num_frames):
        prev_lattice = np.array(lattice_truncated[frame_idx-1])
        curr_lattice = np.array(lattice_truncated[frame_idx])
        box_dims = (prev_lattice + curr_lattice) / 2

        for i, ion_idx in enumerate(target_indices):
            prev_coords = np.array(list(map(float, frames_truncated[frame_idx-1]['atoms'][ion_idx].split()[1:4])))
            curr_coords = np.array(list(map(float, frames_truncated[frame_idx]['atoms'][ion_idx].split()[1:4])))
            displacement = curr_coords - prev_coords
            # Correct for periodic boundary crossing
            correction = box_dims * np.round(displacement / box_dims)
            unwrapped_coords[i, frame_idx, :] = unwrapped_coords[i, frame_idx-1, :] + (displacement - correction)

    displacements = unwrapped_coords - unwrapped_coords[:, 0:1, :]
    msd_total = np.mean(np.sum(displacements**2, axis=2), axis=0)
    msd_x = np.mean(displacements[:, :, 0]**2, axis=0)
    msd_y = np.mean(displacements[:, :, 1]**2, axis=0)
    msd_z = np.mean(displacements[:, :, 2]**2, axis=0)

    relative_time = np.array(timestamps_truncated) - timestamps_truncated[0]
    msd_df = pd.DataFrame({
        'Sample_Name': folder_name,
        'Sample_Time (ps)': timestamps_truncated,
        'Relative_Time (ps)': relative_time,
        'Total_MSD (Å²)': msd_total,
        'X_MSD (Å²)': msd_x,
        'Y_MSD (Å²)': msd_y,
        'Z_MSD (Å²)': msd_z
    })

    # Append to CSV
    msd_df.to_csv(
        output_file,
        mode='a',
        index=False,
        header=not os.path.exists(output_file),
        float_format='%.6f',
        encoding='utf-8-sig'
    )

    print(f"[Success] Sample {folder_name} processed and results saved.")

    # Explicitly free memory
    del traj1_frames, traj2_frames, total_frames, frames_truncated, unwrapped_coords, displacements, msd_df
    gc.collect()

def run_msd_calculation():
    """Main function to orchestrate the MSD calculation for all specified folders."""
    print("=" * 60)
    print(f"Starting MSD Calculation for {len(RUN_FOLDERS)} samples")
    print(f"Target Ion: {TARGET_ION} | Output CSV: {OUTPUT_CSV_PATH}")
    print("=" * 60)

    # Clean up previous results if any
    if os.path.exists(OUTPUT_CSV_PATH):
        os.remove(OUTPUT_CSV_PATH)

    for folder_name in RUN_FOLDERS:
        process_single_sample_for_msd(folder_name, OUTPUT_CSV_PATH)

    print(f"\n" + "=" * 60)
    print(f"MSD calculation complete!")
    print(f"Aggregated results saved to: {OUTPUT_CSV_PATH}")
    print("=" * 60)

# ==============================================================================
# --- 3. Plotting ---
# ==============================================================================
class PlotProperties:
    """A class to handle the styling of matplotlib plots for consistency."""
    def __init__(self, font_type="Times New Roman", font_size=26, axis_ticks_font_size=24,
                 label_x=r'Time / ps', label_y=r'MSD / $\AA^2$', fig_size=(8, 6), legend_size=20):
        self.font_type = font_type
        self.font_size = font_size
        self.axis_ticks_font_size = axis_ticks_font_size
        self.label_x = label_x
        self.label_y = label_y
        self.fig_size = fig_size
        self.legend_size = legend_size

    def apply_style(self):
        """Applies the defined style to the current pyplot figure."""
        plt.figure(figsize=self.fig_size)

        axis_font = {'fontname': self.font_type, 'size': self.font_size, 'weight': 'bold'}

        plt.rc('font', **{'family': self.font_type, 'size': self.legend_size, 'weight': 'bold'})

        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = self.font_type
        plt.rcParams['mathtext.it'] = f'{self.font_type}:italic'
        plt.rcParams['mathtext.bf'] = f'{self.font_type}:bold'

        plt.xlabel(self.label_x, **axis_font)
        plt.ylabel(self.label_y, **axis_font)

        ax = plt.gca()
        thickness = 2
        for spine in ax.spines.values():
            spine.set_linewidth(thickness)

        ax.tick_params(axis='both', direction='in', width=2, length=6, top=True, right=True, labelsize=self.axis_ticks_font_size)

        plt.tight_layout()
        return plt

def extract_msd_from_xcd(file_path, component=None):
    """Extracts MSD data series from an XCD file (for pre-computed MSD)."""
    msd_results = {}
    start_extracting = False
    target_line = 'Total MSD' if component is None else component
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                if target_line in line and 'SERIES_2D' in line:
                    start_extracting = True
                    continue
                if '</SERIES_2D>' in line and start_extracting:
                    break
                if start_extracting and '<POINT_2D XY="' in line:
                    data_str = line.split('"')[1]
                    time_str, msd_str = data_str.split(',')[:2]
                    try:
                        msd_results[float(time_str)] = float(msd_str)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[Error] Failed to extract {target_line} from {os.path.basename(file_path)}: {e}")
    return msd_results

def generate_colors(num_colors):
    """Generates a list of distinct colors from a colormap."""
    cmap = plt.get_cmap('plasma')
    return [cmap(i) for i in np.linspace(0, 0.8, num_colors)]

def load_data_for_plotting(csv_path, xcd_paths, cutoff_time):
    """Loads and combines data from the CSV and additional XCD files for plotting."""
    all_data = {'total': [], 'xx component': [], 'yy component': [], 'zz component': []}

    # Load from CSV
    try:
        df = pd.read_csv(csv_path)
        df_truncated = df[df['Sample_Time (ps)'] >= cutoff_time].copy()
        df_truncated['Relative_Time'] = df_truncated['Sample_Time (ps)'] - cutoff_time

        sample_names = sorted(df_truncated['Sample_Name'].unique())
        for name in sample_names:
            group = df_truncated[df_truncated['Sample_Name'] == name]
            x = group['Relative_Time'].values
            all_data['total'].append((x, group['Total_MSD (Å²)'].values))
            all_data['xx component'].append((x, group['X_MSD (Å²)'].values))
            all_data['yy component'].append((x, group['Y_MSD (Å²)'].values))
            all_data['zz component'].append((x, group['Z_MSD (Å²)'].values))
        print(f"[Info] Loaded {len(sample_names)} samples from {os.path.basename(csv_path)}")
    except Exception as e:
        print(f"[Error] Failed to load CSV data for plotting: {e}")

    # Load from XCD
    for path in xcd_paths:
        if not os.path.exists(path):
            print(f"[Warning] XCD file for plotting not found: {path}")
            continue

        total_msd = extract_msd_from_xcd(path)
        xx_msd = extract_msd_from_xcd(path, 'xx component')
        yy_msd = extract_msd_from_xcd(path, 'yy component')
        zz_msd = extract_msd_from_xcd(path, 'zz component')

        for comp, msd_dict in [('total', total_msd), ('xx component', xx_msd),
                               ('yy component', yy_msd), ('zz component', zz_msd)]:
            if msd_dict:
                times = np.array(sorted(msd_dict.keys()))
                msds = np.array([msd_dict[t] for t in times])
                mask = times >= cutoff_time
                if np.any(mask):
                    rel_times = times[mask] - cutoff_time
                    all_data[comp].append((rel_times, msds[mask]))
        print(f"[Info] Loaded data from XCD: {os.path.basename(path)}")

    return all_data

def plot_style_1_individual_runs(all_data, temperature):
    """Plots each run with a distinct color."""
    plot_style = PlotProperties()
    cutoff_time = SETOFF_TIMES[temperature]
    total_runs = len(all_data['total'])
    colors = generate_colors(total_runs)
    legend_labels = [f"Run {i+1}" for i in range(total_runs)]

    setoff_text = r'$\mathrm{setoff}: ' + f'{cutoff_time:.1f}' + r'\ \mathrm{ps}$'
    legend_title = f'{temperature} K Simulation\n' + setoff_text

    plot_configs = Y_LIMITS_CONFIG[temperature]

    for comp_key in ['total', 'xx component', 'yy component', 'zz component']:
        plt_comp = plot_style.apply_style()
        for i, (x, y) in enumerate(all_data[comp_key]):
            if len(x) > 0 and len(y) > 0:
                plt_comp.plot(x, y, color=colors[i], linestyle='--', linewidth=2, alpha=0.8,
                              marker=".", markersize=10, label=legend_labels[i])

        config = plot_configs[comp_key]
        plt_comp.ylim(config['ylim'])
        plt_comp.yticks(config['yticks'])
        plt_comp.legend(title=legend_title, loc='upper left', ncol=2, framealpha=1, edgecolor='k', fontsize=12, title_fontsize=14)

        title = f"Total MSD – {temperature} K" if comp_key == 'total' else f"{config['label']} Direction – {temperature} K"
        plt_comp.title(title, fontsize=plot_style.font_size, fontweight='bold', y=1.03)
        plt_comp.tight_layout()

def calculate_mean_by_time(data_list, dt, max_time):
    """Calculates the mean of multiple time series using interpolation."""
    if not data_list: return np.array([]), np.array([])

    valid_data = [(t, m) for t, m in data_list if len(t) > 1 and len(m) > 1]
    if not valid_data: return np.array([]), np.array([])

    unified_t = np.arange(0, max_time, dt)
    interpolated_msds = []

    for t, m in valid_data:
        if not np.all(np.diff(t) > 0): continue # Skip if time is not monotonic
        f = interp1d(t, m, kind='linear', bounds_error=False, fill_value="extrapolate")
        interpolated_msds.append(f(unified_t))

    return (unified_t, np.mean(interpolated_msds, axis=0)) if interpolated_msds else (np.array([]), np.array([]))

def plot_style_2_average_trend(all_data, temperature):
    """Plots all runs in grey and their average in red."""
    style = PlotProperties(fig_size=(12, 8))

    # Determine the global time axis limit (shortest of all runs)
    all_max_times = [np.max(t) for t, _ in all_data['total'] if len(t) > 0]
    if not all_max_times:
        print("[Warning] No data available to plot for average trend style.")
        return
    global_time_max = min(all_max_times)

    plot_configs = [('total', 'Total MSD'), ('xx component', 'X[100]'),
                    ('yy component', 'Y[010]'), ('zz component', 'Z[001]')]

    total_ylim = None
    for comp_key, comp_label in plot_configs:
        plt_comp = style.apply_style()

        # Truncate data to global max time for averaging
        truncated_data_for_avg = []
        for x, y in all_data[comp_key]:
            if len(x) > 0:
                mask = x <= global_time_max
                x_trunc, y_trunc = x[mask], y[mask]
                if len(x_trunc) > 0:
                    truncated_data_for_avg.append((x_trunc, y_trunc))
                    plt_comp.plot(x_trunc, y_trunc, 'grey', linestyle='-', linewidth=2, alpha=0.3)

        avg_time, avg_msd = calculate_mean_by_time(truncated_data_for_avg, INTERP_STEP, global_time_max)
        if len(avg_time) > 0:
            plt_comp.plot(avg_time, avg_msd, 'red', linewidth=4, label="Average")

        plt_comp.xlim(0, global_time_max)
        plt_comp.xticks(np.linspace(0, global_time_max, 6))

        # Unify Y-axis for all components based on total MSD
        if comp_key == 'total':
            all_msds = [item for _, y_list in truncated_data_for_avg for item in y_list]
            total_ylim = (0, max(all_msds) * 1.1) if all_msds else (0, 1)

        if total_ylim:
            plt_comp.ylim(total_ylim)
            plt_comp.yticks(np.linspace(total_ylim[0], total_ylim[1], 6))

        plt_comp.legend(loc='upper left', framealpha=1, edgecolor='k', fontsize=style.legend_size)
        plt_comp.title(f"{comp_label} - {temperature}K", fontsize=26, fontweight='bold', y=1.03)

def run_plotting():
    """Main function to orchestrate the plotting of MSD data."""
    print("\n" + "=" * 60)
    print("Starting Plotting")
    print("=" * 60)

    if not os.path.exists(OUTPUT_CSV_PATH):
        print(f"[Error] Cannot start plotting. Required data file not found: {OUTPUT_CSV_PATH}")
        return

    # Load all data for plotting
    all_plot_data = load_data_for_plotting(OUTPUT_CSV_PATH, XCD_FILE_PATHS, CUTOFF_TIME)

    # Generate plots
    print("[Info] Generating Plot Style 1: Individual Runs...")
    plot_style_1_individual_runs(all_plot_data, TEMPERATURE)

    print("[Info] Generating Plot Style 2: Average Trend...")
    plot_style_2_average_trend(all_plot_data, TEMPERATURE)

    print("\nAll plots generated. Displaying now...")
    plt.show()

# ==============================================================================
# --- 4. Main Execution ---
# ==============================================================================
if __name__ == "__main__":
    # Step 1: Calculate MSD from raw trajectory data and save to CSV
    run_msd_calculation()

    # Step 2: Load the generated CSV and other data to create plots
    run_plotting()
