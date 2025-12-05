# -*- coding: utf-8 -*-
"""
unified_md_analyzer.py

A comprehensive script to process and analyze molecular dynamics (MD) simulation data for different temperatures.

This script provides functionalities to:
1.  Handle both interrupted (stitched from multiple files) and complete (single file) simulation runs.
2.  Process data for multiple temperatures (e.g., 500K, 600K, 700K).
3.  Calculate the Mean Squared Displacement (MSD) for all runs and save the results into separate CSV files for each temperature.
4.  Generate and display publication-quality plots showing the individual MSD curves and their average for each temperature and component.
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import gc

# ==============================================================================
# --- 1. Global Configuration ---
# ==============================================================================

TARGET_ION = "Li"
ROOT_FOLDER = r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\Tri_comb\Crystal"

# --- Analysis & Plotting Configuration ---
INTERP_STEP = 1.0  # ps, for averaging with interpolation
PLOT_STYLE = {
    'font_type': "Times New Roman",
    'font_size': 26,
    'axis_ticks_font_size': 26,
    'label_x': "Time (ps)",
    'label_y': r"MSD ($\AA^2$)",
    'fig_size': (12, 8),
    'legend_size': 15
}

# --- Temperature-Specific Configurations ---
TEMPERATURE_CONFIGS = {
    "700K": {
        'TEMP': 700,
        'CUTOFF_TIME': 27.0,
        'RUNS': {
            "run_1": ["supercell_17.xyz", "3D Atomistic.xyz"],
            "run_2": ["PEO_Li_final_supercell Forcite MSD.xcd"],
            "run_4": ["supercell_17.xyz", "3D Atomistic.xyz"],
            "run_5": ["supercell_17.xyz", "3D Atomistic.xyz"],
            "run_6": ["supercell_17.xyz", "3D Atomistic.xyz"],
            "run_7": ["supercell_17.xyz", "3D Atomistic.xyz"],
            "run_9": ["PEO_Li_final_supercell Forcite MSD.xcd"],
        },
        'BASE_PATH': {
            "run_1": os.path.join(ROOT_FOLDER, "supercell_17_700K_run_1"),
            "run_2": r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\PEO_RUN\crystal\10ns\700K\supercell_17_700K_run_2",
            "run_4": os.path.join(ROOT_FOLDER, "supercell_17_700K_run_4"),
            "run_5": os.path.join(ROOT_FOLDER, "supercell_17_700K_run_5"),
            "run_6": os.path.join(ROOT_FOLDER, "supercell_17_700K_run_6"),
            "run_7": os.path.join(ROOT_FOLDER, "supercell_17_700K_run_7"),
            "run_9": r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\PEO_RUN\crystal\10ns\700K\supercell_17_700K_run_9",
        },
        'OUTPUT_CSV': os.path.join(ROOT_FOLDER, "MSD_700K_unified.csv")
    },
    "600K": {
        'TEMP': 600,
        'CUTOFF_TIME': 22.5,
        'RUNS': {
            f"run_{i}": ["supercell_17.xyz", "3D Atomistic.xyz"] for i in range(11, 16)
        },
        'BASE_PATH': {
            f"run_{i}": os.path.join(ROOT_FOLDER, f"supercell_17_600K_run_{i}") for i in range(11, 16)
        },
        'OUTPUT_CSV': os.path.join(ROOT_FOLDER, "MSD_600K_unified.csv")
    }
}


# ==============================================================================
# --- 2. Core Parsing Functions ---
# ==============================================================================

def parse_xyz_file(file_path: str) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Parses an XYZ file, returning frames and their corresponding times."""
    frames, times = [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            try:
                natoms = int(lines[i].strip())
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
                times.append(frame_time)
                i += natoms
            except (ValueError, IndexError):
                i += 1
                continue
    except Exception as e:
        print(f"[Error] Parsing XYZ file {os.path.basename(file_path)} failed: {e}")
    return frames, times


def parse_xcd_file(file_path: str) -> List[List[float]]:
    """Parses an XCD file to extract lattice parameters."""
    lattice_params = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        axis_data = {axis: [] for axis in ['A', 'B', 'C']}
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
        min_len = min(len(d) for d in axis_data.values())
        for i in range(min_len):
            lattice_params.append([axis_data['A'][i], axis_data['B'][i], axis_data['C'][i]])
    except Exception as e:
        print(f"[Error] Parsing XCD file {os.path.basename(file_path)} failed: {e}")
    return lattice_params


def extract_msd_from_xcd(file_path: str) -> Dict[str, pd.Series]:
    """Extracts pre-calculated MSD data from an XCD file."""
    msd_data = {}
    component_map = {
        'Total MSD': 'Total_MSD (Å²)',
        'xx component': 'X_MSD (Å²)',
        'yy component': 'Y_MSD (Å²)',
        'zz component': 'Z_MSD (Å²)'
    }
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for series in root.findall('.//SERIES_2D'):
            series_name = series.get('Name', '')
            for key, col_name in component_map.items():
                if key in series_name:
                    times, msds = [], []
                    for point in series.findall('POINT_2D'):
                        xy_str = point.get('XY', '')
                        if ',' in xy_str:
                            try:
                                t, m = map(float, xy_str.split(','))
                                times.append(t)
                                msds.append(m)
                            except (ValueError, IndexError):
                                continue
                    if times:
                        msd_data[col_name] = pd.Series(msds, index=times)
    except Exception as e:
        print(f"[Error] Extracting MSD from {os.path.basename(file_path)} failed: {e}")
    return msd_data


# ==============================================================================
# --- 3. MSD Calculation Logic ---
# ==============================================================================

def process_run(run_name: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Processes a single run, handling both stitched XYZ and pre-calculated XCD."""
    print(f"\n[Processing] T={config['TEMP']}K, Run: {run_name}")
    base_path = config['BASE_PATH'][run_name]
    file_list = config['RUNS'][run_name]
    cutoff_time = config['CUTOFF_TIME']

    # --- Path 1: Process pre-calculated MSD from a single XCD file ---
    if len(file_list) == 1 and file_list[0].endswith('.xcd'):
        file_path = os.path.join(base_path, file_list[0])
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}. Skipping.")
            return pd.DataFrame()

        msd_components = extract_msd_from_xcd(file_path)
        if not msd_components:
            return pd.DataFrame()

        df = pd.DataFrame(msd_components).sort_index()
        df = df[df.index >= cutoff_time]
        if df.empty:
            print(f"[Warning] No data after cutoff time {cutoff_time}ps for {file_list[0]}")
            return pd.DataFrame()

        df.index.name = 'Absolute_Time (ps)'
        df.reset_index(inplace=True)
        df['Relative_Time (ps)'] = df['Absolute_Time (ps)'] - df['Absolute_Time (ps)'].iloc[0]
        df['Run_Name'] = run_name
        print(f"[Success] Finished processing run {run_name} from XCD.")
        return df

    # --- Path 2: Calculate MSD from stitched XYZ files ---
    all_frames, all_lattice = [], []
    last_timestamp = 0.0

    for file_name in file_list:
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}. Skipping.")
            continue

        if file_name.endswith('.xyz'):
            frames, _ = parse_xyz_file(file_path)
            xcd_equivalent = "PEO_Li_final_supercell Forcite Cell parameters.xcd" if "supercell_17" in file_name else "3D Atomistic Forcite Cell parameters.xcd"
            lattice = parse_xcd_file(os.path.join(base_path, xcd_equivalent))

            if len(frames) != len(lattice):
                print(f"[Warning] Mismatch frames ({len(frames)}) vs lattice ({len(lattice)}) for {file_name}. Skipping.")
                continue

            for frame in frames:
                frame['time'] += last_timestamp
            all_frames.extend(frames)
            all_lattice.extend(lattice)
            if frames:
                last_timestamp = all_frames[-1]['time']

    if not all_frames:
        print(f"[Error] No valid XYZ data for run {run_name}.")
        return pd.DataFrame()

    start_index = next((i for i, f in enumerate(all_frames) if f['time'] >= cutoff_time), -1)
    if start_index == -1:
        return pd.DataFrame()

    frames_trunc, lattice_trunc = all_frames[start_index:], all_lattice[start_index:]
    times_trunc = [f['time'] for f in frames_trunc]

    target_indices = [i for i, atom in enumerate(frames_trunc[0]['atoms']) if atom.split()[0] == TARGET_ION]
    if not target_indices:
        return pd.DataFrame()

    num_ions, num_frames = len(target_indices), len(frames_trunc)
    unwrapped = np.zeros((num_ions, num_frames, 3))

    for i, ion_idx in enumerate(target_indices):
        unwrapped[i, 0, :] = list(map(float, frames_trunc[0]['atoms'][ion_idx].split()[1:4]))

    for frame_idx in range(1, num_frames):
        box_dims = (np.array(lattice_trunc[frame_idx - 1]) + np.array(lattice_trunc[frame_idx])) / 2
        for i, ion_idx in enumerate(target_indices):
            prev_coords = unwrapped[i, frame_idx - 1, :]
            curr_coords_raw = list(map(float, frames_trunc[frame_idx]['atoms'][ion_idx].split()[1:4]))
            prev_coords_raw = list(map(float, frames_trunc[frame_idx-1]['atoms'][ion_idx].split()[1:4]))
            displacement = np.array(curr_coords_raw) - np.array(prev_coords_raw)
            correction = box_dims * np.round(displacement / box_dims)
            unwrapped[i, frame_idx, :] = prev_coords + (displacement - correction)

    displacements = unwrapped - unwrapped[:, 0:1, :]
    df = pd.DataFrame({
        'Run_Name': run_name,
        'Absolute_Time (ps)': times_trunc,
        'Relative_Time (ps)': np.array(times_trunc) - times_trunc[0],
        'Total_MSD (Å²)': np.mean(np.sum(displacements ** 2, axis=2), axis=0),
        'X_MSD (Å²)': np.mean(displacements[:, :, 0] ** 2, axis=0),
        'Y_MSD (Å²)': np.mean(displacements[:, :, 1] ** 2, axis=0),
        'Z_MSD (Å²)': np.mean(displacements[:, :, 2] ** 2, axis=0)
    })
    print(f"[Success] Finished processing run {run_name} from XYZ.")
    return df


# ==============================================================================
# --- 4. Plotting Logic ---
# ==============================================================================

class Plotter:
    def __init__(self, style_config):
        self.style = style_config

    def apply_style(self):
        plt.figure(figsize=self.style['fig_size'])
        axis_font = {'fontname': self.style['font_type'], 'size': self.style['font_size'], 'weight': 'bold'}
        plt.rc('font', family=self.style['font_type'], weight='bold')
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = self.style['font_type']
        plt.xlabel(self.style['label_x'], **axis_font)
        plt.ylabel(self.style['label_y'], **axis_font)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(axis='both', direction='in', width=2, length=6, top=True, right=True,
                       labelsize=self.style['axis_ticks_font_size'])
        plt.tight_layout()
        return plt

    def plot_msd(self, df: pd.DataFrame, temp: int):
        runs = df['Run_Name'].unique()
        max_times = [df[df['Run_Name'] == run]['Relative_Time (ps)'].max() for run in runs]
        global_max_time = min(max_times) if max_times else 0

        plots_def = [
            ('Total_MSD (Å²)', 'Total MSD'), ('X_MSD (Å²)', 'X[100]'),
            ('Y_MSD (Å²)', 'Y[010]'), ('Z_MSD (Å²)', 'Z[001]')
        ]

        for msd_col, title_prefix in plots_def:
            p = self.apply_style()
            all_msds_truncated = []

            for run_name in runs:
                run_data = df[df['Run_Name'] == run_name]
                mask = run_data['Relative_Time (ps)'] <= global_max_time
                x, y = run_data['Relative_Time (ps)'][mask], run_data[msd_col][mask]
                if not x.empty:
                    p.plot(x, y, 'grey', linestyle='-', linewidth=2, alpha=0.5)
                    all_msds_truncated.append((x.values, y.values))

            if all_msds_truncated:
                unified_t = np.arange(0, global_max_time, INTERP_STEP)
                interpolated_msds = []
                for t, m in all_msds_truncated:
                    if len(t) > 1:
                        f = interp1d(t, m, kind='linear', bounds_error=False, fill_value="extrapolate")
                        interpolated_msds.append(f(unified_t))

                if interpolated_msds:
                    avg_msd = np.mean(interpolated_msds, axis=0)
                    p.plot(unified_t, avg_msd, 'red', linewidth=4, label='Average')

            p.xlim(0, global_max_time)
            p.title(f"{title_prefix} - {temp}K", fontsize=self.style['font_size'], fontweight='bold', y=1.03)
            p.legend(loc='upper left', framealpha=1, edgecolor='k', fontsize=self.style['legend_size'])
            p.tight_layout()

    def plot_individual_runs_colored(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Generates plots where each individual run is shown in a distinct color."""
        temp = config['TEMP']
        cutoff_time = config['CUTOFF_TIME']
        runs = sorted(df['Run_Name'].unique())
        num_runs = len(runs)

        # Generate distinct colors for each run
        cmap = plt.get_cmap('plasma')
        colors = [cmap(i) for i in np.linspace(0, 0.8, num_runs)]

        plots_def = [
            ('Total_MSD (Å²)', 'Total MSD'), ('X_MSD (Å²)', 'X[100]'),
            ('Y_MSD (Å²)', 'Y[010]'), ('Z_MSD (Å²)', 'Z[001]')
        ]

        legend_title = f'{temp} K Simulation\n' + r'$\mathrm{setoff}: ' + f'{cutoff_time:.1f}' + r'\ \mathrm{ps}$'

        for msd_col, title_prefix in plots_def:
            p = self.apply_style()
            for i, run_name in enumerate(runs):
                run_data = df[df['Run_Name'] == run_name]
                if not run_data.empty:
                    p.plot(
                        run_data['Relative_Time (ps)'], run_data[msd_col],
                        color=colors[i],
                        linestyle='--',
                        linewidth=2,
                        marker=".",
                        markersize=10,
                        alpha=0.8,
                        label=f"Run {i+1}" # Simple numeric labels
                    )

            p.legend(title=legend_title, loc='upper left', ncol=2, framealpha=1, edgecolor='k', fontsize=12, title_fontsize=14)
            p.title(f"{title_prefix} – {temp} K", fontsize=self.style['font_size'], fontweight='bold', y=1.03)
            p.tight_layout()


# ==============================================================================
# --- 5. Main Execution ---
# ==============================================================================

def main():
    """Main function to run the MSD analysis for all configured temperatures."""
    plotter = Plotter(PLOT_STYLE)

    for temp_key, config in TEMPERATURE_CONFIGS.items():
        print("=" * 70)
        print(f"--- Starting Analysis for {temp_key} ---")
        print("=" * 70)

        all_run_dfs = []
        for run_name in config['RUNS'].keys():
            df = process_run(run_name, config)
            if not df.empty:
                all_run_dfs.append(df)
            gc.collect()

        if not all_run_dfs:
            print(f"\n[Error] No data could be processed for {temp_key}. Skipping.")
            continue

        final_df = pd.concat(all_run_dfs, ignore_index=True)
        final_df.to_csv(config['OUTPUT_CSV'], index=False, float_format='%.6f', encoding='utf-8-sig')
        print(f"\n[Success] All runs for {temp_key} processed and saved to:\n{config['OUTPUT_CSV']}")

        # Plotting
        print(f"\n[Plotting] Generating MSD plots for {temp_key}...")
        print(" -> Style 1: Individual runs in color...")
        plotter.plot_individual_runs_colored(final_df, config)
        print(" -> Style 2: Averaged MSD (grey lines with red average)...")
        plotter.plot_msd(final_df, config['TEMP'])

    print("\n" + "=" * 70)
    print("--- Analysis Complete ---")
    print("=" * 70)
    plt.show()


if __name__ == "__main__":
    main()