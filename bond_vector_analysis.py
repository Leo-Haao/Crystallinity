#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from matplotlib.colors import LogNorm
import os

# Set global plotting parameters
# All text elements will use 'Times New Roman' and be bold.
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'cm'


# --- Cell 1: Data Acquisition and Bond Vector Calculation ---
def calculate_bond_vectors(structure, center_element='C', neighbor_element='C', cutoff=3.0):
    """
    Calculates bond vectors (PBC-folded Cartesian vectors from center to neighbor).

    Args:
    - structure: pymatgen Structure object
    - center_element: str, symbol of the central element
    - neighbor_element: str, symbol of the neighbor element
    - cutoff: float, cutoff distance (Å)

    Returns:
    - all_bond_vecs: numpy array (M, 3), all calculated bond vectors
    """
    # Extract all atomic coordinates and elements
    cart_coords = np.array([site.coords for site in structure.sites])
    elements = np.array([site.specie.symbol for site in structure.sites])

    # Filter indices for center and neighbor atoms
    center_indices = np.where(elements == center_element)[0]
    neighbor_indices = np.where(elements == neighbor_element)[0]

    if len(center_indices) == 0 or len(neighbor_indices) == 0:
        print(f"No atoms found for {center_element}-{neighbor_element} pair.")
        return np.array([])

    lattice_matrix = structure.lattice.matrix
    inv_lattice_matrix = np.linalg.inv(lattice_matrix)

    all_bond_vecs = []

    for i in center_indices:
        diff_cart = cart_coords[neighbor_indices] - cart_coords[i]
        frac_diff = np.dot(diff_cart, inv_lattice_matrix)
        frac_diff_folded = frac_diff - np.round(frac_diff)
        pbc_diff_cart = np.dot(frac_diff_folded, lattice_matrix)

        distances = np.linalg.norm(pbc_diff_cart, axis=1)

        # Exclude self-interaction (if i is in neighbor_indices)
        mask = (distances > 1e-6) & (distances <= cutoff)
        bond_vecs = pbc_diff_cart[mask]

        all_bond_vecs.extend(bond_vecs)

    return np.array(all_bond_vecs)


# --- Cell 2: Plotting Class and Functions ---
class plot_properties:
    def __init__(self, font_type='Times New Roman', font_size=40, axis_ticks_font_size=36,
                 legend_size=20):
        self.font_size = font_size
        self.axis_ticks_font_size = axis_ticks_font_size
        self.font_type = font_type
        self.legend_size = legend_size

    def apply_style_to_ax(self, ax, label_x, label_y):
        """Applies bold 'Times New Roman' styling to the plot axes."""
        axis_font = {'fontname': self.font_type, 'size': self.font_size, 'fontweight': 'bold'}

        # Configure mathtext to use Times New Roman
        plt.rcParams['mathtext.rm'] = self.font_type
        plt.rcParams['mathtext.it'] = self.font_type + ':italic'
        plt.rcParams['mathtext.bf'] = self.font_type + ':bold'

        ax.set_ylabel(label_y, **axis_font)
        ax.set_xlabel(label_x, **axis_font)
        ax.tick_params(labelsize=self.axis_ticks_font_size)

        # Explicitly set tick labels to bold
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')

        thickness = 2
        ax.spines['top'].set_linewidth(thickness)
        ax.spines['right'].set_linewidth(thickness)
        ax.spines['left'].set_linewidth(thickness)
        ax.spines['bottom'].set_linewidth(thickness)
        ax.tick_params(direction='in', width=2, length=6, top=True, right=True)


def draw_projection(x_data, y_data, label_x, label_y, title, xlim=(-4, 4), ylim=(-4, 4)):
    """Draws a single 2D projection scatter plot."""
    print(f"INFO: Generating plot - {title}")
    fig, ax = plt.subplots(figsize=(8, 8))

    style = plot_properties()
    style.apply_style_to_ax(ax, label_x=label_x, label_y=label_y)

    ax.scatter(x_data, y_data, s=5, alpha=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()
    plt.close()


def compute_spherical_coords(all_bond_vecs, cutoff=3.0, dr=0.1, theta_bins=36, phi_bins=72):
    """
    Computes spherical coordinates, density statistics, and applies corrections.
    Returns a 3D histogram and related coordinate arrays.
    """
    r = np.linalg.norm(all_bond_vecs, axis=1)
    mask = r > 1e-6  # Filter out zero-length vectors
    if not np.any(mask):
        print("Warning: All bond vectors have zero length. Cannot compute spherical coordinates.")
        return (None,) * 10

    r = r[mask]
    all_bond_vecs = all_bond_vecs[mask]

    theta = np.arccos(all_bond_vecs[:, 2] / r) * 180 / np.pi
    phi = np.arctan2(all_bond_vecs[:, 1], all_bond_vecs[:, 0]) * 180 / np.pi
    phi = (phi + 360) % 360

    r_bins = np.arange(0, cutoff + dr, dr)
    theta_bins_arr = np.linspace(0, 180, theta_bins + 1)
    phi_bins_arr = np.linspace(0, 360, phi_bins + 1)

    hist, _ = np.histogramdd((r, theta, phi), bins=(r_bins, theta_bins_arr, phi_bins_arr))

    dtheta = np.diff(theta_bins_arr)[0] * np.pi / 180
    dphi = np.diff(phi_bins_arr)[0] * np.pi / 180

    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    theta_centers_rad = (theta_bins_arr[:-1] + theta_bins_arr[1:]) / 2 * np.pi / 180

    sin_theta = np.sin(theta_centers_rad)[np.newaxis, :, np.newaxis]
    correction = r_centers[:, np.newaxis, np.newaxis] ** 2 * dr * sin_theta * dtheta * dphi

    hist_corrected = np.divide(hist, correction, where=correction != 0, out=np.zeros_like(hist))

    theta_centers_deg = theta_centers_rad * 180 / np.pi
    phi_centers = (phi_bins_arr[:-1] + phi_bins_arr[1:]) / 2

    return hist_corrected, r_centers, theta_centers_deg, phi_centers, r_bins, theta_bins_arr, phi_bins_arr, dtheta, dphi, theta_centers_rad


def draw_heatmap(hist_2d, x_centers, y_centers, label_x, label_y, title):
    """Generic heatmap drawing function for r vs. θ or θ vs. φ."""
    print(f"INFO: Generating plot - {title}")
    fig, ax = plt.subplots(figsize=(10, 8))
    style = plot_properties()
    style.apply_style_to_ax(ax, label_x=label_x, label_y=label_y)

    if not np.isfinite(hist_2d).all():
        print(f"Warning: Non-finite values detected in heatmap data. Skipping plot: {title}")
        plt.close()
        return

    im = ax.imshow(hist_2d, origin='lower', extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
                   aspect='auto', cmap='viridis', norm=LogNorm() if np.max(hist_2d) > 100 else None)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density', weight='bold', size=36)
    cbar.ax.tick_params(labelsize=36)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')

    plt.tight_layout()
    plt.show()
    plt.close()


def draw_density_histogram(hist_corrected, r_centers, theta_bins_arr, phi_bins_arr, dtheta, dphi, theta_centers_rad, title):
    """Plots a histogram of density (ρ) vs. volume fraction."""
    print(f"INFO: Generating plot - {title}")
    dr = r_centers[1] - r_centers[0] if len(r_centers) > 1 else 0.1
    sin_theta = np.sin(theta_centers_rad)[np.newaxis, :, np.newaxis]
    num_phi = len(phi_bins_arr) - 1

    v_box = r_centers[:, np.newaxis, np.newaxis] ** 2 * dr * sin_theta * dtheta * dphi * np.ones((len(r_centers), len(theta_centers_rad), num_phi))
    v_total = np.sum(v_box)

    rho_array = hist_corrected.flatten()
    v_box_array = v_box.flatten()

    mask = np.isfinite(rho_array) & (rho_array > 0)
    if not np.any(mask):
        print(f"Warning: No valid density data found. Skipping plot: {title}")
        return

    rho_array = rho_array[mask]
    v_box_array = v_box_array[mask]

    fig, ax = plt.subplots(figsize=(8, 8))
    style = plot_properties()
    style.apply_style_to_ax(ax, label_x='Density (ρ)', label_y='Volume Fraction')

    bins = np.linspace(np.min(rho_array), np.max(rho_array), 30)
    hist, bin_edges = np.histogram(rho_array, bins=bins, weights=v_box_array / v_total)
    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 0.05)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_all_results(all_bond_vecs, label=''):
    """Main plotting function to create all visualizations for a given structure."""
    if len(all_bond_vecs) == 0:
        print("No bond vectors to plot.")
        return

    # Titles for projection plots
    proj_title_xy = f'C-C Bond Vector (XY plane) for {label}'
    proj_title_xz = f'C-C Bond Vector (XZ plane) for {label}'
    proj_title_yz = f'C-C Bond Vector (YZ plane) for {label}'

    # Generate 2D projection plots
    draw_projection(all_bond_vecs[:, 0], all_bond_vecs[:, 1], 'Δx (Å)', 'Δy (Å)', proj_title_xy)
    draw_projection(all_bond_vecs[:, 0], all_bond_vecs[:, 2], 'Δx (Å)', 'Δz (Å)', proj_title_xz)
    draw_projection(all_bond_vecs[:, 1], all_bond_vecs[:, 2], 'Δy (Å)', 'Δz (Å)', proj_title_yz)
    print("2D projections generated.")

    # Calculate spherical coordinates and density
    results = compute_spherical_coords(all_bond_vecs, cutoff=3.0, dr=0.1, theta_bins=36, phi_bins=72)
    hist_corrected, r_centers, theta_centers, phi_centers, _, theta_bins_arr, phi_bins_arr, dtheta, dphi, theta_centers_rad = results

    if hist_corrected is None:
        return

    # Draw r vs. θ heatmap (averaged over φ)
    hist_2d_r_theta = np.mean(hist_corrected, axis=2)
    heatmap_r_theta_title = f'C-C Bond Vector Heatmap (r vs θ) for {label}'
    draw_heatmap(hist_2d_r_theta.T, r_centers, theta_centers, 'r (Å)', 'θ (°)', heatmap_r_theta_title)

    # Find the radius 'r' with the highest total density
    r_densities = np.sum(hist_corrected, axis=(1, 2))
    max_r_idx = np.argmax(r_densities)
    if r_densities[max_r_idx] == 0:
        print(f"Warning: No valid density peak found for {label}. Skipping θ vs φ heatmap.")
        return

    max_r = r_centers[max_r_idx]
    hist_2d_theta_phi = hist_corrected[max_r_idx, :, :]

    # Draw θ vs. φ heatmap for the radius of maximum density
    heatmap_theta_phi_title = f'C-C Bond Vector Heatmap (θ vs φ at r={max_r:.2f} Å) for {label}'
    draw_heatmap(hist_2d_theta_phi, phi_centers, theta_centers, 'φ (°)', 'θ (°)', heatmap_theta_phi_title)

    # Draw the density distribution histogram
    density_hist_title = f'Density Distribution for {label}'
    draw_density_histogram(hist_corrected, r_centers, theta_bins_arr, phi_bins_arr, dtheta, dphi, theta_centers_rad, density_hist_title)

    print("Heatmap and density histogram generated.")


# --- Main Execution Block ---
if __name__ == '__main__':
    # Define the base path for the data files. Note the raw string `r''` to handle backslashes.
    base_path = r'E:\理论计算\MD works\2025.PEO.Li.MD_work\09.bond_vector\Graphit'

    # Define the files to be analyzed with descriptive labels
    file_info = {
        'Crystalline Graphite (Supercell 4x4x1)': os.path.join(base_path, 'graphit_supercell_4x4x1.cif'),
        'Amorphous Graphite (7100K)': os.path.join(base_path, 'graphit_7100K.cif')
    }

    # Loop through each file, generate and display plots
    for label, file_path in file_info.items():
        print(f"--- Processing: {label} ---")
        print(f"File path: {file_path}")
        try:
            if os.path.exists(file_path):
                structure = Structure.from_file(file_path)
                all_bond_vecs = calculate_bond_vectors(structure, center_element='C', neighbor_element='C', cutoff=3.0)
                print(f"Calculated {len(all_bond_vecs)} bond vectors.")

                plot_all_results(all_bond_vecs, label=label)
            else:
                print(f"Error: File not found at '{file_path}'. Please check the path.")

        except Exception as e:
            print(f"An error occurred while processing the file: {e}")
            print("Please check if the file path is correct and the CIF file is not corrupted.")
        print("-" * 50)
