import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from io import StringIO
from monty.json import MSONable

# --- Unified Data Store ---
CRYSTAL_DIFFUSION_DATA = {
    300: {'X': 3.94e-11, 'Y': 8.93e-12, 'Z': 1.23e-09},
    500: {'X': 1.22e-07, 'Y': 8.07e-08, 'Z': 6.80e-07},
    600: {'X': 5.75e-07, 'Y': 3.75e-07, 'Z': 1.36e-06},
    700: {'X': 3.81e-06, 'Y': 3.98e-06, 'Z': 1.01e-05}
}

DATA_STRING = """System	State	T / K	D / cm²·s⁻¹
Total MSD	Crystal	300	7.74E-10
X MSD	Crystal	300	3.94E-11
Y MSD	Crystal	300	8.93E-12
Z MSD	Crystal	300	1.23E-09
Total MSD	Crystal	500	8.82E-07
X MSD	Crystal	500	1.22E-07
Y MSD	Crystal	500	8.07E-08
Z MSD	Crystal	500	6.80E-07
Total MSD	Crystal	600	2.31E-06
X MSD	Crystal	600	5.75E-07
Y MSD	Crystal	600	3.75E-07
Z MSD	Crystal	600	1.36E-06
Total MSD	Crystal	700	1.79E-05
X MSD	Crystal	700	3.81E-06
Y MSD	Crystal	700	3.98E-06
Z MSD	Crystal	700	1.01E-05
Total MSD	Amorphous	300	4.87E-07
X MSD	Amorphous	300	2.06E-07
Y MSD	Amorphous	300	1.29E-07
Z MSD	Amorphous	300	1.53E-07
Total MSD	Amorphous	500	9.08E-06
X MSD	Amorphous	500	2.59E-06
Y MSD	Amorphous	500	2.85E-06
Z MSD	Amorphous	500	3.63E-06
Total MSD	Amorphous	600	2.75E-05
X MSD	Amorphous	600	9.63E-06
Y MSD	Amorphous	600	8.20E-06
Z MSD	Amorphous	600	9.72E-06
Total MSD	Amorphous	700	4.17E-05
X MSD	Amorphous	700	1.25E-05
Y MSD	Amorphous	700	1.32E-05
Z MSD	Amorphous	700	1.62E-05
"""

# --- Plotting Class ---
class PlotProperties(MSONable):
    def __init__(self, font_type='Times New Roman', font_size='26', axis_ticks_font_size='24',
                 label_x="", label_y="", legend_size='20', xlimit=8, ylimit=6):
        self.font_size = font_size
        self.axis_ticks_font_size = axis_ticks_font_size
        self.font_type = font_type
        self.label_x = label_x
        self.label_y = label_y
        self.legend_size = legend_size
        self.xlimit = xlimit
        self.ylimit = ylimit

    def get_plot_style(self):
        plt.figure(figsize=(self.xlimit, self.ylimit))
        axis_font = {'fontname': self.font_type, 'size': self.font_size, 'fontweight': 'bold'}
        axis_ticks_font = {'fontname': self.font_type, 'size': self.axis_ticks_font_size, 'fontweight': 'bold'}
        plt.rc('font', **{'family': [self.font_type], 'size': self.legend_size, 'weight': 'bold'})
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = self.font_type
        plt.rcParams['mathtext.it'] = self.font_type + ':italic'
        plt.rcParams['mathtext.bf'] = self.font_type + ':bold'
        plt.ylabel(self.label_y, **axis_font)
        plt.xlabel(self.label_x, **axis_font)
        plt.xticks(**axis_ticks_font)
        plt.yticks(**axis_ticks_font)
        ax = plt.gca()
        thickness = 2
        ax.spines['top'].set_linewidth(thickness)
        ax.spines['right'].set_linewidth(thickness)
        ax.spines['left'].set_linewidth(thickness)
        ax.spines['bottom'].set_linewidth(thickness)
        ax.get_yaxis().set_tick_params(direction='in', width=2, length=6, right='on')
        ax.get_xaxis().set_tick_params(direction='in', width=2, length=6, top='on')
        plt.tight_layout()
        return plt

# --- Analysis and Plotting Functions ---

def analyze_and_plot_crystal_anisotropy(diff_data):
    def normalize_by_x_with_original(diff_data):
        results = []
        for temp in sorted(diff_data.keys()):
            X = diff_data[temp]['X']
            Y = diff_data[temp]['Y']
            Z = diff_data[temp]['Z']
            D_X = f"{X:.4e}"
            D_Y = f"{Y:.4e}"
            D_Z = f"{Z:.4e}"
            norm_X = 1.0
            norm_Y = round(Y / X, 4)
            norm_Z = round(Z / X, 4)
            results.append({
                'Temperature (K)': temp,
                'D_X (cm²/s)': D_X,
                'D_Y (cm²/s)': D_Y,
                'D_Z (cm²/s)': D_Z,
                'Norm_X (X/X)': norm_X,
                'Norm_Y (Y/X)': norm_Y,
                'Norm_Z (Z/X)': norm_Z
            })
        return pd.DataFrame(results)

    crystal_combined_df = normalize_by_x_with_original(diff_data)

    def extract_heatmap_data(combined_df):
        temps = combined_df['Temperature (K)'].values
        norm_data = combined_df[['Norm_X (X/X)', 'Norm_Y (Y/X)', 'Norm_Z (Z/X)']].values
        anisotropy_strength = [round(max(row) - min(row), 4) for row in norm_data]
        return temps, norm_data, anisotropy_strength

    temps, norm_data, anisotropy_strength = extract_heatmap_data(crystal_combined_df)

    def plot_normalized_heatmap(temps, norm_data):
        plt.figure(figsize=(10, 6))
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 20
        plt.rcParams["axes.linewidth"] = 2
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        bold_font = FontProperties(weight='bold', size=20)
        directions = ['X', 'Y', 'Z']
        im = plt.imshow(norm_data, cmap='RdYlBu', aspect='auto', vmin=0, vmax=32)
        plt.xticks(range(len(directions)), directions, fontproperties=bold_font)
        plt.yticks(range(len(temps)), [f'{t}K' for t in temps], fontproperties=bold_font)
        plt.title('Normalized Diffusion Coefficients (by X)', fontsize=20, fontweight='bold', pad=15)
        for i in range(len(temps)):
            for j in range(len(directions)):
                plt.text(j, i, f'{norm_data[i, j]:.4f}', ha='center', va='center', color='white', fontweight='bold', fontsize=20)
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('Normalized Value (X=1)', fontsize=20, fontweight='bold')
        cbar.ax.tick_params(direction='in', width=2, length=6)
        plt.tight_layout()
        plt.savefig("crystal_normalized_heatmap.png", dpi=300)
        plt.close()

    def plot_anisotropy_trend(temps, anisotropy_strength):
        plt.figure(figsize=(10, 4))
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 18
        plt.rcParams["axes.linewidth"] = 2
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        bold_font = FontProperties(weight='bold', size=18)
        plt.plot(temps, anisotropy_strength, 'o-', color='darkred', linewidth=3, markersize=16, label='Anisotropy Gap (max - min)')
        plt.xlabel('Temperature (K)', fontsize=18, fontweight='bold')
        plt.ylabel('Anisotropy Gap', fontsize=18, fontweight='bold')
        plt.xticks(temps)
        plt.tick_params(direction='in', width=2, length=6, top=True, right=True)
        plt.legend(prop=bold_font, frameon=False, loc='upper right')
        plt.tight_layout()
        plt.savefig("crystal_anisotropy_trend.png", dpi=300)
        plt.close()

    print("="*140)
    print("                         晶态体系扩散系数（原始值+X归一化值）表（含300K外推）")
    print("="*140)
    print(crystal_combined_df.to_string(index=False))
    print("="*140)
    print("\n说明：")
    print("1. D_X/D_Y/D_Z：各方向原始扩散系数（单位：cm²/s），300K为外推值；")
    print("2. 归一化规则：以X为基准（Norm_X=1），Norm_Y=Y/X，Norm_Z=Z/X；")
    print("3. 趋势：随温度升高，各向异性强度变化可从趋势线直观观察。\n")

    plot_normalized_heatmap(temps, norm_data)
    plot_anisotropy_trend(temps, anisotropy_strength)
    print("Crystal anisotropy plots saved.")

def plot_diffusivity_comparison(df):
    plot_total = PlotProperties(label_x='Temperature (K)', label_y='Diffusivity (cm²·s⁻¹)', xlimit=10, ylimit=6)
    plt_total = plot_total.get_plot_style()
    total_data = df[df['System'] == 'Total MSD']
    for state in ['Crystal', 'Amorphous']:
        state_data = total_data[total_data['State'] == state].sort_values('T / K')
        plt_total.plot(state_data['T / K'], state_data['D / cm²·s⁻¹'], marker='o', markersize=12, linewidth=4, label=state)
    plt_total.title('Total Diffusivity vs Temperature', fontsize=26, fontname='Times New Roman', fontweight='bold')
    plt_total.legend(fontsize=plot_total.legend_size)
    plt_total.yscale('log')
    plt_total.tight_layout()
    plt_total.savefig('total_diffusivity_vs_temp.png', dpi=300)
    plt_total.close()

    plot_components = PlotProperties(label_x='Temperature (K)', label_y='Diffusivity (cm²·s⁻¹)', xlimit=12, ylimit=7)
    plt_components = plot_components.get_plot_style()
    directions = {'X MSD': ('#1f77b4', 'o'), 'Y MSD': ('#2ca02c', 'o'), 'Z MSD': ('#ff7f0e', 'o')}
    for direction, (color, marker) in directions.items():
        dir_data = df[df['System'] == direction]
        for state in ['Crystal', 'Amorphous']:
            state_dir_data = dir_data[dir_data['State'] == state].sort_values('T / K')
            if not state_dir_data.empty:
                marker_face = color if state == 'Crystal' else 'white'
                marker_edge = 'none' if state == 'Crystal' else color
                plt_components.plot(state_dir_data['T / K'], state_dir_data['D / cm²·s⁻¹'], color=color, marker=marker, linestyle='-', markersize=15, linewidth=5, markerfacecolor=marker_face, markeredgecolor=marker_edge, markeredgewidth=2, label=f'{direction.replace(" MSD", "")} - {state}')
    plt_components.title('Diffusivity Components vs Temperature', fontsize=26, fontname='Times New Roman', fontweight='bold')
    plt_components.legend(fontsize=18, loc='lower right', frameon=True, edgecolor='black')
    plt_components.yscale('log')
    plt_components.tight_layout()
    plt_components.savefig('diffusivity_components_vs_temp.png', dpi=300)
    plt_components.close()
    print("Diffusivity comparison plots saved.")

def plot_crystal_diffusion_vs_temp(data):
    temperatures = list(data.keys())
    diffusion_data = {
        'Total': [sum(data[t].values()) for t in temperatures],
        'X': [data[t]['X'] for t in temperatures],
        'Y': [data[t]['Y'] for t in temperatures],
        'Z': [data[t]['Z'] for t in temperatures]
    }
    plot_prop = PlotProperties(label_x='Temperature / K', label_y='$D$ / cm²·s⁻¹', xlimit=10, ylimit=8)
    plt_crystal = plot_prop.get_plot_style()
    markers = ['o', 's', '^', 'D']
    colors = ['blue', 'red', 'green', 'purple']
    line_styles = ['-', '-', '-', '-']
    for i, (direction, d_values) in enumerate(diffusion_data.items()):
        plt_crystal.semilogy(temperatures, d_values, marker=markers[i], color=colors[i], linestyle=line_styles[i], linewidth=2, markersize=10, label=direction)
    plt_crystal.legend(loc='best')
    plt_crystal.title('Diffusion Coefficient vs Temperature (Crystal State)', fontname='Times New Roman', fontsize=24, fontweight='bold')
    plt_crystal.savefig('crystal_diffusion_vs_temp_combined.png', dpi=300)
    plt_crystal.close()
    print("Crystal diffusion vs temp plot saved.")

def plot_diffusion_anisotropy(data_crystal, data_amorphous):
    T1, D1_x, D1_y, D1_z = data_crystal
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    ax1.set_xlabel('Temperature (K)', fontsize=24)
    ax1.set_ylabel('Diffusion Coefficient (cm²·s⁻¹)', fontsize=24)
    ax1.plot(T1, D1_x, 'o-', color='blue', markersize=8, linewidth=2, markerfacecolor='none', markeredgewidth=2, label='D(X)')
    ax1.plot(T1, D1_y, 's-', color='green', markersize=8, linewidth=2, markerfacecolor='none', markeredgewidth=2, label='D(Y)')
    ax1.plot(T1, D1_z, '^-', color='red', markersize=8, linewidth=2, markerfacecolor='none', markeredgewidth=2, label='D(Z)')
    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-12, 1e-5)
    ax1.legend(loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig("crystal_diffusion_vs_temp_log_detailed.png", dpi=300)
    plt.close(fig1)

    D1_avg = (D1_x + D1_y + D1_z) / 3
    D1_x_norm = D1_x / D1_avg
    D1_y_norm = D1_y / D1_avg
    D1_z_norm = D1_z / D1_avg
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.set_xlabel('Temperature (K)', fontsize=24)
    ax2.set_ylabel('Normalized D (Anisotropy)', fontsize=24)
    ax2.plot(T1, D1_x_norm, 'd-', color='blue', markersize=8, linewidth=2, markerfacecolor='none', markeredgewidth=2, label='X (Anisotropy)')
    ax2.plot(T1, D1_y_norm, 'd-', color='green', markersize=8, linewidth=2, markerfacecolor='none', markeredgewidth=2, label='Y (Anisotropy)')
    ax2.plot(T1, D1_z_norm, 'd-', color='red', markersize=8, linewidth=2, markerfacecolor='none', markeredgewidth=2, label='Z (Anisotropy)')
    ax2.set_ylim(0, 4)
    ax2.yaxis.set_major_locator(MultipleLocator(1))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax2.tick_params(axis='x', labelsize=24)
    ax2.tick_params(axis='y', labelsize=24)
    ax2.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig("crystal_anisotropy_detailed.png", dpi=300)
    plt.close(fig2)

    T2, D2_x, D2_y, D2_z = data_amorphous
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    ax3.set_xlabel('Temperature (K)', fontsize=24)
    ax3.set_ylabel('Diffusion Coefficient (cm²·s⁻¹)', fontsize=24)
    ax3.plot(T2, D2_x, 'o-', color='blue', markersize=8, linewidth=2, markerfacecolor='none', markeredgewidth=2, label='D(X)')
    ax3.plot(T2, D2_y, 's-', color='green', markersize=8, linewidth=2, markerfacecolor='none', markeredgewidth=2, label='D(Y)')
    ax3.plot(T2, D2_z, '^-', color='red', markersize=8, linewidth=2, markerfacecolor='none', markeredgewidth=2, label='D(Z)')
    ax3.tick_params(axis='x', labelsize=24)
    ax3.tick_params(axis='y', labelsize=24)
    ax3.set_yscale('log')
    ax3.set_ylim(1e-08, 2e-05)
    ax3.legend(loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig("amorphous_diffusion_vs_temp_log_detailed.png", dpi=300)
    plt.close(fig3)

    D2_avg = (D2_x + D2_y + D2_z) / 3
    D2_x_norm = D2_x / D2_avg
    D2_y_norm = D2_y / D2_avg
    D2_z_norm = D2_z / D2_avg
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    ax4.set_xlabel('Temperature (K)', fontsize=24)
    ax4.set_ylabel('Normalized D (Anisotropy)', fontsize=24)
    ax4.plot(T2, D2_x_norm, 'd-', color='blue', markersize=8, linewidth=2, markerfacecolor='none', markeredgewidth=2, label='X (Anisotropy)')
    ax4.plot(T2, D2_y_norm, 'd-', color='green', markersize=8, linewidth=2, markerfacecolor='none', markeredgewidth=2, label='Y (Anisotropy)')
    ax4.plot(T2, D2_z_norm, 'd-', color='red', markersize=8, linewidth=2, markerfacecolor='none', markeredgewidth=2, label='Z (Anisotropy)')
    ax4.set_ylim(0.5, 2)
    ax4.yaxis.set_major_locator(MultipleLocator(0.5))
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax4.tick_params(axis='x', labelsize=24)
    ax4.tick_params(axis='y', labelsize=24)
    ax4.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig("amorphous_anisotropy_detailed.png", dpi=300)
    plt.close(fig4)
    print("Detailed diffusion and anisotropy plots saved.")

# --- Main Execution Block ---
if __name__ == "__main__":
    df_comparison = pd.read_csv(StringIO(DATA_STRING), sep='\t')
    df_comparison['System'] = df_comparison['System'].str.strip()

    analyze_and_plot_crystal_anisotropy(CRYSTAL_DIFFUSION_DATA)
    plot_diffusivity_comparison(df_comparison)
    plot_crystal_diffusion_vs_temp(CRYSTAL_DIFFUSION_DATA)

    T_crystal = np.array(sorted(CRYSTAL_DIFFUSION_DATA.keys()))
    D_crystal_x = np.array([CRYSTAL_DIFFUSION_DATA[t]['X'] for t in T_crystal])
    D_crystal_y = np.array([CRYSTAL_DIFFUSION_DATA[t]['Y'] for t in T_crystal])
    D_crystal_z = np.array([CRYSTAL_DIFFUSION_DATA[t]['Z'] for t in T_crystal])

    amorphous_data = df_comparison[df_comparison['State'] == 'Amorphous']
    T_amorphous = sorted(amorphous_data['T / K'].unique())

    D_amorphous_x = amorphous_data[amorphous_data['System'] == 'X MSD'].sort_values('T / K')['D / cm²·s⁻¹'].values
    D_amorphous_y = amorphous_data[amorphous_data['System'] == 'Y MSD'].sort_values('T / K')['D / cm²·s⁻¹'].values
    D_amorphous_z = amorphous_data[amorphous_data['System'] == 'Z MSD'].sort_values('T / K')['D / cm²·s⁻¹'].values

    crystal_data_for_plot = (T_crystal, D_crystal_x, D_crystal_y, D_crystal_z)
    amorphous_data_for_plot = (T_amorphous, D_amorphous_x, D_amorphous_y, D_amorphous_z)

    plot_diffusion_anisotropy(crystal_data_for_plot, amorphous_data_for_plot)

    print("\nAll analysis complete and all plots have been saved to PNG files.")
