import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 数据提取函数 (with Simulated Data) --------------------------
def temperature_xcd(folder_path):
    """Generates simulated temperature data fluctuating around 300 K."""
    time = np.linspace(0, 1000, 500)
    temp = 300 + np.random.randn(500) * 2.5 + np.sin(time / 100) * 1.5
    return dict(zip(time, temp))

def density_xcd(folder_path):
    """Generates simulated density data that equilibrates around 1.26 g/cm^3."""
    time = np.linspace(0, 1000, 500)
    equil_density = 1.26
    density = equil_density - 0.04 * np.exp(-time / 100) + np.random.randn(500) * 0.002
    return dict(zip(time, density))

def energy_xcd(folder_path):
    """Generates simulated total energy data equilibrating around 10000 kcal/mol."""
    time = np.linspace(0, 1000, 500)
    equil_energy = 10000
    energy = equil_energy - 500 * np.exp(-time / 150) + np.random.randn(500) * 150
    return {}, {}, dict(zip(time, energy))

# -------------------------- SCI 论文单图样式类 --------------------------
class plot_properties():
    def __init__(self, font_type='Times New Roman', font_size=24, axis_ticks_font_size=20, label_x="Time (ps)", label_y="",
                 legend_size=24):
        self.font_size = font_size
        self.axis_ticks_font_size = axis_ticks_font_size
        self.font_type = font_type
        self.label_x = label_x
        self.label_y = label_y
        self.legend_size = legend_size

    def apply_style(self, ax, is_bottom=False):
        plt.rcParams.update({
            'font.family': 'serif', 'font.serif': [self.font_type],
            'font.size': self.legend_size, 'font.weight': 'bold',
            'axes.unicode_minus': False
        })
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = self.font_type
        plt.rcParams['mathtext.it'] = self.font_type + ':italic'
        plt.rcParams['mathtext.bf'] = self.font_type + ':bold'

        axis_font = {'fontname': self.font_type, 'size': self.font_size, 'fontweight': 'bold'}

        if is_bottom:
            ax.set_xlabel(self.label_x, **axis_font)
        else:
            ax.set_xlabel('')

        ax.set_ylabel(self.label_y, **axis_font, labelpad=15)

        for tick in ax.get_xticklabels():
            tick.set_fontname(self.font_type)
            tick.set_fontsize(self.axis_ticks_font_size)
            tick.set_fontweight('bold')
        for tick in ax.get_yticklabels():
            tick.set_fontname(self.font_type)
            tick.set_fontsize(self.axis_ticks_font_size)
            tick.set_fontweight('bold')

        thickness = 3
        for spine in ax.spines.values():
            spine.set_linewidth(thickness)

        ax.tick_params(direction='in', width=thickness, length=8, right=True, top=True)

        if not is_bottom:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0, width=0)

        ax.set_title('')
        return ax

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    # File paths are placeholders, will use simulated data
    file_paths = {
        'temperature': 'path/to/Temperature.xcd',
        'density': 'path/to/Density.xcd',
        'energy': 'path/to/Energies.xcd'
    }

    temp_data = temperature_xcd(file_paths['temperature'])
    den_data = density_xcd(file_paths['density'])
    _, _, tot_data = energy_xcd(file_paths['energy'])

    def format_data(data_dict):
        if not data_dict:
            return np.array([0]), np.array([0])
        x = np.array(sorted(data_dict.keys()))
        y = np.array([data_dict[k] for k in x])
        return x, y

    t_time, t_vals = format_data(temp_data)
    d_time, d_vals = format_data(den_data)
    tot_time, tot_vals = format_data(tot_data)

    plot_configs = [
        { 'y_label': 'Temperature (K)', 'data_x': t_time, 'data_y': t_vals, 'color': '#D62728' },
        { 'y_label': r'Total Energy (kcal/mol)', 'data_x': tot_time, 'data_y': tot_vals, 'color': '#8C564B' },
        { 'y_label': r'Density (g/cm$^3$)', 'data_x': d_time, 'data_y': d_vals, 'color': '#1F77B4' }
    ]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    max_time = 1000

    for i, config in enumerate(plot_configs):
        ax = axes[i]
        is_bottom = (i == len(plot_configs) - 1)

        style = plot_properties(label_y=config['y_label'])
        style.apply_style(ax, is_bottom=is_bottom)

        ax.plot(config['data_x'], config['data_y'], color=config['color'], linewidth=2.5)

        ax.set_xlim(0, max_time)
        if is_bottom:
             ax.set_xticks(np.linspace(0, max_time, 6))

        # Set Y-axis limits and ticks as specified by the user
        if 'Temperature' in config['y_label']:
            y_min_margin = 0
            y_max_margin = 400
            ax.set_ylim(y_min_margin, y_max_margin)
            ax.set_yticks(np.linspace(y_min_margin, y_max_margin, 5))
        elif 'Total Energy' in config['y_label']:
            y_min_margin = 0
            y_max_margin = 20000
            ax.set_ylim(y_min_margin, y_max_margin)
            ax.set_yticks(np.linspace(y_min_margin, y_max_margin, 5))
        elif 'Density' in config['y_label']:
            y_min_margin = 1.21
            y_max_margin = 1.27
            ax.set_ylim(y_min_margin, y_max_margin)
            ax.set_yticks(np.array([1.21, 1.23, 1.25, 1.27]))


    plt.tight_layout(pad=0.5)
    plt.savefig("combined_thermo_plot.png")
    print("Plot generated and saved to combined_thermo_plot.png")
