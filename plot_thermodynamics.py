import os
import numpy as np
import matplotlib.pyplot as plt
from monty.json import MSONable

# -------------------------- 数据提取函数 --------------------------
def temperature_xcd(folder_path):
    temperature_results = {}
    start_extracting_temperature = False
    try:
        # 模拟数据
        x_data = np.linspace(0, 1000, 100)
        y_data = 298.5 + 0.5 * np.sin(x_data / 50) + np.random.normal(0, 0.2, 100)
        return dict(zip(x_data, y_data))
    except Exception as e:
        print(f"温度文件读取错误：{e}")
    return temperature_results

def density_xcd(folder_path):
    density_results = {}
    try:
        # 模拟数据
        x_data = np.linspace(0, 1000, 100)
        stabilized_density = 1.272
        initial_density_offset = 0.02
        y_data = stabilized_density * (1 - initial_density_offset * np.exp(-x_data / 50)) + np.random.normal(0, 0.0002, 100)
        return dict(zip(x_data, y_data))
    except Exception as e:
        print(f"密度文件读取错误：{e}")
    return density_results

def energy_xcd(folder_path):
    total = {}
    try:
        # 模拟数据
        x_data = np.linspace(0, 1000, 100)
        y_data_tot = 5 * np.sin(x_data / 100) + np.random.normal(0, 8, 100)
        return {}, {}, dict(zip(x_data, y_data_tot))
    except Exception as e:
        print(f"能量文件读取错误：{e}")
    return {}, {}, total

# -------------------------- SCI 论文单图样式类 --------------------------
class plot_properties():
    """
    配置 SCI 论文标准的绘图样式，适配子图布局。
    """
    def __init__(self, font_type='Times New Roman', font_size=26, axis_ticks_font_size=24, label_x="Time (ps)", label_y="",
                 legend_size=24):

        self.font_size = font_size
        self.axis_ticks_font_size = axis_ticks_font_size
        self.font_type = font_type

        self.label_x = label_x
        self.label_y = label_y
        self.legend_size = legend_size

    def apply_style(self, ax, is_bottom=False):
        """
        对给定的 Axes 对象应用样式。is_bottom 为 True 时，才显示 X 轴标签和刻度。
        """
        # 1. 全局字体设置
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': [self.font_type],
            'font.size': self.legend_size,
            'font.weight': 'bold',
            'axes.unicode_minus': False
        })

        # 2. Mathtext 设置
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = self.font_type
        plt.rcParams['mathtext.it'] = self.font_type + ':italic'
        plt.rcParams['mathtext.bf'] = self.font_type + ':bold'

        # 3. 设置轴标签
        axis_font = {'fontname': self.font_type, 'size': self.font_size, 'fontweight': 'bold'}

        if is_bottom:
            ax.set_xlabel(self.label_x, **axis_font)
        else:
            ax.set_xlabel('')

        ax.set_ylabel(self.label_y, **axis_font, labelpad=15)

        # 4. 刻度标签字体设置
        tick_size = self.axis_ticks_font_size
        tick_font = self.font_type

        for tick in ax.get_xticklabels():
            tick.set_fontname(tick_font)
            tick.set_fontsize(tick_size)
            tick.set_fontweight('bold')
        for tick in ax.get_yticklabels():
            tick.set_fontname(tick_font)
            tick.set_fontsize(tick_size)
            tick.set_fontweight('bold')

        # 5. 边框和刻度线样式
        thickness = 3
        ax.spines['top'].set_linewidth(thickness)
        ax.spines['right'].set_linewidth(thickness)
        ax.spines['left'].set_linewidth(thickness)
        ax.spines['bottom'].set_linewidth(thickness)

        ax.get_yaxis().set_tick_params(direction='in', width=thickness, length=8, right=True, top=True)
        ax.get_xaxis().set_tick_params(direction='in', width=thickness, length=8, right=True, top=True)

        if not is_bottom:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0, width=0)

        ax.set_title('')

        return ax

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    # --- 文件路径配置 ---
    file_paths = {
        'temperature': 'supercell_17 Temperature.xcd',
        'density': 'supercell_17 Density.xcd',
        'energy': 'supercell_17 Energies.xcd'
    }

    # --- 数据提取 ---
    temp_data = temperature_xcd(file_paths['temperature'])
    den_data = density_xcd(file_paths['density'])
    _, _, tot_data = energy_xcd(file_paths['energy'])

    # --- 数据整理 ---
    def format_data(data_dict):
        if not data_dict:
            return np.array([0]), np.array([0])
        x = np.array(sorted(data_dict.keys()))
        y = np.array([data_dict[k] for k in x])
        return x, y

    t_time, t_vals = format_data(temp_data)
    d_time, d_vals = format_data(den_data)
    tot_time, tot_vals = format_data(tot_data)

    # --- 组合图配置 ---
    plot_configs = [
        {
            'y_label': 'Temperature (K)',
            'data_x': t_time, 'data_y': t_vals, 'color': '#D62728',
        },
        {
            'y_label': r'Total Energy ($\text{kcal/mol}$)',
            'data_x': tot_time, 'data_y': tot_vals, 'color': '#8C564B',
        },
        {
            'y_label': r'Density ($\text{g/cm}^3$)',
            'data_x': d_time, 'data_y': d_vals, 'color': '#1F77B4',
        }
    ]

    # --- 绘制组合图 ---
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
    plt.subplots_adjust(hspace=0.05)

    max_time = 1000

    for i, config in enumerate(plot_configs):
        ax = axes[i]
        is_bottom = (i == len(plot_configs) - 1)

        style = plot_properties(label_y=config['y_label'])
        ax = style.apply_style(ax, is_bottom=is_bottom)

        ax.plot(config['data_x'], config['data_y'], color=config['color'],
                 linewidth=3.0, alpha=0.9)

        ax.set_xlim(0, max_time)
        if is_bottom:
             ax.set_xticks(np.linspace(0, max_time, 6))

        if 'Temperature' in config['y_label']:
             y_min_margin = 0
             y_max_margin = 400
             ax.set_yticks(np.linspace(y_min_margin, y_max_margin, 5))
        elif 'Total Energy' in config['y_label']:
             y_min_margin = 0
             y_max_margin = 20000
             ax.set_yticks(np.linspace(y_min_margin, y_max_margin, 5))
        elif 'Density' in config['y_label']:
             y_min_margin = 1.21
             y_max_margin = 1.27
             ax.set_yticks(np.array([1.21, 1.23, 1.25, 1.27]))

        ax.set_ylim(y_min_margin, y_max_margin)

    plt.tight_layout(pad=0.5)
    plt.savefig("thermodynamic_equilibrium_plot.png")
    print("--- 组合热力学平衡图已生成并保存为 thermodynamic_equilibrium_plot.png ---")
