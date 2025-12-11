import os
import numpy as np
import matplotlib.pyplot as plt
from monty.json import MSONable

# -------------------------- 数据提取函数（保持不变） --------------------------
# 注意：在实际运行环境中，请确保以下文件路径是正确的
def temperature_xcd(folder_path):
    temperature_results = {}
    start_extracting_temperature = False
    try:
        if not os.path.exists(folder_path):
            print(f"温度文件不存在: {folder_path}")
            return temperature_results

        with open(folder_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                if 'Temperature' in line:
                    start_extracting_temperature = True
                    continue
                if '</SERIES_2D>' in line and start_extracting_temperature:
                    start_extracting_temperature = False
                    continue
                if start_extracting_temperature and '<POINT_2D XY="' in line:
                    data_part = line.split('"')[1]
                    time_x, temperature_data = data_part.split(',')
                    x = float(time_x)
                    temperature = float(temperature_data)
                    temperature_results[x] = temperature
    except Exception as e:
        print(f"温度文件读取错误：{e}")
    return temperature_results

def cell_xcd(folder_path):
    length_A_results, length_B_results, length_C_results = {}, {}, {}
    alpha_results, beta_results, gamma_results = {}, {}, {}
    start_extracting = {k: False for k in ['A', 'B', 'C', 'alpha', 'beta', 'gamma']}

    try:
        if not os.path.exists(folder_path):
            print(f"晶胞文件不存在: {folder_path}")
            return {}

        with open(folder_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                if 'Length A' in line: start_extracting = {k: False for k in start_extracting}; start_extracting['A'] = True
                elif 'Length B' in line: start_extracting = {k: False for k in start_extracting}; start_extracting['B'] = True
                elif 'Length C' in line: start_extracting = {k: False for k in start_extracting}; start_extracting['C'] = True
                elif '"alpha"' in line: start_extracting = {k: False for k in start_extracting}; start_extracting['alpha'] = True
                elif 'beta' in line: start_extracting = {k: False for k in start_extracting}; start_extracting['beta'] = True
                elif 'gamma' in line: start_extracting = {k: False for k in start_extracting}; start_extracting['gamma'] = True
                elif '</SERIES_2D>' in line: start_extracting = {k: False for k in start_extracting}; continue

                if '<POINT_2D XY="' in line:
                    data_part = line.split('"')[1]
                    x, val = map(float, data_part.split(','))
                    if start_extracting['A']: length_A_results[x] = val
                    elif start_extracting['B']: length_B_results[x] = val
                    elif start_extracting['C']: length_C_results[x] = val
                    elif start_extracting['alpha']: alpha_results[x] = val
                    elif start_extracting['beta']: beta_results[x] = val
                    elif start_extracting['gamma']: gamma_results[x] = val
    except Exception as e:
        print(f"晶胞文件读取错误：{e}")

    eo_units = 1792
    common_keys = sorted(set(length_A_results.keys()) & set(length_B_results.keys()) &
                         set(length_C_results.keys()) & set(alpha_results.keys()) &
                         set(beta_results.keys()) & set(gamma_results.keys()))
    volume_results = {}
    for key in common_keys:
        la, lb, lc = length_A_results[key], length_B_results[key], length_C_results[key]
        alpha, beta, gamma = np.radians(alpha_results[key]), np.radians(beta_results[key]), np.radians(gamma_results[key])
        volume = la * lb * lc * np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
                                         2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
        volume_results[key] = volume / eo_units
    return volume_results

def density_xcd(folder_path):
    density_results = {}
    start_extracting_density = False
    try:
        if not os.path.exists(folder_path):
            print(f"密度文件不存在: {folder_path}")
            return density_results

        with open(folder_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                if 'Density' in line:
                    start_extracting_density = True
                    continue
                elif '</SERIES_2D>' in line:
                    start_extracting_density = False
                    continue
                if start_extracting_density and '<POINT_2D XY="' in line:
                    data_part = line.split('"')[1]
                    x, val = data_part.split(',')
                    density_results[float(x)] = float(val)
    except Exception as e:
        print(f"密度文件读取错误：{e}")
    return density_results

def energy_xcd(folder_path):
    potential, kinetic, total = {}, {}, {}
    start_extracting = {'potential': False, 'kinetic': False, 'total': False}

    try:
        if not os.path.exists(folder_path):
            print(f"能量文件不存在: {folder_path}")
            return potential, kinetic, total

        with open(folder_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if 'Potential energy' in line: start_extracting = {k: False for k in start_extracting}; start_extracting['potential'] = True
                elif 'Kinetic energy' in line: start_extracting = {k: False for k in start_extracting}; start_extracting['kinetic'] = True
                elif 'Total energy' in line: start_extracting = {k: False for k in start_extracting}; start_extracting['total'] = True
                elif '</SERIES_2D>' in line: start_extracting = {k: False for k in start_extracting}; continue

                if '<POINT_2D XY="' in line:
                    x, val = map(float, line.split('"')[1].split(','))
                    if start_extracting['potential']: potential[x] = val
                    elif start_extracting['kinetic']: kinetic[x] = val
                    elif start_extracting['total']: total[x] = val
    except Exception as e:
        print(f"能量文件读取错误：{e}")
    return potential, kinetic, total

# -------------------------- SCI 论文单图样式类 (保持不变) --------------------------
class plot_properties():
    def __init__(self, font_type='Times New Roman', font_size=36, axis_ticks_font_size=36, label_x="Time (ps)", label_y="",
                 legend_size=36, xlimit=12, ylimit=8):
        self.font_size = font_size
        self.axis_ticks_font_size = axis_ticks_font_size
        self.font_type = font_type
        self.label_x = label_x
        self.label_y = label_y
        self.legend_size = legend_size
        self.figsize = (xlimit, ylimit)

    def apply_style(self, ax, title=""):
        plt.rcParams.update({
            'font.family': 'serif', 'font.serif': [self.font_type], 'font.size': self.legend_size,
            'font.weight': 'bold', 'axes.unicode_minus': False
        })
        plt.rcParams.update({
            'mathtext.default': 'regular', 'mathtext.fontset': 'custom',
            'mathtext.rm': self.font_type, 'mathtext.it': self.font_type + ':italic', 'mathtext.bf': self.font_type + ':bold'
        })
        axis_font = {'fontname': self.font_type, 'size': self.font_size, 'fontweight': 'bold'}
        ax.set_xlabel(self.label_x, **axis_font)
        ax.set_ylabel(self.label_y, **axis_font, labelpad=20)

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname(self.font_type)
            tick.set_fontsize(self.axis_ticks_font_size)
            tick.set_fontweight('bold')

        thickness = 3
        for spine in ax.spines.values():
            spine.set_linewidth(thickness)

        ax.tick_params(direction='in', width=thickness, length=8, top=True, right=True)

        ax.set_title('')
        return ax

# -------------------------- 主程序：生成垂直堆叠的三合一图 --------------------------
if __name__ == "__main__":
    # 文件路径 (请根据实际情况修改)
    # 为确保代码可运行，此处使用占位符路径
    file_paths = {
        'temperature': 'path/to/your/Temperature.xcd',
        'energy': 'path/to/your/Energies.xcd',
        'density': 'path/to/your/Density.xcd'
    }

    # 当文件不存在时，生成模拟数据以供演示
    def generate_mock_data(time_points, start_val, noise_level):
        time = np.linspace(0, 1000, time_points)
        vals = np.full(time_points, start_val) + (np.random.randn(time_points) * noise_level)
        return {t: v for t, v in zip(time, vals)}

    # 提取数据 (若文件不存在，则使用模拟数据)
    temp_data = temperature_xcd(file_paths['temperature'])
    if not temp_data: temp_data = generate_mock_data(500, 300, 5)

    pot_data, kin_data, tot_data = energy_xcd(file_paths['energy'])
    if not tot_data: tot_data = generate_mock_data(500, 10000, 100)

    den_data = density_xcd(file_paths['density'])
    if not den_data: den_data = generate_mock_data(500, 1.24, 0.01)

    # 数据整理
    def format_data(data_dict):
        if not data_dict: return np.array([0]), np.array([0])
        x = np.array(sorted(data_dict.keys()))
        y = np.array([data_dict[k] for k in x])
        return x, y

    t_time, t_vals = format_data(temp_data)
    tot_time, tot_vals = format_data(tot_data)
    d_time, d_vals = format_data(den_data)

    # 整合所需的三个子图配置信息
    stacked_plot_configs = [
        {'y_label': 'Temperature (K)', 'data_x': t_time, 'data_y': t_vals, 'color': '#D62728'},
        {'y_label': r'Total Energy ($\text{kcal/mol}$)', 'data_x': tot_time, 'data_y': tot_vals, 'color': '#8C564B'},
        {'y_label': r'Density ($\text{g/cm}^3$)', 'data_x': d_time, 'data_y': d_vals, 'color': '#1F77B4'}
    ]

    # 统一设置 X 轴时间范围和刻度
    max_time = 1000
    xticks = np.linspace(0, max_time, 6)

    # ========================== 核心修改：创建 3x1 垂直堆叠子图 ==========================

    # 1. 创建 3 行 1 列的子图布局，共享 X 轴，总尺寸 12x18
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # 2. 遍历子图列表并绘制数据
    for i, ax in enumerate(axes):
        config = stacked_plot_configs[i]

        # 实例化样式类并应用 (仅底部子图显示X轴标签)
        x_label = "Time (ps)" if i == 2 else ""
        style = plot_properties(label_x=x_label, label_y=config['y_label'])
        ax = style.apply_style(ax)

        # 绘制数据
        ax.plot(config['data_x'], config['data_y'], color=config['color'], linewidth=3.5, alpha=0.9)

        # 设置统一的 X 轴范围和刻度
        ax.set_xlim(0, max_time)
        ax.set_xticks(xticks)

        # --------------------- 应用您指定的Y轴范围和刻度 ---------------------
        if 'Temperature' in config['y_label']:
            ax.set_ylim(0, 400)
            ax.set_yticks(np.linspace(0, 400, 5))
        elif 'Total Energy' in config['y_label']:
            ax.set_ylim(0, 20000)
            ax.set_yticks(np.linspace(0, 20000, 5))
        elif 'Density' in config['y_label']:
            ax.set_ylim(1.21, 1.27)
            ax.set_yticks(np.array([1.21, 1.23, 1.25, 1.27]))

    # 3. 调整子图间距，使其紧凑美观
    fig.subplots_adjust(hspace=0.1) # 稍微增加一点间距，避免轴线重叠过于拥挤
    plt.tight_layout(pad=0.5) # 调整布局，确保标签不被裁剪

    # 4. 保存并显示图像
    output_filename = "combined_thermo_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"--- 图像已保存为 {output_filename} ---")
    plt.show()
