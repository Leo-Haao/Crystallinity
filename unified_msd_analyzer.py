# -*- coding: utf-8 -*-
"""
unified_msd_analyzer.py

功能：
1. 读取 500K, 600K, 700K 的所有 MSD 数据。
   - 500K: 从指定目录结构中自动发现 XCD 文件。
   - 600K & 700K: 从预定义的 CSV 和 XCD 文件路径加载。
2. 将每个温度下的所有样本数据统一到该温度下最短的共同时间轴上。
3. 计算并绘制每个温度下总 MSD 和 X/Y/Z 分量的平均曲线与所有样本曲线。
4. 将所有温度的平均 MSD 曲线数据点合并输出到一个 CSV 文件中。
5. 直接显示图表，不保存为图片文件。
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# ==============================================================================
# --- 1. 全局配置与数据源 ---
# ==============================================================================
# --- 根目录与输出配置 ---
# 请根据您的文件结构修改这些路径
ROOT_FOLDER = r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents"
OUTPUT_DIR = os.path.join(ROOT_FOLDER, "Tri_comb", "Crystal", "combined_analysis_output")
AVERAGE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "All_Temperatures_Average_MSD_Data.csv")

# --- 通用计算配置 ---
INTERP_STEP = 1.0  # ps, 插值步长

# --- 各温度数据源配置 ---
# 700K 配置 (CSV + XCD)
CONFIG_700K = {
    'TEMP': 700,
    'CUTOFF_TIME': 27.0,
    'CSV_PATH': os.path.join(ROOT_FOLDER, "Tri_comb", "Crystal", "MSD_from_27ps_700K.csv"),
    'XCD_PATHS': [
        os.path.join(ROOT_FOLDER, "PEO_RUN", "crystal", "10ns", "700K", "supercell_17_700K_run_2", "PEO_Li_final_supercell Forcite MSD.xcd")
    ]
}

# 600K 配置 (CSV + XCD)
CONFIG_600K = {
    'TEMP': 600,
    'CUTOFF_TIME': 22.5,
    'CSV_PATH': os.path.join(ROOT_FOLDER, "Tri_comb", "Crystal", "MSD_from_22_5ps_600K.csv"),
    'XCD_PATHS': [
        os.path.join(ROOT_FOLDER, "PEO_RUN", "crystal", "10ns", "600K", "supercell_17_600K_run_3", "PEO_Li_final_supercell Forcite MSD.xcd"),
        os.path.join(ROOT_FOLDER, "PEO_RUN", "crystal", "10ns", "600K", "supercell_17_600K_run_7", "PEO_Li_final_supercell Forcite MSD.xcd")
    ]
}

# 500K 配置 (仅 XCD, 自动扫描)
CONFIG_500K = {
    'TEMP': 500,
    'CUTOFF_TIME': 47.5,
    'DATA_ROOT': os.path.join(ROOT_FOLDER, "PEO_RUN", "crystal", "10ns", "500K")
}

# --- Y轴范围配置 (可选，用于美化图表) ---
Y_LIMITS_CONFIG = {
    500: {
        'total': {'ylim': (0, 1000), 'yticks': np.arange(0, 1001, 200)},
        'xx component': {'ylim': (0, 1000), 'yticks': np.arange(0, 1001, 200)},
        'yy component': {'ylim': (0, 1000), 'yticks': np.arange(0, 1001, 200)},
        'zz component': {'ylim': (0, 1000), 'yticks': np.arange(0, 1001, 200)},
    },
    600: {
        'total': {'ylim': (0, 3000), 'yticks': np.arange(0, 3001, 600)},
        'xx component': {'ylim': (0, 3000), 'yticks': np.arange(0, 3001, 600)},
        'yy component': {'ylim': (0, 3000), 'yticks': np.arange(0, 3001, 600)},
        'zz component': {'ylim': (0, 3000), 'yticks': np.arange(0, 3001, 600)},
    },
    700: {
        'total': {'ylim': (0, 10000), 'yticks': np.arange(0, 10001, 2000)},
        'xx component': {'ylim': (0, 10000), 'yticks': np.arange(0, 10001, 2000)},
        'yy component': {'ylim': (0, 10000), 'yticks': np.arange(0, 10001, 2000)},
        'zz component': {'ylim': (0, 10000), 'yticks': np.arange(0, 10001, 2000)},
    }
}

# ==============================================================================
# --- 2. 绘图样式类 ---
# ==============================================================================
class PlotProperties:
    def __init__(self, font_type="Times New Roman", font_size=26, axis_ticks_font_size=26,
                 label_x="Time (ps)", label_y=r"MSD ($\AA^2$)", fig_size=(12, 8), legend_size=15):
        self.font_type = font_type
        self.font_size = font_size
        self.axis_ticks_font_size = axis_ticks_font_size
        self.label_x = label_x
        self.label_y = label_y
        self.fig_size = fig_size
        self.legend_size = legend_size

    def apply_style(self):
        plt.figure(figsize=self.fig_size)
        axis_font = {'fontname': self.font_type, 'size': self.font_size, 'weight': 'bold'}

        plt.rc('font', family=self.font_type, weight='bold')
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = self.font_type

        plt.xlabel(self.label_x, **axis_font)
        plt.ylabel(self.label_y, **axis_font)

        ax = plt.gca()
        ax.tick_params(axis='both', direction='in', width=2, length=6,
                       top=True, right=True, labelsize=self.axis_ticks_font_size)

        for spine in ax.spines.values():
            spine.set_linewidth(2)

        plt.tight_layout()
        return plt

# ==============================================================================
# --- 3. 工具函数 ---
# ==============================================================================
def extract_msd_data(file_path, component=None):
    """从 XCD 文件中提取总 MSD 或分量 MSD 数据"""
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
                        time = float(time_str)
                        msd = float(msd_str)
                        msd_results[time] = msd
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[错误] 提取 {os.path.basename(file_path)} 的 {target_line} 失败: {e}")
    return msd_results

def calculate_mean_by_time(data_list, dt, max_time):
    """基于统一时间轴进行插值并计算平均值"""
    if not data_list:
        return np.array([]), np.array([])

    valid_data = [(t, m) for t, m in data_list if len(t) > 1 and len(m) > 1]
    if not valid_data:
        return np.array([]), np.array([])

    # 创建统一的时间轴
    unified_t = np.arange(0, max_time + dt, dt)
    interpolated_msds = []

    for t, m in valid_data:
        # 确保时间是单调递增的
        if not np.all(np.diff(t) >= 0): continue
        # 创建插值函数
        f = interp1d(t, m, kind='linear', bounds_error=False, fill_value="extrapolate")
        interpolated_msds.append(f(unified_t))

    if not interpolated_msds:
        return np.array([]), np.array([])

    return unified_t, np.mean(interpolated_msds, axis=0)

def get_global_min_time(all_data):
    """获取所有样本的最大时间中的最小值，作为全局时间上限"""
    all_max_times = []
    for t, _ in all_data['total']:
        if len(t) > 0:
            all_max_times.append(np.max(t))
    return min(all_max_times) if all_max_times else 0

# ==============================================================================
# --- 4. 数据加载核心逻辑 ---
# ==============================================================================
def load_data_for_temp(config):
    """根据配置加载一个温度下的所有 MSD 数据"""
    all_data = {'total': [], 'xx component': [], 'yy component': [], 'zz component': []}
    temp = config['TEMP']
    cutoff_time = config['CUTOFF_TIME']

    # --- 模式1: 加载 CSV 和额外的 XCD 文件 (600K, 700K) ---
    if 'CSV_PATH' in config:
        try:
            df = pd.read_csv(config['CSV_PATH'])
            df_truncated = df[df['Sample_Time (ps)'] >= cutoff_time].copy()

            sample_names = sorted(df_truncated['Sample_Name'].unique())
            for name in sample_names:
                group = df_truncated[df_truncated['Sample_Name'] == name]
                x = group['Relative_Time (ps)'].values
                all_data['total'].append((x, group['Total_MSD (Å²)'].values))
                all_data['xx component'].append((x, group['X_MSD (Å²)'].values))
                all_data['yy component'].append((x, group['Y_MSD (Å²)'].values))
                all_data['zz component'].append((x, group['Z_MSD (Å²)'].values))
            print(f"[信息] {temp}K: 成功加载 {len(sample_names)} 个CSV样本。")
        except Exception as e:
            print(f"[错误] {temp}K: 加载CSV数据失败: {e}")

    # 加载 config 中明确指定的 XCD 文件
    xcd_paths_to_load = config.get('XCD_PATHS', [])

    # --- 模式2: 自动扫描发现 XCD 文件 (500K) ---
    if 'DATA_ROOT' in config:
        data_root = config['DATA_ROOT']
        folder_pattern = rf'supercell_17_{temp}K_run_\d+'
        try:
            subfolders = [d for d in os.listdir(data_root) if re.match(folder_pattern, d)]
            for folder in subfolders:
                folder_path = os.path.join(data_root, folder)
                for dirpath, _, files in os.walk(folder_path):
                    for f in files:
                        if f.endswith('MSD.xcd'):
                            xcd_paths_to_load.append(os.path.join(dirpath, f))
                            break # 每个 run 文件夹只取第一个找到的
            print(f"[信息] {temp}K: 在 {data_root} 发现 {len(xcd_paths_to_load)} 个XCD文件。")
        except Exception as e:
             print(f"[错误] {temp}K: 扫描目录失败: {e}")

    # --- 对所有找到的 XCD 路径进行统一处理 ---
    for path in xcd_paths_to_load:
        if not os.path.exists(path):
            print(f"[警告] {temp}K: XCD文件不存在: {path}")
            continue

        total_msd = extract_msd_data(path)
        xx_msd = extract_msd_data(path, 'xx component')
        yy_msd = extract_msd_data(path, 'yy component')
        zz_msd = extract_msd_data(path, 'zz component')

        for comp, msd_dict in [('total', total_msd), ('xx component', xx_msd),
                               ('yy component', yy_msd), ('zz component', zz_msd)]:
            if msd_dict:
                times = np.array(sorted(msd_dict.keys()))
                msds = np.array([msd_dict[t] for t in times])
                mask = times >= cutoff_time
                if np.any(mask):
                    rel_times = times[mask] - cutoff_time
                    all_data[comp].append((rel_times, msds[mask]))
        print(f"[信息] {temp}K: 成功加载 XCD样本: {os.path.basename(path)}")

    return all_data

# ==============================================================================
# --- 5. 绘图与数据处理核心逻辑 ---
# ==============================================================================
def plot_and_get_avg(all_data, config, global_time_max):
    """绘制所有样本和平均曲线，并返回平均值数据"""
    style = PlotProperties()
    all_avg_data = {}
    temp = config['TEMP']

    plots_map = [
        ('total', 'Total MSD', 'Total_MSD_Avg'),
        ('xx component', 'X[100]', 'X_MSD_Avg'),
        ('yy component', 'Y[010]', 'Y_MSD_Avg'),
        ('zz component', 'Z[001]', 'Z_MSD_Avg')
    ]

    for comp_key, comp_label, avg_col_name in plots_map:
        if not all_data[comp_key]:
            print(f"[信息] {temp}K: 没有找到 {comp_key} 的数据，跳过绘图。")
            continue

        plt_comp = style.apply_style()

        # 绘制所有截断后的独立样本曲线
        truncated_data_for_avg = []
        for x, y in all_data[comp_key]:
            if len(x) > 0:
                mask = x <= global_time_max
                x_trunc, y_trunc = x[mask], y[mask]
                if len(x_trunc) > 0:
                    truncated_data_for_avg.append((x_trunc, y_trunc))
                    plt_comp.plot(x_trunc, y_trunc, 'grey', linestyle='-', linewidth=2, alpha=0.3)

        # 计算并绘制平均曲线
        avg_time, avg_msd = calculate_mean_by_time(truncated_data_for_avg, INTERP_STEP, global_time_max)

        if len(avg_time) > 0:
            all_avg_data[f"Time_{temp}K (ps)"] = avg_time
            all_avg_data[f"{avg_col_name}_{temp}K (Å²)"] = avg_msd
            plt_comp.plot(avg_time, avg_msd, 'red', linewidth=4, label="Average")

        # --- 设置坐标轴和标题 ---
        plt_comp.xlim(0, global_time_max)
        plt_comp.xticks(np.linspace(0, global_time_max, 6))

        # 应用Y轴范围配置
        if temp in Y_LIMITS_CONFIG and comp_key in Y_LIMITS_CONFIG[temp]:
            y_config = Y_LIMITS_CONFIG[temp][comp_key]
            plt_comp.ylim(y_config['ylim'])
            plt_comp.yticks(y_config['yticks'])

        plt_comp.legend(loc='upper left', framealpha=1, edgecolor='k', fontsize=style.legend_size)
        plt_comp.title(f"Crystal {comp_label} - {temp}K", fontsize=style.font_size, fontweight='bold', y=1.03)

    return all_avg_data

# ==============================================================================
# --- 6. 主函数 ---
# ==============================================================================
def main_analyzer():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_avg_data_frames = []

    # 按顺序处理 500K, 600K, 700K
    for config in [CONFIG_500K, CONFIG_600K, CONFIG_700K]:
        temp = config['TEMP']
        print(f"\n======== 开始处理 {temp}K 数据 ========")

        # 1. 加载数据
        all_data = load_data_for_temp(config)

        if not any(all_data.values()):
            print(f"[警告] {temp}K: 未加载到任何有效数据，跳过处理。")
            continue

        # 2. 确定全局时间轴上限
        global_time_max = get_global_min_time(all_data)
        if global_time_max <= 0:
            print(f"[错误] {temp}K: 无法确定全局时间轴上限，跳过绘图。")
            continue
        print(f"[信息] {temp}K: 全局统一时间轴上限: {global_time_max:.2f} ps")

        # 3. 绘图并获取平均值
        avg_data = plot_and_get_avg(all_data, config, global_time_max)

        # 4. 将平均数据转换为 DataFrame 存储
        if avg_data:
            time_key = f"Time_{temp}K (ps)"
            # 创建时必须保证所有列长度一致，先处理好数据
            max_len = max(len(v) for v in avg_data.values())
            processed_avg_data = {k: np.pad(v, (0, max_len - len(v)), 'constant', constant_values=np.nan) for k, v in avg_data.items()}

            avg_df = pd.DataFrame(processed_avg_data)

            # 调整列顺序
            cols_ordered = [time_key]
            for col in ['Total_MSD_Avg', 'X_MSD_Avg', 'Y_MSD_Avg', 'Z_MSD_Avg']:
                key = f"{col}_{temp}K (Å²)"
                if key in avg_df.columns:
                    cols_ordered.append(key)

            all_avg_data_frames.append(avg_df[cols_ordered])

    print("\n======== 正在合并并输出平均值数据 ========")
    if all_avg_data_frames:
        final_avg_df = pd.concat(all_avg_data_frames, axis=1)
        final_avg_df.to_csv(AVERAGE_OUTPUT_PATH, index=False, float_format='%.6f', encoding='utf-8-sig')
        print(f"[成功] 所有平均值数据已保存到:\n{AVERAGE_OUTPUT_PATH}")
    else:
        print("[警告] 未生成任何可供输出的平均值数据。")

    print("\n所有处理完成。即将显示图表...")
    plt.show()

if __name__ == "__main__":
    main_analyzer()
