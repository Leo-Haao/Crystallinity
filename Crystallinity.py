import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import warnings

# 忽略 CIF 文件解析警告
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")


class PlotProperties:
    def __init__(self, font_type="Times New Roman", font_size=26, axis_ticks_size=24,
                 xlabel="Time (ps)", ylabel="Crystallinity Quality", fig_size=(8, 6)):
        self.font_type = font_type
        self.font_size = font_size
        self.axis_ticks_size = axis_ticks_size
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fig_size = fig_size

    def get_plot_style(self):
        plt.figure(figsize=self.fig_size)
        plt.xlabel(self.xlabel, fontname=self.font_type, size=self.font_size, weight="bold")
        plt.ylabel(self.ylabel, fontname=self.font_type, size=self.font_size, weight="bold")
        plt.xticks(fontname=self.font_type, size=self.axis_ticks_size, weight="bold")
        plt.yticks(fontname=self.font_type, size=self.axis_ticks_size, weight="bold")
        ax = plt.gca()
        ax.spines["top"].set_linewidth(2)
        ax.spines["right"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        ax.tick_params(axis="y", direction="in", width=2, length=6, right=True)
        ax.tick_params(axis="x", direction="in", width=2, length=6, top=True)
        plt.tight_layout()
        return plt


def calculate_crystallinity(pattern_x, pattern_y):
    """
    基于峰尖锐度（峰高/半高宽）评估结晶完善程度
    1. 识别所有有效峰
    2. 计算每个峰的尖锐度（峰高/半高宽）
    3. 平均尖锐度归一化，作为结晶完善度
    """
    # 1. 平滑数据（适度保留峰特征）
    y_smoothed = savgol_filter(pattern_y, window_length=9, polyorder=3)

    # 2. 识别晶相峰（低阈值适配尖锐峰，峰宽限制 0.1-5°）
    peak_height_threshold = 0.01 * max(y_smoothed)  # 阈值设为最大强度的 1%
    peaks, props = find_peaks(
        y_smoothed,
        height=peak_height_threshold,
        width=(0.1, 5)  # 峰宽范围 0.1-5°（覆盖实际窄峰）
    )

    if len(peaks) == 0:
        return 0.0  # 无峰，结晶度（完善度）为 0

    # 3. 计算每个峰的尖锐度（峰高 / 半高宽）
    sharpness_values = []
    for i, peak_idx in enumerate(peaks):
        peak_height = props['peak_heights'][i]
        fwhm = props['widths'][i]
        if fwhm == 0:
            continue  # 避免除以 0（理论上不会出现）
        sharpness = peak_height / fwhm
        sharpness_values.append(sharpness)

    if not sharpness_values:
        return 0.0  # 无有效峰

    # 4. 平均尖锐度 & 归一化
    avg_sharpness = np.mean(sharpness_values)
    # 归一化：除以理论最大尖锐度（可根据体系调整，这里取所有帧的最大尖锐度）
    # 若需动态归一化，可先跑所有帧计算全局最大，再重新归一化。这里简化为除以当前最大尖锐度
    max_sharpness = max(sharpness_values) if sharpness_values else 1
    normalized_sharpness = avg_sharpness / max_sharpness

    # 限制在 [0, 1] 范围
    return min(max(normalized_sharpness, 0), 1)


def process_xrd_for_crystallinity(file_path):
    """处理单个 CIF 文件，返回结晶完善度"""
    try:
        structure = Structure.from_file(file_path)
        calculator = XRDCalculator()
        # 计算 XRD 图谱（覆盖所有峰的 2θ 范围）
        pattern = calculator.get_pattern(structure, scaled=False, two_theta_range=(20, 40))
        return calculate_crystallinity(pattern.x, pattern.y)
    except Exception as e:
        print(f"处理 {file_path} 失败: {e}")
        return None


def process_temperature_folder(temp_folder, time_per_frame=5):
    """处理单个温度文件夹，返回时间和结晶完善度数据"""
    times = []
    crystallinities = []
    all_frames = range(1, len(os.listdir(temp_folder)) + 1)  # 假设帧号从 1 开始连续

    for frame in all_frames:
        file_path = os.path.join(temp_folder, f'Frame_{frame}.cif')
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue

        cryst = process_xrd_for_crystallinity(file_path)
        if cryst is not None:
            times.append(frame * time_per_frame)  # 换算为 ps
            crystallinities.append(cryst)

    return times, crystallinities


def main():
    # 定义三个温度的文件夹路径
    temp_folders = {
        "300K": r"E:\坚果云\MD works\2025.PEO.Li.MD_work\07.separated_300_and_500K\lowest Frame cif\300K 0.5ns",
        "500K": r"E:\坚果云\MD works\2025.PEO.Li.MD_work\07.separated_300_and_500K\lowest Frame cif\500K 0.5ns",
        "700K": r"E:\坚果云\MD works\2025.PEO.Li.MD_work\07.separated_300_and_500K\lowest Frame cif\700K 0.5ns"
    }

    # 颜色区分不同温度
    colors = {"300K": "#1f77b4", "500K": "#ff7f0e", "700K": "#2ca02c"}

    # 初始化绘图样式
    plot_style = PlotProperties(
        xlabel="Time (ps)",
        ylabel="Crystallinity ",
        fig_size=(10, 6)
    )
    plt = plot_style.get_plot_style()

    for temp, folder in temp_folders.items():
        times, crystallinities = process_temperature_folder(folder)
        if times and crystallinities:
            plt.plot(
                times, crystallinities,
                marker="o", markersize=4, linestyle="-", color=colors[temp],
                label=f"{temp}"
            )

    # 设置图例和标题
    plt.legend(
        title="Temperature",
        loc="upper left",
        frameon=True,
        edgecolor="black",
        fontsize=16,
        title_fontsize=18
    )
    plt.title("Crystallinity vs Time ",
              fontname=plot_style.font_type, size=plot_style.font_size, weight="bold")
    plt.xlim(0, max([max(times) for times, _ in temp_folders.values() if times]))
    plt.ylim(0, 1)  # 完善度范围 0-1
    plt.show()


if __name__ == "__main__":
    main()