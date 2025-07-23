import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from matplotlib.ticker import MaxNLocator, FuncFormatter
from monty.json import MSONable
from pymatgen.core import Structure, Element, Lattice
from math import pi
import warnings

# 忽略pymatgen可能产生的用户警告
warnings.filterwarnings("ignore", category=UserWarning, module='pymatgen')

def moving_average(x, window_size):
    """移动平均滤波函数"""
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(x, window, 'same')

class PlotProperties(MSONable):
    """绘图样式类，匹配能量/体积图的规范"""
    def __init__(self, font_type='Times New Roman', font_size=26, axis_ticks_font_size=24, label_x="", label_y="", legend_size=15, fig_size=(8, 6)):
        self.font_type = font_type
        self.font_size = font_size
        self.axis_ticks_font_size = axis_ticks_font_size
        self.label_x = label_x
        self.label_y = label_y
        self.legend_size = legend_size
        self.fig_size = fig_size
        self.plt = plt

    def apply_style(self):
        """应用与能量图一致的绘图样式"""
        plt.figure(figsize=self.fig_size)
        ax = plt.gca()

        # 字体设置
        plt.rc('font', family=self.font_type, size=self.legend_size, weight='bold')
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = self.font_type

        plt.xlabel(self.label_x, fontname=self.font_type, size=self.font_size, weight='bold')
        plt.ylabel(self.label_y, fontname=self.font_type, size=self.font_size, weight='bold')
        plt.xticks(fontname=self.font_type, size=self.axis_ticks_font_size, weight='bold')
        plt.yticks(fontname=self.font_type, size=self.axis_ticks_font_size, weight='bold')

        # 边框和刻度样式
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(direction='in', width=2, length=6, right=True, top=True)

        plt.tight_layout()
        return plt, ax

def pbc_distances_numpy(coords1, coords2, lattice_matrix, inv_lattice_matrix, max_r=None):
    """ 用NumPy计算考虑周期性边界条件（PBC）的原子对之间的距离（CPU版本）。 使用 Minimum Image Convention (MIC)。 """
    # 1. 计算所有可能的笛卡尔坐标差值（广播机制实现N1×N2对）
    diff_cart = coords1[:, None, :] - coords2[None, :, :]  # 形状：(N1, N2, 3)

    # 2. 笛卡尔差值 → 分数坐标差值：Δs = Δr · A⁻¹
    frac_diff = np.matmul(diff_cart, inv_lattice_matrix)

    # 3. 周期性折叠（Minimum Image Convention）
    frac_diff_folded = frac_diff - np.round(frac_diff)

    # 4. 分数坐标差值 → 笛卡尔坐标差值：Δr_pbc = Δs_folded · A
    pbc_diff_cart = np.matmul(frac_diff_folded, lattice_matrix)

    # 5. 计算欧几里得距离
    distances = np.linalg.norm(pbc_diff_cart, axis=-1)  # 形状：(N1, N2)

    # 筛选小于max_r的距离
    if max_r is not None:
        distances = distances[distances <= max_r]

    return distances.flatten()  # 展平为一维数组

def calculate_rdf_nr(structure, center_element, neighbor_element, dr=0.1, max_r=5.0):
    """ 用numpy和pymatgen计算g(r)和n(r)（替代PyTorch加速版本） Returns: r_for_nr, nr_list, r_for_gr, gr_list """
    # 提取中心原子和邻近原子坐标
    center_coords = np.array([site.coords for site in structure.sites if site.specie == Element(center_element)])
    neighbor_coords = np.array([site.coords for site in structure.sites if site.specie == Element(neighbor_element)])

    # 检查原子是否存在
    if len(center_coords) == 0 or len(neighbor_coords) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 晶格矩阵和逆矩阵
    lattice = structure.lattice.matrix
    inv_lattice = np.linalg.inv(lattice)

    # 计算所有原子对距离（考虑PBC）
    print(f"正在计算 {len(center_coords)} 个 {center_element} 到 {len(neighbor_coords)} 个 {neighbor_element} 的距离...")
    distances = pbc_distances_numpy(center_coords, neighbor_coords, lattice, inv_lattice, max_r)
    print(f"计算了 {len(distances)} 个有效距离。")

    if len(distances) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 构建直方图（ bins从0到max_r，步长dr）
    bins = np.arange(0, max_r + dr, dr)
    hist, _ = np.histogram(distances, bins=bins)

    # 计算n(r)：累积和 / 中心原子数
    num_center = len(center_coords)
    nr_list = np.cumsum(hist) / num_center
    r_for_nr = bins[1:]  # 桶的右边界

    # 计算g(r)：实际计数 / 理想计数
    # 跳过第一个桶（0, dr]，避免r=0附近的奇点
    hist_for_gr = hist[1:]
    r_mid = (bins[1:-1] + bins[2:]) / 2  # 桶中心

    # 理想计数：ρ * 壳层体积 * 中心原子数
    volume = structure.volume
    rho_neighbor = len(neighbor_coords) / volume  # 邻近原子数密度
    shell_volumes = (4/3 * pi) * ((r_mid + dr/2)**3 - (r_mid - dr/2)**3)
    ideal_counts = rho_neighbor * shell_volumes * num_center

    # 避免除以零
    gr_list = np.where(ideal_counts != 0, hist_for_gr / ideal_counts, 0)

    return r_for_nr, nr_list, r_mid, gr_list

def plot_combined_rdf_nr_data(data_by_temp, atom_type, time_point, y_limits=None):
    """绘制合并了多个温度的g(r)和n(r)图像"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"300K": "#1f77b4", "500K": "#ff7f0e", "700K": "#2ca02c"}

    # 绘制n(r)
    plot_style_nr = PlotProperties(label_x="r (Å)", label_y="n(r)")
    _, ax1 = plot_style_nr.apply_style()
    ax1.set_title(f"Cumulative Number Density (Li+ - {atom_type})", fontname='Times New Roman', size=22, weight='bold', pad=20)

    # 绘制g(r)
    plot_style_gr = PlotProperties(label_x="r (Å)", label_y="g(r)")
    _, ax2 = plot_style_gr.apply_style()
    ax2.set_title(f"Radial Distribution Function (Li+ - {atom_type})", fontname='Times New Roman', size=22, weight='bold', pad=20)

    for temp, data in data_by_temp.items():
        r_nr, nr, r_gr, gr = data
        if len(nr) == 0 or len(gr) == 0:
            continue

        # 平滑处理
        nr_smoothed = moving_average(nr, window_size=5)
        gr_smoothed = moving_average(gr, window_size=5)

        # 插值生成平滑曲线
        nr_spline = UnivariateSpline(r_nr, nr_smoothed, s=0, k=3)
        gr_spline = UnivariateSpline(r_gr, gr_smoothed, s=0, k=3)

        r_nr_smooth = np.linspace(r_nr.min(), r_nr.max(), 500)
        r_gr_smooth = np.linspace(r_gr.min(), r_gr.max(), 500)

        nr_smooth = nr_spline(r_nr_smooth)
        gr_smooth = gr_spline(r_gr_smooth)

        color = colors.get(temp, "k")
        label = f"{temp}"

        ax1.plot(r_nr_smooth, nr_smooth, color=color, linewidth=2, label=f"{temp}")
        ax2.plot(r_gr_smooth, gr_smooth, color=color, linewidth=2, label=f"{temp}")

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 5.0)
        ax.set_xticks(np.arange(0, 5.1, 1.0))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
        ax.legend(fontsize=12, frameon=False, loc='upper left')
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)

    if y_limits:
        ax1.set_ylim(y_limits[0], y_limits[1])
        ax2.set_ylim(y_limits[2], y_limits[3])

    plt.suptitle(f"Time: {time_point}ps", fontname='Times New Roman', size=28, weight='bold', y=1.02)
    plt.tight_layout()
    return plt

def main():
    # 你的数据根目录（CIF文件所在）
    base_data_root = r'E:\坚果云\MD works\2025.PEO.Li.MD_work\07.separated_300_and_500K\lowest Frame cif'
    temperature_folders = {
        "300K": {
            "folder": "300K 0.5ns",
            "frame_time_map": {1: 0, 26: 125, 51: 250, 101: 500}
        },
        "500K": {
            "folder": "500K 0.5ns",
            "frame_time_map": {1: 0, 26: 125, 51: 250, 101: 500}
        },
        "700K": {
            "folder": "700K 0.5ns",
            "frame_time_map": {1: 0, 26: 125, 51: 250, 101: 500}
        },
    }
    atom_mapping = {
        'C': ('Li+ - C', '#1f77b4'),
        'H': ('Li+ - H', '#d62728'),
        'O': ('Li+ - O', '#2ca02c')
    }

    # 计算参数
    center_element = "Li"
    dr = 0.1
    max_r = 5.0

    # 收集所有时间点
    all_time_points = sorted(list(set(t for temp_info in temperature_folders.values() for t in temp_info["frame_time_map"].values())))

    # 第一步：收集所有数据，确定最大范围
    max_values = {atom: {'nr': 0, 'gr': 0} for atom in atom_mapping}
    for temp, temp_info in temperature_folders.items():
        temp_path = os.path.join(base_data_root, temp_info["folder"])
        if not os.path.isdir(temp_path):
            print(f"警告：文件夹不存在 → {temp_path}")
            continue
        for frame_num in temp_info["frame_time_map"]:
            cif_path = os.path.join(temp_path, f"Frame_{frame_num}.cif")
            if not os.path.exists(cif_path):
                continue
            try:
                structure = Structure.from_file(cif_path)
                for atom_type in atom_mapping:
                    r_nr, nr, r_gr, gr = calculate_rdf_nr(
                        structure, center_element, atom_type, dr, max_r
                    )
                    if len(nr) > 0:
                        max_values[atom_type]['nr'] = max(max_values[atom_type]['nr'], np.max(moving_average(nr, 5)))
                    if len(gr) > 0:
                        max_values[atom_type]['gr'] = max(max_values[atom_type]['gr'], np.max(moving_average(gr, 5)))
            except Exception as e:
                print(f"处理 {cif_path} 出错 → {e}")

    # 第二步：按时间点绘制图像
    for time_point in all_time_points:
        for atom_type, (label, color) in atom_mapping.items():
            data_by_temp = {}
            for temp, temp_info in temperature_folders.items():
                # 找到对应时间点的帧号
                frame_num = None
                for f, t in temp_info["frame_time_map"].items():
                    if t == time_point:
                        frame_num = f
                        break
                if frame_num is None:
                    continue

                temp_path = os.path.join(base_data_root, temp_info["folder"])
                cif_path = os.path.join(temp_path, f"Frame_{frame_num}.cif")
                if not os.path.exists(cif_path):
                    continue

                try:
                    structure = Structure.from_file(cif_path)
                    data_by_temp[temp] = calculate_rdf_nr(
                        structure, center_element, atom_type, dr, max_r
                    )
                except Exception as e:
                    print(f"处理 {cif_path} 出错 → {e}")

            if not data_by_temp:
                continue

            y_nr_max = np.ceil(max_values[atom_type]['nr'] * 1.1)
            y_gr_max = np.ceil(max_values[atom_type]['gr'] * 1.1)
            y_limits = (0, y_nr_max, 0, y_gr_max)

            plt = plot_combined_rdf_nr_data(data_by_temp, atom_type, time_point, y_limits)
            if plt:
                plt.show()

if __name__ == "__main__":
    main()
