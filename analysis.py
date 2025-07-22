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

# --- 周期性边界条件距离计算函数（NumPy版本，CPU计算） ---
def pbc_distances_numpy(coords1, coords2, lattice_matrix, inv_lattice_matrix, max_r=None):
    """
    用NumPy计算考虑周期性边界条件（PBC）的原子对之间的距离（CPU版本）。
    使用 Minimum Image Convention (MIC)。
    """
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


# --- NumPy版本直方图函数 ---
def custom_histogram_numpy(data, bins):
    """
    用NumPy实现直方图计算（CPU版本）。
    """
    # 获取每个数据点的桶索引（区间为(left, right]）
    indices = np.digitize(data, bins, right=True)  # 返回1-based索引

    # 筛选有效索引（落在[bins[0], bins[-1]]范围内）
    valid_mask = (indices >= 1) & (indices <= len(bins) - 1)
    binned_indices = indices[valid_mask] - 1  # 转换为0-based索引

    # 计数
    hist = np.bincount(binned_indices, minlength=len(bins) - 1).astype(np.float64)

    # 确保长度正确
    if len(hist) < len(bins) - 1:
        hist = np.pad(hist, (0, len(bins) - 1 - len(hist)), mode='constant')
    elif len(hist) > len(bins) - 1:
        hist = hist[:len(bins) - 1]

    return hist


# --- 主函数：计算RDF和n(r)（NumPy版本） ---
def calculate_oh_rdf_numpy(structure: Structure, center_element: str, neighbor_element: str, dr: float, max_r: float = 10.0):
    """
    用NumPy计算径向分布函数g(r)和累积数n(r)（CPU版本）。
    """
    # CPU端数据准备
    center_coords = np.array([site.coords for site in structure.sites if site.specie == Element(center_element)])
    neighbor_coords = np.array([site.coords for site in structure.sites if site.specie == Element(neighbor_element)])

    lattice_matrix = structure.lattice.matrix
    inv_lattice_matrix = np.linalg.inv(lattice_matrix)

    # 检查原子是否存在
    if len(center_coords) == 0:
        print(f"警告：结构中没有 {center_element} 原子。")
        return np.array([]), np.array([]), np.array([]), np.array([])
    if len(neighbor_coords) == 0:
        print(f"警告：结构中没有 {neighbor_element} 原子。")
        return np.array([]), np.array([]), np.array([]), np.array([])

    num_center = len(center_coords)
    num_neighbor = len(neighbor_coords)
    volume = structure.volume
    if volume <= 0:
        print("警告：结构体积无效。")
        return np.array([]), np.array([]), np.array([]), np.array([])
    rho_neighbor = num_neighbor / volume  # 邻近原子密度

    # 计算所有原子对距离（考虑PBC）
    print(f"正在计算 {num_center} 个 {center_element} 到 {num_neighbor} 个 {neighbor_element} 的距离...")
    all_distances = pbc_distances_numpy(
        center_coords, neighbor_coords, lattice_matrix, inv_lattice_matrix, max_r=max_r
    )
    print(f"计算了 {len(all_distances)} 个有效距离。")

    if len(all_distances) == 0:
        print(f"警告：{max_r} Å 范围内无有效距离。")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 构建直方图
    bins = np.arange(0, max_r + dr, dr, dtype=np.float64)
    hist = custom_histogram_numpy(all_distances, bins)

    # 计算n(r)：累积和 + 归一化
    nr_list = np.cumsum(hist) / num_center
    r_for_nr = bins[1:]  # 桶的右边界

    # 计算g(r)：实际计数 / 理想计数
    hist_for_gr = hist[1:]  # 跳过第一个桶
    r_for_gr = (bins[1:-1] + bins[2:]) / 2  # 桶的中心点
    actual_dr = dr

    # 理想情况下的球壳体积内原子数
    shell_volumes = (4/3 * pi) * ((r_for_gr + actual_dr/2)**3 - (r_for_gr - actual_dr/2)** 3)
    ideal_counts = rho_neighbor * shell_volumes * num_center

    # 避免除以0
    gr_list = np.where(ideal_counts != 0, hist_for_gr / ideal_counts, 0.0)

    return r_for_nr, nr_list, r_for_gr, gr_list


class PlotProperties(MSONable):
    """绘图样式类，匹配能量/体积图的规范"""
    def __init__(self, font_type='Times New Roman', font_size=26,
                 axis_ticks_font_size=24, label_x="", label_y="",
                 legend_size=15, fig_size=(8, 6)):
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


def moving_average(x, window_size):
    """移动平均滤波函数"""
    window = np.ones(int(window_size)) / int(window_size)
    return np.convolve(x, window, 'same')


def plot_rdf_nr_data(r_for_nr, nr_list, r_for_gr, gr_list, color, label,
                     temperature, atom_type, time_point, y_limits=None):
    """绘制g(r)和n(r)图像（整合到你的绘图框架）"""
    # 平滑处理
    nr_smoothed = moving_average(nr_list, window_size=5)
    gr_smoothed = moving_average(gr_list, window_size=5)

    # 插值生成平滑曲线
    nr_spline = UnivariateSpline(r_for_nr, nr_smoothed, s=3, k=3)
    gr_spline = UnivariateSpline(r_for_gr, gr_smoothed, s=3, k=3)

    r_nr_smooth = np.linspace(r_for_nr.min(), r_for_nr.max(), 500)
    r_gr_smooth = np.linspace(r_for_gr.min(), r_for_gr.max(), 500)

    nr_smooth = nr_spline(r_nr_smooth)
    gr_smooth = gr_spline(r_gr_smooth)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制n(r)
    plot_style_nr = PlotProperties(label_x="r (Å)", label_y="n(r)")
    plt_nr, ax1 = plot_style_nr.apply_style()
    ax1.plot(r_nr_smooth, nr_smooth, color=color, linewidth=2, label=label)
    ax1.scatter(r_for_nr, nr_smoothed, color=color, alpha=0.6, s=30)
    ax1.set_title(f"Cumulative Number Density (Li+ - {atom_type})",
                 fontname='Times New Roman', size=22, weight='bold', pad=20)

    # 绘制g(r)
    plot_style_gr = PlotProperties(label_x="r (Å)", label_y="g(r)")
    plt_gr, ax2 = plot_style_gr.apply_style()
    ax2.plot(r_gr_smooth, gr_smooth, color=color, linewidth=2, label=label)
    ax2.scatter(r_for_gr, gr_smoothed, color=color, alpha=0.6, s=30)
    ax2.set_title(f"Radial Distribution Function (Li+ - {atom_type})",
                 fontname='Times New Roman', size=22, weight='bold', pad=20)

    # 统一坐标轴设置
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 5.0)
        ax.set_xticks(np.arange(0, 5.1, 1.0))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
        ax.legend(fontsize=12, frameon=False, loc='upper left')
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)

    # 设置Y轴范围
    if y_limits:
        ax1.set_ylim(y_limits[0], y_limits[1])
        ax2.set_ylim(y_limits[2], y_limits[3])

    plt.suptitle(f"{temperature} - {time_point}ps",
                fontname='Times New Roman', size=28, weight='bold', y=1.02)
    plt.tight_layout()
    return plt, (nr_smooth.max(), gr_smooth.max())


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

    # 第一步：收集所有数据，确定最大范围
    max_values = {atom: {'nr': 0, 'gr': 0} for atom in atom_mapping}

    for temp, temp_info in temperature_folders.items():
        temp_path = os.path.join(base_data_root, temp_info["folder"])
        if not os.path.isdir(temp_path):
            print(f"警告：文件夹不存在 → {temp_path}")
            continue

        for atom_type, (_, _) in atom_mapping.items():
            for frame_num in temp_info["frame_time_map"]:
                # CIF文件路径（根据你的命名规则）
                cif_path = os.path.join(temp_path, f"Frame_{frame_num}.cif")
                if not os.path.exists(cif_path):
                    print(f"警告：CIF文件不存在 → {cif_path}")
                    continue

                # 加载结构并计算RDF/n(r)
                try:
                    structure = Structure.from_file(cif_path)
                    r_nr, nr, r_gr, gr = calculate_oh_rdf_numpy(
                        structure, center_element, atom_type, dr, max_r
                    )
                    if len(nr) == 0 or len(gr) == 0:
                        continue

                    # 平滑后的值用于确定范围
                    nr_smooth = moving_average(nr, 5)
                    gr_smooth = moving_average(gr, 5)
                    max_values[atom_type]['nr'] = max(max_values[atom_type]['nr'], np.max(nr_smooth))
                    max_values[atom_type]['gr'] = max(max_values[atom_type]['gr'], np.max(gr_smooth))
                except Exception as e:
                    print(f"处理 {cif_path} 出错 → {e}")
                    continue

    # 第二步：绘制所有图像
    for temp, temp_info in temperature_folders.items():
        temp_path = os.path.join(base_data_root, temp_info["folder"])
        if not os.path.isdir(temp_path):
            continue

        for atom_type, (label, color) in atom_mapping.items():
            # 获取统一的Y轴范围（向上取整）
            y_nr_max = np.ceil(max_values[atom_type]['nr'] * 2) / 2
            y_gr_max = np.ceil(max_values[atom_type]['gr'] * 2) / 2
            y_limits = (0, y_nr_max, 0, y_gr_max)

            for frame_num, time_point in temp_info["frame_time_map"].items():
                cif_path = os.path.join(temp_path, f"Frame_{frame_num}.cif")
                if not os.path.exists(cif_path):
                    continue

                try:
                    structure = Structure.from_file(cif_path)
                    r_nr, nr, r_gr, gr = calculate_oh_rdf_numpy(
                        structure, center_element, atom_type, dr, max_r
                    )
                    if len(nr) == 0 or len(gr) == 0:
                        continue

                    # 绘制图像
                    plt, _ = plot_rdf_nr_data(
                        r_nr, nr, r_gr, gr, color, label,
                        temp, atom_type, time_point, y_limits
                    )
                    plt.show()
                except Exception as e:
                    print(f"绘图出错 → {e}")
                    continue


if __name__ == "__main__":
    main()
