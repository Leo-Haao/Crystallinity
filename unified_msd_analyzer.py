import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import pandas as pd
from scipy import stats
from matplotlib.font_manager import FontProperties

# ----------------- 常量定义 -----------------
k_boltzmann = 1.38064852e-23
k_boltzmann_eV = 8.617333262e-5


# ----------------- 绘图样式类 -----------------
class PlotProperties:
    def __init__(self, font_type='Times New Roman', font_size=26,
                 axis_ticks_font_size=24, label_x="", label_y="",
                 legend_size=18, xlimit=8, ylimit=6):
        self.font_type = font_type
        self.font_size = font_size
        self.axis_ticks_font_size = axis_ticks_font_size
        self.label_x = label_x
        self.label_y = label_y
        self.legend_size = legend_size
        self.xlimit = xlimit
        self.ylimit = ylimit

    def apply_style(self):
        plt.figure(figsize=(self.xlimit, self.ylimit))
        ax = plt.gca()

        # 设置字体风格
        plt.rc('font', family=self.font_type, size=self.legend_size, weight='bold')
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = self.font_type
        plt.rcParams['mathtext.it'] = f'{self.font_type}:italic'
        plt.rcParams['mathtext.bf'] = f'{self.font_type}:bold'

        # 坐标轴标签样式
        axis_font = {
            'fontname': self.font_type,
            'size': self.font_size,
            'fontweight': 'bold'
        }
        plt.xlabel(self.label_x, **axis_font)
        plt.ylabel(self.label_y, **axis_font)

        # 坐标轴刻度样式
        axis_ticks_font = {
            'fontname': self.font_type,
            'size': self.axis_ticks_font_size,
            'fontweight': 'bold'
        }
        plt.xticks(**axis_ticks_font)
        plt.yticks(**axis_ticks_font)

        # 设置轴线内向刻度
        ax.get_xaxis().set_tick_params(direction='in', width=2, length=6, top='on')
        ax.get_yaxis().set_tick_params(direction='in', width=2, length=6, right='on')

        # 设置轴线粗细
        thickness = 2
        ax.spines['top'].set_linewidth(thickness)
        ax.spines['right'].set_linewidth(thickness)
        ax.spines['left'].set_linewidth(thickness)
        ax.spines['bottom'].set_linewidth(thickness)

        plt.tight_layout()
        return plt, ax


# ----------------- MSD数据绘图类 -----------------
class DataPlotter:
    """
    统一的 MSD 数据绘图类，支持科学计数法Y轴格式，并精确控制刻度为一位小数。
    - 科学计数法标签位于图的左上角。
    - 图例位于图的右上角。
    """
    def __init__(self, temperatures, msd_x, msd_y, msd_z):
        self.temperatures = temperatures
        self.msd_x = msd_x
        self.msd_y = msd_y
        self.msd_z = msd_z
        self.results = self._calculate_msd_ratios()

    def _calculate_msd_ratios(self):
        # 防止除以零
        msd_y_safe = np.where(self.msd_y == 0, 1e-300, self.msd_y)
        msd_z_safe = np.where(self.msd_z == 0, 1e-300, self.msd_z)

        ratio_xy = np.array(self.msd_x) / msd_y_safe
        ratio_xz = np.array(self.msd_x) / msd_z_safe
        ratio_yz = np.array(self.msd_y) / msd_z_safe

        results = []
        for i, T in enumerate(self.temperatures):
            results.append({
                'temperature': T,
                'ratio_xy': ratio_xy[i],
                'ratio_xz': ratio_xz[i],
                'ratio_yz': ratio_yz[i]
            })
        return results

    def plot_msd_temperature(self, title="MSD vs Temperature", filename=None):
        plot_style = PlotProperties(
            label_x="Temperature (K)",
            label_y=r"MSD (m²)",
            xlimit=8,
            ylimit=6
        )
        plt, ax = plot_style.apply_style()

        y_max = max(max(self.msd_x), max(self.msd_y), max(self.msd_z)) * 1.1

        # 设置 X 轴刻度和范围
        plt.xlim(min(self.temperatures) - 50, max(self.temperatures) + 50)
        ax.set_xticks(self.temperatures)

        # 统一的 Y 轴科学计数法格式
        if y_max > 0:
            # 找到最大的 10 的幂
            max_exponent = np.floor(np.log10(y_max))

            # 左上角添加科学计数法标签
            ax.text(0.01, 1.01, f"1e{int(max_exponent):+d}",
                    transform=ax.transAxes,
                    verticalalignment='bottom', horizontalalignment='left',
                    fontsize=plot_style.axis_ticks_font_size)

            # 归一化 MSD 值
            msd_x_scaled = self.msd_x / (10 ** max_exponent)
            msd_y_scaled = self.msd_y / (10 ** max_exponent)
            msd_z_scaled = self.msd_z / (10 ** max_exponent)

            # 格式化 Y 轴刻度为一位小数
            def y_tick_formatter(y, pos):
                return f"{y:.1f}"

            ax.yaxis.set_major_formatter(FuncFormatter(y_tick_formatter))

            # 设置 Y 轴范围和刻度
            y_max_scaled = max(msd_x_scaled.max(), msd_y_scaled.max(), msd_z_scaled.max()) * 1.5
            plt.ylim(0, y_max_scaled)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))

        else:
            msd_x_scaled, msd_y_scaled, msd_z_scaled = self.msd_x, self.msd_y, self.msd_z
            plt.ylim(0, 1)

        # 绘制各方向MSD曲线
        plt.plot(self.temperatures, msd_x_scaled, 'b-o', linewidth=2.5, label='MSD x [100]')
        plt.plot(self.temperatures, msd_y_scaled, 'r--o', linewidth=2.5, label='MSD y [010]')
        plt.plot(self.temperatures, msd_z_scaled, 'g-.o', linewidth=2.5, label='MSD z [001]')

        # 图例设置
        plt.legend(fontsize=plot_style.legend_size, frameon=False, loc='upper right')

        plt.title(title, fontname=plot_style.font_type, size=22, weight='bold', pad=20)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        return plt

    def plot_msd_ratios(self, title="MSD Ratios vs Temperature", filename=None):
        temps = [r['temperature'] for r in self.results]
        ratio_xy = [r['ratio_xy'] for r in self.results]
        ratio_xz = [r['ratio_xz'] for r in self.results]
        ratio_yz = [r['ratio_yz'] for r in self.results]

        plot_style = PlotProperties(
            label_x="Temperature (K)",
            label_y="MSD Ratio",
            xlimit=8,
            ylimit=6
        )
        plt, ax = plot_style.apply_style()

        x = np.arange(len(temps))
        width = 0.25

        # 绘制比值柱状图
        ax.bar(x - width, ratio_xy, width, label='MSD x / MSD y', color='b')
        ax.bar(x, ratio_xz, width, label='MSD x / MSD z', color='r')
        ax.bar(x + width, ratio_yz, width, label='MSD y / MSD z', color='g')

        ax.set_xticks(x)
        ax.set_xticklabels(temps)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_ylim(0, 1.8)
        # 图例设置
        plt.legend(fontsize=plot_style.legend_size, frameon=False, loc='upper right')

        plt.title(title, fontname=plot_style.font_type, size=22, weight='bold', pad=20)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        return plt

# ----------------- 阿伦尼乌斯拟合与绘图 (Total D) -----------------
def plot_total_arrhenius(temperatures, D, title, extrap_temp, filename=None):
    inv_temp = 1 / temperatures
    ln_D = np.log(D)

    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_temp, ln_D)
    Ea = -slope * k_boltzmann_eV
    D0 = np.exp(intercept)

    inv_T_extrap = 1 / extrap_temp
    ln_D_extrap = slope * inv_T_extrap + intercept
    D_extrap = np.exp(ln_D_extrap)

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    bold_font = FontProperties(weight='bold', size=16)

    plt.scatter(inv_temp * 1000, ln_D, color='blue', s=100, marker='o', label='Original Data')

    all_inv_temps = np.concatenate((inv_temp, [inv_T_extrap]))
    x_fit = np.linspace(all_inv_temps.min(), all_inv_temps.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit * 1000, y_fit, 'r-', linewidth=2.5, label=f'Fitting Line (R²={r_value ** 2:.4f})')

    plt.scatter(inv_T_extrap * 1000, ln_D_extrap, color='green', s=120, marker='*',
                label=f'Extrapolated at {extrap_temp}K: D={D_extrap:.2e} cm²/s')

    ax = plt.gca()
    ax.set_xlabel(r'1/T × 10⁻³ (K⁻¹)', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'ln(D) (cm²/s)', fontsize=16, fontweight='bold')
    ax.tick_params(direction='in', width=2, length=6, which='both', top=True, right=True)

    text_str = (f'Arrhenius Equation: ln(D) = ln(D₀) - Eₐ/(k_B T)\n'
                f'Activation Energy Eₐ = {Ea:.4f} eV\n'
                f'Pre-exponential Factor D₀ = {D0:.4e} cm²/s')
    plt.text(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=17,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
             fontweight='bold')

    plt.legend(prop=bold_font, frameon=False, loc='upper right')
    plt.title(title, fontsize=16, fontweight='bold', pad=10)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"=== {title} 拟合与外推结果 ===")
    print(f"拟合得到的活化能 Ea = {Ea:.4f} eV")
    print(f"指前因子 D0 = {D0:.4e} cm²/s")
    print(f"{extrap_temp}K外推得到的扩散系数 D = {D_extrap:.4e} cm²/s")

    return plt

# ----------------- 阿伦尼乌斯拟合与绘图 (Directional D) -----------------
def fit_and_plot_directional(directions_data, title, extrap_temp, kB=8.617333262e-5, filename=None):
    colors = {'X': 'blue', 'Y': 'red', 'Z': 'green'}
    labels = {'X': 'X[100]', 'Y': 'Y[010]', 'Z': 'Z[001]'}
    Ea_dict, D0_dict, D_extrap_dict = {}, {}, {}

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    bold_font = FontProperties(weight='bold', size=14)

    for dir_name, (T_list, D_list) in directions_data.items():
        inv_T = 1 / np.array(T_list)
        ln_D = np.log(np.array(D_list))

        slope, intercept, r_value, _, _ = stats.linregress(inv_T, ln_D)
        Ea = -slope * kB
        D0 = np.exp(intercept)
        Ea_dict[dir_name] = Ea
        D0_dict[dir_name] = D0

        inv_T_extrap = 1 / extrap_temp
        ln_D_extrap = slope * inv_T_extrap + intercept
        D_extrap = np.exp(ln_D_extrap)
        D_extrap_dict[dir_name] = D_extrap

        ax.scatter(inv_T * 1000, ln_D, color=colors[dir_name], s=100, marker='o',
                   label=f'{labels[dir_name]} Original Data')

        T_fit = np.linspace(
            min(min(T_list), extrap_temp),
            max(max(T_list), extrap_temp),
            100
        )
        inv_T_fit = 1 / T_fit
        ln_D_fit = slope * inv_T_fit + intercept
        ax.plot(inv_T_fit * 1000, ln_D_fit, color=colors[dir_name], linewidth=2.5,
                label=f'{labels[dir_name]} Fit (R²={r_value ** 2:.4f})')

        ax.scatter(inv_T_extrap * 1000, ln_D_extrap, color=colors[dir_name], s=120, marker='*',
                   label=f'{labels[dir_name]} Extrap at {extrap_temp}K: D={D_extrap:.2e} cm²/s')

    ax.set_xlabel(r'1/T × 10⁻³ (K⁻¹)', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'ln(D) (cm²/s)', fontsize=16, fontweight='bold')
    ax.tick_params(direction='in', width=2, length=6, which='both', top=True, right=True)
    ax.legend(prop=bold_font, frameon=False, loc='lower left')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n=== {title} 拟合与外推结果 ===")
    for dir_name in directions_data.keys():
        print(f"{labels[dir_name]}:")
        print(f"  活化能 Eₐ = {Ea_dict[dir_name]:.4f} eV")
        print(f"  指前因子 D₀ = {D0_dict[dir_name]:.4e} cm²/s")
        print(f"  {extrap_temp}K外推扩散系数 D = {D_extrap_dict[dir_name]:.4e} cm²/s")

    return plt

if __name__ == "__main__":
    # --- Crystalline Data ---
    crystal_temps = np.array([500, 600, 700])
    # MSD data in Å²
    crystal_msd_x_A2 = np.array([59.100, 415.000, 1100.000])
    crystal_msd_y_A2 = np.array([36.200, 261.000, 1329.000])
    crystal_msd_z_A2 = np.array([355.000, 1213.000, 3121.000])
    # Convert Å² to m²
    crystal_msd_x = crystal_msd_x_A2 * 1e-20
    crystal_msd_y = crystal_msd_y_A2 * 1e-20
    crystal_msd_z = crystal_msd_z_A2 * 1e-20

    # Diffusion coefficient data in cm²/s
    crystal_D_total = np.array([8.8150E-07, 4.0072E-06, 1.0754E-05])
    crystal_D_directions = {
        'X': (crystal_temps, np.array([3.6650E-07, 2.7800E-06, 5.5995E-06])),
        'Y': (crystal_temps, np.array([2.4150E-07, 1.8040E-06, 8.2185E-06])),
        'Z': (crystal_temps, np.array([2.0360E-06, 7.4375E-06, 1.8445E-05]))
    }

    # --- Amorphous Data ---
    amorphous_temps = np.array([300, 500, 600])
    # MSD data in Å²
    amorphous_msd_x_A2 = np.array([100.464, 1407.869, 4137.335])
    amorphous_msd_y_A2 = np.array([67.429, 1514.262, 3135.327])
    amorphous_msd_z_A2 = np.array([91.611, 1810.520, 3327.161])
    # Convert Å² to m²
    amorphous_msd_x = amorphous_msd_x_A2 * 1e-20
    amorphous_msd_y = amorphous_msd_y_A2 * 1e-20
    amorphous_msd_z = amorphous_msd_z_A2 * 1e-20

    # Diffusion coefficient data in cm²/s
    amorphous_D_total = np.array([4.870E-07, 9.122E-06, 2.201E-05])

    # Calculate missing D for Amorphous Z at 600K
    # D_sum(600K) = 6.603E-05, D_x(600K) = 2.702E-05, D_y(600K) = 1.757E-05
    d_z_600k = 6.603E-05 - 2.702E-05 - 1.757E-05

    amorphous_D_directions = {
        'X': (amorphous_temps, np.array([6.180E-07, 7.679E-06, 2.702E-05])),
        'Y': (amorphous_temps, np.array([3.855E-07, 9.213E-06, 1.757E-05])),
        'Z': (amorphous_temps, np.array([4.575E-07, 1.047E-05, d_z_600k]))
    }

    # --- Analysis and Plotting ---

    # Crystalline plots
    crystal_plotter = DataPlotter(crystal_temps, crystal_msd_x, crystal_msd_y, crystal_msd_z)
    crystal_plotter.plot_msd_temperature(title="Crystal MSD vs Temperature", filename="crystalline_msd_vs_temp.png")
    crystal_plotter.plot_msd_ratios(title="Crystal MSD Ratios vs Temperature", filename="crystalline_msd_ratios.png")
    plot_total_arrhenius(crystal_temps, crystal_D_total, "Arrhenius Fitting of Crystal (Total D)", 300, filename="crystalline_arrhenius_total.png")
    fit_and_plot_directional(crystal_D_directions, "Arrhenius Fitting of Crystal", 300, filename="crystalline_arrhenius_directional.png")

    # Amorphous plots
    amorphous_plotter = DataPlotter(amorphous_temps, amorphous_msd_x, amorphous_msd_y, amorphous_msd_z)
    amorphous_plotter.plot_msd_temperature(title="Amorphous MSD vs Temperature", filename="amorphous_msd_vs_temp.png")
    amorphous_plotter.plot_msd_ratios(title="Amorphous MSD Ratios vs Temperature", filename="amorphous_msd_ratios.png")
    plot_total_arrhenius(amorphous_temps, amorphous_D_total, "Arrhenius Fitting of Amorphous (Total D)", 700, filename="amorphous_arrhenius_total.png")
    fit_and_plot_directional(amorphous_D_directions, "Arrhenius Fitting of Amorphous", 700, filename="amorphous_arrhenius_directional.png")

    print("Script finished. All plots have been saved as PNG files.")
