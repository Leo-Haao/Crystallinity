# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 配置区 =================
ROOT_TRI = r"E:\Materials Studio Projects\PEO_project_1_2_Files\Documents\Tri_comb\Crystal"
TEMPS = [500, 600, 700]
COMPONENTS = ['X', 'Y', 'Z']
COMP_COLORS = {'X': '#2980b9', 'Y': '#c0392b', 'Z': '#27ae60'}

AVG_DOWNSAMPLE = 10

# ================= 核心：单分量独立搜索算法 =================
def find_best_window_for_component(time_ns, msd_data, t_max=10.0):
    """
    为单个分量寻找斜率最接近 1.0 的线性区间。
    """
    best_start, best_end = 5.0, 10.0
    min_slope_err = float('inf')

    # 排除起始瞬态 (t < 1.0 ns)
    search_starts = np.arange(1.0, 6.1, 0.1)

    for t_start in search_starts:
        # 窗口长度至少 3ns 以保证统计可靠性
        for t_end in np.arange(t_start + 3.0, t_max + 0.01, 0.1):
            mask = (time_ns >= t_start) & (time_ns <= t_end)

            # 归零处理
            ay_v = msd_data - msd_data[0]
            valid = (time_ns[mask] > 0) & (ay_v[mask] > 0)

            if np.sum(valid) < 15: continue

            log_t = np.log10(time_ns[mask][valid])
            log_msd = np.log10(ay_v[mask][valid])

            # 线性拟合
            slope, intercept = np.polyfit(log_t, log_msd, 1)

            # 评分：越接近 1 越好
            slope_err = abs(slope - 1.0)

            if slope_err < min_slope_err:
                min_slope_err = slope_err
                best_start, best_end = t_start, t_end

    return (best_start, best_end)

# ================= 样式函数 (保持原有格式) =================
def apply_beauty_style(ax, title_text="", is_first=False):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-2, 10); ax.set_ylim(1e-1, 5e3)
    ax.set_xlabel('Time (ns)', fontsize=38, fontweight='bold', labelpad=10)

    if is_first:
        label = r'MSD ($\mathbf{\mathring{A}^2}$)'
        ax.set_ylabel(label, fontsize=38, fontweight='bold', labelpad=10)
        ax.tick_params(axis='y', labelleft=True)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis='y', labelleft=False)

    ax.tick_params(axis='both', direction='in', which='major', length=15, width=4,
                   labelsize=30, pad=10, top=False, right=False)
    ax.tick_params(axis='both', which='minor', direction='in', length=8, width=2,
                   top=False, right=False)
    for spine in ax.spines.values(): spine.set_linewidth(4)

    if title_text:
        ax.text(0.05, 0.9, title_text, transform=ax.transAxes, fontsize=32, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

def main():
    plt.rc('font', family='Times New Roman', weight='bold')
    plt.rcParams.update({'mathtext.fontset': 'stix', 'mathtext.default': 'it'})

    print("\n" + "="*85)
    print(f"{'Temp':<6} | {'Comp':<5} | {'Best Range (ns)':<18} | {'Slope (a)':<10} | {'D (cm^2/s)'}")
    print("-"*85)

    fig, axes = plt.subplots(1, 3, figsize=(24, 18), sharey=True)
    plt.subplots_adjust(wspace=0.1, left=0.08, right=0.98, bottom=0.1, top=0.95)

    for i, temp in enumerate(TEMPS):
        ax = axes[i]
        f_avg = os.path.join(ROOT_TRI, f"Average_MSD_{temp}K.csv")

        if os.path.exists(f_avg):
            df_avg = pd.read_csv(f_avg)
            time_raw = df_avg.iloc[:, 0].values / 1000.0

            for comp in COMPONENTS:
                msd_raw = df_avg[f'{comp}_MSD'].values

                # 1. 为每个分量寻找独立的黄金窗口
                t_start, t_end = find_best_window_for_component(time_raw, msd_raw)

                # 2. 准备绘图数据 (Downsample)
                time_plot = time_raw[::AVG_DOWNSAMPLE]
                msd_plot = (msd_raw - msd_raw[0])[::AVG_DOWNSAMPLE]

                ax.plot(time_plot, msd_plot, color=COMP_COLORS[comp], linewidth=10, label=f'Avg {comp}', zorder=5)

                # 3. 在其专属窗口内进行拟合
                mask = (time_raw >= t_start) & (time_raw <= t_end)
                ay_v = msd_raw - msd_raw[0]
                valid = (time_raw[mask] > 0) & (ay_v[mask] > 0)

                log_t = np.log10(time_raw[mask][valid])
                log_msd = np.log10(ay_v[mask][valid])
                slope, intercept = np.polyfit(log_t, log_msd, 1)

                d_comp = (10**intercept / 2.0) * 1e-7
                print(f"{temp:<6} | {comp:<5} | {t_start:>5.1f} - {t_end:<5.1f} | {slope:<10.4f} | {d_comp:.4e}")

                # 4. 绘制对应的拟合虚线
                fit_x = np.logspace(np.log10(t_start), np.log10(t_end), 50)
                fit_y = 10**(slope * np.log10(fit_x) + intercept)
                ax.plot(fit_x, fit_y, '--', color='black', linewidth=4, alpha=0.7, zorder=10)

        ax.legend(loc='lower right', fontsize=26)
        apply_beauty_style(ax, title_text=f'T={temp}K', is_first=(i==0))

    print("="*85)
    plt.show()

if __name__ == "__main__":
    main()