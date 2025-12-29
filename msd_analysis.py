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

DOWNSAMPLE = 100
AVG_DOWNSAMPLE = 10
CHUNK_SIZE = 500000

# ================= 终极优化：高精度双端扫描 + 线性度约束 =================
def auto_select_fit_range_precision(time_ns, msd_x, msd_y, msd_z, t_max=10.0):
    """
    MODIFIED based on feedback:
    1. Prioritizes finding a window where the average slope of the log-log plot is closest to 1.0,
       which is characteristic of the diffusive regime.
    2. Starts the search at t > 1.0 ns to exclude the initial, non-diffusive transient regime.
    3. Uses slope variance and R-squared as secondary metrics to ensure a high-quality, consistent fit.
    """
    best_start, best_end = 1.0, 9.5
    min_score = float('inf')

    # Start search later in time (t > 1 ns) to find the diffusive regime.
    for t_start in np.arange(1.0, 6.6, 0.1):
        # Scan with a window of 2.5ns to 6ns length.
        for t_end in np.arange(t_start + 2.5, min(t_start + 6.0, t_max + 0.1), 0.1):

            mask = (time_ns >= t_start) & (time_ns <= t_end)
            slopes = []
            r_sq_list = []

            for msd_data in [msd_x, msd_y, msd_z]:
                # Pre-process: zero the MSD
                ay_v = msd_data - msd_data[0]
                valid = (time_ns[mask] > 0) & (ay_v[mask] > 0)

                if np.sum(valid) < 10: # Ensure enough data points for a stable fit
                    slopes.append(999); r_sq_list.append(0); continue

                log_t = np.log10(time_ns[mask][valid])
                log_msd = np.log10(ay_v[mask][valid])

                # Perform linear fit on log-log data
                coeffs, residuals, _, _, _ = np.polyfit(log_t, log_msd, 1, full=True)

                # Calculate R² to check linearity
                y_mean = np.mean(log_msd)
                ss_tot = np.sum((log_msd - y_mean)**2)
                r_sq = 1 - (residuals[0] / ss_tot) if ss_tot > 0 else 0

                slopes.append(coeffs[0])
                r_sq_list.append(r_sq)

            if 999 in slopes: continue # Skip windows with invalid fits

            # --- REFINED SCORING LOGIC ---
            # Primary goal: Minimize the deviation of EACH individual slope from the ideal value of 1.
            # This is a more robust way to ensure all components are in the diffusive regime.
            # We use Mean Squared Error for this penalty.

            slopes_array = np.array(slopes)
            slope_mse = np.mean((slopes_array - 1.0)**2) # Penalize deviation of *each* slope from 1

            avg_r_sq = np.mean(r_sq_list)

            # Weights to prioritize finding slopes near 1.
            W_slope_mse = 10.0
            W_r_squared = 1.0

            # Score = Weighted sum of penalties. Lower is better.
            # The variance is implicitly penalized by the MSE term, so we can simplify the score.
            score = (slope_mse * W_slope_mse) + ((1 - avg_r_sq) * W_r_squared)

            if score < min_score:
                min_score = score
                best_start, best_end = t_start, t_end

    return (best_start, best_end)

# ================= 样式函数 =================
def apply_beauty_style(ax, title_text="", ylabel_type="total", is_first=False):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-2, 10); ax.set_ylim(1e-1, 5e3)
    ax.set_xlabel('Time (ns)', fontsize=38, fontweight='bold', labelpad=10)

    if is_first:
        label = r'MSD$_{\mathbf{%s}}$ ($\mathbf{\mathring{A}^2}$)' % ylabel_type.lower() if ylabel_type in COMPONENTS else r'MSD ($\mathbf{\mathring{A}^2}$)'
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

    data_cache = {}
    dynamic_fit_ranges = {}

    # 1. 地毯式搜索最优同步窗口
    for temp in TEMPS:
        print(f"正在对 {temp}K 进行高精度同步时窗搜索...")

        f_avg = os.path.join(ROOT_TRI, f"Average_MSD_{temp}K.csv")
        if os.path.exists(f_avg):
            df_avg = pd.read_csv(f_avg)
            common_x = df_avg.iloc[:, 0].values / 1000.0

            # 使用高精度版本
            best_range = auto_select_fit_range_precision(common_x, df_avg['X_MSD'].values,
                                                        df_avg['Y_MSD'].values, df_avg['Z_MSD'].values)
            dynamic_fit_ranges[temp] = best_range
            print(f"  > 最佳同步区间: {best_range[0]:.2f} - {best_range[1]:.2f} ns")

            data_cache[temp] = {'avg': {}}
            for c in COMPONENTS:
                data_cache[temp]['avg'][f'x_{c}'] = common_x[::AVG_DOWNSAMPLE]
                data_cache[temp]['avg'][f'y_{c}'] = df_avg[f'{c}_MSD'].values[::AVG_DOWNSAMPLE]

    # --- 绘图 ---
    print("\n" + "="*65)
    print(f"{'Temp':<6} | {'Comp':<5} | {'Slope (alpha)':<14} | {'D (cm^2/s)':<15}")
    print("-"*65)

    fig, axes = plt.subplots(1, 3, figsize=(24, 18), sharey=True)
    plt.subplots_adjust(wspace=0.1, left=0.08, right=0.98, bottom=0.1, top=0.95)

    for i, temp in enumerate(TEMPS):
        ax = axes[i]
        fit_start, fit_end = dynamic_fit_ranges[temp]

        for comp in COMPONENTS:
            ax_v = np.array(data_cache[temp]['avg'][f'x_{comp}'])
            ay_v = np.array(data_cache[temp]['avg'][f'y_{comp}']) - data_cache[temp]['avg'][f'y_{comp}'][0]

            ax.plot(ax_v, ay_v, color=COMP_COLORS[comp], linewidth=10, label=f'Avg {comp}', zorder=5)

            mask = (ax_v >= fit_start) & (ax_v <= fit_end) & (ay_v > 0)
            if any(mask):
                slope, intercept = np.polyfit(np.log10(ax_v[mask]), np.log10(ay_v[mask]), 1)
                d_comp = (10**intercept / 2.0) * 1e-7
                print(f"{temp:<6} | {comp:<5} | {slope:<14.4f} | {d_comp:.4e}")

                # 绘制反向延长拟合虚线
                fit_x_plot = np.logspace(np.log10(1e-2), np.log10(fit_end), 100)
                fit_y_plot = 10**(slope * np.log10(fit_x_plot) + intercept)
                ax.plot(fit_x_plot, fit_y_plot, '--', color='black', linewidth=4, alpha=0.6, zorder=10)

        ax.legend(loc='lower right', fontsize=26)
        apply_beauty_style(ax, title_text=f'T={temp}K', is_first=(i==0))

    print("="*65)
    plt.show()

if __name__ == "__main__":
    main()