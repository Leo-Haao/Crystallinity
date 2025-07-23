#Design Parameters: Pre - exponential factor D0 = 1e-09 m²/s, Activation energy in x - direction Ea_x = 0.5 eV, Activation energy in y - direction Ea_y = 0.55 eV, Activation energy in z - direction Ea_z = 0.6 eV
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from matplotlib.ticker import MaxNLocator, FuncFormatter
import pandas as pd

#Defining the Boltzmann constant
k_boltzmann = 1.38064852e-23
k_boltzmann_eV = 8.617333262e-5
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

        plt.rc('font', family=self.font_type, size=self.legend_size, weight='bold')
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = self.font_type
        plt.rcParams['mathtext.it'] = f'{self.font_type}:italic'
        plt.rcParams['mathtext.bf'] = f'{self.font_type}:bold'

        axis_font = {
            'fontname': self.font_type,
            'size': self.font_size,
            'fontweight': 'bold'
        }
        plt.xlabel(self.label_x, **axis_font)
        plt.ylabel(self.label_y, **axis_font)

        axis_ticks_font = {
            'fontname': self.font_type,
            'size': self.axis_ticks_font_size,
            'fontweight': 'bold'
        }
        plt.xticks(**axis_ticks_font)
        plt.yticks(** axis_ticks_font)
        ax.get_xaxis().set_tick_params(direction='in', width=2, length=6, top='on')
        ax.get_yaxis().set_tick_params(direction='in', width=2, length=6, right='on')

        thickness = 2
        ax.spines['top'].set_linewidth(thickness)
        ax.spines['right'].set_linewidth(thickness)
        ax.spines['left'].set_linewidth(thickness)
        ax.spines['bottom'].set_linewidth(thickness)

        plt.tight_layout()
        return plt, ax

#Calculate the mean square displacement (MSD) and its ratio in three directions (x/y/z) at different temperatures
class DiffusionModel:
    def __init__(self, D0, Ea_x, Ea_y, Ea_z):
        self.D0 = D0
        self.Ea_x = Ea_x
        self.Ea_y = Ea_y
        self.Ea_z = Ea_z

    def diffusion_coefficient(self, T, Ea):
        exponent = -Ea / (k_boltzmann_eV * T)
        exponent = np.clip(exponent, -700, 700)
        return self.D0 * np.exp(exponent)

    def msd(self, T, Ea, time=1.0):
        D = self.diffusion_coefficient(T, Ea)
        return 6 * D * time

    def calculate_msd_ratios(self, temperatures):
        results = []
        for T in temperatures:
            msd_x = self.msd(T, self.Ea_x)
            msd_y = self.msd(T, self.Ea_y)
            msd_z = self.msd(T, self.Ea_z)
            results.append({
                'temperature': T,
                'msd_x': msd_x,
                'msd_y': msd_y,
                'msd_z': msd_z,
                'ratio_xy': msd_x / msd_y,
                'ratio_xz': msd_x / msd_z,
                'ratio_yz': msd_y / msd_z
            })
        return results

    def plot_msd_temperature(self, temperatures, title="MSD vs Temperature"):
        msd_x = np.array([self.msd(T, self.Ea_x) for T in temperatures])
        msd_y = np.array([self.msd(T, self.Ea_y) for T in temperatures])
        msd_z = np.array([self.msd(T, self.Ea_z) for T in temperatures])

        plot_style = PlotProperties(
            label_x="Temperature (K)",
            label_y="MSD (m²)",
            xlimit=8,
            ylimit=6
        )
        plt, ax = plot_style.apply_style()

        y_max = max([msd_x.max(), msd_y.max(), msd_z.max()]) * 1.1
        plt.ylim(0, y_max)

        plt.plot(temperatures, msd_x, 'b-o', linewidth=2.5, label='MSD_x')
        plt.plot(temperatures, msd_y, 'r--o', linewidth=2.5, label='MSD_y')
        plt.plot(temperatures, msd_z, 'g-.o', linewidth=2.5, label='MSD_z')

        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        plt.legend(fontsize=plot_style.legend_size, frameon=False, loc='upper left')

        plt.title(title, fontname=plot_style.font_type, size=22, weight='bold', pad=20)
        return plt

    def plot_msd_ratios(self, temperatures, title="MSD Ratios vs Temperature"):
        results = self.calculate_msd_ratios(temperatures)
        temps = [r['temperature'] for r in results]
        ratio_xy = [r['ratio_xy'] for r in results]
        ratio_xz = [r['ratio_xz'] for r in results]
        ratio_yz = [r['ratio_yz'] for r in results]

        plot_style = PlotProperties(
            label_x="Temperature (K)",
            label_y="MSD Ratio",
            xlimit=8,
            ylimit=6
        )
        plt, ax = plot_style.apply_style()

        x = np.arange(len(temps))
        width = 0.25

        rects1 = ax.bar(x - width, ratio_xy, width, label='MSD_x / MSD_y', color='b')
        rects2 = ax.bar(x, ratio_xz, width, label='MSD_x / MSD_z', color='r')
        rects3 = ax.bar(x + width, ratio_yz, width, label='MSD_y / MSD_z', color='g')

        ax.set_xticks(x)
        ax.set_xticklabels(temps)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        plt.legend(fontsize=plot_style.legend_size, frameon=False, loc='upper right')
        plt.title(title, fontname=plot_style.font_type, size=22, weight='bold', pad=20)
        return plt

#Back-calculation of activation energies from MSD data (by fitting the Arrhenius equation) to verify accuracy
class ActivationEnergyExtractor:
    @staticmethod
    def arrhenius_equation(T, D0, Ea):
        exponent = -Ea / (k_boltzmann_eV * T)
        exponent = np.clip(exponent, -700, 700)
        return D0 * np.exp(exponent)

    @staticmethod
    def msd_to_diffusion(msd_values, time=1.0):
        msd_array = np.array(msd_values)
        msd_array = np.maximum(msd_array, 1e-300)
        return msd_array / (6 * time)

    def extract_activation_energy(self, temperatures, msd_values, time=1.0):
        diffusion_coeffs = self.msd_to_diffusion(msd_values, time)

        valid_indices = diffusion_coeffs > 0
        if not np.any(valid_indices):
            raise ValueError("All diffusion coefficient values are zero or negative, cannot fit")

        valid_temps = np.array(temperatures)[valid_indices]
        valid_diff = diffusion_coeffs[valid_indices]

        try:
            popt, pcov = curve_fit(
                self.arrhenius_equation,
                valid_temps,
                valid_diff,
                p0=[1e-10, 0.5]
            )
        except RuntimeError as e:
            print(f"Fitting failed: {e}")
            return {'D0': 1e-10, 'Ea': 0.5, 'D0_error': np.nan, 'Ea_error': np.nan}

        D0_fit, Ea_fit = popt
        perr = np.sqrt(np.diag(pcov))

        return {
            'D0': D0_fit,
            'Ea': Ea_fit,
            'D0_error': perr[0],
            'Ea_error': perr[1]
        }

    def plot_activation_energy_fit(self, temperatures, msd_values, direction='x', time=1.0):
        diffusion_coeffs = self.msd_to_diffusion(msd_values, time)
        result = self.extract_activation_energy(temperatures, msd_values, time)

        T_fit = np.linspace(min(temperatures), max(temperatures), 200)
        D_fit = self.arrhenius_equation(T_fit, result['D0'], result['Ea'])

        plot_style = PlotProperties(
            label_x="1/T (1/K)",
            label_y="ln(D)",
            xlimit=8,
            ylimit=6
        )
        plt, ax = plot_style.apply_style()

        valid_indices = diffusion_coeffs > 0
        valid_temps = np.array(temperatures)[valid_indices]
        valid_diff = diffusion_coeffs[valid_indices]

        plt.scatter(1 / valid_temps, np.log(valid_diff),
                    color='k', s=100, edgecolors='r', linewidth=2, label='Data')
        plt.plot(1 / T_fit, np.log(D_fit), 'r-', linewidth=2.5,
                 label=f'Fit: Ea = {result["Ea"]:.4f} eV')

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        plt.legend(fontsize=plot_style.legend_size, frameon=False, loc='best')
        plt.title(f'Arrhenius Plot ({direction}-direction)',
                  fontname=plot_style.font_type, size=22, weight='bold', pad=20)
        return plt


if __name__ == "__main__":
    D0 = 1e-9
    Ea_x = 0.5
    Ea_y = 0.55
    Ea_z = 0.6

    print(f"Design Parameters: Pre - exponential factor D0 = {D0} m²/s, Activation energy in x - direction Ea_x = {Ea_x} eV, Activation energy in y - direction Ea_y = {Ea_y} eV, Activation energy in z - direction Ea_z = {Ea_z} eV\n")

    temperatures = [300, 400, 500, 600, 700, 800, 900]

    model = DiffusionModel(D0, Ea_x, Ea_y, Ea_z)

    results = model.calculate_msd_ratios(temperatures)
    print("MSD Calculation Results:")
    for res in results:
        print(f"\nTemperature: {res['temperature']} K")
        print(f"MSD_x: {res['msd_x']:.2e} m²")
        print(f"MSD_y: {res['msd_y']:.2e} m²")
        print(f"MSD_z: {res['msd_z']:.2e} m²")
        print(f"Ratio (x/y): {res['ratio_xy']:.4f}; (x/z): {res['ratio_xz']:.4f}; (y/z): {res['ratio_yz']:.4f}")

    table_data = {
        "Temperature (K)": [res['temperature'] for res in results],
        "MSD_x (m²)": [f"{res['msd_x']:.2e}" for res in results],
        "MSD_y (m²)": [f"{res['msd_y']:.2e}" for res in results],
        "MSD_z (m²)": [f"{res['msd_z']:.2e}" for res in results],
        "Ratio (x/y)": [f"{res['ratio_xy']:.4f}" for res in results],
        "Ratio (x/z)": [f"{res['ratio_xz']:.4f}" for res in results],
        "Ratio (y/z)": [f"{res['ratio_yz']:.4f}" for res in results]
    }

    df = pd.DataFrame(table_data)

    print("\nMSD Calculation Results Table:")
    print(df.to_string())

    plt_msd = model.plot_msd_temperature(temperatures)
    plt_msd.show()

    plt_ratios = model.plot_msd_ratios(temperatures)
    plt_ratios.show()

    # extractor = ActivationEnergyExtractor()
    # simulated_msd_x = [model.msd(T, Ea_x) for T in temperatures]
    # plt_ea = extractor.plot_activation_energy_fit(temperatures, simulated_msd_x, direction='x')
    # plt_ea.show()
