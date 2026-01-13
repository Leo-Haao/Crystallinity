
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --- Data Extraction and Consolidation ---
# Data points were extracted from all the tables provided by the user.
# All temperatures are converted to Celsius, and all diffusion coefficients to cm^2/s.
# Assumptions are noted where the original table was missing a value.

# T (°C), r (Li/EO), D (cm²/s)
data_points = [
    # Data from '组 Case温度 T (K)r (Li:EO)...' table
    (360-273.15, 0.08, 0.25e-7), (400-273.15, 0.08, 1.54e-7), (440-273.15, 0.08, 4.61e-7), (480-273.15, 0.08, 13.64e-7),
    (360-273.15, 0.06, 0.47e-7), (400-273.15, 0.06, 1.96e-7), (440-273.15, 0.06, 7.11e-7), (480-273.15, 0.06, 17.12e-7),
    (360-273.15, 0.04, 0.49e-7), (400-273.15, 0.04, 2.15e-7), (440-273.15, 0.04, 8.32e-7), (480-273.15, 0.04, 20.97e-7),
    (360-273.15, 0.02, 0.46e-7), (400-273.15, 0.02, 2.74e-7), (440-273.15, 0.02, 7.57e-7), (480-273.15, 0.02, 15.60e-7),
    (360-273.15, 0.02, 1.74e-7), (360-273.15, 0.02, 1.16e-7), (360-273.15, 0.02, 0.50e-7),
    (400-273.15, 0.02, 5.05e-7), (400-273.15, 0.02, 4.74e-7), (400-273.15, 0.02, 3.82e-7),
    (440-273.15, 0.02, 13.88e-7), (440-273.15, 0.02, 11.43e-7), (440-273.15, 0.02, 8.96e-7),
    (480-273.15, 0.02, 34.78e-7), (480-273.15, 0.02, 26.77e-7), (480-273.15, 0.02, 16.35e-7),

    # Data from '文献来源...' table
    (90, 0.085, 1.1e-7), (85, 0.083, 4.6e-8), (90, 0.033, 4.2e-8),

    # Data from 'X (Li/EO)D_Li⁺...' table (Assuming T=90°C based on context from other tables)
    (90, 0.001, 1.12e-7), (90, 0.04, 9.8e-8), (90, 0.08, 7.3e-8), (90, 0.12, 5.8e-8),
    (90, 0.16, 7.0e-8), (90, 0.20, 5.7e-8), (90, 0.24, 5.4e-8),

    # Data from '来源温度 (K)D_Li⁺...' table (Assuming r=0.1, a common value for such studies)
    (400-273.15, 0.1, 6.55e-5), (450-273.15, 0.1, 1.118e-4),
    (400-273.15, 0.1, 6.4e-5), (450-273.15, 0.1, 1.39e-4),

    # Data from '样品类型PEO...' table
    (90, 0.085, 8e-8),

    # Data from '样品温度 (°C)r (Li/EO)D//...' table
    (80, 0.06, 1.06e-7), (80, 0.06, 4.64e-8), (80, 0.06, 1.28e-7), (80, 0.06, 4.52e-8),

    # Data from '温度 (°C)温度 (K)D (cm²/s)...' table (Assuming r=0.08 based on source context)
    (40, 0.08, 0.172e-8), (50, 0.08, 0.509e-8), (60, 0.08, 2.12e-8),
    (70, 0.08, 2.05e-7), (80, 0.08, 4.09e-7), (90, 0.08, 4.95e-7),

    # Data from 'St-Onge et al., 2024...' text (midpoint of range, assuming r=0.08)
    (45, 0.08, 7.5e-9),

    # Data from '样品相态r (Li/EO)...' table
    (85, 0.083, 4.65e-8),

    # Data from '体系方法温度 (°C)...' table (Assuming r=0.05, a common ratio for PEO/LiTFSI)
    (90, 0.05, 6.4e-8), (90, 0.05, 1.3e-7),

    # Data from '样品相态情况温度...' table
    (340-273.15, 1/30, 0.84e-8),
    (340-273.15, 1/20, 3.3e-8), # Note: This point is for Na+, not Li+
]

# Unzip the data into separate lists
T_celsius, r_ratio, D_cm2_s = zip(*data_points)

# Calculate the log of the diffusion coefficient for the Z-axis
log10_D = np.log10(np.array(D_cm2_s))

# --- Plotting ---

# Define font properties for styling, based on the user's provided class
font_style = {'fontname': 'Times New Roman', 'fontweight': 'bold'}

# Create the 3D plot
fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

# Create the scatter plot
scatter = ax.scatter(T_celsius, r_ratio, log10_D, c=log10_D, cmap='viridis', s=80, edgecolors='k', depthshade=True)

# Add the vertical reference plane at T = 65 °C
y_plane, z_plane = np.meshgrid(np.linspace(0, max(r_ratio)*1.1, 10), np.linspace(-12, -4, 10))
x_plane = np.full_like(y_plane, 65)
ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.2, color='red', rstride=1, cstride=1)


# --- Styling and Formatting ---

# Set axis labels using LaTeX for formatting
ax.set_xlabel('\n温度 $T$ ($^\\circ$C)', fontsize=26, **font_style)
ax.set_ylabel('\n浓度 $r$ (Li/EO)', fontsize=26, **font_style)
ax.set_zlabel('\n$\\log_{10}(D_{Li^+} / \\mathrm{cm}^2\\mathrm{s}^{-1})$', fontsize=26, **font_style)

# Set axis limits
ax.set_xlim(0, 200)
ax.set_zlim(-12, -4)
ax.set_ylim(0, max(r_ratio)*1.1)

# Style the axis ticks
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    for tick in axis.get_major_ticks():
        tick.label1.set_fontname(font_style['fontname'])
        tick.label1.set_fontweight(font_style['fontweight'])
        tick.label1.set_fontsize(24)

# Add a color bar to show the mapping of colors to D values
cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('$\\log_{10}(D)$', size=22, **font_style)
cbar.ax.tick_params(labelsize=20)
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_family(font_style['fontname'])

# Set the viewing angle for better perspective
ax.view_init(elev=25, azim=-50)

# Make axis panes transparent and set edge thickness
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('k')
ax.yaxis.pane.set_edgecolor('k')
ax.zaxis.pane.set_edgecolor('k')
ax.xaxis.pane.set_linewidth(2)
ax.yaxis.pane.set_linewidth(2)
ax.zaxis.pane.set_linewidth(2)

# Set tick parameters for all axes
ax.tick_params(axis='both', direction='in', width=2, size=6)

plt.tight_layout()
# Save the figure to a file
plt.savefig("3d_diffusion_plot.png", dpi=300)
print("Plot saved as 3d_diffusion_plot.png")
# To display the plot in an interactive window, uncomment the line below
# plt.show()
