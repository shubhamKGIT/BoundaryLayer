import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns 


ENERGY_DENSITY = 17.9  # KJ/cm3 
BURN_RATE = 1.2     # cm/s
REACTANT_DENSITY = 2.276e-3   #kg/cm3

p = np.linspace(10, 1000, 20)   # power rating in kW
w = np.linspace(5, 305, 30)   # weight

t, s = np.meshgrid(p, w)
z = (t/t + s/s)

vol_burn_rate = t/ENERGY_DENSITY    #cm3/s
cross_section = vol_burn_rate/BURN_RATE   #cm2

fuel_vol = s/REACTANT_DENSITY  # cm3 


coil_len = (fuel_vol/cross_section)/ 100   # m


fig = plt.figure("Parametric Surface")
# ax = Axes3D(fig)
# ax = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d')


h = ax.plot_wireframe(t, s, np.log10(coil_len), cmap='jet', edgecolor='b', linewidth = 0.8)
r = ax.plot_surface(t, s, np.log10(fuel_vol/0.25) - 4, facecolors = cm.jet(z), linewidth=0, alpha = 0.4)

# h = ax.plot_wireframe(t, s, z, cmap='jet', edgecolor='k')
# fig.colorbar(h)

ax.set_xlabel("Heat Release Rate (kW)", fontweight='bold', fontsize=10)
ax.set_ylabel("Fuel Weight (kg)", fontweight='bold', fontsize=10)
ax.set_zlabel("log10 [Coil Length (m)]", fontweight='bold', fontsize=10)
# ax.set_xscale("log")

# ax.set_zlim(0, 1000)
ax.set_title("Reactor Parametric Design Space",  fontweight='bold', fontsize=14)
ax.set_xlim(10, 1000)
ax.set_ylim(3, 305)

plt.savefig('nasaProj_parametric_plot')
# ax.axis('auto')
plt.show()