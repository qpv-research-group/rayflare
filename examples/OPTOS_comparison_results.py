
import numpy as np
import os
# solcore imports
from solcore.structure import Layer
from solcore import material
from solcore import si
from solcore.absorption_calculator import calculate_rat, OptiStack
from structure import Interface, BulkLayer, Structure
from matrix_formalism.process_structure import process_structure
from matrix_formalism.multiply_matrices import calculate_RAT
from angles import theta_summary

import matplotlib.pyplot as plt
import seaborn as sns

from cycler import cycler

palhf = sns.color_palette("hls", 4)

cols = cycler('color', palhf)

params = {'legend.fontsize': 'small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small',
          'axes.prop_cycle': cols}

plt.rcParams.update(params)



angle_degrees_in = 8

wavelengths = np.linspace(900, 1200, 30)

sim_fig6 = np.loadtxt('data/optos_fig6_sim.csv', delimiter=',')
sim_fig7 = np.loadtxt('data/optos_fig7_sim.csv', delimiter=',')
sim_fig8 = np.loadtxt('data/optos_fig8_sim.csv', delimiter=',')

rayflare_fig6 = np.loadtxt('fig6_rayflare.txt')
rayflare_fig7 = np.loadtxt('fig7_rayflare.txt')
rayflare_fig8 = np.loadtxt('fig8_rayflare.txt')

# planar
Si = material('Si_OPTOS')()
Air = material('Air')()
struc = OptiStack([Layer(si('200um'), Si)], substrate=Air)

RAT = calculate_rat(struc, wavelength=wavelengths, coherent=True)

fig = plt.figure()
plt.plot(sim_fig6[:,0], sim_fig6[:,1], '--', color=palhf[0], label= 'OPTOS - rear grating')
plt.plot(wavelengths, rayflare_fig6, '-o', color=palhf[0], label='RayFlare - rear grating', fillstyle='none')
plt.plot(sim_fig7[:,0], sim_fig7[:,1], '--', color=palhf[1],  label= 'OPTOS - front pyramids')
plt.plot(wavelengths, rayflare_fig7, '-o', color=palhf[1],  label= 'RayFlare - front pyramids', fillstyle='none')
plt.plot(sim_fig8[:,0], sim_fig8[:,1], '--', color=palhf[2],  label= 'OPTOS - grating + pyramids')
plt.plot(wavelengths, rayflare_fig8, '-o', color=palhf[2],  label= 'RayFlare - grating + pyramids', fillstyle='none')
plt.plot(wavelengths, RAT['A_per_layer'][1], '-k', label='Planar')
plt.legend(loc='lower left')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorption in Si')
plt.xlim([900, 1200])
plt.ylim([0, 1])
fig.savefig('OPTOScomp.pdf', bbox_inches='tight', format='pdf')
plt.show()


