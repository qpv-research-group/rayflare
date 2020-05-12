
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
palhf = sns.cubehelix_palette(3, start=.5, rot=-.9)

angle_degrees_in = 8

wavelengths = np.linspace(900, 1200, 30)

sim_fig6 = np.loadtxt('data/optos_fig6_sim.csv', delimiter=',')
sim_fig7 = np.loadtxt('data/optos_fig7_sim.csv', delimiter=',')


rayflare_fig6 = np.loadtxt('fig6_rayflare.txt')
rayflare_fig7 = np.loadtxt('fig7_rayflare.txt')

# planar
Si = material('Si_OPTOS')()
Air = material('Air')()
struc = OptiStack([Layer(si('200um'), Si)], substrate=Air)

RAT = calculate_rat(struc, wavelength=wavelengths, coherent=True)

plt.figure()
plt.plot(sim_fig6[:,0], sim_fig6[:,1], '--', color=palhf[0], label= 'OPTOS - rear grating')
plt.plot(wavelengths, rayflare_fig6, color=palhf[0], label='RayFlare - rear grating')
plt.plot(wavelengths, rayflare_fig6, 'o', color=palhf[0])
plt.plot(sim_fig7[:,0], sim_fig7[:,1], '--', color=palhf[1],  label= 'OPTOS - front pyramids')
plt.plot(wavelengths, rayflare_fig7, color=palhf[1],  label= 'RayFlare - front pyramids')
plt.plot(wavelengths, rayflare_fig7, 'o', color=palhf[1])
plt.plot(wavelengths, RAT['A_per_layer'][1], '-k', label='Planar')
plt.legend(loc='upper right')
plt.show()


