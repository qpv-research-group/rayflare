import numpy as np
import os
# solcore imports
from solcore.structure import Layer
from solcore import material
from solcore.light_source import LightSource
from solcore.constants import q

from textures.standard_rt_textures import planar_surface, regular_pyramids
from structure import Interface, BulkLayer, Structure
from matrix_formalism.process_structure import process_structure
from matrix_formalism.multiply_matrices import calculate_RAT
from options import default_options

import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler


pal = sns.cubehelix_palette(10, start=.5, rot=-.9)

cols = cycler('color', pal)

plt.rcParams['axes.prop_cycle'] = cols

bulkthick = 300e-6
#font = {'family' : 'Lato Medium',
#        'size'   : 14}
#matplotlib.rc('font', **font)

# matrix multiplication
wavelengths = np.linspace(300, 1800, 200)*1e-9

options = default_options
options.nm_spacing = 0.5
options.wavelengths = wavelengths
options.project_name = 'test_matrix2'
options.n_rays = 1000
options.n_theta_bins = 30
options.phi_symmetry = np.pi/4
options.I_thresh = 1e-8
options.lookuptable_angles = 200
options.parallel = True
options.c_azimuth = 0.001
options.theta_in = 0*np.pi/180
options.phi_in = 'all'
options.only_incidence_angle = False

Ge = material('Ge')()
GaAs = material('GaAs')()
GaInP = material('GaInP')(In=0.5)
Air = material('Air')()
SiN = material('Si3N4')()

Ta2O5 = material('410', nk_db=True)()
MgF2 = material('203', nk_db=True)()
SiGeSn = material('SiGeSn')()

# stack based on doi:10.1038/s41563-018-0115-4
front_materials = [Layer(120e-9, MgF2), Layer(74e-9, Ta2O5), Layer(464e-9, GaInP), Layer(1682e-9, GaAs), Layer(1289e-9, SiGeSn)]
back_materials = [Layer(100E-9, SiN)]

# whether pyramids are upright or inverted is relative to front incidence.
# so if the same etch is applied to both sides of a slab of silicon, one surface
# will have 'upright' pyramids and the other side will have 'not upright' (inverted)
# pyramids in the model


## TMM, matrix framework

front_surf = Interface('TMM', layers=front_materials, name = 'GaInP_GaAs_SiGeSn_TMM',
                       coherent=True)
back_surf = Interface('TMM', layers=back_materials, name = 'SiN_Air_TMM',
                      coherent=True)


bulk_Si = BulkLayer(bulkthick, Ge, name = 'Ge_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Air)

process_structure(SC, options)

results_TMM_Matrix = calculate_RAT(SC, options)

results_per_pass = results_TMM_Matrix[1]
R_per_pass = np.sum(results_per_pass['r'][0], 2)
R_0 = R_per_pass[0]
R_escape = np.sum(R_per_pass[1:, :], 0)

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)
results_per_layer_back = np.sum(results_per_pass['a'][1], 0)
total = results_TMM_Matrix[0].R[0] + results_TMM_Matrix[0].T[0] + results_TMM_Matrix[0].A_bulk[0] +  np.sum(results_per_layer_front, 1)

plt.figure()
plt.plot(options['wavelengths']*1e9, results_TMM_Matrix[0].R[0], label='R')
plt.plot(options['wavelengths']*1e9, results_TMM_Matrix[0].T[0], label='T')
plt.plot(options['wavelengths']*1e9, results_TMM_Matrix[0].A_bulk[0], label='A_bulk (Ge)')
#plt.plot(options['wavelengths']*1e9, R_0, '--', label='R_0')
#plt.plot(options['wavelengths']*1e9, R_escape, '--', label='R_escape')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,0] + results_per_layer_front[:,1], label='ARC')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='InGaP')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,3], label='GaAs')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,4], label='SiGeSn')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflection / Absorption')
plt.legend(loc='upper right')
plt.show()


## RT TMM

surf = regular_pyramids(elevation_angle=0, upright=True) # [texture, reverse]

front_surf = Interface('RT_TMM', layers=front_materials, texture=surf, name = 'GaInP_GaAs_SiGeSn_RT',
                       coherent=True)
back_surf = Interface('RT_TMM', layers=back_materials, texture = surf, name = 'SiN_Air_RT',
                      coherent=True)


SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Air)

process_structure(SC, options)

results_RT = calculate_RAT(SC, options)

results_per_pass = results_RT[1]
R_per_pass = np.sum(results_per_pass['r'][0], 2)
R_0 = R_per_pass[0]
R_escape = np.sum(R_per_pass[1:, :], 0)

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)
results_per_layer_back = np.sum(results_per_pass['a'][1], 0)
total = results_RT[0].R[0] + results_RT[0].A_bulk[0] +  np.sum(results_per_layer_front, 1)

plt.figure()
plt.plot(options['wavelengths']*1e9, results_RT[0].R[0], label='R')
plt.plot(options['wavelengths']*1e9, results_RT[0].T[0], label='T')
plt.plot(options['wavelengths']*1e9, results_RT[0].A_bulk[0], label='A_bulk (Ge)')
#plt.plot(options['wavelengths']*1e9, R_0, '--', label='R_0')
#plt.plot(options['wavelengths']*1e9, R_escape, '--', label='R_escape')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,0] + results_per_layer_front[:,1], label='ARC')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='InGaP')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,3], label='GaAs')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,4], label='SiGeSn')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflection / Absorption')
#plt.legend()
plt.show()



## RCWA

front_surf = Interface('RCWA', layers=front_materials, name = 'GaInP_GaAs_SiGeSn_RCWA',
                       coherent=True, d_vectors = ((500,0), (0,500)), rcwa_orders=3)
back_surf = Interface('RCWA', layers=back_materials, name = 'SiN_Air_RCWA',
                      coherent=True, d_vectors = ((500,0), (0,500)), rcwa_orders=3)



SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Air)

process_structure(SC, options)

results_RCWA_Matrix = calculate_RAT(SC, options)

results_per_pass = results_RCWA_Matrix[1]
R_per_pass = np.sum(results_per_pass['r'][0], 2)
R_0 = R_per_pass[0]
R_escape = np.sum(R_per_pass[1:, :], 0)

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)
results_per_layer_back = np.sum(results_per_pass['a'][1], 0)
total = results_RCWA_Matrix[0].R[0] + results_RCWA_Matrix[0].T[0] + results_RCWA_Matrix[0].A_bulk[0] +  np.sum(results_per_layer_front, 1)

plt.figure()
plt.plot(options['wavelengths']*1e9, results_RCWA_Matrix[0].R[0], label='R')
plt.plot(options['wavelengths']*1e9, results_RCWA_Matrix[0].T[0], label='T')
plt.plot(options['wavelengths']*1e9, results_RCWA_Matrix[0].A_bulk[0], label='A_bulk (Ge)')
#plt.plot(options['wavelengths']*1e9, R_0, '--', label='R_0')
#plt.plot(options['wavelengths']*1e9, R_escape, '--', label='R_escape')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,0] +  results_per_layer_front[:,1], label='ARC')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='InGaP')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,3], label='GaAs')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,4], label='SiGeSn')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflection / Absorption')
#plt.legend()
plt.show()


from solcore.absorption_calculator import calculate_rat


## pure TMM
all_layers = front_materials + [Layer(bulkthick, Ge)] + back_materials

coh_list = len(front_materials)*['c'] + ['i'] + ['c']

TMM_res = calculate_rat(all_layers, wavelength=wavelengths*1e9, substrate=Air, no_back_reflection=False, angle=options['theta_in']*180/np.pi, coherent=False,
                        coherency_list=coh_list, pol=options['pol'])

plt.figure()
plt.plot(options['wavelengths']*1e9, TMM_res['R'], label='R')
plt.plot(options['wavelengths']*1e9, TMM_res['T'], label='T')
plt.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][len(front_materials)+1], label='Ge')
plt.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][1] + TMM_res['A_per_layer'][2], label='ARC')
plt.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][3], label='InGaP')
plt.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][4], label='GaAs')
plt.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][5], label='SiGeSn')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflection / Absorption')
#plt.legend()
plt.show()


from sparse import load_npz

RTmat = load_npz('/home/phoebe/Documents/rayflare/results/test_matrix2/GaInP_GaAs_SiGeSn_RTfrontRT.npz')

TMMmat = load_npz('/home/phoebe/Documents/rayflare/results/test_matrix2/GaInP_GaAs_SiGeSn_TMMfrontRT.npz')

RTmat_0 = RTmat[0].todense()

TMMmat_0 = TMMmat[0].todense()