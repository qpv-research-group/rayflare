import numpy as np
import os
# solcore imports
from solcore.structure import Layer
from solcore import material
from solcore.light_source import LightSource
from solcore.constants import q

from textures.standard_rt_textures import regular_pyramids
from structure import Interface, BulkLayer, Structure
from matrix_formalism.process_structure import process_structure
from matrix_formalism.multiply_matrices import calculate_RAT
from options import default_options

import matplotlib.pyplot as plt
import seaborn as sns

#font = {'family' : 'Lato Medium',
#        'size'   : 14}
#matplotlib.rc('font', **font)

# matrix multiplication
wavelengths = np.linspace(300, 1300, 50)*1e-9

options = default_options
options.nm_spacing = 0.5
options.wavelengths = wavelengths
options.project_name = 'test_matrix'
options.n_rays = 1e4
options.n_theta_bins = 20
options.phi_symmetry = np.pi/4
options.I_thresh = 1e-10
options.lookuptable_angles = 200
options.parallel = True
options.c_azimuth = 0.05

Si = material('Si')()
GaAs = material('GaAs')()
GaInP = material('GaInP')(In=0.5)
Ag = material('Ag')()
SiN = material('Si3N4')()
Air = material('Air')()

# stack based on doi:10.1038/s41563-018-0115-4
front_materials = [Layer(60e-9, SiN), Layer(100E-9, GaInP), Layer(200e-9, GaAs)]
back_materials = []

# whether pyramids are upright or inverted is relative to front incidence.
# so if the same etch is applied to both sides of a slab of silicon, one surface
# will have 'upright' pyramids and the other side will have 'not upright' (inverted)
# pyramids in the model


front_surf = Interface('TMM', layers=front_materials, name = 'GaInP_GaAs',
                       coherent=True)
back_surf = Interface('TMM', layers=back_materials, name = 'Si_Ag',
                      coherent=True)


bulk_Si = BulkLayer(20e-6, Si, name = 'Si_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options)

results = calculate_RAT(SC, options)

results_per_pass = results[1]



R_per_pass = np.sum(results_per_pass['r'][0], 2)
R_0 = R_per_pass[0]
R_escape = np.sum(R_per_pass[1:, :], 0)

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)
total = results[0].R[0] + results[0].T[0] + results[0].A_bulk[0] +  np.sum(results_per_layer_front, 1)

plt.figure()
plt.plot(options['wavelengths']*1e9, results[0].R[0], label='R')
plt.plot(options['wavelengths']*1e9, results[0].T[0], label='T')
plt.plot(options['wavelengths']*1e9, results[0].A_bulk[0], label='A_bulk (Si)')
plt.plot(options['wavelengths']*1e9, R_0, '--', label='R_0')
plt.plot(options['wavelengths']*1e9, R_escape, '--', label='R_escape')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,0], label='SiN')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,1], label='InGaP')
plt.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='GaAs')

plt.legend()
plt.show()

#  pure TMM comparison

layers = [Layer(60e-9, SiN), Layer(100E-9, GaInP), Layer(200e-9, GaAs), Layer(20e-6, Si)]

from solcore.absorption_calculator import OptiStack, calculate_rat

optist = OptiStack(layers, incidence=Air, substrate=Ag, no_back_reflection=False)
TMM_res = calculate_rat(optist, options['wavelengths']*1e9, angle=options['theta_in']*180/np.pi, pol=options['pol'],
                        coherent=False, coherency_list=['c', 'c', 'c', 'i'], no_back_reflection=False)

plt.figure()
plt.plot(options['wavelengths']*1e9, TMM_res['R'], label='R')
plt.plot(options['wavelengths']*1e9, TMM_res['T'], label='T')
plt.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][4], label='A_bulk (Si)')
plt.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][1], label='SiN')
plt.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][2], label='InGaP')
plt.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][3], label='GaAs')
plt.legend()
plt.show()