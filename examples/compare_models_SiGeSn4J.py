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

params  = {'legend.fontsize': 'small',
         'axes.labelsize': 'small',
         'axes.titlesize':'small',
         'xtick.labelsize':'small',
         'ytick.labelsize':'small',
           'axes.prop_cycle': cols}

plt.rcParams.update(params)


bulkthick = 300e-6
#font = {'family' : 'Lato Medium',
#        'size'   : 14}
#matplotlib.rc('font', **font)

# matrix multiplication
wavelengths = np.linspace(300, 1850, 200)*1e-9


options = default_options
options.nm_spacing = 0.5
options.wavelengths = wavelengths
options.project_name = 'test_matrix2'
options.n_rays = 50000
options.n_theta_bins = 30
options.phi_symmetry = np.pi/4
options.I_thresh = 1e-5
options.lookuptable_angles = 200
options.parallel = True
options.c_azimuth = 0.001
options.theta_in = 0*np.pi/180
options.phi_in = 'all'
options.only_incidence_angle = False

Ge = material('Ge')()
GaAs = material('GaAs')()
GaInP = material('GaInP')(In=0.5)
Ag = material('Ag')()
SiN = material('Si3N4')()
Air = material('Air')()
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
back_surf = Interface('TMM', layers=back_materials, name = 'SiN_Ag_TMM',
                      coherent=True)


bulk_Si = BulkLayer(bulkthick, Ge, name = 'Ge_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

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

fig, axes = plt.subplots(2, 2, figsize=(9,7))
ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[1,0]
ax4 = axes[1,1]
#fig = plt.subplots(figsize=(8,6))
#plt.subplot(221)
ax1.plot(options['wavelengths']*1e9, results_TMM_Matrix[0].R[0], label='R')

#ax1.plot(options['wavelengths']*1e9, R_0, '--', label='R_0')
#ax1.plot(options['wavelengths']*1e9, R_escape, '--', label='R_escape')
ax1.plot(options['wavelengths']*1e9, results_per_layer_front[:,0] + results_per_layer_front[:,1], label='ARC')
ax1.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='InGaP')
ax1.plot(options['wavelengths']*1e9, results_per_layer_front[:,3], label='GaAs')
ax1.plot(options['wavelengths']*1e9, results_per_layer_front[:,4], label='SiGeSn')
ax1.plot(options['wavelengths']*1e9, results_TMM_Matrix[0].A_bulk[0], label='Ge')
ax1.plot(options['wavelengths']*1e9, results_TMM_Matrix[0].T[0], label='T')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Reflection / Absorption')
#ax1.legend(loc='upper right')
ax1.set_title('a)', loc = 'left')
#plt.show()


## RT TMM

surf = regular_pyramids(elevation_angle=0, upright=True) # [texture, reverse]

front_surf = Interface('RT_TMM', layers=front_materials, texture=surf, name = 'GaInP_GaAs_SiGeSn_RT',
                       coherent=True)
back_surf = Interface('RT_TMM', layers=back_materials, texture = surf, name = 'SiN_Ag_RT_50k',
                      coherent=True)


SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

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

#ax2.subplot(222)
ax2.plot(options['wavelengths']*1e9, results_RT[0].R[0], label='R')

#ax2.plot(options['wavelengths']*1e9, R_0, '--', label='R_0')
#ax2.plot(options['wavelengths']*1e9, R_escape, '--', label='R_escape')
ax2.plot(options['wavelengths']*1e9, results_per_layer_front[:,0] + results_per_layer_front[:,1], label='ARC')
ax2.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='InGaP')
ax2.plot(options['wavelengths']*1e9, results_per_layer_front[:,3], label='GaAs')
ax2.plot(options['wavelengths']*1e9, results_per_layer_front[:,4], label='SiGeSn')
ax2.plot(options['wavelengths']*1e9, results_RT[0].A_bulk[0], label='Ge')
ax2.plot(options['wavelengths']*1e9, results_RT[0].T[0], label='T')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Reflection / Absorption')
ax2.set_title('b)', loc = 'left')
#ax2.legend()
#plt.show()



## RCWA

front_surf = Interface('RCWA', layers=front_materials, name = 'GaInP_GaAs_SiGeSn_RCWA',
                       coherent=True, d_vectors = ((500,0), (0,500)), rcwa_orders=3)
back_surf = Interface('RCWA', layers=back_materials, name = 'SiN_Ag_RCWA',
                      coherent=True, d_vectors = ((500,0), (0,500)), rcwa_orders=3)



SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

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

#ax3.subplot(223)
ax3.plot(options['wavelengths']*1e9, results_RCWA_Matrix[0].R[0], label='R')


#ax3.plot(options['wavelengths']*1e9, R_0, '--', label='R_0')
#ax3.plot(options['wavelengths']*1e9, R_escape, '--', label='R_escape')
ax3.plot(options['wavelengths']*1e9, results_per_layer_front[:,0] +  results_per_layer_front[:,1], label='ARC')
ax3.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='InGaP')
ax3.plot(options['wavelengths']*1e9, results_per_layer_front[:,3], label='GaAs')
ax3.plot(options['wavelengths']*1e9, results_per_layer_front[:,4], label='SiGeSn')
ax3.plot(options['wavelengths']*1e9, results_RCWA_Matrix[0].A_bulk[0], label='Ge')
ax3.plot(options['wavelengths']*1e9, results_RCWA_Matrix[0].T[0], label='T')
ax3.set_xlabel('Wavelength (nm)')
ax3.set_ylabel('Reflection / Absorption')
ax3.set_title('c)', loc = 'left')
#plt.legend()
#plt.show()


from solcore.absorption_calculator import calculate_rat, OptiStack


## pure TMM
all_layers = front_materials + [Layer(bulkthick, Ge)] + back_materials

coh_list = len(front_materials)*['c'] + ['i'] + ['c']

OS_layers = OptiStack(all_layers, substrate=Ag, no_back_reflection=False)

TMM_res = calculate_rat(OS_layers, wavelength=wavelengths*1e9,
                        no_back_reflection=False, angle=options['theta_in']*180/np.pi, coherent=False,
                        coherency_list=coh_list, pol=options['pol'])

#ax4.subplot(224)
ax4.plot(options['wavelengths']*1e9, TMM_res['R'], label='R')

ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][1] + TMM_res['A_per_layer'][2], label='ARC')
ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][3], label='InGaP')
ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][4], label='GaAs')
ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][5], label='SiGeSn')
ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][len(front_materials)+1], label='Ge')
ax4.plot(options['wavelengths']*1e9, TMM_res['T'], label='T')
ax4.set_xlabel('Wavelength (nm)')
ax4.set_ylabel('Reflection / Absorption')
ax4.set_title('d)', loc = 'left')

handles, labels = ax4.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0, 0, 0.42, 0.46), loc='upper right')

fig.savefig('model_validation2.pdf', bbox_inches='tight', format='pdf')
plt.show()


from sparse import load_npz

RTmat = load_npz('/home/phoebe/Documents/rayflare/results/test_matrix2/GaInP_GaAs_SiGeSn_RTfrontRT.npz')

TMMmat = load_npz('/home/phoebe/Documents/rayflare/results/test_matrix2/GaInP_GaAs_SiGeSn_TMMfrontRT.npz')

RTmat_0 = RTmat[0].todense()

TMMmat_0 = TMMmat[0].todense()