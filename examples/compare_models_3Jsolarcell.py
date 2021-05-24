import numpy as np

# solcore imports
from solcore.structure import Layer
from solcore import material

# rayflare imports
from rayflare.textures import planar_surface
from rayflare.structure import Interface, BulkLayer, Structure
from rayflare.matrix_formalism import process_structure, calculate_RAT
from rayflare.transfer_matrix_method import tmm_structure
from rayflare.options import default_options

# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

pal = sns.color_palette('husl', 5)

cols = cycler('color', pal)

params  = {'axes.prop_cycle': cols}

plt.rcParams.update(params)

# Thickness of bottom Ge layer
bulkthick = 300e-6

wavelengths = np.linspace(300, 1850, 200)*1e-9

# set options
options = default_options()
options.wavelengths = wavelengths
options.project_name = 'method_comparison'
options.n_rays = 250
options.n_theta_bins = 3
options.lookuptable_angles = 100
options.parallel = True
options.c_azimuth = 0.001

# set up Solcore materials
Ge = material('Ge')()
GaAs = material('GaAs')()
GaInP = material('GaInP')(In=0.5)
Ag = material('Ag')()
SiN = material('Si3N4')()
Air = material('Air')()
Ta2O5 = material('TaOx1')() # Ta2O5 (SOPRA database)
MgF2 = material('MgF2')() # MgF2 (SOPRA database)


front_materials = [Layer(120e-9, MgF2), Layer(74e-9, Ta2O5), Layer(464e-9, GaInP),
                   Layer(1682e-9, GaAs)]
back_materials = [Layer(100E-9, SiN)]

# make figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(9,7))
ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[1,0]
ax4 = axes[1,1]

# TMM with redistribution matrix method

front_surf = Interface('TMM', layers=front_materials, name = 'GaInP_GaAs_TMM',
                       coherent=True)
back_surf = Interface('TMM', layers=back_materials, name = 'SiN_Ag_TMM',
                      coherent=True)


bulk_Ge = BulkLayer(bulkthick, Ge, name = 'Ge_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options)

results_TMM_Matrix = calculate_RAT(SC, options)

results_per_pass = results_TMM_Matrix[1]

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)
ax1.plot(options['wavelengths']*1e9, results_TMM_Matrix[0].R[0], label='R')
ax1.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='InGaP')
ax1.plot(options['wavelengths']*1e9, results_per_layer_front[:,3], label='GaAs')
ax1.plot(options['wavelengths']*1e9, results_TMM_Matrix[0].A_bulk[0], label='Ge')
ax1.plot(options['wavelengths']*1e9, results_TMM_Matrix[0].T[0], label='T')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Reflection / Absorption')
ax1.set_title('a) TMM + matrix formalism', loc = 'left')


## RT with TMM lookup tables

surf = planar_surface() # [texture, flipped texture]

front_surf = Interface('RT_TMM', layers=front_materials, texture=surf, name = 'GaInP_GaAs_RT',
                       coherent=True)
back_surf = Interface('RT_TMM', layers=back_materials, texture = surf, name = 'SiN_Ag_RT_50k',
                      coherent=True)

SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options)

results_RT = calculate_RAT(SC, options)

results_per_pass = results_RT[1]

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)

ax2.plot(options['wavelengths']*1e9, results_RT[0].R[0], label='R')
ax2.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='InGaP')
ax2.plot(options['wavelengths']*1e9, results_per_layer_front[:,3], label='GaAs')
ax2.plot(options['wavelengths']*1e9, results_RT[0].A_bulk[0], label='Ge')
ax2.plot(options['wavelengths']*1e9, results_RT[0].T[0], label='T')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Reflection / Absorption')
ax2.set_title('b) Ray-tracing/TMM + matrix formalism', loc = 'left')


## RCWA

front_surf = Interface('RCWA', layers=front_materials, name = 'GaInP_GaAs_RCWA',
                       coherent=True, d_vectors = ((500,0), (0,500)), rcwa_orders=2)
back_surf = Interface('RCWA', layers=back_materials, name = 'SiN_Ag_RCWA',
                      coherent=True, d_vectors = ((500,0), (0,500)), rcwa_orders=2)


SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options)

results_RCWA_Matrix = calculate_RAT(SC, options)

results_per_pass = results_RCWA_Matrix[1]
R_per_pass = np.sum(results_per_pass['r'][0], 2)

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)


ax3.plot(options['wavelengths']*1e9, results_RCWA_Matrix[0].R[0], label='R')
ax3.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='InGaP')
ax3.plot(options['wavelengths']*1e9, results_per_layer_front[:,3], label='GaAs')
ax3.plot(options['wavelengths']*1e9, results_RCWA_Matrix[0].A_bulk[0], label='Ge')
ax3.plot(options['wavelengths']*1e9, results_RCWA_Matrix[0].T[0], label='T')
ax3.set_xlabel('Wavelength (nm)')
ax3.set_ylabel('Reflection / Absorption')
ax3.set_title('c) RCWA + matrix formalism', loc = 'left')

## pure TMM
all_layers = front_materials + [Layer(bulkthick, Ge)] + back_materials

coh_list = len(front_materials)*['c'] + ['i'] + ['c']

options.coherency_list = coh_list
options.coherent = False

TMM_setup = tmm_structure(all_layers, incidence=Air, transmission=Ag, no_back_reflection=False)

TMM_res = TMM_setup.calculate(options)

ax4.plot(options['wavelengths']*1e9, TMM_res['R'], label='R')

ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][:,2], label='InGaP')
ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][:,3], label='GaAs')
ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][:,4], label='Ge')
ax4.plot(options['wavelengths']*1e9, TMM_res['T'], label='T')
ax4.set_xlabel('Wavelength (nm)')
ax4.set_ylabel('Reflection / Absorption')
ax4.set_title('d) Only TMM (Solcore)', loc = 'left')

handles, labels = ax4.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0, 0, 0.42, 0.46), loc='upper right')

plt.show()