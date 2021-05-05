import numpy as np

# solcore imports
from solcore.structure import Layer
from solcore import material

# rayflare imports
from rayflare.textures.standard_rt_textures import planar_surface
from rayflare.structure import Interface, BulkLayer, Structure
from rayflare.matrix_formalism.process_structure import process_structure
from rayflare.matrix_formalism.multiply_matrices import calculate_RAT
from rayflare.options import default_options

# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import xarray as xr
import os

pal = sns.color_palette('husl', 6)

cols = cycler('color', pal)

params  = {'axes.prop_cycle': cols}

plt.rcParams.update(params)

# Thickness of bottom Ge layer
bulkthick = 30e-9


wavelengths = np.linspace(400, 800, 20)*1e-9

pal2 = sns.cubehelix_palette(len(wavelengths), start=.5, rot=-.9)

# set options
options = default_options()
options.wavelengths = wavelengths
options.project_name = 'back_incidence_alllayers'
options.n_rays = 2500
options.n_theta_bins = 3
options.lookuptable_angles = 100
options.parallel = True
options.c_azimuth = 0.001
options.theta_in = 0
options.pol = 's'
options.depth_spacing = 1e-9

# set up Solcore materials
Ge = material('Ge')()
GaAs = material('GaAs')()
GaInP = material('GaInP')(In=0.5)
Ag = material('Ag')()
SiN = material('Si3N4')()
Air = material('Air')()
Ta2O5 = material('TaOx1')() # Ta2O5 (SOPRA database)
MgF2 = material('MgF2')() # MgF2 (SOPRA database)


front_materials = [Layer(100e-9, MgF2), Layer(50e-9, GaInP), Layer(100e-9, Ta2O5), Layer(200e-9, GaAs)]
back_materials = [Layer(50E-9, GaInP)]

# make figure with subplots
fig1, axes = plt.subplots(2, 2, figsize=(9,7))
ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[1,0]
ax4 = axes[1,1]

fig2, axes2 = plt.subplots(2, 2, figsize=(9,7))
ax5 = axes2[0,0]
ax6 = axes2[0,1]
ax7 = axes2[1,0]
ax8 = axes2[1,1]

# TMM, matrix framework

front_surf = Interface('TMM', layers=front_materials, name = 'absorbing_front',
                       coherent=True, prof_layers=[1,2,3,4])
back_surf = Interface('TMM', layers=back_materials, name = 'absorbing_back',
                      coherent=True, prof_layers=[1])


bulk_Ge = BulkLayer(bulkthick, Ge, name = 'SiN_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options)

results_TMM_Matrix = calculate_RAT(SC, options)

results_per_pass = results_TMM_Matrix[1]

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)
results_per_layer_back = np.sum(results_per_pass['a'][1], 0)

ax1.plot(options['wavelengths']*1e9, results_TMM_Matrix[0].R[0], label='R')
ax1.plot(options['wavelengths']*1e9, results_per_layer_front[:,0] + results_per_layer_front[:,1], label='ARC/InGaP')
ax1.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='Ta2O5')
ax1.plot(options['wavelengths']*1e9, results_per_layer_front[:,3], label='GaAs')
ax1.plot(options['wavelengths']*1e9, results_TMM_Matrix[0].A_bulk[0], label='SiN')
ax1.plot(options['wavelengths']*1e9, results_per_layer_back[:,0], label='InGaP')
ax1.plot(options['wavelengths']*1e9, results_TMM_Matrix[0].T[0], label='T')
ax1.plot(options['wavelengths']*1e9, np.sum(results_per_layer_front, 1), '-b', label='total')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Reflection / Absorption')
ax1.set_title('a) TMM + matrix formalism', loc = 'left')


profile = results_TMM_Matrix[2]

prof_plot = profile[0]

depths = np.linspace(0, len(prof_plot[0, :]) * options['depth_spacing'] * 1e9, len(prof_plot[0, :]))

for i1 in np.arange(len(wavelengths)):
    ax5.plot(depths, prof_plot[i1, :], color=pal2[i1],
            label=str(round(options['wavelengths'][i1] * 1e9, 1)))

ax5.set_ylabel('Absorbed energy density (nm$^{-1}$)')
# ax5.legend(title='Wavelength (nm)')
ax5.set_xlabel('Distance into surface (nm)')

profdatapath = os.path.join('/home/phoebe/RayFlare_results', options.project_name, front_surf.name + 'frontprofmat.nc')
profdatapath_r = os.path.join('/home/phoebe/RayFlare_results', options.project_name, front_surf.name + 'rearprofmat.nc')

prof_dataset = xr.load_dataset(profdatapath)
prof_dataset_r = xr.load_dataset(profdatapath_r)

intgr = prof_dataset['intgr']
intgr_r = prof_dataset_r['intgr']

ax1.plot(options['wavelengths']*1e9, intgr[:,0], '--r')
ax1.plot(options['wavelengths']*1e9, intgr_r[:,0], '--r')

integrated_A_TMM = np.trapz(prof_plot, depths, axis=1)

## RT with TMM lookup tables

surf = planar_surface() # [texture, flipped texture]

front_surf = Interface('RT_TMM', layers=front_materials, texture=surf, name = 'GaInP_GaAs_RT',
                       coherent=True, prof_layers=[1,2,3,4])
back_surf = Interface('RT_TMM', layers=back_materials, texture = surf, name = 'SiN_Ag_RT_50k',
                      coherent=True, prof_layers=[1])

SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options)

results_RT = calculate_RAT(SC, options)

results_per_pass = results_RT[1]

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)

ax2.plot(options['wavelengths']*1e9, results_RT[0].R[0], label='R')

ax2.plot(options['wavelengths']*1e9, results_per_layer_front[:,0] + results_per_layer_front[:,1], label='ARC/InGaP')
ax2.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='Ta2O5')
ax2.plot(options['wavelengths']*1e9, results_per_layer_front[:,3], label='GaAs')
ax2.plot(options['wavelengths']*1e9, results_RT[0].A_bulk[0], label='SiN')
ax2.plot(options['wavelengths']*1e9, results_per_layer_back[:,0], label='InGaP')
ax2.plot(options['wavelengths']*1e9, results_RT[0].T[0], label='T')
ax2.plot(options['wavelengths']*1e9, np.sum(results_per_layer_front, 1), '-b', label='total')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Reflection / Absorption')
ax2.set_title('b) Ray-tracing/TMM + matrix formalism', loc = 'left')

profile = results_RT[2]

prof_plot = profile[0]

depths = np.linspace(0, len(prof_plot[0, :]) * options['depth_spacing'] * 1e9, len(prof_plot[0, :]))

for i1 in np.arange(len(wavelengths)):
    ax6.plot(depths, prof_plot[i1, :], color=pal2[i1],
            label=str(round(options['wavelengths'][i1] * 1e9, 1)))

ax6.set_ylabel('Absorbed energy density (nm$^{-1}$)')
# ax6.legend(title='Wavelength (nm)')
ax6.set_xlabel('Distance into surface (nm)')

profdatapath = os.path.join('/home/phoebe/RayFlare_results', options.project_name, front_surf.name + 'frontprofmat.nc')
profdatapath_r = os.path.join('/home/phoebe/RayFlare_results', options.project_name, front_surf.name + 'rearprofmat.nc')

prof_dataset = xr.load_dataset(profdatapath)
prof_dataset_r = xr.load_dataset(profdatapath_r)

intgr = prof_dataset['intgr']
intgr_r = prof_dataset_r['intgr']

ax2.plot(options['wavelengths']*1e9, intgr[:,0], '--r')
ax2.plot(options['wavelengths']*1e9, intgr_r[:,0], '--r')

integrated_A_RT = np.trapz(prof_plot, depths, axis=1)

## RCWA

front_surf = Interface('RCWA', layers=front_materials, name = 'GaInP_GaAs_RCWA',
                       coherent=True, d_vectors = ((500,0), (0,500)), rcwa_orders=2, prof_layers=[1,2,3,4])
back_surf = Interface('RCWA', layers=back_materials, name = 'SiN_Ag_RCWA',
                      coherent=True, d_vectors = ((500,0), (0,500)), rcwa_orders=2, prof_layers=[1])


SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

options.parallel = False

process_structure(SC, options)

results_RCWA_Matrix = calculate_RAT(SC, options)

results_per_pass = results_RCWA_Matrix[1]
R_per_pass = np.sum(results_per_pass['r'][0], 2)

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)


ax3.plot(options['wavelengths']*1e9, results_RCWA_Matrix[0].R[0], label='R')
ax3.plot(options['wavelengths']*1e9, results_per_layer_front[:,0] + results_per_layer_front[:,1], label='ARC/InGaP')
ax3.plot(options['wavelengths']*1e9, results_per_layer_front[:,2], label='Ta2O5')
ax3.plot(options['wavelengths']*1e9, results_per_layer_front[:,3], label='GaAs')
ax3.plot(options['wavelengths']*1e9, results_RT[0].A_bulk[0], label='SiN')
ax3.plot(options['wavelengths']*1e9, results_per_layer_back[:,0], label='InGaP')
ax3.plot(options['wavelengths']*1e9, results_RCWA_Matrix[0].T[0], label='T')
ax3.plot(options['wavelengths']*1e9, np.sum(results_per_layer_front, 1), '-b', label='total')
ax3.set_xlabel('Wavelength (nm)')
ax3.set_ylabel('Reflection / Absorption')
ax3.set_title('c) RCWA + matrix formalism', loc = 'left')

profile = results_RCWA_Matrix[2]

prof_plot = profile[0]

depths = np.linspace(0, len(prof_plot[0, :]) * options['depth_spacing'] * 1e9, len(prof_plot[0, :]))

for i1 in np.arange(len(wavelengths)):
    ax7.plot(depths, prof_plot[i1, :], color=pal2[i1],
            label=str(round(options['wavelengths'][i1] * 1e9, 1)))

ax7.set_ylabel('Absorbed energy density (nm$^{-1}$)')
# ax5.legend(title='Wavelength (nm)')
ax7.set_xlabel('Distance into surface (nm)')

profdatapath = os.path.join('/home/phoebe/RayFlare_results', options.project_name, front_surf.name + 'frontprofmat.nc')
profdatapath_r = os.path.join('/home/phoebe/RayFlare_results', options.project_name, front_surf.name + 'rearprofmat.nc')

prof_dataset = xr.load_dataset(profdatapath)
prof_dataset_r = xr.load_dataset(profdatapath_r)

intgr = prof_dataset['intgr']
intgr_r = prof_dataset_r['intgr']

ax3.plot(options['wavelengths']*1e9, intgr[:,0], '--r')
ax3.plot(options['wavelengths']*1e9, intgr_r[:,0], '--r')

integrated_A_RCWA = np.trapz(prof_plot, depths, axis=1)

from rayflare.transfer_matrix_method import tmm_structure


## pure TMM (from Solcore)
all_layers = front_materials + [Layer(bulkthick, Ge)] + back_materials

coh_list = len(front_materials)*['c'] + ['i'] + ['i']

options.coherent = False
options.coherency_list = coh_list

OS_layers = tmm_structure(all_layers, incidence=Air, transmission=Ag, no_back_reflection=False)

TMM_res = OS_layers.calculate(options, profile=True, layers=[1,2,3,4,5,6])

ax4.plot(options['wavelengths']*1e9, TMM_res['R'], label='R')

ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][:,0] + TMM_res['A_per_layer'][:,1], label='ARC/InGaP')
ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][:,2], label='Ta2O5')
ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][:,3], label='GaAs')
ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][:,4], label='SiN')
ax4.plot(options['wavelengths']*1e9, TMM_res['A_per_layer'][:,5], label='back')
ax4.plot(options['wavelengths']*1e9, TMM_res['T'], label='T')
ax4.plot(options['wavelengths']*1e9, np.sum(TMM_res['A_per_layer'][:,:4], 1), '-b', label='total')
ax4.set_xlabel('Wavelength (nm)')
ax4.set_ylabel('Reflection / Absorption')
ax4.set_title('d) Only TMM (Solcore)', loc = 'left')

handles, labels = ax4.get_legend_handles_labels()
fig1.legend(handles, labels, bbox_to_anchor=(0, 0, 0.42, 0.46), loc='upper right')


for i1 in np.arange(len(wavelengths)):
    ax8.plot(depths, TMM_res['profile'][i1, :len(depths)], color=pal2[i1],
            label=str(round(options['wavelengths'][i1] * 1e9, 1)))

ax8.set_ylabel('Absorbed energy density (nm$^{-1}$)')
# ax8.legend(title='Wavelength (nm)')
ax8.set_xlabel('Distance into surface (nm)')


integrated_A_TMMstruct = np.trapz(TMM_res['profile'][:, :len(depths)], depths, axis=1)
#

# fig.savefig('model_validation2.pdf', bbox_inches='tight', format='pdf')
ax1.plot(options['wavelengths']*1e9, integrated_A_TMM, '--k', label='TMM')
ax2.plot(options['wavelengths']*1e9, integrated_A_RT, '--k', label='RT')
ax3.plot(options['wavelengths']*1e9, integrated_A_RCWA, '--k', label='RCWA')
ax4.plot(options['wavelengths']*1e9, integrated_A_TMMstruct, '--k', label='TMM struct')
fig1.show()
plt.figure()
fig2.show()

plt.figure()
plt.plot(options['wavelengths']*1e9, integrated_A_TMM, label='TMM')
plt.plot(options['wavelengths']*1e9, integrated_A_RT, label='RT')
plt.plot(options['wavelengths']*1e9, integrated_A_RCWA, label='RCWA')
plt.plot(options['wavelengths']*1e9, integrated_A_TMMstruct, label='TMM struct')
plt.legend()
plt.show()

