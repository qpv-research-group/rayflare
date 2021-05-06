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
from rayflare.transfer_matrix_method import tmm_structure
from rayflare.angles import make_angle_vector, theta_summary

# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import xarray as xr
import os
from sparse import load_npz
import matplotlib as mpl

pal = sns.color_palette('husl', 6)

cols = cycler('color', pal)

params  = {'axes.prop_cycle': cols}

plt.rcParams.update(params)

# Thickness of bottom Ge layer
bulkthick = 30e-9


wavelengths = np.linspace(640, 850, 8)*1e-9

pal2 = sns.cubehelix_palette(len(wavelengths), start=.5, rot=-.9)

theta_in = 1
# set options
options = default_options()
options.wavelengths = wavelengths
options.project_name = 'back_incidence_s'
options.n_rays = 4500
options.n_theta_bins = 20
options.lookuptable_angles = 100
options.parallel = True
options.c_azimuth = 0.25
options.theta_in = theta_in
options.phi_in = 0.7
options.pol = 's'
options.depth_spacing = 1e-9
options.only_incidence_angle = False
options.nx = 5
options.ny = 5

_, _, angle_vector = make_angle_vector(options.n_theta_bins,
                                       options.phi_symmetry, options.c_azimuth)

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

## pure TMM (from Solcore)
all_layers = front_materials + [Layer(bulkthick, Ge)] + back_materials

coh_list = len(front_materials)*['c'] + ['i'] + ['c']

options.coherent = False
options.coherency_list = coh_list

options.theta_in = angle_vector[np.argmin(np.abs(angle_vector[:,1] - theta_in)),1]

OS_layers = tmm_structure(all_layers, incidence=Air,
                          transmission=Ag, no_back_reflection=False)

TMM_res = OS_layers.calculate(options, profile=True, layers=[1,2,3,4,5,6])

options.coherent = True
OS_front = tmm_structure(front_materials, incidence=Air, transmission=SiN,
                         no_back_reflection=False)
TMM_res_front = OS_front.calculate(options, profile=True, layers=[1,2,3,4])
OS_back = tmm_structure(front_materials[::-1], incidence=SiN, transmission=Air,
                        no_back_reflection=False)
TMM_res_back = OS_back.calculate(options, profile=True, layers=[1,2,3,4])

plt.figure()
plt.plot()

options.theta_in = theta_in
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
prof = prof_dataset['profile']
intgr_r = prof_dataset_r['intgr']
prof_r = prof_dataset_r['profile']


plt.figure()
plt.subplot(121)
prof[:,:,0].plot(vmin=0)
plt.subplot(122)
prof_r[:,:,0].plot(vmin=0)
plt.title('TMM')
plt.show()


integrated_prof = np.trapz(prof.data, depths, axis=1)
integrated_prof_r = np.trapz(prof_r.data, depths, axis=1)

# wl_ind = -1
# plt.figure()
# plt.plot(depths, TMM_res['profile'][wl_ind, :len(depths)], '--r', label='TMM')
# plt.plot(depths, prof_plot[wl_ind, :], '-k', label='RCWA total')
# plt.plot(depths, results_per_pass['a_prof'][0][0,wl_ind,:], label='0')
# plt.plot(depths, results_per_pass['a_prof'][0][1,wl_ind,:], label='1')
# plt.plot(depths, results_per_pass['a_prof'][0][2,wl_ind,:])
# plt.plot(depths, results_per_pass['a_prof'][0][3,wl_ind,:])
# plt.legend()
# plt.show()


plt.figure()
plt.plot(intgr)
plt.plot(integrated_prof, '--')
plt.title('Matrix_TMM')
plt.show()

plt.figure()
plt.plot(intgr_r)
plt.plot(integrated_prof_r, '--')
plt.title('Matrix_TMM')
plt.ylim(0,1)
plt.show()

ax1.plot(options['wavelengths']*1e9, intgr[:,0], '--r', label='front int')
ax1.plot(options['wavelengths']*1e9, intgr_r[:,0], '--y', label='rear int')

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
prof = prof_dataset['profile']
intgr_r = prof_dataset_r['intgr']
prof_r = prof_dataset_r['profile']

integrated_prof = np.trapz(prof.data, depths, axis=2)
integrated_prof_r = np.trapz(prof_r.data, depths, axis=2)

plt.figure()
plt.subplot(121)
prof[:,0,:].plot(vmin=0)
plt.subplot(122)
prof_r[:,0,:].plot(vmin=0)
plt.title('RT_TMM')
plt.show()

plt.figure()
plt.plot(intgr_r)
plt.plot(integrated_prof_r, '--')
plt.title('RT_TMM')
plt.ylim(0,1)
plt.show()

ax2.plot(options['wavelengths']*1e9, intgr[:,0], '--r', label='front int')
ax2.plot(options['wavelengths']*1e9, intgr_r[:,0], '--y', label='rear int')

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
ax3.plot(options['wavelengths']*1e9, results_RCWA_Matrix[0].A_bulk[0], label='SiN')
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
prof = prof_dataset['profile']
intgr_r = prof_dataset_r['intgr']
prof_r = prof_dataset_r['profile']

integrated_prof = np.trapz(prof.data, depths, axis=1)

plt.figure()
plt.subplot(121)
prof[:,:,0].plot(vmin=0)
plt.subplot(122)
prof_r[:,:,0].plot(vmin=0)
plt.title('RCWA')
plt.show()

plt.figure()
plt.plot(intgr)
plt.plot(integrated_prof, '--')
plt.title('RCWA')
plt.show()

ax3.plot(options['wavelengths']*1e9, intgr[:,0], '--r', label='front int')
ax3.plot(options['wavelengths']*1e9, intgr_r[:,0], '--y', label='rear int')


integrated_A_RCWA = np.trapz(prof_plot, depths, axis=1)


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
plt.plot(options['wavelengths']*1e9, integrated_A_TMMstruct, '-k', label='TMM struct')
plt.legend()
plt.show()


pal2 = sns.cubehelix_palette(5, start=.5, rot=-.9, reverse=True)

cols = cycler('color', pal2)

params  = {'axes.prop_cycle': cols}

plt.rcParams.update(params)



# plt.figure()
# plt.plot(wavelengths*1e9, np.sum(results_per_pass['a'][0],2).T)
# plt.plot(wavelengths*1e9, np.sum(results_per_pass['a'][0],(0,2)), '-k')
# plt.plot(wavelengths*1e9, np.sum(results_per_layer_front, 1), '--r', label='total')
# plt.plot(wavelengths*1e9, np.sum(TMM_res['A_per_layer'][:,:4], 1), 'o', label='total')
# plt.show()

int_per_pass = np.trapz(results_per_pass['a_prof'][0], depths, 2)

plt.figure()
plt.plot(wavelengths*1e9, np.sum(results_per_pass['a'][0],2).T)
plt.plot(wavelengths*1e9, int_per_pass.T, '--')
plt.show()


mat_path_RT = os.path.join('/home/phoebe/RayFlare_results', options.project_name, 'GaInP_GaAs_RT' + 'frontRT.npz')
mat_path_TMM = os.path.join('/home/phoebe/RayFlare_results', options.project_name, 'absorbing_front' + 'frontRT.npz')
mat_path_RCWA = os.path.join('/home/phoebe/RayFlare_results', options.project_name, 'GaInP_GaAs_RCWA' + 'frontRT.npz')


mat_path_RT = os.path.join('/home/phoebe/RayFlare_results', options.project_name, 'SiN_Ag_RT_50k' + 'frontRT.npz')
mat_path_TMM = os.path.join('/home/phoebe/RayFlare_results', options.project_name, 'absorbing_back' + 'frontRT.npz')
mat_path_RCWA = os.path.join('/home/phoebe/RayFlare_results', options.project_name, 'SiN_Ag_RCWA' + 'frontRT.npz')


wl_to_plot = 650e-9

wl_index = np.argmin(np.abs(wavelengths-wl_to_plot))

sprs_front = load_npz(mat_path_RT)

full_f = sprs_front[wl_index].todense()
# full_r = sprs_rear[wl_index].todense()

summat = theta_summary(full_f, angle_vector, options['n_theta_bins'], "front")

## reflection

#whole_mat = xr.concat((summat, summat_back), dim=r'$\theta_{in}$')
summat_back = summat[options['n_theta_bins']:]

whole_mat_imshow = summat_back.rename({r'$\theta_{in}$': 'a', r'$\theta_{out}$': 'b'})

whole_mat_imshow = whole_mat_imshow.assign_coords(a=np.sin(whole_mat_imshow.coords['a']).data,
                                                  b=np.sin(whole_mat_imshow.coords['b']).data)


whole_mat_imshow = whole_mat_imshow.rename({'a': r'$\sin(\theta_{in})$', 'b': r'$\sin(\theta_{out})$'})


palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
palhf.reverse()
seamap = mpl.colors.ListedColormap(palhf)
fig = plt.figure(figsize=(12,3.5))
ax = fig.add_subplot(1,3,1, aspect='equal')
ax.text(-0.15, 1, 'RT')
ax = whole_mat_imshow.plot.imshow(ax=ax, cmap=seamap)


sprs_front = load_npz(mat_path_TMM)

full_f = sprs_front[wl_index].todense()
# full_r = sprs_rear[wl_index].todense()

summat = theta_summary(full_f, angle_vector, options['n_theta_bins'], "front")

## reflection

#whole_mat = xr.concat((summat, summat_back), dim=r'$\theta_{in}$')
summat_back = summat[options['n_theta_bins']:]

whole_mat_imshow = summat_back.rename({r'$\theta_{in}$': 'a', r'$\theta_{out}$': 'b'})

whole_mat_imshow = whole_mat_imshow.assign_coords(a=np.sin(whole_mat_imshow.coords['a']).data,
                                                  b=np.sin(whole_mat_imshow.coords['b']).data)


whole_mat_imshow = whole_mat_imshow.rename({'a': r'$\sin(\theta_{in})$', 'b': r'$\sin(\theta_{out})$'})


ax2 = fig.add_subplot(1,3,2, aspect='equal')
ax2.text(-0.15, 1, 'TMM')
ax2 = whole_mat_imshow.plot.imshow(ax=ax2, cmap=seamap)

sprs_front = load_npz(mat_path_RCWA)

full_f = sprs_front[wl_index].todense()
# full_r = sprs_rear[wl_index].todense()

summat = theta_summary(full_f, angle_vector, options['n_theta_bins'], "front")

## reflection

#whole_mat = xr.concat((summat, summat_back), dim=r'$\theta_{in}$')
summat_back = summat[options['n_theta_bins']:]

whole_mat_imshow = summat_back.rename({r'$\theta_{in}$': 'a', r'$\theta_{out}$': 'b'})

whole_mat_imshow = whole_mat_imshow.assign_coords(a=np.sin(whole_mat_imshow.coords['a']).data,
                                                  b=np.sin(whole_mat_imshow.coords['b']).data)


whole_mat_imshow = whole_mat_imshow.rename({'a': r'$\sin(\theta_{in})$', 'b': r'$\sin(\theta_{out})$'})



ax3 = fig.add_subplot(1,3,3, aspect='equal')
ax3.text(-0.15, 1, 'RCWA')
ax3 = whole_mat_imshow.plot.imshow(ax=ax3, cmap=seamap)


plt.show()




sprs_front = load_npz(mat_path_RT)

full_f = sprs_front[wl_index].todense()
# full_r = sprs_rear[wl_index].todense()

summat = theta_summary(full_f, angle_vector, options['n_theta_bins'], "front")

## reflection

#whole_mat = xr.concat((summat, summat_back), dim=r'$\theta_{in}$')
summat_back = summat[:options['n_theta_bins']]

whole_mat_imshow = summat_back.rename({r'$\theta_{in}$': 'a', r'$\theta_{out}$': 'b'})

whole_mat_imshow = whole_mat_imshow.assign_coords(a=np.sin(whole_mat_imshow.coords['a']).data,
                                                  b=np.sin(whole_mat_imshow.coords['b']).data)


whole_mat_imshow = whole_mat_imshow.rename({'a': r'$\sin(\theta_{in})$', 'b': r'$\sin(\theta_{out})$'})


palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
palhf.reverse()
seamap = mpl.colors.ListedColormap(palhf)
fig = plt.figure(figsize=(12,3.5))
ax = fig.add_subplot(1,3,1, aspect='equal')
ax.text(-0.15, 1, 'RT')
ax = whole_mat_imshow.plot.imshow(ax=ax, cmap=seamap)


sprs_front = load_npz(mat_path_TMM)

full_f = sprs_front[wl_index].todense()
# full_r = sprs_rear[wl_index].todense()

summat = theta_summary(full_f, angle_vector, options['n_theta_bins'], "front")

## reflection

#whole_mat = xr.concat((summat, summat_back), dim=r'$\theta_{in}$')
summat_back = summat[:options['n_theta_bins']]

whole_mat_imshow = summat_back.rename({r'$\theta_{in}$': 'a', r'$\theta_{out}$': 'b'})

whole_mat_imshow = whole_mat_imshow.assign_coords(a=np.sin(whole_mat_imshow.coords['a']).data,
                                                  b=np.sin(whole_mat_imshow.coords['b']).data)


whole_mat_imshow = whole_mat_imshow.rename({'a': r'$\sin(\theta_{in})$', 'b': r'$\sin(\theta_{out})$'})


ax2 = fig.add_subplot(1,3,2, aspect='equal')
ax2.text(-0.15, 1, 'TMM')
ax2 = whole_mat_imshow.plot.imshow(ax=ax2, cmap=seamap)

sprs_front = load_npz(mat_path_RCWA)

full_f = sprs_front[wl_index].todense()
# full_r = sprs_rear[wl_index].todense()

summat = theta_summary(full_f, angle_vector, options['n_theta_bins'], "front")

## reflection

#whole_mat = xr.concat((summat, summat_back), dim=r'$\theta_{in}$')
summat_back = summat[:options['n_theta_bins']]

whole_mat_imshow = summat_back.rename({r'$\theta_{in}$': 'a', r'$\theta_{out}$': 'b'})

whole_mat_imshow = whole_mat_imshow.assign_coords(a=np.sin(whole_mat_imshow.coords['a']).data,
                                                  b=np.sin(whole_mat_imshow.coords['b']).data)


whole_mat_imshow = whole_mat_imshow.rename({'a': r'$\sin(\theta_{in})$', 'b': r'$\sin(\theta_{out})$'})



ax3 = fig.add_subplot(1,3,3, aspect='equal')
ax3.text(-0.15, 1, 'RCWA')
ax3 = whole_mat_imshow.plot.imshow(ax=ax3, cmap=seamap)


plt.show()
