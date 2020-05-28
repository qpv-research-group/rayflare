import numpy as np
import os
# solcore imports
from solcore.structure import Layer
from solcore import material
from solcore.light_source import LightSource
from solcore.constants import q

from textures.standard_rt_textures import regular_pyramids
from structure import Interface, BulkLayer, Structure
from matrix_formalism.multiply_matrices import calculate_RAT
from matrix_formalism import process_structure
from solcore.material_system import create_new_material
from options import default_options

import matplotlib.pyplot as plt
import seaborn as sns

from cycler import cycler

pal = sns.cubehelix_palette()

cols = cycler('color', pal)

params = {'legend.fontsize': 'small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small',
          'axes.prop_cycle': cols}

plt.rcParams.update(params)

# cur_path = os.path.dirname(os.path.abspath(__file__))
# # new materials from data
# create_new_material('Perovskite_CsBr_1p6eV', os.path.join(cur_path, 'data/CsBr10p_1to2_n_shifted.txt'), os.path.join(cur_path, 'data/CsBr10p_1to2_k_shifted.txt'))
# create_new_material('ITO_lowdoping', os.path.join(cur_path, 'data/model_back_ito_n.txt'), os.path.join(cur_path, 'data/model_back_ito_k.txt'))
# create_new_material('Ag_Jiang', os.path.join(cur_path, 'data/Ag_UNSW_n.txt'), os.path.join(cur_path, 'data/Ag_UNSW_k.txt'))
# create_new_material('aSi_i', os.path.join(cur_path, 'data/model_i_a_silicon_n.txt'),os.path.join(cur_path, 'data/model_i_a_silicon_k.txt'))
# create_new_material('aSi_p', os.path.join(cur_path, 'data/model_p_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_p_a_silicon_k.txt'))
# create_new_material('aSi_n', os.path.join(cur_path, 'data/model_n_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_n_a_silicon_k.txt'))
# create_new_material('MgF2_RdeM', os.path.join(cur_path, 'data/MgF2_RdeM_n.txt'), os.path.join(cur_path, 'data/MgF2_RdeM_k.txt'))
# create_new_material('C60', os.path.join(cur_path, 'data/C60_Ren_n.txt'), os.path.join(cur_path, 'data/C60_Ren_k.txt'))
# create_new_material('IZO', os.path.join(cur_path, 'data/IZO_Ballif_rO2_10pcnt_n.txt'), os.path.join(cur_path, 'data/IZO_Ballif_rO2_10pcnt_k.txt'))
#

#font = {'family' : 'Lato Medium',
#        'size'   : 14}
#matplotlib.rc('font', **font)

# matrix multiplication
wavelengths = np.linspace(300, 1200, 150)*1e-9

options = default_options
options.nm_spacing = 0.5
options.wavelengths = wavelengths
options.project_name = 'Perovskite_Si_1e6_shifted'
options.n_rays = 500000
options.n_theta_bins = 50
options.phi_symmetry = np.pi/4
options.I_thresh = 1e-4
options.lookuptable_angles = 200
options.parallel = True
options.only_incidence_angle = True
options.random_ray_position = False
options.random_angles = False
options.nx = 21
options.ny = 21
options.c_azimuth = 0.25

Si = material('Si')()
Air = material('Air')()
MgF2 = material('MgF2_RdeM')()
ITO_back = material('ITO_lowdoping')()
Perovskite = material('Perovskite_CsBr_1p6eV')()
Ag = material('Ag_Jiang')()
aSi_i = material('aSi_i')()
aSi_p = material('aSi_p')()
aSi_n = material('aSi_n')()
LiF = material('LiF')()
IZO = material('IZO')()
C60 = material('C60')()

# materials with constant n, zero k
Spiro = [12e-9, np.array([0,1]), np.array([1.65, 1.65]), np.array([0,0])]
SnO2 = [10e-9, np.array([0,1]), np.array([2, 2]), np.array([0,0])]

# stack based on doi:10.1038/s41563-018-0115-4
front_materials = [Layer(100e-9, MgF2), Layer(110e-9, IZO),
                   SnO2,
                   Layer(15e-9, C60),
                   Layer(1e-9, LiF),
                   Layer(440e-9, Perovskite),
                   Spiro,
                   Layer(6.5e-9, aSi_n), Layer(6.5e-9, aSi_i)]
back_materials = [Layer(6.5e-9, aSi_i), Layer(6.5e-9, aSi_p), Layer(240e-9, ITO_back)]

# whether pyramids are upright or inverted is relative to front incidence.
# so if the same etch is applied to both sides of a slab of silicon, one surface
# will have 'upright' pyramids and the other side will have 'not upright' (inverted)
# pyramids in the model

surf = regular_pyramids(elevation_angle=55, upright=True)
surf_back = regular_pyramids(elevation_angle=55, upright=False)

front_surf = Interface('RT_TMM', texture = surf, layers=front_materials, name = 'Perovskite_aSi',
                       coherent=True)
back_surf = Interface('RT_TMM', texture = surf_back, layers=back_materials, name = 'aSi_ITO_2',
                      coherent=True)



bulk_Si = BulkLayer(260e-6, Si, name = 'Si_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options)

results = calculate_RAT(SC, options)

RAT = results[0]
results_per_pass = results[1]



R_per_pass = np.sum(results_per_pass['r'][0], 2)
R_0 = R_per_pass[0]
R_escape = np.sum(R_per_pass[1:, :], 0)

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)[:,[0,1,3,5,7,8]]

results_per_layer_back = np.sum(results_per_pass['a'][1], 0)


allres = np.flip(np.hstack((R_0[:,None], R_escape[:,None], results_per_layer_front, RAT['A_bulk'].T,
                    results_per_layer_back, RAT['T'].T)),1)

# calculated photogenerated current (Jsc with 100% EQE)

spectr_flux = LightSource(source_type='standard', version='AM1.5g', x=wavelengths,
                           output_units='photon_flux_per_m', concentration=1).spectrum(wavelengths)[1]

Jph_Si = q * np.trapz(RAT['A_bulk'][0] * spectr_flux, wavelengths)/10 # mA/cm2
Jph_Perovskite =  q * np.trapz(results_per_layer_front[:,3] * spectr_flux, wavelengths)/10 # mA/cm2

pal = sns.cubehelix_palette(13, start=.5, rot=-.9)
pal.reverse()

from scipy.ndimage.filters import gaussian_filter1d

ysmoothed = gaussian_filter1d(allres, sigma=2, axis=0)

bulk_A_text= ysmoothed[:,4]

# plot total R, A, T
fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
ax.stackplot(options['wavelengths']*1e9, allres.T,
              labels=['Ag', 'ITO', 'aSi-n', 'aSi-i', 'c-Si (bulk)', 'aSi-i', 'aSi-p',
                      'Perovskite','C$_{60}$','IZO',
            'MgF$_2$', 'R$_{escape}$', 'R$_0$'], colors = pal)
lgd=ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('R/A/T')
ax.set_xlim(300, 1200)
ax.set_ylim(0, 1)
ax.text(530, 0.5, 'Perovskite: \n' + str(round(Jph_Perovskite,1)) + ' mA/cm$^2$', ha='center')
ax.text(900, 0.5, 'Si: \n' + str(round(Jph_Si,1)) + ' mA/cm$^2$', ha='center')

fig.savefig('perovskite_Si_summary.pdf', bbox_inches='tight', format='pdf')
plt.show()


label = [str(x) if (x) % 5 == 0 else None for x in np.arange(22)]

pal = sns.cubehelix_palette(24, start=2.8, rot=0.7)
pal.reverse()
fig=plt.figure(figsize=(8, 3.5))
ax=fig.add_subplot(1,2,1)
ax.stackplot(options['wavelengths']*1e9, R_per_pass[1:25], colors=pal, labels=label)
ax.set_xlim([1000, 1200])
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Escape reflection')
ax.set_ylim([0,0.3])
ax.text(960, 0.3, 'a)')
ax.legend(loc='upper left')
#fig.savefig('perovskite_Si_R_per_pass.pdf', bbox_inches='tight', format='pdf')



#label = [str(x) if (x) % 5 == 0 else None for x in np.arange(25)]
#pal = sns.cubehelix_palette(25, start=2.8, rot=0.7)
#pal.reverse()
#fig=plt.figure()
ax2=fig.add_subplot(1,2,2)
ax2.stackplot(options['wavelengths']*1e9, results_per_pass['A'][0][0:24], colors=pal, labels=label)
ax2.set_xlim([1000, 1200])
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Bulk (Si) absorption per pass')
ax2.set_ylim([0,1])
ax2.text(960, 1, 'b)')
#ax2.legend(loc='upper right')
fig.savefig('perovskite_Si_A_R_per_pass.pdf', bbox_inches='tight', format='pdf')
plt.show()


if len(front_surf.prof_layers) > 0:
    profile = results[2]
    bpf = results[3]  # bulk profule
    layer_widths = []

    for i1, struct in enumerate(SC):
        if type(struct) == Interface:
            layer_widths.append((np.array(struct.widths)*1e9).tolist())

    #plt.show()
    import seaborn as sns

    offset = np.cumsum([0]+layer_widths[0])
    prof_plot = profile[0]
    print(profile[0].shape)
    material_labels = ['Perovskite']
    pal2 = sns.cubehelix_palette(4, start=.5, rot=-.9)

    fig = plt.figure()
    ax = plt.subplot(111)
    j1=0
    for j1,i1 in enumerate([0,5,19]):
          ax.plot(prof_plot[i1,:], color=pal2[j1+1], label = str(round(options['wavelengths'][i1]*1e9,1)))
          j1+=1
    ax.set_ylabel('Absorbed energy density (nm$^{-1}$)')
    ax.legend(title='Wavelength (nm)')
    ax.set_xlabel('Distance into surface (nm)')
    ax.set_ylim(0,0.017)
    ax.set_xlim(0,750)


    plt.show()

from angles import theta_summary, make_angle_vector
from config import results_path
from sparse import load_npz
import xarray as xr

import matplotlib as mpl
palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
palhf.reverse()
seamap = mpl.colors.ListedColormap(palhf)

_, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                       options['c_azimuth'])


sprs_front = load_npz(os.path.join(results_path, options['project_name'], SC[0].name + 'frontRT.npz'))
sprs_rear = load_npz(os.path.join(results_path, options['project_name'], SC[0].name + 'rearRT.npz'))

wl_to_plot = 1100e-9

wl_index = np.argmin(np.abs(wavelengths-wl_to_plot))

full_f = sprs_front[wl_index].todense()
full_r = sprs_rear[wl_index].todense()

#summat= theta_summary(full_f, angle_vector, options['n_theta_bins'], "front")

summat = theta_summary(full_r, angle_vector, options['n_theta_bins'], "rear")

## reflection

#whole_mat = xr.concat((summat, summat_back), dim=r'$\theta_{in}$')
summat_back = summat[options['n_theta_bins']:]

whole_mat_imshow = summat_back.rename({r'$\theta_{in}$': 'a', r'$\theta_{out}$': 'b'})

whole_mat_imshow = whole_mat_imshow.assign_coords(a=np.sin(whole_mat_imshow.coords['a']).data,
                                                  b=np.sin(whole_mat_imshow.coords['b']).data)


whole_mat_imshow = whole_mat_imshow.rename({'a': r'$\sin(\theta_{in})$', 'b': r'$\sin(\theta_{out})$'})

#whole_mat_imshow = whole_mat_imshow.interp(theta_in = np.linspace(0, np.pi, 100), theta_out =  np.linspace(0, np.pi, 100))

#whole_mat_imshow = whole_mat_imshow.rename({'theta_in': r'$\theta_{in}$', 'theta_out' : r'$\theta_{out}$'})





import matplotlib as mpl
palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
palhf.reverse()
seamap = mpl.colors.ListedColormap(palhf)
fig = plt.figure(figsize=(10,3.5))
ax = fig.add_subplot(1,2,1, aspect='equal')
ax.text(-0.15, 1, 'a)')
ax = whole_mat_imshow.plot.imshow(ax=ax, cmap=seamap)
#ax = plt.subplot(212)
#fig.savefig('perovskite_Si_frontsurf_rearR.pdf', bbox_inches='tight', format='pdf')
#ax = Tth.plot.imshow(ax=ax)

#plt.show()


## transmission

#whole_mat = xr.concat((summat, summat_back), dim=r'$\theta_{in}$')
summat_back = summat[:options['n_theta_bins']]

whole_mat_imshow = summat_back.rename({r'$\theta_{in}$': 'a', r'$\theta_{out}$': 'b'})

whole_mat_imshow = whole_mat_imshow.assign_coords(a=np.sin(whole_mat_imshow.coords['a']).data,
                                                  b=np.sin(whole_mat_imshow.coords['b']).data)


whole_mat_imshow = whole_mat_imshow.rename({'a': r'$\sin(\theta_{in})$', 'b': r'$\sin(\theta_{out})$'})

#whole_mat_imshow = whole_mat_imshow.interp(theta_in = np.linspace(0, np.pi, 100), theta_out =  np.linspace(0, np.pi, 100))

#whole_mat_imshow = whole_mat_imshow.rename({'theta_in': r'$\theta_{in}$', 'theta_out' : r'$\theta_{out}$'})





import matplotlib as mpl
palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
palhf.reverse()
seamap = mpl.colors.ListedColormap(palhf)
#fig = plt.figure(figsize=(5,4))
ax2 = fig.add_subplot(1,2,2, aspect='equal')
ax2.text(-0.15, 1, 'b)')
ax2 = whole_mat_imshow.plot.imshow(ax=ax2, cmap=seamap)
#ax = plt.subplot(212)
fig.savefig('perovskite_Si_frontsurf_rearRT.pdf', bbox_inches='tight', format='pdf')
#ax = Tth.plot.imshow(ax=ax)

plt.show()