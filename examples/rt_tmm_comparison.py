import numpy as np
import os

from solcore.structure import Layer
from solcore import material, si
from solcore.light_source import LightSource
from solcore.constants import q

from rayflare.textures import regular_pyramids
from rayflare.structure import Interface, BulkLayer, Structure
from rayflare.matrix_formalism import calculate_RAT, process_structure
from rayflare.options import default_options

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

cur_path = os.path.dirname(os.path.abspath(__file__))
# new materials from data (only need to add once, uncomment following lines to do so:

# from solcore.material_system import create_new_material
# create_new_material('Perovskite_CsBr_1p6eV', os.path.join(cur_path, 'data/CsBr10p_1to2_n_shifted.txt'), os.path.join(cur_path, 'data/CsBr10p_1to2_k_shifted.txt'))
# create_new_material('ITO_lowdoping', os.path.join(cur_path, 'data/model_back_ito_n.txt'), os.path.join(cur_path, 'data/model_back_ito_k.txt'))
# create_new_material('Ag_Jiang', os.path.join(cur_path, 'data/Ag_UNSW_n.txt'), os.path.join(cur_path, 'data/Ag_UNSW_k.txt'))
# create_new_material('aSi_i', os.path.join(cur_path, 'data/model_i_a_silicon_n.txt'),os.path.join(cur_path, 'data/model_i_a_silicon_k.txt'))
# create_new_material('aSi_p', os.path.join(cur_path, 'data/model_p_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_p_a_silicon_k.txt'))
# create_new_material('aSi_n', os.path.join(cur_path, 'data/model_n_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_n_a_silicon_k.txt'))
# create_new_material('MgF2_RdeM', os.path.join(cur_path, 'data/MgF2_RdeM_n.txt'), os.path.join(cur_path, 'data/MgF2_RdeM_k.txt'))
# create_new_material('C60', os.path.join(cur_path, 'data/C60_Ren_n.txt'), os.path.join(cur_path, 'data/C60_Ren_k.txt'))
# create_new_material('IZO', os.path.join(cur_path, 'data/IZO_Ballif_rO2_10pcnt_n.txt'), os.path.join(cur_path, 'data/IZO_Ballif_rO2_10pcnt_k.txt'))


# matrix multiplication
wavelengths = np.linspace(300, 1200, 50)*1e-9

options = default_options()
options.wavelengths = wavelengths
options.project_name = 'perovskite_Si_example'
options.n_rays = 2000
options.n_theta_bins = 30
options.nx = 2
options.ny = 2
options.depth_spacing = 1e-9

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

# materials with constant n, zero k. Layer width is in nm.
Spiro = [12, np.array([0,1]), np.array([1.65, 1.65]), np.array([0,0])]
SnO2 = [10, np.array([0,1]), np.array([2, 2]), np.array([0,0])]

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

front_surf = Interface('RT_TMM', texture = surf, layers=front_materials, name = 'Perovskite_aSi_widthcorr',
                       coherent=True, prof_layers=np.arange(1, 10))
# NOTE: depending on your computer, calculation the absorption profiles in the front surface may cause
# memory-related errors as it uses extremely large matrices. Hopefully this can be resolved in the future but if this is
# an issue, replace the above definition of the front surface so that the profiles aren't calculated:

# front_surf = Interface('RT_TMM', texture = surf, layers=front_materials, name = 'Perovskite_aSi_widthcorr',
#                        coherent=True)

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
plt.show()



## ray-tracing

from rayflare.ray_tracing import rt_structure
from rayflare.textures import regular_pyramids
from rayflare.options import default_options

nxy = 5

calc = True

# setting options
options = default_options()
options.wavelengths = np.linspace(300, 1201, 50) * 1e-9
options.nx = nxy
options.ny = nxy
options.n_rays = 2 * nxy ** 2
options.depth_spacing = si('1um')
options.parallel = True

Spiro = [1.65, 0]
SnO2 = [2, 0]


if calc:

    print('RT')
    triangle_surf = regular_pyramids(55, upright=True, size=1)
    triangle_surf_back = regular_pyramids(55, upright=False, size=1)

    # set up ray-tracing options
    # rtstr = rt_structure(textures=[triangle_surf]*9 + [triangle_surf_back]*3,
    #                      materials= [MgF2,
    #                                 IZO,
    #                                 SnO2,
    #                                 C60,
    #                                 LiF,
    #                                 Perovskite,
    #                                 Spiro,
    #                                 aSi_n,
    #                                 aSi_i,
    #                                 Si,
    #                                 aSi_i,
    #                                 aSi_p,
    #                                 ITO_back],
    #                      widths=[100e-9, 110e-9, 10e-9, 12e-9, 15e-9, 1e-9, 440e-9, 12e-9, 6.5e-9, 6.5e-9,
    #                              260e-6,
    #                              6.5e-6, 6.5e-6, 240e-9], incidence=Air, transmission=Ag)

    rtstr = rt_structure(textures=[triangle_surf, triangle_surf_back],
                        materials = [Si],
                        widths=[si('300um')], incidence=Air, transmission=Ag)
    result = rtstr.calculate(options)
    #result = result_new


    #result = np.vstack((options['wavelengths']*1e9, result['R'], result['R0'], result['T'], result['A_per_layer'][:,0])).T

else:
    # result = np.loadtxt('data/rayflare_fullrt_300um_2umpyramids_300_1200nm_3')
    0



fig=plt.figure(figsize=(9,3.7))
plt.subplot(1,1,1)
plt.plot(wavelengths*1e9, result['R'], '-o', color=pal[0], label=r'R$_{total}$', fillstyle='none')
plt.plot(wavelengths*1e9, result['R0'], '-o', color=pal[1], label=r'R$_0$', fillstyle='none')
plt.plot(wavelengths*1e9, result['T'], '-o', color=pal[2], label=r'T', fillstyle='none')
plt.plot(wavelengths*1e9, result['A_per_layer'][:,0], '-o', color=pal[3], label=r'A', fillstyle='none')

plt.title('a)', loc='left')
plt.plot(-1, -1, '-ok', label='RayFlare')
plt.plot(-1, -1, '--k', label='PVLighthouse')
plt.xlabel('Wavelength (nm)')
plt.ylabel('R / A / T')
plt.ylim(0, 1)
plt.xlim(300, 1200)

plt.legend()
plt.show()
