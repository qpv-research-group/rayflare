import numpy as np
import os
# solcore imports
from solcore.structure import Layer
from solcore import material
from solcore.light_source import LightSource
from solcore.constants import q
from solcore.material_system.create_new_material import create_new_material

from textures.standard_rt_textures import regular_pyramids
from structure import Interface, BulkLayer, Structure
from process_structure import process_structure, calculate_RAT

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
font = {'family' : 'Lato Medium',
        'size'   : 14}
matplotlib.rc('font', **font)

# matrix multiplication
wavelengths = np.linspace(300, 1200, 112)*1e-9
options = {'nm_spacing': 0.5,
           'project_name': 'UC_PC',
           'calc_profile': False,
           'n_theta_bins': 100,
           'c_azimuth': 0.25,
           'pol': 'u',
           'wavelengths': wavelengths,
           'theta_in': 1e-6, 'phi_in': 1e-6,
           'I_thresh': 0.001,
           'coherent': True,
           'coherency_list': None,
           'lookuptable_angles': 200,
           #'prof_layers': [1,2],
           'n_rays': 100000,
           'random_angles': False,
           'nx': 5, 'ny': 5,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi/2,
           'only_incidence_angle': True
           }

cur_path = os.path.dirname(os.path.abspath(__file__))
# new materials from data
create_new_material('Perovskite_CsBr', os.path.join(cur_path, 'data/CsBr10p_1to2_n.txt'), os.path.join(cur_path, 'data/CsBr10p_1to2_k.txt'))
create_new_material('ITO_lowdoping', os.path.join(cur_path, 'data/model_back_ito_n.txt'), os.path.join(cur_path, 'data/model_back_ito_k.txt'))
create_new_material('Ag_Jiang', os.path.join(cur_path, 'data/Ag_UNSW_n.txt'), os.path.join(cur_path, 'data/Ag_UNSW_k.txt'))
create_new_material('aSi_i', os.path.join(cur_path, 'data/model_i_a_silicon_n.txt'),os.path.join(cur_path, 'data/model_i_a_silicon_k.txt'))
create_new_material('aSi_p', os.path.join(cur_path, 'data/model_p_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_p_a_silicon_k.txt'))
create_new_material('aSi_n', os.path.join(cur_path, 'data/model_n_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_n_a_silicon_k.txt'))
create_new_material('MgF2_RdeM', os.path.join(cur_path, 'data/MgF2_RdeM_n.txt'), os.path.join(cur_path, 'data/MgF2_RdeM_k.txt'))
create_new_material('C60', os.path.join(cur_path, 'data/C60_Ren_n.txt'), os.path.join(cur_path, 'data/C60_Ren_k.txt'))
create_new_material('IZO', os.path.join(cur_path, 'data/IZO_Ballif_rO2_10pcnt_n.txt'), os.path.join(cur_path, 'data/IZO_Ballif_rO2_10pcnt_k.txt'))


Si = material('Si')()
Air = material('Air')()
MgF2 = material('MgF2_RdeM')()
ITO_back = material('ITO_lowdoping')()
Perovskite = material('Perovskite_CsBr')()
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

front_surf = Interface('RT_TMM', texture = surf, layers=front_materials, name = 'Perovskite_aSi_1e6',
                       coherent=True, prof_layers = np.arange(1,10))
back_surf = Interface('RT_TMM', texture = surf_back, layers=back_materials, name = 'aSi_ITO_1e6',

                      coherent=True, prof_layers = np.arange(1,4))


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


allres = np.flip(np.hstack((RAT['R'].T, results_per_layer_front, RAT['A_bulk'].T,
                    results_per_layer_back, RAT['T'].T)),1)

light_source = LightSource(source_type='standard', version='AM1.5d', x=options['wavelengths'],
                           output_units='photon_flux_per_m', concentration=1) # define the input light source as AM1.5G


# calculated photogenerated current (Jsc with 100% EQE)
photon_flux = light_source._get_photon_flux_per_m(options['wavelengths'])
Jph_Si = q*np.trapz(photon_flux*RAT['A_bulk'], options['wavelengths'])[0]*1000/(100**2)  # A/m^2
Jph_Perovskite = q*np.trapz(photon_flux*results_per_layer_front[:,3], options['wavelengths'])*1000/(100**2)
#

pal = sns.cubehelix_palette(13, start=.5, rot=-.9)
pal.reverse()

from scipy.ndimage.filters import gaussian_filter1d

ysmoothed = gaussian_filter1d(allres, sigma=1, axis=0)

bulk_A_text= ysmoothed[:,4]

# plot total R, A, T
fig = plt.figure()
ax = plt.subplot(111)
ax.stackplot(options['wavelengths']*1e9, ysmoothed.T,
              labels=['Ag', 'ITO', 'aSi-n', 'aSi-i', 'c-Si (bulk)', 'aSi-i', 'aSi-p',
                      'Perovskite','C$_{60}$','IZO',
            'MgF$_2$', 'R$_{escape}$', 'R$_0$'], colors = pal)
lgd=ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Absorption')
ax.set_xlim(300, 1200)
ax.set_ylim(0, 1)
ax.text(530, 0.5, 'Perovskite: \n' + str(round(Jph_Perovskite,1)) + ' mA/cm$^2$', ha='center')
ax.text(900, 0.5, 'Si: \n' + str(round(Jph_Si,1)) + ' mA/cm$^2$', ha='center')

fig.savefig('samplefigure.png', bbox_inches='tight', format='png')
plt.show()

# plot absorption profiles

if options['calc_profile']:
    profile = results[2]
    bpf = results[3]  # bulk profule
    layer_widths = []

    for i1, struct in enumerate(SC):
        if type(struct) == Interface:
            layer_widths.append((np.array(struct.widths)*1e9).tolist())

    #plt.show()
    import seaborn as sns
    import matplotlib

    offset = np.cumsum([0]+layer_widths[0])
    prof_plot = profile[0]
    prof_plot = prof_plot.assign_coords(z=np.arange(0, np.sum(layer_widths[0]), options['nm_spacing']))
    material_labels = ['MgF$_2$', 'IZO', 'SnO$_2$', 'C$_{60}$', 'LiF', 'Perovskite', 'Spiro-TTB', 'aSi-n', 'aSi-i']
    pal2 = sns.cubehelix_palette(4, start=.5, rot=-.9)



    fig = plt.figure()
    ax = plt.subplot(111)
    j1=0
    for j1,i1 in enumerate([0,30,111]):
          z = np.arange(0, np.sum(layer_widths[0]), options['nm_spacing'])
          ax.plot(z, prof_plot[i1,:], color=pal2[j1+1], label = str(round(options['wavelengths'][i1]*1e9,1)))
          j1+=1
    ax.set_ylabel('Absorbed energy density (nm$^{-1}$)')
    ax.legend(title='Wavelength (nm)')
    ax.set_xlabel('Distance into surface (nm)')
    ax.set_ylim(0,0.017)
    ax.set_xlim(0,750)
    for i1 in [0,1,3,5,7]:
        ax.text(offset[i1], np.max(prof_plot[0].sel(z=offset[i1]))+0.0003, material_labels[i1], rotation=0)


    plt.show()

from angles import theta_summary, make_angle_vector
from config import results_path
from sparse import load_npz
import xarray as xr

_, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                       options['c_azimuth'])

sprs = load_npz(os.path.join(results_path, options['project_name'], SC[0].name + 'rearRT.npz'))

full = sprs[99].todense()


summat, Rsum, Tsum = theta_summary(full, angle_vector)

Rth = summat[0:100,:]
Tth = summat[100:, :]
Rth = xr.DataArray(Rth, dims=[r'$\sin(\theta_{out})$', r'$\sin(\theta_{in})$'], coords={r'$\sin(\theta_{out})$': np.linspace(0,1,100),
                                                                            r'$\sin(\theta_{in})$': np.linspace(0,1,100)})
Tth = xr.DataArray(Tth, dims=[r'$\sin(\theta_{out})$', r'$\sin(\theta_{in})$'], coords={r'$\sin(\theta_{out})$': np.linspace(0,1,100),
                                                                            r'$\sin(\theta_{in})$': np.linspace(0,1,100)})

import matplotlib as mpl
palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
palhf.reverse()
seamap = mpl.colors.ListedColormap(palhf)
fig = plt.figure()
ax = plt.subplot(111)
ax = Rth.plot.imshow(ax=ax, cmap=seamap)
#ax = plt.subplot(212)
fig.savefig('matrix.png', bbox_inches='tight', format='png')
#ax = Tth.plot.imshow(ax=ax)



front_surf = Interface('RT_TMM', texture = surf, layers=front_materials, name = 'Perovskite_aSi_1e6',
                       coherent=True, prof_layers = np.arange(1,10))
back_surf = Interface('Mirror', texture = surf_back, layers=back_materials, name = 'mirror')

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options)

results = calculate_RAT(SC, options)
RAT = results[0]
results_per_pass = results[1]

R_per_pass = np.sum(results_per_pass['r'][0], 2)
R_0 = R_per_pass[0]
R_escape = np.sum(R_per_pass[1:, :], 0)
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)[:,[0,1,3,5,7,8]]
results_per_layer_back = np.sum(results_per_pass['a'][1], 0)
allres = np.flip(np.hstack((R_0[:,None], R_escape[:,None], results_per_layer_front, RAT['A_bulk'].T,
                    results_per_layer_back, RAT['T'].T)),1)

ysmoothed = gaussian_filter1d(allres, sigma=1, axis=0)
bulk_A_flat = ysmoothed[:,1]


front_surf = Interface('RT_TMM', texture = surf, layers=front_materials, name = 'Perovskite_aSi_1e6',
                       coherent=True, prof_layers = np.arange(1,10))
back_surf = Interface('RT_TMM', texture = surf_back, layers=back_materials, name = 'aSi_ITO_1e6',

                      coherent=True, prof_layers = np.arange(1,4))


bulk_Si = BulkLayer(360e-6, Si, name = 'Si_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options)

results = calculate_RAT(SC, options)

RAT = results[0]
results_per_pass = results[1]


# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)[:,[0,1,3,5,7,8]]

results_per_layer_back = np.sum(results_per_pass['a'][1], 0)


allres = np.flip(np.hstack((RAT['R'].T, results_per_layer_front, RAT['A_bulk'].T,
                    results_per_layer_back, RAT['T'].T)),1)

ysmoothed = gaussian_filter1d(allres, sigma=1, axis=0)

bulk_A_text_thicker= ysmoothed[:,4]


font = {'family' : 'Lato Medium',
        'size'   : 16}
matplotlib.rc('font', **font)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(options['wavelengths']*1e9, bulk_A_flat, color= pal[0], linewidth=2,
        label='Perfect back mirror')
ax.plot(options['wavelengths']*1e9, bulk_A_text, color= pal[5], linewidth=2,
        label='Textured back surface')
ax.plot(options['wavelengths']*1e9, bulk_A_text_thicker, color= pal[10], linewidth=2,
        label=r'Text. back + 360 $\mu$m c-Si')

lgd=ax.legend(loc='bottom left')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Absorption in c-Si')
ax.set_xlim(900, 1200)
ax.set_ylim(0, 1)

fig.savefig('textcomp.png', bbox_inches='tight', format='png')
plt.show()