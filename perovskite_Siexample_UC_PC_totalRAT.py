import numpy as np
from solcore.structure import Layer
from transfer_matrix_method.lookup_table import make_TMM_lookuptable
from solcore import material
from solcore.material_system import create_new_material
import seaborn as sns
import matplotlib as mpl
import matplotlib
font = {'family' : 'Lato Medium',
        'size'   : 14}
matplotlib.rc('font', **font)
from solcore.absorption_calculator.nk_db import search_db

from structure import Interface, RTgroup, BulkLayer, Structure
from ray_tracing.rt_lookup import RTSurface, RT
import math
from matrix_formalism.multiply_matrices import matrix_multiplication
from time import time
from sparse import stack, COO, save_npz, concatenate
from angles import make_angle_vector
import xarray as xr
from config import results_path
import os

import matplotlib.pyplot as plt

### structure ###
# Air
# ---- pyramids: layers are GaInP, GaAs
# Si
# ---- pyramids: layers are GaAs
# Air

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
           'I_thresh': 1e-4,
           'coherent': True,
           'coherency_list': None,
           'lookuptable_angles': 500,
           #'prof_layers': [1,2],
           'n_rays': 1000000,
           'random_angles': False,
           'nx': 5, 'ny': 5,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi/2,
           'only_incidence_angle': True
           }

theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                                       options['c_azimuth'])
#create_new_material('Perovskite_CsBr', 'examples/CsBr10p_1to2_n.txt', 'examples/CsBr10p_1to2_k.txt')
#create_new_material('ITO_lowdoping', 'examples/model_back_ito_n.txt', 'examples/model_back_ito_k.txt')
#create_new_material('Ag_Jiang', 'examples/Ag_UNSW_n.txt', 'examples/Ag_UNSW_k.txt')
#create_new_material('aSi_i', 'examples/model_i_a_silicon_n.txt', 'examples/model_i_a_silicon_k.txt')
#create_new_material('aSi_p', 'examples/model_p_a_silicon_n.txt', 'examples/model_p_a_silicon_k.txt')
#create_new_material('aSi_n', 'examples/model_n_a_silicon_n.txt', 'examples/model_n_a_silicon_k.txt')
#create_new_material('MgF2_RdeM', 'examples/MgF2_RdeM_n.txt', 'examples/MgF2_RdeM_k.txt')
#create_new_material('C60', 'examples/C60_Ren_n.txt', 'examples/C60_Ren_k.txt')
#create_new_material('IZO', 'examples/IZO_Ballif_rO2_10pcnt_n.txt', 'examples/IZO_Ballif_rO2_10pcnt_k.txt')


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
Spiro = [12e-9, np.array([0,1]), np.array([1.65, 1.65]), np.array([0,0])]
SnO2 = [10e-9, np.array([0,1]), np.array([2, 2]), np.array([0,0])]

char_angle = math.radians(55)
Lx = 1
Ly = 1
h = Lx*math.tan(char_angle)/2
x = np.array([0, Lx/2, Lx, 0, Lx])
y = np.array([0, Ly/2, 0, Ly, Ly])
z = np.array([0, h, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf = RTSurface(Points)

Lx = 1
Ly = 1
h = Lx*math.tan(char_angle)/2
x = np.array([0, Lx/2, Lx, 0, Lx])
y = np.array([0, Ly/2, 0, Ly, Ly])
z = np.array([0, -h, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf_back = RTSurface(Points)

front_materials = [Layer(100e-9, MgF2), Layer(110e-9, IZO),
                   SnO2,
                   Layer(15e-9, C60),
                   Layer(1e-9, LiF),
                   Layer(440e-9, Perovskite),
                   Spiro,
                   Layer(6.5e-9, aSi_n), Layer(6.5e-9, aSi_i)]
back_materials = [Layer(6.5e-9, aSi_i), Layer(6.5e-9, aSi_p), Layer(240e-9, ITO_back)]


front_surf = Interface('RT_TMM', texture = surf, layers=front_materials, name = 'Perovskite_aSi_1e6',
                       coherent=True, prof_layers = [6])
back_surf = Interface('RT_TMM', texture = surf_back, layers=back_materials, name = 'aSi_ITO_1e6',
                      coherent=True, prof_layers = [1,2,3])

#back_surf = Interface('Mirror', texture = None, layers=[], name = 'mirror',
#                      coherent=True, prof_layers = [])


bulk_Si = BulkLayer(260e-6, Si, name = 'Si_bulk')

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

# making lookuptable

for i1, struct in enumerate(SC):
    if type(struct) == Interface:
        # is this an interface type which requires a lookup table?
        if struct.method == 'Mirror':
            diag = np.array([1]*int(len(angle_vector)/2))
            Rsp = stack([COO(np.diag(diag)) for _ in range(len(options['wavelengths']))])
            Tsp = COO([],[],(len(options['wavelengths']),int(len(angle_vector)/2),int(len(angle_vector)/2)))
            RTsp = concatenate((Rsp,Tsp), axis=1)
            Asp = COO([],[],(len(options['wavelengths']),0,int(len(angle_vector)/2)))
            savepath_RT = os.path.join(results_path, options['project_name'], struct.name + 'front' + 'RT.npz')
            savepath_A = os.path.join(results_path, options['project_name'], struct.name + 'front' + 'A.npz')
            save_npz(savepath_RT, RTsp)
            save_npz(savepath_A, Asp)

        if struct.method == 'RT_TMM':
            print('Making lookuptable for element ' + str(i1) + ' in structure')
            if i1 == 0:
                incidence = SC.incidence
            else:
                incidence = SC[i1-1].material # bulk material above

            if i1 == (len(SC) - 1):
                substrate = SC.transmission
            else:
                substrate = SC[i1+1].material # bulk material below

            coherent = struct.coherent
            if not coherent:
                coherency_list = struct.coherency_list
            else:
                coherency_list = None
            prof_layers = struct.prof_layers

            allres = make_TMM_lookuptable(struct.layers, substrate, incidence, struct.name,
                                          options, coherent, coherency_list, prof_layers)
            if i1 == 0:
                palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
                palhf.reverse()
                seamap = mpl.colors.ListedColormap(palhf)

                allres_plot = allres.assign_coords(angle=allres.coords['angle'] * 180 / np.pi)
                fig1 = plt.figure()
                labels = ['a) R (front)', 'b) T (front)', 'c) R (back)', 'd) T (back)']
                lab_ind = 0
                for j1, side in enumerate([1, -1]):
                    ax1 = fig1.add_subplot(2, 2, j1 + 1)
                    cax=allres_plot['R'].sel(side=side, pol='u').plot.imshow('wl', 'angle', ax=ax1,
                                                                         vmin=0, vmax=1.001,
                                                                         cmap=seamap
                                                                         )
                    print(np.max(allres_plot['R'].sel(side=side, pol='u')))
                    #cbar = plt.colorbar(fig1)
                    #cbar.remove()
                    if side == 1:
                        ax1.set_ylabel('Angle (°)')
                    else:
                        ax1.set_ylabel('')
                    ax1.set_xlabel('')
                    ax1.get_xaxis().set_visible(False)

                    ax1.set_title(labels[lab_ind])
                    lab_ind += 1
                    ax1.yaxis.set_ticks([0, 45, 90])
                    ax2 = fig1.add_subplot(2, 2, j1 + 3)
                    allres_plot['T'].sel(side=side, pol='u').plot.imshow('wl', 'angle', ax=ax2,
                                                                         vmin=0, vmax=1.001,
                                                                         cmap=seamap)
                    ax2.set_xlabel('Wavelength (nm)')
                    if side == 1:
                        ax2.set_ylabel('Angle (°)')
                    else:
                        ax2.set_ylabel('')
                    ax2.set_title(labels[lab_ind])
                    lab_ind += 1
                    ax2.yaxis.set_ticks([0, 45, 90])
                fig1.savefig('LT1')
                fig2 = plt.figure()
                for j1, side in enumerate([1, -1]):

                    for i1, layer in enumerate(prof_layers):
                        ax = fig2.add_subplot(len(prof_layers), 2, 2 * (i1) + j1 + 1)
                        allres_plot['Alayer'].sel(side=side,
                                                  pol='u', layer=layer).plot.imshow('wl', 'angle', ax=ax,
                                                                                    vmin=0, vmax=1.001,
                                                                         cmap=seamap)
                        ax.set_xlabel('')
                        if side == 1:
                            ax.set_ylabel('Angle (°)')
                        else:
                            ax.set_ylabel('')
                        # ax.set_title('')
                        ax.yaxis.set_ticks([0, 45, 90])
                        ax.set_title('')
                    ax.set_xlabel('Wavelength (nm)')
                    plt.show()
                fig2.savefig('LT2')

# make matrices by ray tracing

for i1, struct in enumerate(SC):
    if type(struct) == Interface:
        # is this an interface type which requires a lookup table?
        if struct.method == 'RT_TMM':
            print('Ray tracing with TMM lookup table for element ' + str(i1) + ' in structure')
            if i1 == 0:
                incidence = SC.incidence
            else:
                incidence = SC[i1-1].material # bulk material above

            if i1 == (len(SC) - 1):
                substrate = SC.transmission
                which_sides = ['front']
            else:
                substrate = SC[i1+1].material # bulk material below
                which_sides = ['front', 'rear']

            coherent = struct.coherent
            if not coherent:
                coherency_list = struct.coherency_list
            else:
                coherency_list = None

            if len(struct.prof_layers) > 0:
                prof = True
            else:
                prof = False

            n_abs_layers = len(struct.layers)

            group = RTgroup(textures=[struct.texture])
            for side in which_sides:
                if side == 'front' and i1 == 0 and options['only_incidence_angle']:
                    only_incidence_angle = True
                else:
                    only_incidence_angle = False
                print(only_incidence_angle)
                RT(group, incidence, substrate, struct.name, options, 1, side,
                   n_abs_layers, prof, only_incidence_angle)

        if struct.method == 'RT_Fresnel':
            print('Ray tracing with Fresnel equations for element ' + str(i1) + ' in structure')
            if i1 == 0:
                incidence = SC.incidence
            else:
                incidence = SC[i1-1].material # bulk material above

            if i1 == (len(SC) - 1):
                substrate = SC.transmission
                which_sides = ['front']
            else:
                substrate = SC[i1+1].material # bulk material below
                which_sides = ['front', 'rear']

            group = RTgroup(textures=[struct.texture])
            for side in which_sides:
                RT(group, incidence, substrate, struct.name, options, 0, side, 0, False)


# matrix multiplication
bulk_mats = []
bulk_widths = []
layer_widths = []
n_layers = []
layer_names = []

for i1, struct in enumerate(SC):
    if type(struct) == BulkLayer:
        bulk_mats.append(struct.material)
        bulk_widths.append(struct.width)
    if type(struct) == Interface:
        layer_names.append(struct.name)
        n_layers.append(len(struct.layers))

if options['calc_profile']:
    for i1, struct in enumerate(SC):
        if type(struct) == Interface:
            layer_widths.append((np.array(struct.widths)*1e9).tolist())


RAT, results_per_pass = matrix_multiplication(bulk_mats, bulk_widths, options,
                                                           layer_widths, n_layers, layer_names)


R_per_pass = np.sum(results_per_pass['r'][0], 2)
R_0 = R_per_pass[0]
R_escape = np.sum(R_per_pass[1:, :], 0)
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)[:,[0,1,3,5,7,8]]
results_per_layer_back = np.sum(results_per_pass['a'][1], 0)

allres = np.flip(np.hstack((R_0[:,None], R_escape[:,None], results_per_layer_front, RAT['A_bulk'].T,
                    results_per_layer_back, RAT['T'].T)),1)

from solcore.light_source import LightSource
from solcore.constants import q
light_source = LightSource(source_type='standard', version='AM1.5d', x=options['wavelengths'],
                           output_units='photon_flux_per_m', concentration=1) # define the input light source as AM1.5G


photon_flux = light_source._get_photon_flux_per_m(options['wavelengths'])
Jph_Si = q*np.trapz(photon_flux*RAT['A_bulk'], options['wavelengths'])[0]*1000/(100**2)  # A/m^2
Jph_Perovskite = q*np.trapz(photon_flux*results_per_layer_front[:,3], options['wavelengths'])*1000/(100**2)
#

# Plot
import seaborn as sns
#pal = sns.cubehelix_palette(13, start = 0)
#pal = sns.color_palette()
pal = sns.cubehelix_palette(13, start=.5, rot=-.9)
pal.reverse()
sns.palplot(pal)

from scipy.ndimage.filters import gaussian_filter1d

ysmoothed = gaussian_filter1d(allres, sigma=1, axis=0)

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

fig.savefig('samplefigure', bbox_inches='tight')
plt.show()


# redistribution matrix
from sparse import load_npz
from config import results_path
import os
from angles import make_angle_vector, theta_summary

sprs = load_npz(os.path.join(results_path, options['project_name'], SC[0].name + 'rearRT.npz'))

full = sprs[0].todense()



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
fig.savefig('matrix', bbox_inches='tight')
#ax = Tth.plot.imshow(ax=ax)
