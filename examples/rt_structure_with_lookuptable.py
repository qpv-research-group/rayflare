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
wavelengths = np.linspace(300, 1200, 30)*1e-9

options = default_options()
options.wavelengths = wavelengths
options.project_name = 'perovskite_Si_example'

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

## ray-tracing

from rayflare.ray_tracing import rt_structure
from rayflare.textures import regular_pyramids

nxy = 25

calc = True

# setting options
options.wavelengths = wavelengths
options.nx = nxy
options.ny = nxy
options.n_rays = 2 * nxy ** 2
options.depth_spacing = si('1um')
options.parallel = True

Spiro = [1.65, 0]
SnO2 = [2, 0]

front_layers = [Layer(100e-9, MgF2),
    Layer(110e-9, IZO),
    Layer(15e-9, C60),
    Layer(1e-9, LiF),
    Layer(440e-9, Perovskite),
    Layer(6.5e-9, aSi_n),
    Layer(6.5e-9, aSi_i)]


triangle_surf = regular_pyramids(55, upright=True, size=1,
                                 interface_layers=front_layers
                                 )
triangle_surf_back = regular_pyramids(55, upright=False, size=1,
                                      interface_layers=[Layer(200e-9, Ag)])

# set up ray-tracing options
rtstr = rt_structure(textures=[triangle_surf, triangle_surf_back],
                     materials= [Si],
                    widths=[260e-6],
                     incidence=Air, transmission=Ag,
                     use_TMM=True, options=options, save_location="current")
result = rtstr.calculate(options)
#result = result_new

#result = np.vstack((options['wavelengths']*1e9, result['R'], result['R0'], result['T'], result['A_per_layer'][:,0])).T
checking = result["A_per_interface"]
#
fig=plt.figure(figsize=(9,3.7))
plt.subplot(1,1,1)
plt.plot(wavelengths*1e9, result['R'], '-o', color=pal[0], label=r'R$_{total}$', fillstyle='none')
plt.plot(wavelengths*1e9, result['R0'], '-o', color=pal[1], label=r'R$_0$', fillstyle='none')
plt.plot(wavelengths*1e9, result['T'], '-o', color=pal[2], label=r'T', fillstyle='none')
plt.plot(wavelengths*1e9, result['A_per_layer'][:,0], '-o', color=pal[3], label=r'A', fillstyle='none')
plt.plot(wavelengths*1e9, result['A_per_interface'][0], '-o')
plt.plot(wavelengths*1e9, result['A_per_interface'][1], '--o')

plt.title('a)', loc='left')
plt.plot(-1, -1, '-ok', label='RayFlare')
plt.plot(-1, -1, '--k', label='PVLighthouse')
plt.xlabel('Wavelength (nm)')
plt.ylabel('R / A / T')
plt.ylim(0, 1)
plt.xlim(300, 1200)

plt.legend()
plt.show()

# one_wl = checking[6]
#
# n_rays_absorbed = np.zeros(len(one_wl))
#
# for i1, layer_data in enumerate(one_wl):
#     n_rays_absorbed[i1] = len(layer_data)
#
#     if len(one_wl[i1]) > 0:
#         data = np.stack(one_wl[i1])
#         data_per_layer = np.mean(data, axis=0)

# first entry will always be just zeros
