import numpy as np
import os

from solcore.structure import Layer
from solcore import material, si
from solcore.light_source import LightSource
from solcore.constants import q

from rayflare.textures import regular_pyramids, planar_surface
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
wavelengths = np.linspace(300, 1200, 40)*1e-9

options = default_options()
options.wavelengths = wavelengths
options.project_name = 'perovskite_Si_example'
options.phi_symmetry = np.pi/2

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

triangle_surf = regular_pyramids(elevation_angle=55, upright=True, size=1,
                                 interface_layers=front_layers
                                 )

triangle_surf_inc = regular_pyramids(elevation_angle=55, upright=True, size=1,
                                 interface_layers=front_layers,
                                     coherency_list=["i"]*len(front_layers)
                                 )

triangle_surf_back = regular_pyramids(elevation_angle=55, upright=False, size=1,
                                      interface_layers=[Layer(200e-9, Ag)])

planar_surf = planar_surface(size=1)

# set up ray-tracing options
rtstr = rt_structure(textures=[triangle_surf, planar_surf, triangle_surf_back],
                     materials= [Si, Si],
                    widths=[130e-6, 130e-6],
                     incidence=Air, transmission=Ag,
                     use_TMM=True, options=options, save_location="current")
result = rtstr.calculate(options)

options.project_name = "inc"

rtstr_inc = rt_structure(textures=[triangle_surf_inc, planar_surf, triangle_surf_back],
                     materials= [Si, Si],
                    widths=[130e-6, 130e-6],
                     incidence=Air, transmission=Ag,
                     use_TMM=True, options=options, save_location="current")
result_inc = rtstr_inc.calculate(options)
#result = result_new

#result = np.vstack((options['wavelengths']*1e9, result['R'], result['R0'], result['T'], result['A_per_layer'][:,0])).T
checking = result["A_per_interface"]
#
fig=plt.figure(figsize=(9,3.7))
plt.subplot(1,1,1)
plt.plot(wavelengths*1e9, result['R'], '-o', color=pal[0], label=r'R$_{total}$', fillstyle='none')
plt.plot(wavelengths*1e9, result['R0'], '-o', color=pal[1], label=r'R$_0$', fillstyle='none')
plt.plot(wavelengths*1e9, result['T'], '-o', color=pal[2], label=r'T', fillstyle='none')
plt.plot(wavelengths*1e9, np.sum(result['A_per_layer'],1), '-o', color=pal[3], label=r'A', fillstyle='none')
plt.plot(wavelengths*1e9, result['A_per_interface'][0], '-o')
plt.plot(wavelengths*1e9, result['A_per_interface'][1], '-o')

plt.plot(wavelengths*1e9, result_inc['R'], '--o', color=pal[0], label=r'R$_{total}$', fillstyle='none')
plt.plot(wavelengths*1e9, result_inc['R0'], '--o', color=pal[1], label=r'R$_0$', fillstyle='none')
plt.plot(wavelengths*1e9, result_inc['T'], '--o', color=pal[2], label=r'T', fillstyle='none')
plt.plot(wavelengths*1e9, np.sum(result_inc['A_per_layer'],1), '--o', color=pal[3], label=r'A', fillstyle='none')
plt.plot(wavelengths*1e9, result_inc['A_per_interface'][0], '--o')
plt.plot(wavelengths*1e9, result_inc['A_per_interface'][1], '--o')

plt.title('a)', loc='left')
plt.plot(-1, -1, '-ok', label='RayFlare')
plt.plot(-1, -1, '--k', label='PVLighthouse')
plt.xlabel('Wavelength (nm)')
plt.ylabel('R / A / T')
plt.ylim(0, 1)
plt.xlim(300, 1200)

plt.legend()
plt.show()

import xarray as xr
lookuptable = xr.open_dataset(os.path.join(rtstr_inc.save_location, "int_0" + ".nc"))

data = lookuptable.loc[dict(side=1, pol="u")].interp(
        angle=0.2, wl=1100
    )

a = lookuptable["Alayer"].loc[dict(side=-1, pol="u")]
print(a.where(a==a.max(), drop=True))
b = a.sum(dim="layer")

from rayflare.transfer_matrix_method import tmm_structure


ost = front_layers[::-1]

options.pol = "u"
options.coherent = False
options.coherency_list = ["i"]*len(front_layers)
options.theta_in = 1.2

a = tmm_structure(ost, incidence=Si, transmission=Air, no_back_reflection=False)
a_sharp = tmm_structure([Layer(440e-9, Perovskite)], incidence=Si, transmission=Air, no_back_reflection=False)


res = a.calculate(options)
options.coherent = False
options.coherency_list = ["i"]
# res_sharp = a_sharp.calculate(options)

# total A seems fine? Because scaled....
plt.figure()
plt.plot(wavelengths*1e9, res["A_per_layer"])
plt.plot(wavelengths*1e9, res["R"], "--k")
plt.plot(wavelengths*1e9, res["T"], "--r")
# plt.plot(wavelengths*1e9, res_sharp["R"], 'r--')
# plt.plot(wavelengths*1e9, res_sharp["A_per_layer"], 'b--')
# plt.plot(wavelengths*1e9, res_sharp["A_per_layer"], 'r--')
plt.legend(["aSi_n", "Perovskite","MgF2"])
plt.ylim(-1,2)
plt.show()


from solcore.absorption_calculator.tmm_core_vec import inc_tmm, inc_absorp_in_each_layer

n_list = [np.array([4.976, 3.94 , 3.614, 3.52 ]),
          np.array([3.0027+3.2014j  , 3.9528+0.32338j , 3.5126+0.039199j,
       3.3643+0.023508j]),
       #    np.array([3.0031+3.2018j  , 3.9528+0.32444j , 3.5126+0.041007j,
       # 3.3643+0.026026j]),
          np.array([1.74      +0.96j      , 2.47963404+0.18781702j,
       2.21130799+0.j        , 2.1370813 +0.j        ]),
       #    np.array([1.40881337+0.j, 1.39190178+0.j, 1.3878255 +0.j, 1.38526595+0.j]),
       #    np.array([2.10211415+0.59159079j, 2.02373836+0.03972636j,
       # 1.9585232 +0.0187167j , 1.93801332+0.01336343j]),
       #    np.array([2.49880534+0.13326956j, 2.00757794+0.00918188j,
       # 1.80529446+0.j        , 1.56865962+0.j        ]),
          np.array([1.439164+0.001162j, 1.421732+0.000569j, 1.418722+0.000379j,
       1.417674+0.000282j]),
          np.array([1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j])]
d_list = [np.inf,
          6.5,
          # 6.5,
          440.0,
          # 1.0, 14.999999999999998, 110.0,
          99.99999999999999,
          np.inf]
c_list = ['i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i']
c_list = ["i"]*5
th_0 = 1
lam_vac = [300, 600,  900, 1200]

out = inc_tmm("s", n_list, d_list, c_list, th_0, lam_vac)

A_per_layer = np.array(inc_absorp_in_each_layer(out))

# plt.figure()
# plt.plot(lam_vac, out["R"], '-k')
# plt.plot(lam_vac, A_per_layer[0,:], '--r')
# plt.plot(lam_vac, out["T"], '--r')
# plt.plot(lam_vac, A_per_layer[1:-1].T)
# plt.show()
