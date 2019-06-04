import xarray as xr
import numpy as np
from solcore import material
from solcore.structure import Layer
from transfer_matrix_method.transfer_matrix import calculate_rat, OptiStack, calculate_absorption_profile
import matplotlib.pyplot as plt
from transfer_matrix_method.lookup_table import make_TMM_lookuptable
from transfer_matrix_method.tmm_core_vec import absorp_analytic_fn

wavelengths = np.linspace(500, 1200, 600)
options =  {'wavelengths': wavelengths, 'I_thresh': 1e-4,
            'nx': 5, 'ny': 5, 'max_passes': 100, 'parallel': False, 'n_rays': 40000,
            'phi_symmetry': np.pi/2, 'n_theta_bins': 100, 'c_azimuth': 0.25,
            'random_angles': True, 'lookuptable_angles': 1000, 'pol': 's',
            'coherent': True, 'coherency_list': ['c', 'c'], 'struct_name': 'testing',
            'prof_layers': [1,2], 'surf_name': 'GaInPGaAsonSi'}#,


Ge = material('Ge')()
GaAs = material('GaAs')()
GaInP = material('GaInP')(In=0.5)
Air = material('Air')()
Si = material('Si')()

#layers = [Layer(100e-9, GaAs), Layer(500e-9, Ge)]
layers = [Layer(500e-9, GaInP), Layer(700e-9, GaAs)]

n_layers = len(layers)

substrate = Si
incidence = Air


allres = make_TMM_lookuptable(layers, substrate, incidence, options)


#unpol = allres.reduce(np.mean, 'pol').assign_coords(pol='u').expand_dims('pol')

#allres = allres.merge(unpol)

for side in [1,-1]:
    plt.figure()
    plt.subplot(2, 2, 1)
    allres['R'].sel(side=side, pol='s').plot.imshow('wl', 'angle')
    plt.subplot(2, 2, 3)
    allres['T'].sel(side=side, pol='s').plot.imshow('wl', 'angle')

    plt.subplot(2, 2, 2)
    allres['R'].sel(side=side, pol='p').plot.imshow('wl', 'angle')
    plt.subplot(2, 2, 4)
    allres['T'].sel(side=side, pol='p').plot.imshow('wl', 'angle')
    plt.show()

for side in [1, -1]:

    plt.figure()
    for i1 in range(1, n_layers + 1):
        plt.subplot(n_layers, 1, i1)
        allres['Alayer'].sel(side=side, pol = 'u', layer=i1).plot.imshow('wl', 'angle')
    plt.show()

angles = np.arange(0, options['lookuptable_angles'], 25)

#wavelengths in nm so alpha should be in nm^-1
a1 = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u', coeff = 'a1', layer=1)].Aprof[angles].data
A2 = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u', coeff = 'A2', layer=1)].Aprof[angles].data
a3 = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u', coeff = 'a3', layer=1)].Aprof[angles].data
A1 = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u', coeff = 'A1', layer=1)].Aprof[angles].data
A3_r = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u', coeff = 'A3_r', layer=1)].Aprof[angles].data
A3_i = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u', coeff = 'A3_i', layer=1)].Aprof[angles].data

R = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u')].R[angles]
T = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u')].T[angles]
A = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u')].Alayer[angles]

depth=np.linspace(0,500,500)
#
# a_z = absorp_analytic_fn().set(A1, A2, A3, a1, a3).run(depth)
#
# plt.figure()
# plt.plot(a_z.T)
# plt.legend(angles)


plt.figure()
plt.plot(angles, R)
plt.plot(angles, A[:,0])
#plt.plot(angles, A[:,1])
plt.legend(['R', 'A1', 'A2'])

A2/a1

# import xarray as xr
# import numpy as np
# from solcore import material
# from solcore.structure import Layer
# from transfer_matrix_method.transfer_matrix import calculate_rat, OptiStack
# import matplotlib.pyplot as plt
# from transfer_matrix_method.lookup_table import make_TMM_lookuptable
#
# wavelengths = np.linspace(400, 1200, 10)
# options =  {'wavelengths': wavelengths, 'I_thresh': 1e-4,
#             'nx': 5, 'ny': 5, 'max_passes': 100, 'parallel': False, 'n_rays': 40000,
#             'phi_symmetry': np.pi/2, 'n_theta_bins': 100, 'c_azimuth': 0.25,
#             'random_angles': True, 'lookuptable_angles': 200, 'pol': 's',
#             'coherent': True, 'coherency_list': None, 'struct_name': 'testing'}#,
#
# Ge = material('Ge')()
# GaAs = material('GaAs')()
# Air = material('Air')()
#
#
# layers = [Layer(100e-9, Ge), Layer(50e-9, GaAs)]
#
#
# substrate = Ge
# incidence = Air
#
# a = OptiStack(layers, substrate=substrate, incidence=incidence)
#
# layer_indices = [1,2]
#
# coherent = False
# coherency_list = ['c', 'i']
#
# output = calculate_rat(a, wavelengths, 0.5, 'u', coherent, coherency_list, True, layer_indices)

layers = [Layer(500e-9, GaAs)]
check_profile = OptiStack(layers, substrate=substrate, incidence=incidence, no_back_reflection=False)
res = calculate_absorption_profile(check_profile, wavelengths, angle=0.2, pol='u', z_limit=500, steps_size=1,
                                   no_back_reflection=False)
plt.figure()
plt.plot(res['position'], res['absorption'][0])

res_param_s = calculate_rat(check_profile, wavelengths, angle=0.2, pol='s', profile=True, layers=[1])
params_s = res_param_s['profile_coeff']
res_param_p = calculate_rat(check_profile, wavelengths, angle=0.2, pol='u', profile=True, layers=[1])
params_p = res_param_p['profile_coeff']

test_params = params_p[:,0,0]

A1 = test_params[0]
A2 = test_params[1]
A3_r = test_params[2]
A3_i = test_params[3]
a1 = test_params[4]
a3 = test_params[5]

z = np.linspace(0,500,500)

part1 = A1 * np.exp(a1 * z)
part2 = A2 * np.exp(-a1 * z)
part3 = (A3_r + 1j*A3_i) * np.exp(1j * a3 * z)
part4 = (A3_r - 1j*A3_i) * np.exp(-1j * a3 * z)

plt.plot(z, part1+part2+part3+part4)

lookuptable = xr.open_dataset('C://Users//pmpea//Box Sync//Optics package//results//testing//GaAsGaAsstack_prof2.nc')
test_params = lookuptable['Aprof'].loc[dict(side=1, pol='u')].sel(angle=0.2, wl=600, layer=1,method='nearest').data

A1 = test_params[0]
A2 = test_params[1]
A3_r = test_params[2]
A3_i = test_params[3]
a1 = test_params[4]
a3 = test_params[5]

z = np.linspace(0,500,500)

part1 = A1 * np.exp(a1 * z)
part2 = A2 * np.exp(-a1 * z)
part3 = (A3_r + 1j*A3_i) * np.exp(1j * a3 * z)
part4 = (A3_r - 1j*A3_i) * np.exp(-1j * a3 * z)

plt.plot(z, part1+part2+part3+part4)
plt.legend(['poyn/absor', 'from params', 'loaded lookuptable'])
plt.show()
