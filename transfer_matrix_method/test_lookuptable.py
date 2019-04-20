import xarray as xr
import numpy as np
from solcore import material
from solcore.structure import Layer
from transfer_matrix_method.transfer_matrix import calculate_rat, OptiStack
import matplotlib.pyplot as plt
from transfer_matrix_method.lookup_table import make_TMM_lookuptable
from transfer_matrix_method.tmm_core_vec import absorp_analytic_fn

wavelengths = np.linspace(400, 1200, 150)
options =  {'wavelengths': wavelengths, 'I_thresh': 1e-4,
            'nx': 5, 'ny': 5, 'max_passes': 100, 'parallel': False, 'n_rays': 40000,
            'phi_symmetry': np.pi/2, 'n_theta_bins': 100, 'c_azimuth': 0.25,
            'random_angles': True, 'lookuptable_angles': 200, 'pol': 's',
            'coherent': False, 'coherency_list': ['c', 'c'], 'struct_name': 'testing'}#,

Ge = material('Ge')()
GaAs = material('GaAs')()
Air = material('Air')()
Si = material('Si')()

layers = [Layer(100e-9, GaAs), Layer(500e-9, Ge)]

n_layers = len(layers)

substrate = Si
incidence = Air

surf_name = 'GaAsGestack_prof'

prof_layers = [1,2]

allres = make_TMM_lookuptable(layers, substrate, incidence, surf_name, options, profile=True, prof_layers=prof_layers)

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
a1 = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u', coeff = 'a1', layer=2)].Aprof[angles].data
A2 = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u', coeff = 'A2', layer=2)].Aprof[angles].data
a3 = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u', coeff = 'a3', layer=2)].Aprof[angles].data
A1 = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u', coeff = 'A1', layer=2)].Aprof[angles].data
A3 = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u', coeff = 'A3', layer=2)].Aprof[angles].data

R = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u')].R[angles]
T = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u')].T[angles]
A = allres.loc[dict(side=1, wl=wavelengths[70], pol = 'u')].Alayer[angles]

depth=np.linspace(0,500,500)

a_z = absorp_analytic_fn().set(A1, A2, A3, a1, a3).run(depth)

plt.figure()
plt.plot(a_z.T)
plt.legend(angles)


plt.figure()
plt.plot(angles, R)
plt.plot(angles, A[:,0])
plt.plot(angles, A[:,1])
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
