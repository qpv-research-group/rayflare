import xarray as xr
import numpy as np
from solcore import material
from solcore.structure import Layer
from transfer_matrix_method.transfer_matrix import calculate_rat, OptiStack
import matplotlib.pyplot as plt
from transfer_matrix_method.lookup_table import make_TMM_lookuptable

wavelengths = np.linspace(400, 1200, 150)
options =  {'wavelengths': wavelengths, 'I_thresh': 1e-4,
            'nx': 5, 'ny': 5, 'max_passes': 100, 'parallel': False, 'n_rays': 40000,
            'phi_symmetry': np.pi/2, 'n_theta_bins': 100, 'c_azimuth': 0.25,
            'random_angles': True, 'lookuptable_angles': 200, 'pol': 'u',
            'coherent': True, 'coherency_list': None, 'struct_name': 'testing'}#,

Ge = material('Ge')()
GaAs = material('GaAs')()
Air = material('Air')()


layers = [Layer(100e-9, Ge), Layer(50e-9, GaAs)]

n_layers = len(layers)

substrate = Ge
incidence = Air

surf_name = 'GeGaAsstack'

allres = make_TMM_lookuptable(layers, substrate, incidence, surf_name, options)

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

