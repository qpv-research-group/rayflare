import xarray as xr
import numpy as np
from solcore import material
from solcore.structure import Layer
from transfer_matrix_method.transfer_matrix import calculate_rat, OptiStack
import matplotlib.pyplot as plt

wavelengths = np.linspace(400, 1000, 100)
options =  {'wavelengths': wavelengths, 'I_thresh': 1e-4,
            'nx': 5, 'ny': 5, 'max_passes': 100, 'parallel': False, 'n_rays': 40000,
            'phi_symmetry': np.pi/2, 'n_theta_bins': 100, 'c_azimuth': 0.25,
            'random_angles': True, 'lookuptable_angles': 200, 'pol': 'u',
            'coherent': True, 'coherency_list': None}#,


Ge = material('Ge')()
GaAs = material('GaAs')()
Air = material('Air')()

wl = options['wavelengths']
pol = options['pol']
coherent = options['coherent']
coherency_list = options['coherency_list']


layers = OptiStack([Layer(100e-9, Ge), Layer(50e-9, GaAs)], substrate = Ge, incidence=Air)
n_angles = options['lookuptable_angles']
thetas = np.linspace(0, np.pi/2, n_angles)
#results = xr.DataArray(np.empty((2, len(wavelengths), n_angles)), dims=['side', 'wl', 'theta'],
#                             coords={'side': [1, -1], 'wl': wavelengths, 'theta': thetas})

n_layers = 2

# can group by angle, already vectorized over wavelength
loop_res = np.empty(len(thetas))

R_loop = np.empty((len(wavelengths), n_angles))
T_loop = np.empty((len(wavelengths), n_angles))
Alayer_loop = np.empty((n_angles, len(wavelengths), n_layers))

for i1, theta in enumerate(thetas):
    res = calculate_rat(layers, wavelengths, angle=theta, pol=pol,
                     coherent=coherent, coherency_list=coherency_list, no_back_reflection=False)
    R_loop[:, i1] = res['R']
    T_loop[:, i1] = res['T']
    Alayer_loop[i1, :, :] = res['A_per_layer'].T

# sometimes get very small negative values (like -1e-20)
R_loop[R_loop<0] = 0
T_loop[T_loop<0] = 0
Alayer_loop[Alayer_loop<0] = 0

R = xr.DataArray(R_loop, dims=['wl', 'angle'], coords={'wl': wavelengths, 'angle': thetas,
    'sin_theta': (['angle'], np.sin(thetas))}, name='R')
T = xr.DataArray(T_loop, dims=['wl', 'angle'], coords={'wl': wavelengths, 'angle': thetas,
    'sin_theta': (['angle'], np.sin(thetas))}, name='T')
Alayer = xr.DataArray(Alayer_loop, dims=['angle', 'wl', 'layer'],
                      coords={'wl': wavelengths, 'angle': thetas,
                              'sin_theta': (['angle'], np.sin(thetas)),
                              'layer': range(1, n_layers+1)}, name='Alayer').transpose('wl', 'angle', 'layer')

allres = xr.merge([R, T, Alayer])

plt.figure()
plt.subplot(2,1,1)
allres['R'].plot.imshow('wl', 'angle')
plt.subplot(2,1,2)
allres['T'].plot.imshow('wl', 'angle')
plt.show()

plt.figure()
for i1 in range(1, n_layers+1):
    plt.subplot(n_layers,1,i1)
    allres['Alayer'].sel(layer=i1).plot.imshow('wl', 'angle')
plt.show()

allres.to_netcdf(path='lookuptable.nc', mode='w')
