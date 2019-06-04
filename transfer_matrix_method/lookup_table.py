import xarray as xr
import numpy as np
from solcore import material
from solcore.structure import Layer
from transfer_matrix_method.transfer_matrix import calculate_rat, OptiStack
import matplotlib.pyplot as plt
import os
from config import results_path

def make_TMM_lookuptable(layers, transmission, incidence, options):

    surf_name = options['surf_name']
    wavelengths = options['wavelengths']
    #pol = options['pol']
    coherent = options['coherent']
    coherency_list = options['coherency_list']
    n_angles = options['lookuptable_angles']
    thetas = np.linspace(0, np.pi/2, n_angles)
    if options['prof_layers'] is not None:
        profile = True
        prof_layers = options['prof_layers']
    else:
        profile = False
        prof_layers = None

    n_layers = len(layers)
    optlayers = OptiStack(layers, substrate=transmission, incidence=incidence)
    optlayers_flip = OptiStack(layers[::-1], substrate=incidence, incidence=transmission)
    optstacks = [optlayers, optlayers_flip]

    if coherency_list is not None:
        coherency_lists = [coherency_list, coherency_list[::-1]]
    else:
        coherency_lists = [None, None]
    # can calculate by angle, already vectorized over wavelength
    sides = [1, -1]
    pols = ['s', 'p']
    

    R = xr.DataArray(np.empty((2, 2, len(wavelengths), n_angles)), 
                     dims=['side', 'pol', 'wl', 'angle'],
                     coords={'side': [1, -1], 'pol': pols, 'wl': wavelengths, 'angle': thetas},
                     name='R')
    T = xr.DataArray(np.empty((2, 2, len(wavelengths), n_angles)), 
                     dims=['side', 'pol', 'wl', 'angle'],
                     coords={'side': [1, -1], 'pol': pols, 'wl': wavelengths, 'angle': thetas},
                     name='T')
    Alayer = xr.DataArray(np.empty((2, 2, n_angles, len(wavelengths), n_layers)),
                          dims=['side', 'pol', 'angle', 'wl', 'layer'],
                          coords={'side': [1, -1], 'pol': pols,
                                  'wl': wavelengths, 
                                  'angle': thetas,
                                  'layer': range(1, n_layers + 1)}, name='Alayer')

    if profile:
        Aprof = xr.DataArray(np.empty((2, 2, n_angles, 6, len(prof_layers), len(wavelengths))),
                              dims=['side', 'pol', 'angle', 'coeff', 'layer', 'wl'],
                          coords={'side': [1, -1], 'pol': pols,
                                  'wl': wavelengths,
                                  'angle': thetas,
                                  'layer': prof_layers,
                                  'coeff': ['A1', 'A2', 'A3_r', 'A3_i', 'a1', 'a3']}, name='Aprof')


    for i1, side in enumerate(sides):
        R_loop = np.empty((len(wavelengths), n_angles))
        T_loop = np.empty((len(wavelengths), n_angles))
        Alayer_loop = np.empty((n_angles, len(wavelengths), n_layers),dtype=np.complex_)
        if profile:
            Aprof_loop = np.empty((n_angles, 6, len(prof_layers), len(wavelengths)))

        for i2, pol in enumerate(pols):

            for i3, theta in enumerate(thetas):
                res = calculate_rat(optstacks[i1], wavelengths, angle=theta, pol=pol,
                                coherent=coherent, coherency_list=coherency_lists[i1], profile=profile, layers=prof_layers)
                R_loop[:, i3] = np.real(res['R'])
                T_loop[:, i3] = np.real(res['T'])
                Alayer_loop[i3, :, :] = np.real(res['A_per_layer'].T)
                if profile:
                    Aprof_loop[i3, :, :, :] = res['profile_coeff']

            # sometimes get very small negative values (like -1e-20)
            R_loop[R_loop<0] = 0
            T_loop[T_loop<0] = 0
            Alayer_loop[Alayer_loop<0] = 0

            if side == -1:
                Alayer_loop = np.flip(Alayer_loop, axis = 2)
                if profile:
                    Aprof_loop = np.flip(Aprof_loop, axis = 2)

            R.loc[dict(side=side, pol=pol)] = R_loop
            T.loc[dict(side=side, pol=pol)] = T_loop
            Alayer.loc[dict(side=side, pol=pol)] = Alayer_loop

            if profile:
                Aprof.loc[dict(side=side, pol=pol)] = Aprof_loop
                Aprof.transpose('side', 'pol', 'wl', 'angle', 'layer', 'coeff')

    Alayer = Alayer.transpose('side', 'pol', 'wl', 'angle', 'layer')

    if profile:
        allres = xr.merge([R, T, Alayer, Aprof])
    else:
        allres = xr.merge([R, T, Alayer])

    unpol = allres.reduce(np.mean, 'pol').assign_coords(pol='u').expand_dims('pol')
    allres = allres.merge(unpol)
    structpath = os.path.join(results_path, options['struct_name'])
    if not os.path.isdir(structpath):
        os.mkdir(structpath)
    savepath = os.path.join(structpath, surf_name + '.nc')
    allres.to_netcdf(savepath)

    return allres
