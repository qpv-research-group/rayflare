import xarray as xr
import numpy as np
from rayflare.transfer_matrix_method.tmm import tmm_structure
import os
from solcore.absorption_calculator import OptiStack


def make_TMM_lookuptable(layers, incidence, transmission, surf_name, options, structpath,
                         coherent=True, coherency_list=None, prof_layers=None, sides=None):
    """
    Takes a layer stack and calculates and stores lookup tables for use with the ray-tracer.

    :param layers: a list of layers. These can be Solcore 'Layer' objects, or any other layer format accepted \
    by the Solcore class 'OptiStack'.
    :param incidence: semi-incidence medium. Should be an isntance of a Solcore material object
    :param transmission: semi-infinite transmission medium. Should be an instance of a Solcore material object
    :param surf_name: name of the surfaces, for storing the lookup table (string).
    :param options: dictionary of options
    :param coherent: boolean. True if all the layers in the stack (excluding the semi-inifinite incidence and \
    transmission medium) are coherent, False otherwise. Default True.
    :param coherency_list: list. List of 'c' (coherent) and 'i' (incoherent) for each layer excluding incidence and \
    transmission media. Only needs to be provided if coherent = False. Default = None
    :param prof_layers: Indices of the layers in which the parameters relating to the absorption profile should be \
    calculated and stored. Layer 0 is the incidence medium.
    :param sides: List of which sides of incidence should all parameters be calculated for; 1 indicates incidence from \
    the front and -1 is rea    if not os.path.isdir(structpath):
        os.mkdir(structpath)r incidence. Default = [1, -1]
    :return: xarray Dataset with the R, A, T and (if relevant) absorption profile coefficients for each \
    wavelength, angle, polarization, side of incidence.
    """

    if sides is None:
        sides = [1, -1]

    savepath = os.path.join(structpath, surf_name + '.nc')
    if os.path.isfile(savepath):
        print('Existing lookup table found')
        allres = xr.open_dataset(savepath)
    else:
        wavelengths = options['wavelengths']
        n_angles = options['lookuptable_angles']
        thetas = np.linspace(0, np.pi/2, n_angles)
        if prof_layers is not None:
            profile = True
        else:
            profile = False

        n_layers = len(layers)
        optlayers = OptiStack(layers, substrate=transmission, incidence=incidence)
        optlayers_flip = OptiStack(layers[::-1], substrate=incidence, incidence=transmission)
        optstacks = [optlayers, optlayers_flip]

        if coherency_list is not None:
            coherency_lists = [coherency_list, coherency_list[::-1]]
        else:
            coherency_lists = [['c']*n_layers]*2
        # can calculate by angle, already vectorized over wavelength
        pols = ['s', 'p']

        R = xr.DataArray(np.empty((2, 2, len(wavelengths), n_angles)),
                         dims=['side', 'pol', 'wl', 'angle'],
                         coords={'side': sides, 'pol': pols, 'wl': wavelengths*1e9, 'angle': thetas},
                         name='R')
        T = xr.DataArray(np.empty((2, 2, len(wavelengths), n_angles)),
                         dims=['side', 'pol', 'wl', 'angle'],
                         coords={'side': sides, 'pol': pols, 'wl': wavelengths*1e9, 'angle': thetas},
                         name='T')
        Alayer = xr.DataArray(np.empty((2, 2, n_angles, len(wavelengths), n_layers)),
                              dims=['side', 'pol', 'angle', 'wl', 'layer'],
                              coords={'side': sides, 'pol': pols,
                                      'wl': wavelengths*1e9,
                                      'angle': thetas,
                                      'layer': range(1, n_layers + 1)}, name='Alayer')

        if profile:
            Aprof = xr.DataArray(np.empty((2, 2, n_angles, 6, len(prof_layers), len(wavelengths))),
                                  dims=['side', 'pol', 'angle', 'coeff', 'layer', 'wl'],
                              coords={'side': sides, 'pol': pols,
                                      'wl': wavelengths*1e9,
                                      'angle': thetas,
                                      'layer': prof_layers,
                                      'coeff': ['A1', 'A2', 'A3_r', 'A3_i', 'a1', 'a3']}, name='Aprof')

        pass_options = {}

        pass_options['wavelengths'] = wavelengths
        pass_options['depth_spacing'] = 1e5

        for i1, side in enumerate(sides):
            R_loop = np.empty((len(wavelengths), n_angles))
            T_loop = np.empty((len(wavelengths), n_angles))
            Alayer_loop = np.empty((n_angles, len(wavelengths), n_layers))
            if profile:
                Aprof_loop = np.empty((n_angles, 6, len(prof_layers), len(wavelengths)))

            pass_options['coherent'] = coherent
            pass_options['coherency_list'] = coherency_lists[i1]

            for pol in pols:

                for i3, theta in enumerate(thetas):

                    pass_options['pol'] = pol
                    pass_options['theta_in'] = theta

                    tmm_struct =  tmm_structure(optstacks[i1])
                    res = tmm_struct.calculate(pass_options, profile=profile, layers=prof_layers)
                    R_loop[:, i3] = np.real(res['R'])
                    T_loop[:, i3] = np.real(res['T'])
                    Alayer_loop[i3, :, :] = np.real(res['A_per_layer'])
                    if profile:
                        Aprof_loop[i3, :, :, :] = np.real(res['profile_coeff'])

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

        allres.to_netcdf(savepath)

    return allres
