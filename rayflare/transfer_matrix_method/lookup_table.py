# Copyright (C) 2021-2024 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU Lesser General Public License (LGPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: p.pearce@unsw.edu.au

import xarray as xr
import numpy as np
from rayflare.transfer_matrix_method.tmm import tmm_structure
import os
from solcore.absorption_calculator import OptiStack

import logging
logging.basicConfig(format='%(levelname)s: %(message)s',level=logging.INFO)

def make_TMM_lookuptable(
    layers,
    incidence,
    transmission,
    surf_name,
    options,
    structpath,
    coherent=True,
    coherency_list=None,
    prof_layers=None,
    sides=None,
    overwrite=False,
):
    """
    Takes a layer stack and calculates and stores lookup tables for use with the ray-tracer.

    :param layers: a list of layers. These can be Solcore 'Layer' objects, or any other layer format accepted \
    by the Solcore class 'OptiStack'.
    :param incidence: semi-incidence medium. Should be an instance of a Solcore material object
    :param transmission: semi-infinite transmission medium. Should be an instance of a Solcore material object
    :param surf_name: name of the surfaces, for storing the lookup table (string).
    :param options: dictionary or State object containing user options
    :param structpath: file path where matrices will be stored or loaded from
    :param coherent: boolean. True if all the layers in the stack (excluding the semi-infinite incidence and \
    transmission medium) are coherent, False otherwise. Default True.
    :param coherency_list: list. List of 'c' (coherent) and 'i' (incoherent) for each layer excluding incidence and \
    transmission media. Only needs to be provided if coherent = False. Default = None
    :param prof_layers: Indices of the layers in which the parameters relating to the absorption profile should be \
    calculated and stored. Layer 0 is the incidence medium.
    :param sides: List of which sides of incidence should all parameters be calculated for; 1 indicates incidence from \
    the front and -1 is rear incidence. Default = [1, -1]
    :param overwrite: boolean. If True, existing saved lookup tables will be overwritten. Default = False.
    :return: xarray Dataset with the R, A, T and (if relevant) absorption profile coefficients for each \
    wavelength, angle, polarization, side of incidence.
    """

    if sides is None:
        sides = [1, -1]

    savepath = os.path.join(structpath, surf_name + ".nc")
    if os.path.isfile(savepath) and not overwrite:
        logging.info("Existing lookup table found")
        allres = xr.open_dataset(savepath)
    else:
        wavelengths = options["wavelength"]
        n_angles = options["lookuptable_angles"]
        thetas = np.linspace(0, (np.pi / 2) - 1e-3, n_angles)
        if prof_layers is not None:
            profile = True
            prof_layers_rev = len(layers) - np.array(prof_layers[::-1]) + 1
            prof_layer_list = [prof_layers, prof_layers_rev.tolist()]
        else:
            profile = False
            prof_layer_list = [None, None]

        n_layers = len(layers)
        optlayers = OptiStack(layers, substrate=transmission, incidence=incidence)
        optlayers_flip = OptiStack(
            layers[::-1], substrate=incidence, incidence=transmission
        )
        optstacks = [optlayers, optlayers_flip]

        if coherency_list is not None:
            coherency_list = np.array(coherency_list)
            coherency_list[np.array(optlayers.widths) == 0] = "c" # incoherent implementation doesn't work/
            # make sense for zero thickness, coherent implementation will correctly ignore the layer
            coherency_lists = [coherency_list.tolist(), coherency_list[::-1].tolist()]
        else:
            coherency_lists = [["c"] * n_layers] * 2
        # can calculate by angle, already vectorized over wavelength
        pols = ["s", "p"]

        R = xr.DataArray(
            np.empty((2, 2, len(wavelengths), n_angles)),
            dims=["side", "pol", "wl", "angle"],
            coords={
                "side": sides,
                "pol": pols,
                "wl": wavelengths * 1e9,
                "angle": thetas,
            },
            name="R",
        )
        T = xr.DataArray(
            np.empty((2, 2, len(wavelengths), n_angles)),
            dims=["side", "pol", "wl", "angle"],
            coords={
                "side": sides,
                "pol": pols,
                "wl": wavelengths * 1e9,
                "angle": thetas,
            },
            name="T",
        )
        Alayer = xr.DataArray(
            np.empty((2, 2, n_angles, len(wavelengths), n_layers)),
            dims=["side", "pol", "angle", "wl", "layer"],
            coords={
                "side": sides,
                "pol": pols,
                "wl": wavelengths * 1e9,
                "angle": thetas,
                "layer": range(1, n_layers + 1),
            },
            name="Alayer",
        )

        if profile:
            Aprof = xr.DataArray(
                np.empty((2, 2, n_angles, 6, len(prof_layers), len(wavelengths))),
                dims=["side", "pol", "angle", "coeff", "layer", "wl"],
                coords={
                    "side": sides,
                    "pol": pols,
                    "wl": wavelengths * 1e9,
                    "angle": thetas,
                    "layer": prof_layers,
                    "coeff": ["A1", "A2", "A3_r", "A3_i", "a1", "a3"],
                },
                name="Aprof",
            )

        pass_options = {}

        pass_options["wavelength"] = wavelengths
        pass_options["depth_spacing"] = 1e5
        # we don't actually want to calculate a profile, so the depth spacing
        # doesn't matter, but it needs to be set to something. Larger value means we don't make extremely large arrays
        # no reason during the calculation

        for i1, side in enumerate(sides):
            prof_layer_side = prof_layer_list[i1]
            R_loop = np.empty((len(wavelengths), n_angles))
            T_loop = np.empty((len(wavelengths), n_angles))
            Alayer_loop = np.empty((n_angles, len(wavelengths), n_layers))
            if profile:
                Aprof_loop = np.empty((n_angles, 6, len(prof_layers), len(wavelengths)))

            pass_options["coherent"] = coherent
            pass_options["coherency_list"] = coherency_lists[i1]

            for pol in pols:

                for i3, theta in enumerate(thetas):

                    pass_options["pol"] = pol
                    pass_options["theta_in"] = theta

                    tmm_struct = tmm_structure(optstacks[i1])
                    res = tmm_struct.calculate(
                        pass_options, profile=profile, layers=prof_layer_side
                    )

                    R_loop[:, i3] = np.real(res["R"])
                    T_loop[:, i3] = np.real(res["T"])
                    Alayer_loop[i3, :, :] = np.real(res["A_per_layer"])

                    if profile:
                        Aprof_loop[i3, :, :, :] = np.real(res["profile_coeff"])

                # sometimes get very small negative values (like -1e-20)
                R_loop[R_loop < 0] = 0
                T_loop[T_loop < 0] = 0
                Alayer_loop[Alayer_loop < 0] = 0

                if side == -1:
                    Alayer_loop = np.flip(Alayer_loop, axis=2)
                    # layers were upside down to do calculation; want labelling to be with
                    # respect to side = 1 for consistency
                    if profile:
                        Aprof_loop = np.flip(Aprof_loop, axis=2)

                R.loc[dict(side=side, pol=pol)] = R_loop
                T.loc[dict(side=side, pol=pol)] = T_loop
                Alayer.loc[dict(side=side, pol=pol)] = Alayer_loop

                if profile:
                    Aprof.loc[dict(side=side, pol=pol)] = Aprof_loop
                    Aprof.transpose("side", "pol", "wl", "angle", "layer", "coeff")

        Alayer = Alayer.transpose("side", "pol", "wl", "angle", "layer")

        if profile:
            allres = xr.merge([R, T, Alayer, Aprof])
        else:
            allres = xr.merge([R, T, Alayer])

        unpol = allres.reduce(np.mean, "pol").assign_coords(pol="u").expand_dims("pol")
        allres = allres.merge(unpol)

        allres.to_netcdf(savepath)

    return allres
