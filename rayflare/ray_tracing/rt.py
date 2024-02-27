# Copyright (C) 2021 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU General Public License (GPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: p.pearce@unsw.edu.au


import cProfile as c_profile

# In outer section of code
# pr = c_profile.Profile()
import numpy as np
import os
from scipy.spatial import Delaunay
from cmath import sin, cos, sqrt, acos, atan
from math import atan2
from random import random
from itertools import product
import xarray as xr
from sparse import COO, save_npz, stack
from joblib import Parallel, delayed
from copy import deepcopy
from warnings import warn
from solcore.state import State

from rayflare.angles import fold_phi, make_angle_vector, overall_bin
from rayflare.utilities import get_matrices_or_paths, get_savepath, get_wavelength
from rayflare.transfer_matrix_method.lookup_table import make_TMM_lookuptable
from .analytical_approximation import analytical_front_surface, lambertian_scattering, calculate_lambertian_profile

from rayflare import logger

def RT(
    group,
    incidence,
    transmission,
    surf_name,
    options,
    structpath,
    Fr_or_TMM=0,
    front_or_rear="front",
    n_absorbing_layers=0,
    calc_profile=None,
    only_incidence_angle=False,
    widths=None,
    save=True,
    overwrite=False,
):
    """Calculates the reflection/transmission and absorption redistribution matrices for an interface using
    either a previously calculated TMM lookup table or the Fresnel equations.

    :param group: an RTgroup object containing the surface textures
    :param incidence: incidence medium
    :param transmission: transmission medium
    :param surf_name: name of the surface (to save matrices)
    :param options: user options (State object)
    :param structpath: file path where matrices will be stored or loaded from
    :param Fr_or_TMM: whether to use the Fresnel equations (0) or a TMM lookup table (1)
    :param front_or_rear: whether light is incident from the front or rear
    :param n_absorbing_layers: for a structure with multiple interface layers, where a TMM lookuptable is used, the number of layers in \
    the interface
    :param calc_profile: whether to save the relevant information to calculate the depth-dependent absorption \
    profile. List of layers where the profile should be calculated, or otherwise None
    :param only_incidence_angle: if True, the ray-tracing will only be performed for the incidence theta and phi \
    specified in the options.
    :param widths: if using TMM, width of the surface layers (in nm)
    :param save: whether to save redistribution matrices (True/False), default True
    :param overwrite: whether to overwrite existing matrices (True/False), default False

    :return: Number of returns depends on whether absorption profiles are being calculated; the first two items are
             always returned, the final one only if a profile is being calcualted.

                - allArrays: the R/T redistribution matrix at each wavelength, indexed as (wavelength, angle_bin_out, angle_bin_in)
                - absArrays: the absorption redistribution matrix (total absorption per layer), indexed as (wavelength, layer_out, angle_bin_in)
                - allres: xarray dataset storing the absorption profile data
    """

    existing_mats, path_or_mats = get_matrices_or_paths(
        structpath, surf_name, front_or_rear, calc_profile, overwrite
    )

    if existing_mats and not overwrite:
        return path_or_mats

    else:
        get_wavelength(options)
        wavelengths = options["wavelength"]
        n_rays = options["n_rays"]
        nx = options["nx"]
        ny = options["ny"]
        n_angles = int(np.ceil(n_rays / (nx * ny)))

        phi_sym = options["phi_symmetry"]
        n_theta_bins = options["n_theta_bins"]
        c_az = options["c_azimuth"]
        pol = options["pol"]

        if not options["parallel"]:
            n_jobs = 1

        else:
            n_jobs = options.n_jobs if "n_jobs" in options else -1

        if calc_profile is not None:
            depth_spacing = options["depth_spacing"] * 1e9  # convert from m to nm
        else:
            depth_spacing = None

        if front_or_rear == "front":
            side = 1
        else:
            side = -1

        if Fr_or_TMM == 1:
            lookuptable = xr.open_dataset(os.path.join(structpath, surf_name + ".nc"))
            if front_or_rear == "rear":
                # side gets flipped here
                lookuptable = lookuptable.assign_coords(side=np.flip(lookuptable.side))
        else:
            lookuptable = None

        theta_spacing = options.theta_spacing if "theta_spacing" in options else "sin"

        theta_intv, phi_intv, angle_vector = make_angle_vector(
            n_theta_bins, phi_sym, c_az, theta_spacing
        )

        if only_incidence_angle:
            logger.info("Calculating matrix only for incidence theta/phi")
            if options["theta_in"] == 0:
                th_in = 0.0001
            else:
                th_in = options["theta_in"]

            angles_in = angle_vector[: int(len(angle_vector) / 2), :]
            n_reps = int(np.ceil(n_angles / len(angles_in)))
            thetas_in = np.tile(th_in, n_reps)
            n_angles = n_reps

            if options["phi_in"] == "all":
                # get relevant phis
                phis_in = np.tile(options["phi_in"], n_reps)
            else:
                if options["phi_in"] == 0:
                    phis_in = np.tile(0.0001, n_reps)

                else:
                    phis_in = np.tile(options["phi_in"], n_reps)

        else:
            if options["random_ray_angles"]:
                thetas_in = np.random.random(n_angles) * np.pi / 2
                phis_in = np.random.random(n_angles) * 2 * np.pi
            else:
                angles_in = angle_vector[: int(len(angle_vector) / 2), :]
                if n_angles / len(angles_in) < 1:
                    warn(
                        "The number of rays is not sufficient to populate the redistribution matrix!"
                    )
                n_reps = int(np.ceil(n_angles / len(angles_in)))
                thetas_in = np.tile(angles_in[:, 1], n_reps)[:n_angles]
                phis_in = np.tile(angles_in[:, 2], n_reps)[:n_angles]

        if front_or_rear == "front":
            mats = [incidence]
        else:
            mats = [transmission]

        if group.materials is not None:
            for mat_i in group.materials:
                mats.append(mat_i)

        if front_or_rear == "front":
            mats.append(transmission)
        else:
            mats.append(incidence)

        # list of lists: first in tuple is front incidence
        if front_or_rear == "front":
            surfaces = [x[0] for x in group.textures]

        else:
            surfaces = [x[1] for x in group.textures]

        nks = np.empty((len(mats), len(wavelengths)), dtype=complex)

        for i1, mat in enumerate(mats):
            nks[i1] = mat.n(wavelengths) + 1j * mat.k(wavelengths)

        h = max(surfaces[0].Points[:, 2])
        x_limits = (
            options["x_limits"]
            if "x_limits" in options
            else [
                surfaces[0].x_min + 0.01 * surfaces[0].Lx,
                surfaces[0].x_max - 0.01 * surfaces[0].Lx,
            ]
        )
        y_limits = (
            options["y_limits"]
            if "y_limits" in options
            else [
                surfaces[0].y_min + 0.01 * surfaces[0].Ly,
                surfaces[0].y_max - 0.01 * surfaces[0].Ly,
            ]
        )

        if options["random_ray_position"]:
            xs = np.random.uniform(x_limits[0], x_limits[1], nx)
            ys = np.random.uniform(y_limits[0], y_limits[1], ny)

        else:
            xs = np.linspace(x_limits[0], x_limits[1], nx)
            ys = np.linspace(y_limits[0], y_limits[1], ny)

        allres = Parallel(n_jobs=n_jobs)(
            delayed(RT_wl)(
                i1,
                wavelengths[i1],
                n_angles,
                nx,
                ny,
                widths,
                thetas_in,
                phis_in,
                h,
                xs,
                ys,
                nks,
                surfaces,
                pol,
                phi_sym,
                theta_intv,
                phi_intv,
                angle_vector,
                Fr_or_TMM,
                n_absorbing_layers,
                lookuptable,
                calc_profile,
                depth_spacing,
                side,
            )
            for i1 in range(len(wavelengths))
        )

        allArrays = stack([item[0] for item in allres])
        absArrays = stack([item[1] for item in allres])

        if save:
            save_npz(path_or_mats[0], allArrays)
            save_npz(path_or_mats[1], absArrays)

        if Fr_or_TMM > 0 and calc_profile is not None:
            profile = xr.concat([item[3] for item in allres], "wl")
            intgr = xr.concat([item[4] for item in allres], "wl")
            intgr.name = "intgr"
            profile.name = "profile"

            intgr = intgr.where(intgr > 0, 0)
            profile = profile.where(profile > 0, 0)

            allres = xr.merge([intgr, profile])

            if save:
                allres.to_netcdf(path_or_mats[2])

            return allArrays, absArrays, allres

        else:
            return allArrays, absArrays


def RT_wl(
    i1,
    wl,
    n_angles,
    nx,
    ny,
    widths,
    thetas_in,
    phis_in,
    h,
    xs,
    ys,
    nks,
    surfaces,
    pol,
    phi_sym,
    theta_intv,
    phi_intv,
    angle_vector,
    Fr_or_TMM,
    n_abs_layers,
    lookuptable,
    calc_profile,
    depth_spacing,
    side,
):
    logger.info(f"RT calculation for wavelength = {wl * 1e9} nm")

    theta_out = np.zeros((n_angles, nx * ny))
    phi_out = np.zeros((n_angles, nx * ny))
    A_surface_layers = np.zeros((n_angles, nx * ny, n_abs_layers))
    theta_local_incidence = np.zeros((n_angles, nx * ny))

    for i2 in range(n_angles):

        theta = thetas_in[i2]
        phi = phis_in[i2]
        r = abs((h + 1e-8) / cos(theta))
        r_a_0 = np.real(
            np.array(
                [r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)]
            )
        )
        for c, vals in enumerate(product(xs, ys)):
            _, th_o, phi_o, surface_A = single_ray_interface(
                vals[0],
                vals[1],
                nks[:, i1],
                r_a_0,
                theta,
                phi,
                surfaces,
                pol,
                wl,
                Fr_or_TMM,
                lookuptable,
            )

            if th_o < 0:  # can do outside loup with np.where
                th_o = -th_o
                phi_o = phi_o + np.pi
            theta_out[i2, c] = th_o
            phi_out[i2, c] = phi_o
            A_surface_layers[i2, c] = surface_A[0]
            theta_local_incidence[i2, c] = np.real(surface_A[1])

    phi_out = fold_phi(phi_out, phi_sym)
    phis_in = fold_phi(phis_in, phi_sym)

    if side == -1:
        not_absorbed = np.where(theta_out < (np.pi + 0.1))
        thetas_in = np.pi - thetas_in
        # phis_in = np.pi-phis_in # unsure about this part

        theta_out[not_absorbed] = np.pi - theta_out[not_absorbed]
        # phi_out = np.pi-phi_out # unsure about this part

    theta_local_incidence = np.abs(theta_local_incidence)
    n_thetas = len(theta_intv) - 1

    if Fr_or_TMM > 0:
        # now we need to make bins for the absorption
        theta_intv = np.append(theta_intv, 11)
        phi_intv = phi_intv + [np.array([0])]

    # xarray: can use coordinates in calculations using apply!
    binned_theta_in = np.digitize(thetas_in, theta_intv, right=True) - 1

    binned_theta_out = np.digitize(theta_out, theta_intv, right=True) - 1
    # -1 to give the correct index for the bins in phi_intv

    phi_in = xr.DataArray(
        phis_in,
        coords={"theta_bin": (["angle_in"], binned_theta_in)},
        dims=["angle_in"],
    )

    bin_in = (
        phi_in.groupby("theta_bin")
        .map(overall_bin, args=(phi_intv, angle_vector[:, 0]))
        .data
    )

    phi_out = xr.DataArray(
        phi_out,
        coords={"theta_bin": (["angle_in", "position"], binned_theta_out)},
        dims=["angle_in", "position"],
    )

    bin_out = (
        phi_out.groupby("theta_bin")
        .map(overall_bin, args=(phi_intv, angle_vector[:, 0]))
        .data
    )

    out_mat = np.zeros((len(angle_vector), int(len(angle_vector) / 2)))
    # everything is coming in from above so we don't need 90 -> 180 in incoming bins
    A_mat = np.zeros((n_abs_layers, int(len(angle_vector) / 2)))

    n_rays_in_bin = np.zeros(int(len(angle_vector) / 2))
    n_rays_in_bin_abs = np.zeros(int(len(angle_vector) / 2))

    binned_local_angles = np.digitize(theta_local_incidence, theta_intv, right=True) - 1

    local_angle_mat = np.zeros(
        (int((len(theta_intv) - 1) / 2), int(len(angle_vector) / 2))
    )

    if side == 1:
        offset = 0
    else:
        offset = int(len(angle_vector) / 2)

    for l1 in range(len(thetas_in)):
        for l2 in range(nx * ny):
            n_rays_in_bin[bin_in[l1] - offset] += 1
            if binned_theta_out[l1, l2] <= (n_thetas - 1):
                # reflected or transmitted
                out_mat[bin_out[l1, l2], bin_in[l1] - offset] += 1

            else:
                # absorbed in one of the surface layers
                n_rays_in_bin_abs[bin_in[l1] - offset] += 1
                per_layer = A_surface_layers[l1, l2]
                A_mat[:, bin_in[l1] - offset] += per_layer
                local_angle_mat[binned_local_angles[l1, l2], bin_in[l1] - offset] += 1

    # normalize
    out_mat = np.divide(out_mat, n_rays_in_bin, where=n_rays_in_bin != 0)
    overall_abs_frac = np.divide(
        n_rays_in_bin_abs, n_rays_in_bin, where=n_rays_in_bin != 0
    )
    abs_scale = np.divide(
        overall_abs_frac, np.sum(A_mat, 0), where=np.sum(A_mat, 0) != 0
    )

    intgr = np.divide(
        np.sum(A_mat, 0),
        n_rays_in_bin_abs,
        where=n_rays_in_bin_abs != 0,
        out=np.zeros_like(n_rays_in_bin_abs),
    )
    A_mat = abs_scale * A_mat
    out_mat[np.isnan(out_mat)] = 0
    A_mat[np.isnan(A_mat)] = 0

    out_mat = COO.from_numpy(out_mat)  # sparse matrix
    A_mat = COO.from_numpy(A_mat)

    if Fr_or_TMM > 0:
        local_angle_mat = np.divide(
            local_angle_mat,
            np.sum(local_angle_mat, 0),
            where=np.sum(local_angle_mat, 0) != 0,
            out=np.zeros_like(local_angle_mat),
        )
        local_angle_mat[np.isnan(local_angle_mat)] = 0
        local_angle_mat = COO.from_numpy(local_angle_mat)

        if calc_profile is not None:
            n_a_in = int(len(angle_vector) / 2)
            thetas = angle_vector[:n_a_in, 1]
            unique_thetas = np.unique(thetas)

            profile = make_profiles_wl(
                unique_thetas,
                n_a_in,
                side,
                widths,
                local_angle_mat,
                wl,
                lookuptable,
                pol,
                depth_spacing,
                calc_profile,
            )

            profile = profile.rename({"dim_0": "z"})

            intgr = xr.DataArray(
                intgr,
                dims=["global_index"],
                coords={"global_index": np.arange(0, n_a_in)},
            ).fillna(0)

            return out_mat, A_mat, local_angle_mat, profile, intgr

        else:
            return out_mat, A_mat, local_angle_mat

    else:
        return out_mat, A_mat


def make_lookuptable_rt_structure(
    textures, materials, incidence, transmission, options, save_location="default", overwrite=False,
):

    inc_for_lookuptable = []
    trn_for_lookuptable = []
    layers_for_lookuptable = []
    coherent_for_lookuptable = []
    coherency_list_for_lookuptable = []
    names_for_lookuptable = []
    prof_layers_for_lookuptable = []
    tmm_or_fresnel = []
    n_layers = []
    widths = []
    prof_layers_list = []

    for i1, text in enumerate(textures):
        if hasattr(text[0], "interface_layers"):
            if i1 > 0:
                inc_for_lookuptable.append(materials[i1 - 1])
            else:
                inc_for_lookuptable.append(incidence)

            layers_for_lookuptable.append(text[0].interface_layers)

            if i1 < len(textures) - 1:
                trn_for_lookuptable.append(materials[i1])

            else:
                trn_for_lookuptable.append(transmission)

            if hasattr(text[0], "coherency_list"):
                coherent_for_lookuptable.append(False)
                coherency_list_for_lookuptable.append(text[0].coherency_list)

            else:
                coherent_for_lookuptable.append(True)
                coherency_list_for_lookuptable.append(None)

            if hasattr(text[0], "prof_layers"):
                prof_layers_for_lookuptable.append(text[0].prof_layers)
                prof_layers_list.append(text[0].prof_layers)

            else:
                prof_layers_for_lookuptable.append(None)
                prof_layers_list.append(None)

            names_for_lookuptable.append(text[0].name + "int_{}".format(i1))
            tmm_or_fresnel.append(1)
            n_layers.append(len(text[0].interface_layers))

            widths.append([x.width * 1e9 for x in text[0].interface_layers])

        else:
            tmm_or_fresnel.append(0)
            n_layers.append(0)
            prof_layers_list.append(None)
            widths.append(None)

    savepath = get_savepath(save_location, options["project_name"])

    for (layers, inc, trn, coh, coh_list, name, prof_layers) in zip(
        layers_for_lookuptable,
        inc_for_lookuptable,
        trn_for_lookuptable,
        coherent_for_lookuptable,
        coherency_list_for_lookuptable,
        names_for_lookuptable,
        prof_layers_for_lookuptable,
    ):

        make_TMM_lookuptable(
            layers,
            inc,
            trn,
            name,
            options,
            savepath,
            coherent=coh,
            coherency_list=coh_list,
            prof_layers=prof_layers,
            sides=None,
            overwrite=overwrite,
        )

    return tmm_or_fresnel, savepath, n_layers, prof_layers_list, widths


def calculate_interface_profiles(
    data_prof_layers,
    A_in_prof_layers,
    prof_layer_list_i,
    local_thetas_i,
    directions_i,
    z_list,
    offsets,
    lookuptable,
    # wl,
    pol,
    depth_spacing,
):
    def profile_per_layer(x, z, offset, side):
        layer_index = x.coords["layer"].item(0) - 1

        part1 = x[:, 0] * np.exp(x[:, 4] * z[layer_index])
        part2 = x[:, 1] * np.exp(-x[:, 4] * z[layer_index])
        part3 = (x[:, 2] + 1j * x[:, 3]) * np.exp(1j * x[:, 5] * z[layer_index])
        part4 = (x[:, 2] - 1j * x[:, 3]) * np.exp(-1j * x[:, 5] * z[layer_index])
        result = np.real(part1 + part2 + part3 + part4)

        if side == -1:
            result = np.flip(result, 1)
        return result.reduce(np.sum, axis=0).assign_coords(
            dim_0=z[layer_index] + offset[layer_index]
        )

    def profile_per_angle(x, z, offset, side):
        by_layer = x.groupby("layer").map(
            profile_per_layer, z=z, offset=offset, side=side
        )
        return by_layer

    th_array = np.abs(local_thetas_i)
    front_incidence = np.where(directions_i == 1)[0]
    rear_incidence = np.where(directions_i == -1)[0]

    # need to scale absorption profile for each ray depending on
    # how much intensity was left in it when that ray was absorbed (this is done for total absorption inside
    # single_ray_stack)

    if len(front_incidence) > 0:

        A_lookup_front = lookuptable.Alayer.loc[
            dict(side=1, pol=pol, layer=prof_layer_list_i)
        ].interp(angle=th_array[front_incidence] #, wl=wl * 1e9
                 )# )
        data_front = data_prof_layers[front_incidence]

        ## CHECK! ##
        non_zero = xr.where(A_lookup_front > 1e-10, A_lookup_front, np.nan)

        scale_factor = (
            np.divide(data_front, non_zero).mean(dim="layer", skipna=True).data
        )  # can get slight differences in values between layers due to lookuptable resolution

        # layers because lookuptable angles are not exactly the same as the angles of the rays when absorbed. Take mean.
        # TODO: check what happens when one of these is zero or almost zero?

        # note that if a ray is absorbed in the interface on the first pass, the absorption per layer
        # recorded in A_interfaces will be LARGER than the A from the lookuptable because the lookuptable
        # includes front surface reflection, and by definition if the ray was absorbed it was not reflected
        # so the sum of the absorption per layer recorded in A_interfaces is 1 while the sum of the absorption in the
        # lookuptable is 1 - R - T.

        params_front = lookuptable.Aprof.loc[
            dict(side=1, pol=pol, layer=prof_layer_list_i)
        ].interp(angle=th_array[front_incidence],
                 # wl=wl * 1e9
                 )

        s_params = params_front.loc[
            dict(coeff=["A1", "A2", "A3_r", "A3_i"])
        ]  # have to scale these to make sure integrated absorption is correct
        c_params = params_front.loc[
            dict(coeff=["a1", "a3"])
        ]  # these should not be scaled

        scale_res = s_params * scale_factor[:, None, None]

        params_front = xr.concat((scale_res, c_params), dim="coeff")

        ans_front = (
            params_front.groupby("angle", squeeze=False)
            .map(profile_per_angle, z=z_list, offset=offsets, side=1)
            .drop_vars("coeff")
        )

        profile_front = ans_front.reduce(np.sum, ["angle"]).fillna(0)

    else:
        profile_front = 0

    if len(rear_incidence) > 0:

        A_lookup_back = lookuptable.Alayer.loc[
            dict(side=-1, pol=pol, layer=prof_layer_list_i)
        ].interp(angle=th_array[rear_incidence],
                 # wl=wl * 1e9,
                 )

        data_back = data_prof_layers[rear_incidence]

        non_zero = xr.where(A_lookup_back > 1e-10, A_lookup_back, np.nan)

        scale_factor = (
            np.divide(data_back, non_zero).mean(dim="layer", skipna=True).data
        )  # can get slight differences in values between layers

        params_back = lookuptable.Aprof.loc[
            dict(side=-1, pol=pol, layer=prof_layer_list_i)
        ].interp(angle=th_array[rear_incidence],
                 # wl=wl * 1e9,
                 )

        s_params = params_back.loc[
            dict(coeff=["A1", "A2", "A3_r", "A3_i"])
        ]  # have to scale these to make sure integrated absorption is correct
        c_params = params_back.loc[
            dict(coeff=["a1", "a3"])
        ]  # these should not be scaled

        scale_res = s_params * scale_factor[:, None, None]

        params_back = xr.concat((scale_res, c_params), dim="coeff")

        ans_back = (
            params_back.groupby("angle", squeeze=False)
            .map(profile_per_angle, z=z_list, offset=offsets, side=-1)
            .drop_vars("coeff")
        )

        profile_back = ans_back.reduce(np.sum, ["angle"]).fillna(0)

    else:

        profile_back = 0

    profile = profile_front + profile_back

    integrated_profile = np.sum(profile.reduce(np.trapz, dim="dim_0", dx=depth_spacing))

    A_corr = np.sum(A_in_prof_layers)

    scale_profile = np.real(
        np.divide(
            A_corr,
            integrated_profile.data,
            where=integrated_profile.data > 0,
            out=np.zeros_like(A_corr),
        )
    )

    interface_profile = scale_profile * profile.reduce(np.sum, dim="layer")

    return interface_profile.data


class rt_structure:
    """Set up structure for RT calculations.

    :param textures: list of surface textures. Each entry in the list is another list of two RTSurface objects,
        describing what the surface looks like for front and rear incidence, respectively
    :param materials: list of Solcore materials for each layer (excluding the incidence and transmission medium)
    :param widths: list widths of the layers in m
    :param incidence: incidence medium (Solcore material)
    :param transmission: transmission medium (Solcore material)
    :param options: dictionary/object with options for the calculation; only used if pre-computing lookup tables using TMM
    :param use_TMM: if True, use TMM to pre-compute lookup tables for the structure. Surface layers
            most be specified in the relevant textures.
    :param save_location: location to save the lookup tables; only used if pre-computing lookup tables using TMM
    :param overwrite: if True, overwrite any existing lookup tables; only used if pre-computing lookup tables using TMM
    """

    def __init__(
        self,
        textures,
        materials,
        widths,
        incidence,
        transmission,
        options=None,
        use_TMM=False,
        save_location="default",
        overwrite=False,
    ):

        self.textures = textures
        self.widths = widths

        mats = [incidence]
        for mati in materials:
            mats.append(mati)
        mats.append(transmission)

        self.mats = mats

        surfs_no_offset = [deepcopy(x[0]) for x in textures]
        # this is stupid but I don't know how else to do it. Custom deepcopy implemented for RTSurface

        cum_width = np.cumsum([0] + widths) * 1e6  # convert to um

        surfaces = []

        for i1, text in enumerate(surfs_no_offset):
            text.shift(cum_width[i1])
            surfaces.append(text)

        self.surfaces = surfaces
        self.surfs_no_offset = surfs_no_offset
        self.cum_width = cum_width
        self.width = np.sum(widths)

        if use_TMM:
            logger.info("Pre-computing TMM lookup table(s)")

            if options is None:
                raise (ValueError("Must provide options to pre-compute lookup tables"))

            else:
                (
                    self.tmm_or_fresnel,
                    self.save_location,
                    self.n_interface_layers,
                    self.prof_layers,
                    self.interface_layer_widths,
                ) = make_lookuptable_rt_structure(
                    textures, materials, incidence, transmission, options, save_location, overwrite,
                )

        else:
            self.tmm_or_fresnel = [0] * len(textures)  # no lookuptables

        if options.lambertian_approximation:
            self.lambertian_results = lambertian_scattering(self, save_location, options)

        else:
            self.lambertian_results = None

    def calculate(self, options):
        """Calculates the reflected, absorbed and transmitted intensity of the structure for the wavelengths and angles
        defined.

        :param options: options for the calculation. Relevant entries are:

           - wavelength: Wavelengths (in m) in which calculate the data. An array.
           - theta_in: Polar angle (in radians) of the incident light.
           - phi_in: azimuthal angle (in radians) of the incident light.
           - I_thresh: once the intensity reaches this fraction of the incident light, the light is considered to be absorbed.
           - lambertian_approximation: if 0 (default), keep following the ray until it is absorbed or escapes. Otherwise,
             assume the ray is Lambertian after this many traversals of the bulk.
           - pol: Polarisation of the light: 's', 'p' or 'u'.
           - depth_spacing_bulk: depth spacing for absorption profile calculations in the bulk (m)
           - depth_spacing: depth spacing for absorption profile calculations in interface layers (m)
           - nx and ny: number of points to scan across the surface in the x and y directions (integers)
           - random_ray_position: True/False. instead of scanning across the surface, choose nx*ny points randomly
           - randomize_surface: True/False. Randomize the ray position in the x/y direction before each surface interaction
           - parallel: True/False. Whether to execute calculations in parallel
           - n_jobs: n_jobs argument for Parallel function of joblib. Controls how many threads are used.
           - x_limits: x limits (in same units as the size of the RTSurface) between which incident rays will be generated
           - y_limits: y limits (in same units as the size of the RTSurface) between which incident rays will be generated

        :return: A dictionary with the R, A and T at the specified wavelengths and angle.
        """

        if isinstance(options, dict):
            options = State(options)

        get_wavelength(options)
        wavelengths = options.wavelength
        theta = options.theta_in
        phi = options.phi_in
        I_thresh = options.I_thresh
        periodic = options["periodic"] if "periodic" in options else 1
        depth_spacing_interfaces = (
            options["depth_spacing"] * 1e9 if "depth_spacing" in options else 1
        )
        lambertian_approximation = options.lambertian_approximation
        analytical_rt = options.analytical_ray_tracing

        if not options["parallel"]:
            n_jobs = 1

        else:
            n_jobs = options.n_jobs if "n_jobs" in options else -1

        widths = self.widths[:]
        widths.insert(0, 0)
        widths.append(0)
        widths = 1e6 * np.array(widths)  # convert to um

        z_space = 1e6 * options["depth_spacing_bulk"]  # convert from m to um
        z_pos = np.arange(0, sum(widths), z_space)

        mats = self.mats[:]
        surfaces = self.surfaces[:]

        if sum(self.tmm_or_fresnel) > 0:
            name_list = [x.name for x in surfaces]

            tmm_args = [
                1,
                self.tmm_or_fresnel,
                self.save_location,
                name_list,
                self.n_interface_layers,
                self.prof_layers,
                self.interface_layer_widths,
                depth_spacing_interfaces,
            ]

        else:
            tmm_args = [0, 0, 0, 0, 0, 0]

        nks = np.empty((len(mats), len(wavelengths)), dtype=complex)
        alphas = np.empty((len(mats), len(wavelengths)))
        # R = np.zeros(len(wavelengths))
        # T = np.zeros(len(wavelengths))
        #
        # absorption_profiles = np.zeros((len(wavelengths), len(z_pos)))
        # A_layer = np.zeros((len(wavelengths), len(widths)))

        for i1, mat in enumerate(mats):
            nks[i1] = mat.n(wavelengths) + 1j * mat.k(wavelengths)
            alphas[i1] = mat.k(wavelengths) * 4 * np.pi / (wavelengths * 1e6)

        h = max(surfaces[0].Points[:, 2])
        r = abs((h + 1e-8) / cos(theta))
        r_a_0 = np.real(
            np.array(
                [r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)]
            )
        )

        x_limits = (
            options["x_limits"]
            if "x_limits" in options
            else [
                surfaces[0].x_min + 0.01 * surfaces[0].Lx,
                surfaces[0].x_max - 0.01 * surfaces[0].Lx,
            ]
        )
        y_limits = (
            options["y_limits"]
            if "y_limits" in options
            else [
                surfaces[0].y_min + 0.01 * surfaces[0].Ly,
                surfaces[0].y_max - 0.01 * surfaces[0].Ly,
            ]
        )

        nx = options["nx"]
        ny = options["ny"]

        if options["random_ray_position"]:
            xs = np.random.uniform(x_limits[0], x_limits[1], nx)
            ys = np.random.uniform(y_limits[0], y_limits[1], ny)

        else:
            xs = np.linspace(x_limits[0], x_limits[1], nx)
            ys = np.linspace(y_limits[0], y_limits[1], ny)

        # need to calculate r_a and r_b
        # a total of n_rays will be traced; this is divided by the number of x and y points to scan so we know
        # how many times we need to repeat
        n_reps = int(np.ceil(options["n_rays"] / (nx * ny)))

        pol = options["pol"]
        randomize = options["randomize_surface"]

        initial_mat = (
            options["initial_material"] if "initial_material" in options else 0
        )
        initial_dir = (
            options["initial_direction"] if "initial_direction" in options else 1
        )

        cum_width = np.cumsum([0] + widths)

        depths = []
        depth_indices = []

        # this should not be happening in here! Waste of time, same for all wavelengths & rays!
        for i1 in range(len(widths)):
            depth_indices.append(
                (z_pos < np.cumsum(widths)[i1]) & (z_pos >= np.cumsum(widths)[i1 - 1])
            )
            depths.append(z_pos[depth_indices[i1]] - np.cumsum(widths)[i1 - 1])

        # pr.enable()
        allres = Parallel(n_jobs=n_jobs)(
            delayed(parallel_inner)(
                nks[:, i1],
                alphas[:, i1],
                r_a_0,
                surfaces,
                widths,
                cum_width,
                z_pos,
                depths,
                depth_indices,
                I_thresh,
                pol,
                nx,
                ny,
                n_reps,
                xs,
                ys,
                randomize,
                initial_mat,
                initial_dir,
                periodic,
                lambertian_approximation,
                analytical_rt,
                tmm_args + [wavelengths[i1]],
                self.lambertian_results,
            )
            for i1 in range(len(wavelengths))
        )
        # pr.disable()

        Is = np.stack([item[0] for item in allres])
        absorption_profiles = np.stack([item[1] for item in allres])
        A_layer = np.stack([item[2] for item in allres])
        thetas = np.stack([item[3] for item in allres])
        phis = np.stack([item[4] for item in allres])
        n_passes = np.stack([item[5] for item in allres])
        n_interactions = np.stack([item[6] for item in allres])
        A_interfaces = [item[7] for item in allres]
        # local_thetas = [item[8] for item in allres]
        # directions = [item[9] for item in allres]
        profile_interfaces = [item[8] for item in allres]


        if sum(self.tmm_or_fresnel) > 0:

            A_per_interface = [
                np.zeros((len(wavelengths), n_l)) for n_l in self.n_interface_layers
            ]

            interface_profiles = [[] for _ in self.n_interface_layers]

            for i1, per_wl in enumerate(A_interfaces):  # loop through wavelengths

                for j1, interf in enumerate(per_wl):  # loop through interfaces
                    A_per_interface[j1][i1] = interf

                    if len(profile_interfaces[i1][j1]) > 0:
                        # print("prof shape", profile_interfaces[i1][j1].shape)
                        interface_profiles[j1].append(profile_interfaces[i1][j1])

            interface_profiles = [
                np.stack(x) if len(x) > 0 else x for x in interface_profiles
            ]

        else:
            A_per_interface = 0
            interface_profiles = 0

        non_abs = np.logical_and(~np.isnan(thetas), np.abs(thetas) != 10)

        refl = np.logical_and(
            non_abs,
            np.less_equal(np.real(thetas), np.pi / 2, where=non_abs),
        )
        trns = np.logical_and(
            non_abs, np.greater(np.real(thetas), np.pi / 2, where=non_abs)
        )

        R = np.real(Is * refl).T / (n_reps * nx * ny)
        T = np.real(Is * trns).T / (n_reps * nx * ny)
        R = np.sum(R, 0)
        T = np.sum(T, 0)

        refl_0 = (
            non_abs
            * np.less_equal(np.real(thetas), np.pi / 2, where=non_abs)
            * (n_passes == 0)
        )
        R0 = np.real(Is * refl_0).T / (n_reps * nx * ny)
        R0 = np.sum(R0, 0)

        absorption_profiles[absorption_profiles < 0] = 0

        A_layer[:, 1] = A_layer[:, 1]

        # process A_interfaces

        if np.any(np.abs(thetas) == 10):  # Lambertian scattering

            # +ve if travelling down, about to hit rear surface.
            # stop normal RT right before next interaction with surface, but AFTER taking into account bulk
            # absorption on this pass

            direction = np.sign(thetas[np.abs(thetas)==10])[0]
            lambertian_RAT = self.lambertian_results[0].sel(direction=direction)
            lambertian_A1 = self.lambertian_results[1].sel(direction=direction)
            lambertian_A2 = self.lambertian_results[2].sel(direction=direction)

            # where np.abs(thetas) == 10, want to divide remaining intensity in Is at the same location correctly between
            # reflection, interface absorption, and transmission
            n_lambertian = np.sum(np.abs(thetas) == 10, axis=1)
            I_lambertian = np.array([np.sum(Is[i1][np.abs(thetas[i1]) == 10]) for i1 in
                            range(len(wavelengths))])  / (nx * ny * n_reps)
            I_RAT = lambertian_RAT * I_lambertian

            R += I_RAT.sel(event='R')
            T += I_RAT.sel(event='T')
            A_layer[:, 1] += I_RAT.sel(event='A_bulk')

            if sum(self.tmm_or_fresnel) > 0:
                add_frontsurf = (I_lambertian * lambertian_A1).data.T
                add_backsurf = (I_lambertian * lambertian_A2).data.T

                A_per_interface[0] = A_per_interface[0] + add_frontsurf
                A_per_interface[1] = A_per_interface[-1] + add_backsurf

            # add_profile = calculate_lambertian_profile(self, I_RAT, wavelengths, direction,
            #                                            self.lambertian_results[3])


        return {
            "R": R,
            "T": T,
            "A_per_layer": A_layer[:, 1:-1],
            "profile": absorption_profiles / 1e3,
            "thetas": thetas,
            "phis": phis,
            "R0": R0,
            "n_passes": n_passes,
            "n_interactions": n_interactions,
            "A_per_interface": A_per_interface,
            # "A_interfaces": A_interfaces,
            "interface_profiles": interface_profiles,
            "Is": Is,
        }

    def calculate_profile(self, options):
        prof_results = self.calculate(options)
        return prof_results


def make_tmm_args(arg_list):
    # print("TMM lookup tables used for interfaces: {}".format([i1 for i1, x in enumerate(arg_list[1]) if x == 1]))
    # construct additional arguments to be passed to ray-tracer: wavelength, lookuptables, and to use TMM (1)
    additional_tmm_args = []
    prof_layers = []
    interface_layer_widths = []

    for i1, val in enumerate(arg_list[1]):

        if val == 1:

            structpath = arg_list[2]
            surf_name = arg_list[3][i1] + "int_{}".format(i1)
            lookuptable = xr.open_dataset(os.path.join(structpath, surf_name + ".nc")).loc[dict(wl=arg_list[-1]*1e9)].load()
            additional_tmm_args.append(
                {"Fr_or_TMM": 1, "lookuptable": lookuptable}
            )
            prof_layers.append(arg_list[5][i1])
            interface_layer_widths.append(arg_list[6][i1])

        else:
            additional_tmm_args.append({})
            prof_layers.append(None)
            interface_layer_widths.append(None)

    return additional_tmm_args, prof_layers, interface_layer_widths, arg_list[7]


def parallel_inner(
    nks,
    alphas,
    r_a_0,
    surfaces,
    widths,
    cum_width,
    z_pos,
    depths,
    depth_indices,
    I_thresh,
    pol,
    nx,
    ny,
    n_reps,
    xs,
    ys,
    randomize,
    initial_mat,
    initial_dir,
    periodic,
    lambertian_approximation,
    analytical_rt,
    tmm_args=None,
    lambertian_results=None,
):

    # # generally same across wavelengths, but can be changed by analytical
    # # ray tracing happening first
    if initial_dir == 1 and initial_mat > 0:
        surf_index = initial_mat
        z_offset = -cum_width[initial_mat - 1] - 1e-8
        # print('z_offset', z_offset, r_a_0)

    elif initial_dir == 1 and initial_mat == 0:
        surf_index = 0
        z_offset = r_a_0[2]

    else:
        surf_index = initial_mat - 1
        z_offset = -cum_width[initial_mat] + 1e-8


    # # different for each ray, but same across wavelengths - should pregenerate as an array
    # r_b = np.array([xs, ys, np.zeros_like(xs)])
    # r_a = r_a_0[:, None] + np.array([xs, ys, np.zeros_like(xs)])
    # r_b = np.array([x, y, 0])
    #
    # d = (r_b - r_a) / np.linalg.norm(
    #     r_b - r_a
    # )  # initial_dir (unit vector) of ray. Always downwards!
    d = -r_a_0 / np.linalg.norm(r_a_0)

    # r_a_x = r_a_0[0] + xs
    # r_a_y = r_a_0[1] + ys

    if initial_dir != 1:
        d[2] = -d[2]

    if tmm_args is None:
        tmm_args = [0]

    profile_arrays = [[] for _ in range(len(surfaces))]

    if tmm_args[0] > 0:
        (
            additional_tmm_args,
            prof_layer_list,
            interface_layer_widths,
            depth_spacing_int,
        ) = make_tmm_args(tmm_args)

        A_in_interfaces = [np.zeros(n_l) for n_l in tmm_args[4]]

        # pre-generate z positions and arrays if necessary
        z_lists = [[] for _ in range(len(surfaces))]
        offsets = [[] for _ in range(len(surfaces))]

        for j1, prof_layer_i in enumerate(prof_layer_list):

            if prof_layer_i is None:
                # will not calculate profiles for this list.
                z_lists[j1] = 0
                offsets[j1] = 0

            else:

                n_z_points = 0
                widths_int = interface_layer_widths[j1]
                for k1, l_w in enumerate(widths_int):
                    z_lists[j1].append(
                        xr.DataArray(np.arange(0, l_w, depth_spacing_int))
                    )
                    if k1 + 1 in prof_layer_i:
                        n_z_points += len(z_lists[j1][-1])

                profile_arrays[j1] = np.zeros(n_z_points)
                offsets[j1] = np.cumsum([0] + widths_int)[:-1]

    else:
        additional_tmm_args = [{} for _ in range(len(surfaces))]
        A_in_interfaces = 0

    logger.info("Calculating next wavelength...")

    # thetas and phis divided into
    thetas = np.zeros(n_reps * nx * ny)
    phis = np.zeros(n_reps * nx * ny)
    n_passes = np.zeros(n_reps * nx * ny)
    n_interactions = np.zeros(n_reps * nx * ny)
    A_layer = np.zeros(len(widths))
    Is = np.zeros(n_reps * nx * ny)

    A_interfaces = [[] for _ in range(len(surfaces) + 1)]
    local_thetas = [[] for _ in range(len(surfaces) + 1)]
    directions = [[] for _ in range(len(surfaces) + 1)]

    profiles = np.zeros(len(z_pos))

    # analytical front surface ray-tracing should take place here, and modify
    # the variables passed to single_ray_stack accordingly to start in the first
    # bulk layer
    # Will need to modify:
    # - x/y (randomize)
    # - r_a_0
    # - randomize should be forced True if using analytical front surface
    # - initial_mat
    # - number of rays should be modified to account for those which have already been
    #   reflected or absorbed in the front surface
    if analytical_rt:

        if initial_dir == 1:
            surf_index = 0
            n0 = nks[initial_mat]
            n1 = nks[initial_mat + 1]

        else:
            surf_index = initial_mat - 1
            n0 = nks[initial_mat - 1]
            n1 = nks[initial_mat]

        if tmm_args[0] > 0:
            n_layers = tmm_args[4][surf_index]

        else:
            n_layers = 0

        z_pos_first = depths[initial_mat + initial_dir]
        front_surf = surfaces[surf_index]

        # r_in = np.copy(r_a_0)
        # r_in[2] = -r_in[2]


        (thetas, phis, Is, n_interactions, n_passes, A_layer_scaled, profiles, A_surf) = (
            analytical_front_surface(
            front_surf,
                                             d, # not sure this is right
                                             n0,
                                             n1,
                                             pol,
                                             analytical_rt,
                                             n_layers,
                                             initial_dir,
            n_reps*nx*ny,
            z_pos_first,
            widths[initial_mat + initial_dir],
            alphas[initial_mat + initial_dir],
            I_thresh,
                                            **additional_tmm_args[surf_index]))


                    # now need to pass the rays which were not absorbed or reflected to single_ray_stack
                    # transmitted rays also need to traverse the bulk.

                    # there will be fewer reps because some rays will be absorbed or reflected in the front surface,
                    # or the first traversal of the bulk

        transmitted_inds = np.where(thetas > np.pi / 2)[0]

        if np.sum(A_surf) > 0:

            A_interfaces[initial_mat + initial_dir].append(
                A_surf*n_reps*nx*ny)

        if len(transmitted_inds) == 0:
            A_layer[initial_mat + initial_dir] = A_layer_scaled

          # keep ray-tracing with the rays which have not been reflected/absorbed by the time they reach the
        # second surface in the stack

        else:
            # print('Switch to normal RT')
            r_a_new = [0, 0, front_surf.zcov - 1e-8] # this will get randomized, need to provide something
            # A_layer[initial_mat + initial_dir] = A_layer_RT * len(transmitted_inds)/(n_reps*nx*ny)
            A_layer[initial_mat + initial_dir] = A_layer_scaled

            for ind in transmitted_inds:

                # need to change r_a_0 so that incident angle is correct! Not normal incidence

                # d we want from theta and phi:
                d_inc = initial_dir*np.real(np.array([sin(thetas[ind])*cos(phis[ind]), sin(thetas[ind])*sin(phis[ind]), cos(thetas[ind])]))

                (
                    I,
                    profile,
                    A_per_layer,
                    th_o,
                    phi_o,
                    n_pass,
                    n_interact,
                    A_interface_array,
                    A_interface_index,
                    th_local,
                    direction,
                ) = single_ray_stack(
                    0,
                    0,
                    nks,
                    alphas,
                    r_a_new,
                    d_inc,
                    surfaces,
                    additional_tmm_args,
                    widths,
                    cum_width,
                    z_pos,
                    depths,
                    depth_indices,
                    I_thresh,
                    pol,
                    True,
                    initial_mat + initial_dir,
                    initial_dir,
                    surf_index + initial_dir,
                    periodic,
                    lambertian_approximation,
                    n_passes[ind],
                    n_interactions[ind],
                    Is[ind]
                )

                A_interfaces[A_interface_index].append(A_interface_array)
                profiles += profile / (n_reps * nx * ny)
                thetas[ind] = th_o
                phis[ind] = phi_o
                Is[ind] = np.real(I)
                A_layer += A_per_layer / (n_reps * nx * ny)
                n_passes[ind] = n_pass
                n_interactions[ind] = n_interact

    else:
        for j1 in range(n_reps):
            offset = j1 * nx * ny

            for c, vals in enumerate(product(xs, ys)):

                (
                    I,
                    profile,
                    A_per_layer,
                    th_o,
                    phi_o,
                    n_pass,
                    n_interact,
                    A_interface_array,
                    A_interface_index,
                    th_local,
                    direction,
                ) = single_ray_stack(
                    vals[0],
                    vals[1],
                    nks,
                    alphas,
                    [r_a_0[0] + vals[0], r_a_0[1] + vals[1], z_offset],
                    d,
                    surfaces,
                    additional_tmm_args,
                    widths,
                    cum_width,
                    z_pos,
                    depths,
                    depth_indices,
                    I_thresh,
                    pol,
                    randomize,
                    initial_mat,
                    initial_dir,
                    surf_index,
                    periodic,
                    lambertian_approximation,
                )

                A_interfaces[A_interface_index].append(A_interface_array)
                profiles += profile / (n_reps * nx * ny)
                thetas[c + offset] = th_o
                phis[c + offset] = phi_o
                Is[c + offset] = np.real(I)
                A_layer += A_per_layer / (n_reps * nx * ny)
                n_passes[c + offset] = n_pass
                n_interactions[c + offset] = n_interact
                local_thetas[A_interface_index].append(np.real(th_local))
                directions[A_interface_index].append(direction)

    A_interfaces = A_interfaces[1:]
    # index 0 are all entries for non-interface-absorption events.
    local_thetas = local_thetas[1:]
    directions = directions[1:]

    if tmm_args[0] > 0:
        # process A_interfaces

        for i1, layer_data in enumerate(A_interfaces):
            # A_interfaces is a list of lists; [[list of absorption events in interface 1],
            # [list of absorption events in interface 2], ...].

            if len(layer_data) > 0:

                data = np.stack(layer_data)

                A_in_interfaces[i1] = np.sum(data, axis=0) / (n_reps * nx * ny)

                if prof_layer_list[i1] is not None:

                    lookuptable = additional_tmm_args[i1]["lookuptable"]
                    # wl = additional_tmm_args[i1]["wl"]

                    A_in_profile_layers = A_in_interfaces[i1][
                        np.array(prof_layer_list[i1]) - 1
                    ]  # total absorption per layer
                    data_profile_layers = data[
                        :, np.array(prof_layer_list[i1]) - 1
                    ]  # information on individual absorption events (rays)

                    z_list = z_lists[i1]
                    offset = offsets[i1]

                    profile_arrays[i1] = calculate_interface_profiles(
                        data_profile_layers,
                        A_in_profile_layers,
                        prof_layer_list[i1],
                        np.array(local_thetas[i1]),
                        np.array(directions[i1]),
                        z_list,
                        offset,
                        lookuptable,
                        # wl,
                        pol,
                        depth_spacing_int,
                    )

    return (
        Is,
        profiles,
        A_layer,
        thetas,
        phis,
        n_passes,
        n_interactions,
        A_in_interfaces,
        profile_arrays,
    )


def normalize(x):
    if sum(x > 0):
        x = x / sum(x) - x.coords["A"]
    return x


def make_profiles_wl(
    unique_thetas,
    n_a_in,
    side,
    widths,
    angle_distmat,
    wl,
    lookuptable,
    pol,
    depth_spacing,
    prof_layers,
):
    # widths and depth_spacing are passed in nm!

    def profile_per_layer(xx, z, offset, side, non_zero):
        layer_index = xx.coords["layer"].item(0) - 1
        x = xx[non_zero]
        part1 = x[:, 0] * np.exp(x[:, 4] * z[layer_index])
        part2 = x[:, 1] * np.exp(-x[:, 4] * z[layer_index])
        part3 = (x[:, 2] + 1j * x[:, 3]) * np.exp(1j * x[:, 5] * z[layer_index])
        part4 = (x[:, 2] - 1j * x[:, 3]) * np.exp(-1j * x[:, 5] * z[layer_index])
        result = np.real(part1 + part2 + part3 + part4)
        if side == -1:
            result = np.flip(result, 1)
        return result.reduce(np.sum, axis=0).assign_coords(
            dim_0=z[layer_index] + offset[layer_index]
        )

    def profile_per_angle(x, z, offset, side, nz):
        i2 = x.coords["global_index"].item(0)
        non_zero = np.where(nz[:, i2])[0]
        by_layer = x.groupby("layer").map(
            profile_per_layer, z=z, offset=offset, side=side, non_zero=non_zero
        )
        return by_layer

    def scale_func(x, scale_params):
        return x.data[:, None, None] * scale_params

    def select_func(x, const_params):
        return (x.data[:, None, None] != 0) * const_params

    pr = xr.DataArray(
        angle_distmat.todense(),
        dims=["local_theta", "global_index"],
        coords={"local_theta": unique_thetas, "global_index": np.arange(0, n_a_in)},
    )

    # lookuptable layers are 1-indexed

    data = lookuptable.loc[dict(side=1, pol=pol)].interp(
        angle=pr.coords["local_theta"], wl=wl * 1e9
    )

    params = (
        data["Aprof"]
        .drop_vars(["layer", "side", "angle", "pol"])
        .transpose("local_theta", "layer", "coeff")
    )

    s_params = params.loc[
        dict(coeff=["A1", "A2", "A3_r", "A3_i"])
    ]  # have to scale these to make sure integrated absorption is correct
    c_params = params.loc[dict(coeff=["a1", "a3"])]  # these should not be scaled

    scale_res = pr.groupby("global_index").map(scale_func, scale_params=s_params)
    const_res = pr.groupby("global_index").map(select_func, const_params=c_params)

    params = xr.concat((scale_res, const_res), dim="coeff").assign_coords(
        layer=np.arange(1, len(widths) + 1)
    )
    params = params.transpose("local_theta", "global_index", "layer", "coeff")

    z_list = []

    for l_w in widths:
        z_list.append(xr.DataArray(np.arange(0, l_w, depth_spacing)))

    offsets = np.cumsum([0] + widths)[:-1]

    xloc = params.loc[dict(coeff="A1")].reduce(np.sum, "layer")
    nz = xloc != 0

    ans = (
        params.loc[dict(layer=prof_layers)]
        .groupby("global_index")
        .map(profile_per_angle, z=z_list, offset=offsets, side=side, nz=nz)
        .drop_vars("coeff")
    )
    ans = ans.fillna(0)

    profile = ans.reduce(np.sum, "layer")
    profile = xr.where(profile >= 0, profile, 0)

    return profile.T


class RTSurface:
    """Class which is used to store information about the surface which is used for ray-tracing."""

    def __init__(self, Points, interface_layers=None, **kwargs):
        """Initializes the surface.
        Parameters:

        :param Points: A numpy array of shape (n, 3) where n is the number of points on the surface. The columns are the
                        x, y and z coordinates of the points.
        :param interface_layers: a list of layers (typically, Solcore Layer objects) which are on the interface. Optional.
        :param coverage_height: The height at which the surface is expected to cover the whole unit cell. If this is not
                                provided (None), this function will try to guess the coverage height by finding the height
                                at which both the x and y coordinate are minimized.
        """

        if "height_distribution" in kwargs:
        #     return from a probability distribution instead of fixed values. This
        #     will change the height of the pyramids (assume only pyramids for now)
        #     but not the size of the unit cell. Simplices also do not change.
        #     Changing: Points, P_0s, P_1s, P_2s, crossP, N, z_min, z_max, z_cov if
        #     it's not 0 (but for pyramids it is).
            self.distribution = kwargs["height_distribution"]

        else:
            self.distribution = None

        tri = Delaunay(Points[:, [0, 1]])
        self.simplices = tri.simplices
        self.Points = Points
        self.original_Points = deepcopy(Points)
        self.height_ind = np.argmax(np.abs(self.Points[:,2]))
        self.P_0s = Points[tri.simplices[:, 0]]
        self.P_1s = Points[tri.simplices[:, 1]]
        self.P_2s = Points[tri.simplices[:, 2]]
        self.crossP = np.cross(self.P_1s - self.P_0s, self.P_2s - self.P_0s)
        self.N = self.crossP / np.linalg.norm(self.crossP, axis=1)[:, None]
        self.size = self.P_0s.shape[0]
        self.Lx = abs(min(Points[:, 0]) - max(Points[:, 0]))
        self.Ly = abs(min(Points[:, 1]) - max(Points[:, 1]))
        self.x_min = min(Points[:, 0])
        self.x_max = max(Points[:, 0])
        self.y_min = min(Points[:, 1])
        self.y_max = max(Points[:, 1])
        self.z_min = min(Points[:, 2])
        self.z_max = max(Points[:, 2])

        # zcov is the height at which the surface covers the whole unit cell; i.e. it is safe to aim a ray at the unit
        # cell at this height and be sure that it will hit the surface. The method below works well for regular textures
        # like regular pyramids but doesn't work well for e.g. AFM scans, hyperhemisphere

        if "coverage_height" in kwargs:
            self.zcov = kwargs["coverage_height"]

        # catch exception here in case the surface is not regular
        else:
            self.zcov = Points[:, 2][
                np.all(
                    np.array(
                        [
                            Points[:, 0] == min(Points[:, 0]),
                            Points[:, 1] == min(Points[:, 1]),
                        ]
                    ),
                    axis=0,
                )
            ][0]

        if "name" in kwargs:
            self.name = kwargs["name"]

        else:
            self.name = ""

        if interface_layers is not None:
            self.interface_layers = interface_layers

            if "coherency_list" in kwargs:
                self.coherency_list = kwargs["coherency_list"]

            if "prof_layers" in kwargs:
                self.prof_layers = kwargs["prof_layers"]

    def __deepcopy__(self, memo):
        copy = type(self)(Points=self.Points, coverage_height=self.zcov)
        memo[id(self)] = copy

        keys = self.__dict__.keys()

        for key in keys:
            if key != "interface_layers":
                setattr(copy, key, deepcopy(getattr(self, key), memo))

        return copy

    def find_area(self):
        xyz = np.stack((self.P_0s, self.P_1s, self.P_2s))
        cos_theta = np.sum((xyz[0] - xyz[1]) * (xyz[2] - xyz[1]), 1)

        theta = np.arccos(cos_theta)
        self.area = np.sum(
            (
                0.5
                * np.linalg.norm(xyz[0] - xyz[1], axis=1)
                * np.linalg.norm(xyz[2] - xyz[1], axis=1)
                * np.sin(theta)
            )
        ) / (self.Lx * self.Ly)

    def shift(self, z_shift):
        self.Points[:, 2] = self.Points[:, 2] - z_shift
        self.P_0s = self.Points[self.simplices[:, 0]]
        self.P_1s = self.Points[self.simplices[:, 1]]
        self.P_2s = self.Points[self.simplices[:, 2]]
        self.crossP = np.cross(self.P_1s - self.P_0s, self.P_2s - self.P_0s)
        self.z_min = min(self.Points[:, 2])
        self.z_max = max(self.Points[:, 2])

        self.zcov = self.zcov - z_shift

    def refresh(self):

        if self.distribution is not None:

            new_height = np.random.choice(self.distribution["h"],
                                             p=self.distribution["p"])

            scaling = np.abs(new_height/(self.original_Points[:,2][self.height_ind] - self.zcov))

            self.Points[:,2] = scaling*(self.original_Points[:,2] - self.zcov) + self.zcov

            self.P_0s = self.Points[self.simplices[:, 0]]
            self.P_1s = self.Points[self.simplices[:, 1]]
            self.P_2s = self.Points[self.simplices[:, 2]]
            self.crossP = np.cross(self.P_1s - self.P_0s, self.P_2s - self.P_0s)
            self.N = self.crossP / np.linalg.norm(self.crossP, axis=1)[:, None]
            self.z_min = min(self.Points[:, 2])
            self.z_max = max(self.Points[:, 2])


def calc_R(n1, n2, theta, pol):
    theta_t = np.arcsin((n1 / n2) * np.sin(theta))
    if pol == "s":
        Rs = (
                np.abs(
                    (n1 * np.cos(theta) - n2 * np.cos(theta_t))
                    / (n1 * np.cos(theta) + n2 * np.cos(theta_t))
                )
                ** 2
        )
        return Rs

    if pol == "p":
        Rp = (
                np.abs(
                    (n1 * np.cos(theta_t) - n2 * np.cos(theta))
                    / (n1 * np.cos(theta_t) + n2 * np.cos(theta))
                )
                ** 2
        )
        return Rp

    else:
        Rs = (
                np.abs(
                    (n1 * np.cos(theta) - n2 * np.cos(theta_t))
                    / (n1 * np.cos(theta) + n2 * np.cos(theta_t))
                )
                ** 2
        )
        Rp = (
                np.abs(
                    (n1 * np.cos(theta_t) - n2 * np.cos(theta))
                    / (n1 * np.cos(theta_t) + n2 * np.cos(theta))
                )
                ** 2
        )
        return (Rs + Rp) / 2


def exit_side(r_a, d, Lx, Ly):
    n = np.array(
        [[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]]
    )  # surface normals: top, right, bottom, left
    p_0 = np.array(
        [[0, Ly, 0], [Lx, 0, 0], [0, 0, 0], [0, 0, 0]]
    )  # points on each plane
    denom = np.sum(d * n, axis=1)
    denom[denom == 0] = 1e-12
    t = np.sum((p_0 - r_a) * n, axis=1) / denom  # r_intersect = r_a + t*d
    which_intersect = t > 0  # only want intersections of forward-travelling ray
    t[~which_intersect] = float(
        "inf"
    )  # set others to inf to avoid finding when doing min
    which_side = np.argmin(t)  # find closest plane

    return which_side, t[which_side]


def calc_angle(x):
    v1 = np.array([0, 1])
    return np.math.atan2(np.linalg.det([x, v1]), np.dot(x, v1))  # - 180 to 180


def single_ray_stack(
    x,
    y,
    nks,
    alphas,
    r_a,
    d,
    surfaces,
    tmm_kwargs_list,
    widths,
    cum_width,
    z_pos,
    depths,
    depth_indices,
    I_thresh,
    pol="u",
    randomize=False,
    mat_i=0,
    direction=1,
    surf_index=0,
    periodic=1,
    lambertian_approximation=0,
    n_passes=0,
    n_interactions=0,
    I_in=1,
):

    single_surface = {0: single_cell_check, 1: single_interface_check}
    # use single_cell_check if not periodic, single_interface_check if is periodic

    # final_res = 0: reflection
    # final_res = 1: transmission
    # This should get a list of surfaces and materials (optical constants, alpha + widths); there is one less surface than material
    # minimum is one surface and two materials
    # direction = 1: travelling down
    # direction = -1: travelling up

    #      incidence medium (0)
    # surface 0  ---------
    #           material 1
    # surface 1  ---------
    #           material 2
    # surface 2  ---------

    # should end when either material = final material (len(materials)-1) & direction == 1 or
    # material = 0 & direction == -1

    # print("NEW RAY")

    profile = np.zeros(len(z_pos))
    # do everything in microns
    A_per_layer = np.zeros(len(widths))

    # direction: start travelling downwards; 1 = down, -1 = up

    # surf_below = mat_i
    # surf_above = mat_i - 1

    A_interface_array = 0
    A_interface_index = 0

    # max_below = np.max(surfaces[surf_below].Points[:, 2])
    # min_above = np.min(surfaces[surf_above].Points[:, 2])

    # all of this can be done outside - will be the same for every ray!
    # print(max_below, min_above)

    ########

    # same for all wavelengths & rays

    # generally same across wavelengths, but can be changed by analytical
    # ray tracing happening first
    # if direction == 1 and mat_i > 0:
    #     surf_index = mat_i
    #     z_offset = -cum_width[mat_i - 1] - 1e-8
    #     # print('z_offset', z_offset, r_a_0)
    #
    # elif direction == 1 and mat_i == 0:
    #     surf_index = 0
    #     z_offset = r_a_0[2]
    #
    # else:
    #     surf_index = mat_i - 1
    #     z_offset = -cum_width[mat_i] + 1e-8

    # different for each ray, but same across wavelengths - should pregenerate as an array
    # r_a = r_a_0 + np.array([x, y, 0])
    # r_b = np.array([x, y, 0])
    #
    # d = (r_b - r_a) / np.linalg.norm(
    #     r_b - r_a
    # )  # direction (unit vector) of ray. Always downwards!

    # print(d, -r_a_0/np.linalg.norm(r_a_0))
    # d = -r_a_0/np.linalg.norm(r_a_0)
    # want to translate along d so r_a is in between the correct surfaces
    # first translate to z = 0:

    # nd = -r_a[2] / d[2]
    #
    # # print(r_a[2], d[2], nd)
    #
    # r_a = r_a + nd * d
    #
    # r_a[2] = z_offset
    #
    # if direction != 1:
    #     d[2] = -d[2]

    ##### above here outside

    # don't need direction_i, mat_i after this.

    stop = False
    I = I_in

    while not stop:

        surf = surfaces[surf_index]

        # if periodic:

        if randomize and (n_passes > 0):
            h = surf.z_max - surf.z_min + 0.1
            r_b = [
                np.random.rand() * surf.Lx,
                np.random.rand() * surf.Ly,
                surf.zcov,
            ]

            if d[2] == 0:
                # ray travelling parallel to surface
                # print("parallel ray")
                d[2] = -direction * 1e-3  # make it not parallel in the right direction

            n_z = np.ceil(abs(h / d[2]))
            # print('before', r_a)
            r_a = r_b - n_z * d
            # print('after', r_a)


        if periodic:

            r_a[0] = r_a[0] - surf.Lx * (
                (r_a[0] + d[0] * (surf.zcov - r_a[2]) / d[2]) // surf.Lx
            )
            r_a[1] = r_a[1] - surf.Ly * (
                (r_a[1] + d[1] * (surf.zcov - r_a[2]) / d[2]) // surf.Ly
            )

        if direction == 1:
            ni = nks[mat_i]
            nj = nks[mat_i + 1]

        else:
            ni = nks[mat_i - 1]
            nj = nks[mat_i]

        # theta is overall angle of inc, will be some normal float value for R or T BUT will be an
        # array describing absorption per layer if absorption happens
        # th_local is local angle w.r.t surface normal at that point on surface

        res, theta, phi, r_a, d, th_local, n_interactions, _ = single_surface[periodic](
            r_a,
            d,
            ni,
            nj,
            surf,
            surf.Lx,
            surf.Ly,
            direction,
            surf.zcov,
            pol,
            n_interactions,
            **tmm_kwargs_list[surf_index]
        )


        if res == 0:  # reflection
            direction = -direction  # changing direction due to reflection

            # staying in the same material, so mat_i does not change, but surf_index does
            surf_index = surf_index + direction


        elif res == 1:  # transmission
            surf_index = surf_index + direction
            mat_i = mat_i + direction

        elif res == 2:  # absorption
            stop = True  # absorption in an interface (NOT a bulk layer!)
            A_interface_array = (
                I * theta[:] / np.sum(theta)
            )  # if absorbed, theta contains information about A_per_layer
            A_interface_index = surf_index + 1
            theta = None
            I = 0

        if direction == 1 and mat_i == (len(widths) - 1):
            stop = True  # have ended with transmission

        elif direction == -1 and mat_i == 0:
            stop = True  # have ended with reflection

        # print("phi", np.real(atan2(d[1], d[0])))

        if not stop:
            I_b = I

            DA, stop, I, theta = traverse(
                widths[mat_i],
                theta,
                alphas[mat_i],
                x,
                y,
                I,
                depths[mat_i],
                I_thresh,
                direction,
            )

            # traverse bulk layer. Possibility of absorption; in this case will return stop = True
            # and theta = None

            A_per_layer[mat_i] = np.real(A_per_layer[mat_i] + I_b - I)
            profile[depth_indices[mat_i]] = np.real(
                profile[depth_indices[mat_i]] + DA
            )

            n_passes = n_passes + 1

            if lambertian_approximation and n_passes >= lambertian_approximation:
                # choose a direction randomly, with probability determined by Lambertian scattering
                stop = True
                theta = 10*direction # +ve if travelling down, about to hit rear surface.
                # stop right before next interaction with surface, but AFTER taking into account bulk
                # absorption on this pass

    return (
        I,
        profile,  # bulk profile only. Profile in interfaces gets calculated after ray-tracing is done.
        A_per_layer,  # absorption in bulk layers only, not interfaces
        theta,  # global theta
        phi,  # global phi
        n_passes,
        n_interactions,
        A_interface_array,
        A_interface_index,
        th_local,
        direction,
    )


def single_ray_interface(
    x, y, nks, r_a_0, theta, phi, surfaces, pol, wl, Fr_or_TMM, lookuptable
):
    direction = 1  # start travelling downwards; 1 = down, -1 = up
    mat_index = 0  # start in first medium
    surf_index = 0
    stop = False
    I = 1

    # could be done before to avoid recalculating every time
    r_a = r_a_0 + np.array([x, y, 0])
    r_b = np.array(
        [x, y, 0]
    )  # set r_a and r_b so that ray has correct angle & intersects with first surface
    d = (r_b - r_a) / np.linalg.norm(r_b - r_a)  # direction (unit vector) of ray

    while not stop:

        surf = surfaces[surf_index]

        r_a[0] = r_a[0] - surf.Lx * (
            (r_a[0] + d[0] * (surf.zcov - r_a[2]) / d[2]) // surf.Lx
        )
        r_a[1] = r_a[1] - surf.Ly * (
            (r_a[1] + d[1] * (surf.zcov - r_a[2]) / d[2]) // surf.Ly
        )

        res, theta, phi, r_a, d, theta_loc, _, _ = single_interface_check(
            r_a,
            d,
            nks[mat_index],
            nks[mat_index + 1],
            surf,
            surf.Lx,
            surf.Ly,
            direction,
            surf.zcov,
            pol,
            0,
            wl,
            Fr_or_TMM,
            lookuptable,
        )

        if res == 0:  # reflection
            direction = -direction  # changing direction due to reflection

            # staying in the same material, so mat_index does not change, but surf_index does
            surf_index = surf_index + direction

            surface_A = [0, 10]

        if res == 1:  # transmission
            surf_index = surf_index + direction
            mat_index = mat_index + direction  # is this right?

            surface_A = [0, 10]

        if res == 2:
            surface_A = [
                theta,
                theta_loc,
            ]  # passed a list of absorption per layer in theta
            stop = True
            theta = 10  # theta returned by single_interface_check is actually list of absorption per layer

        if direction == 1 and mat_index == 1:

            stop = True
            # have ended with transmission

        elif direction == -1 and mat_index == 0:
            stop = True

    return I, theta, phi, surface_A


def traverse(width, theta, alpha, x, y, I_i, positions, I_thresh, direction):
    stop = False
    ratio = alpha / np.real(np.abs(cos(theta)))
    DA_u = I_i * ratio * np.exp((-ratio * positions))
    I_back = I_i * np.exp(-ratio * width)

    if I_back < I_thresh:
        stop = True
        theta = None

    if direction == -1:
        DA_u = np.flip(DA_u)

    intgr = np.trapz(DA_u, positions)

    DA = np.divide(
        (I_i - I_back) * DA_u, intgr, where=intgr > 0, out=np.zeros_like(DA_u)
    )

    return DA, stop, I_back, theta


def decide_RT_Fresnel(n0, n1, theta, d, N, side, pol, rnd, wl=None, lookuptable=None):
    ratio = np.clip(np.real(n1) / np.real(n0), -1, 1)

    if abs(theta) > np.arcsin(ratio):
        R = 1
    else:
        R = calc_R(n0, n1, abs(theta), pol)

    # print('local theta/R', theta, R, n0, n1, d, N, side)
    # if np.real(n1) == 1:
    #     print('local theta', theta, R)

    if rnd <= R:  # REFLECTION
        d = np.real(d - 2 * np.dot(d, N) * N)

    else:  # TRANSMISSION)
        # transmission, refraction
        # for now, ignore effect of k on refraction
        tr_par = (np.real(n0) / np.real(n1)) * (d - np.dot(d, N) * N)
        tr_perp = -sqrt(1 - np.linalg.norm(tr_par) ** 2) * N
        side = -side
        d = np.real(tr_par + tr_perp)

    d = d / np.linalg.norm(d)

    return d, side, None  # never absorbed, A = False


def decide_RT_TMM(n0, n1, theta, d, N, side, pol, rnd, wl, lookuptable):
    data = lookuptable.loc[dict(side=side, pol=pol)].sel(
        angle=abs(theta), method="nearest",
    )

    R = np.real(data["R"].data.item(0))
    T = np.real(data["T"].data.item(0))
    A_per_layer = np.real(data["Alayer"].data)

    if rnd <= R:  # REFLECTION

        d = np.real(d - 2 * np.dot(d, N) * N)
        d = d / np.linalg.norm(d)
        A = None

    elif (rnd > R) & (rnd <= (R + T)):  # TRANSMISSION
        # transmission, refraction
        # tr_par = (np.real(n0) / np.real(n1)) * (d - np.dot(d, N) * N)
        tr_par = (n0 / n1) * (d - np.dot(d, N) * N)
        tr_perp = -sqrt(1 - np.linalg.norm(tr_par) ** 2) * N

        side = -side
        d = np.real(tr_par + tr_perp)
        d = d / np.linalg.norm(d)
        A = None

    else:
        # absorption
        A = A_per_layer

    return d, side, A


def single_interface_check(
    r_a,
    d,
    ni,
    nj,
    tri,
    Lx,
    Ly,
    side,
    z_cov,
    pol,
    n_interactions=0,
    wl=None,
    Fr_or_TMM=0,
    lookuptable=None,
):
    decide = {0: decide_RT_Fresnel, 1: decide_RT_TMM}

    d0 = d
    intersect = True
    checked_translation = False
    # [top, right, bottom, left]
    translation = np.array([[0, -Ly, 0], [-Lx, 0, 0], [0, Ly, 0], [Lx, 0, 0]])
    n_misses = 0
    i1 = 0

    while intersect:
        i1 = i1 + 1
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # there will be divide by 0/multiply by inf - this is fine but gives lots of warnings
            result = check_intersect(r_a, d, tri)
        if result is False and not checked_translation:
            if i1 > 1:
                which_side, _ = exit_side(r_a, d, Lx, Ly)
                r_a = r_a + translation[which_side]
                # if random pyramid, need to change surface at this point
                tri.refresh()
                checked_translation = True

            else:
                if n_misses < 100:
                    # misses surface. Try again
                    if d[2] < 0:  # coming from above
                        r_a = np.array(
                            [
                                np.random.rand() * Lx,
                                np.random.rand() * Ly,
                                tri.z_max + 0.01,
                            ]
                        )
                    else:
                        r_a = np.array(
                            [
                                np.random.rand() * Lx,
                                np.random.rand() * Ly,
                                tri.z_min - 0.01,
                            ]
                        )
                    n_misses += 1
                    i1 = 0

                else:
                    # ray keeps missing, probably because it's travelling (almost) exactly perpendicular to surface.
                    # assume it is reflected back into layer it came from
                    d[2] = -d[2]

                    o_t = np.real(acos(d[2] / np.linalg.norm(d)))
                    o_p = np.real(atan2(d[1], d[0]))
                    return 0, o_t, o_p, r_a, d, 0, n_interactions, side

        elif result is False and checked_translation:

            if (side == 1 and d[2] < 0 and r_a[2] > tri.z_min) or (
                side == -1 and d[2] > 0 and r_a[2] < tri.z_max
            ):
                # going down but above surface

                if r_a[0] > Lx or r_a[0] < 0:
                    r_a[0] = (
                        r_a[0] % Lx
                    )  # translate back into until cell before doing any additional translation
                if r_a[1] > Ly or r_a[1] < 0:
                    r_a[1] = (
                        r_a[1] % Ly
                    )  # translate back into until cell before doing any additional translation
                ex, t = exit_side(r_a, d, Lx, Ly)

                r_a = r_a + t * d + translation[ex]
                # also change surface here
                tri.refresh()

                checked_translation = True

            else:

                o_t = np.real(acos(d[2] / np.linalg.norm(d)))
                o_p = np.real(atan2(d[1], d[0]))

                if np.sign(d0[2]) == np.sign(d[2]):
                    intersect = False
                    final_res = 1

                else:
                    intersect = False
                    final_res = 0

                if r_a[0] > Lx or r_a[0] < 0:
                    r_a[0] = (
                        r_a[0] % Lx
                    )  # translate back into until cell before next ray
                if r_a[1] > Ly or r_a[1] < 0:
                    r_a[1] = (
                        r_a[1] % Ly
                    )  # translate back into until cell before next ray

                return (
                    final_res,
                    o_t,  # theta with respect to horizontal
                    o_p,
                    r_a,
                    d,
                    theta,  # LOCAL incidence angle
                    n_interactions,
                    side,
                )  # theta is LOCAL incidence angle (relative to texture)

        else:

            # there has been an intersection
            n_interactions += 1

            intersn = result[0]  # coordinate of the intersection (3D)

            theta = result[1]

            N = (
                result[2] * side
            )  # so angles get worked out correctly, relative to incident face normal

            if side == 1:
                n0 = ni
                n1 = nj

            else:
                n0 = nj
                n1 = ni

            rnd = random()

            d, side, A = decide[Fr_or_TMM](
                n0, n1, theta, d, N, side, pol, rnd, wl, lookuptable
            )

            r_a = np.real(
                intersn + d / 1e9
            )  # this is to make sure the raytracer doesn't immediately just find the same intersection again

            checked_translation = False  # reset, need to be able to translate the ray back into the unit cell again if necessary

            if A is not None:
                # intersect = False
                # checked_translation = True
                final_res = 2
                o_t = A
                o_p = 0

                return (
                    final_res,
                    o_t, # A array, NOT theta (only in the case of absorption)
                    o_p,
                    r_a,
                    d,
                    theta,  # LOCAL incidence angle
                    n_interactions,
                    side,
                )


def single_cell_check(
    r_a,
    d,
    ni,
    nj,
    tri,
    Lx,
    Ly,
    side,
    z_cov,
    pol,
    n_interactions=0,
    wl=None,
    Fr_or_TMM=0,
    lookuptable=None,
):
    decide = {0: decide_RT_Fresnel, 1: decide_RT_TMM}

    theta = 0  # needs to be assigned so no issue with return in case of miss
    # print('side', side)
    d0 = d
    intersect = True
    n_ints_loop = 0
    # [top, right, bottom, left]
    n_misses = 0
    i1 = 0

    while intersect:
        i1 = i1 + 1
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # there will be divide by 0/multiply by inf - this is fine but gives lots of warnings
            result = check_intersect(r_a, d, tri)

        if result is False:

            n_misses += 1

            o_t = np.real(acos(d[2] / np.linalg.norm(d)))
            o_p = np.real(atan2(d[1], d[0]))

            if np.sign(d0[2]) == np.sign(d[2]):
                intersect = False
                final_res = 1

            else:
                intersect = False
                final_res = 0

            return (
                final_res,
                o_t,
                o_p,
                r_a,
                d,
                theta,
                n_interactions,
                side,
            )  # theta is LOCAL incidence angle (relative to texture)

        else:

            # there has been an intersection
            n_interactions += 1
            n_ints_loop += 1

            intersn = result[0]  # coordinate of the intersection (3D)

            theta = result[1]

            N = (
                result[2] * side
            )  # so angles get worked out correctly, relative to incident face normal

            if side == 1:
                n0 = ni
                n1 = nj

            else:
                n0 = nj
                n1 = ni

            rnd = random()

            d, side, A = decide[Fr_or_TMM](
                n0, n1, theta, d, N, side, pol, rnd, wl, lookuptable
            )

            r_a = np.real(
                intersn + d / 1e9
            )  # this is to make sure the raytracer doesn't immediately just find the same intersection again

            if A is not None:
                # intersect = False
                # checked_translation = True
                final_res = 2
                o_t = A
                o_p = 0

                return (
                    final_res,
                    o_t,  # A array
                    o_p,
                    r_a,
                    d,
                    theta,  # LOCAL incidence angle
                    n_interactions,
                    side,
                )


def check_intersect(r_a, d, tri):
    # all the stuff which is only surface-dependent (and not dependent on incoming direction) is
    # in the surface object tri.
    D = np.tile(-d, (tri.size, 1))
    pref = 1 / np.sum(D * tri.crossP, axis=1)
    corner = r_a - tri.P_0s
    t = pref * np.sum(tri.crossP * corner, axis=1)
    u = pref * np.sum(np.cross(tri.P_2s - tri.P_0s, D) * corner, axis=1)
    v = pref * np.sum(np.cross(D, tri.P_1s - tri.P_0s) * corner, axis=1)

    which_intersect = (
        (u + v <= 1) & (np.all(np.vstack((u, v)) >= -1e-10, axis=0)) & (t > 0)
    )
    # get errors if set exactly to zero.
    if sum(which_intersect) > 0:

        t = t[which_intersect]
        ind = np.argmin(t)
        t = min(t)

        intersn = r_a + t * d

        N = tri.N[which_intersect][ind]

        theta = atan(
            np.linalg.norm(np.cross(N, -d)) / np.dot(N, -d)
        )  # in radians, angle relative to plane

        return [intersn, theta, N]
    else:
        return False