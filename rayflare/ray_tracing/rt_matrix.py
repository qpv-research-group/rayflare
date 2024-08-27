# Copyright (C) 2021-2024 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU Lesser General Public License (LGPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: p.pearce@unsw.edu.au

import numpy as np
import os
from cmath import sin, cos
from itertools import product
import xarray as xr
from sparse import COO, save_npz, stack
from joblib import Parallel, delayed
from warnings import warn

from rayflare.angles import fold_phi, make_angle_vector, overall_bin
from rayflare.utilities import get_matrices_or_paths
from .rt_common import Ray, single_interface_check, make_pol_vectors

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
    analytical=False,
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
        wavelengths = options.wavelength
        n_rays = options.n_rays
        nx = options.nx
        ny = options.ny
        n_angles = int(np.ceil(n_rays / (nx * ny)))

        phi_sym = options.phi_symmetry
        n_theta_bins = options.n_theta_bins
        c_az = options.c_azimuth

        pol = options.pol

        # if not (pol[0] == 1 or pol[1] == 1):
        #     logger.warning("Warning: you have specificied unpolarized/partially polarized light. "
        #                    "The ARRM RT class does not currently take into account polarization "
        #                    "changes at interfaces.")

        if not options.parallel:
            n_jobs = 1

        else:
            n_jobs = options.n_jobs if "n_jobs" in options else -1

        if calc_profile is not None:
            depth_spacing = options.depth_spacing * 1e9  # convert from m to nm
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
            if options.theta_in == 0:
                th_in = 0.0001
            else:
                th_in = options.theta_in

            angles_in = angle_vector[: int(len(angle_vector) / 2), :]
            n_reps = int(np.ceil(n_angles / len(angles_in)))
            thetas_in = np.tile(th_in, n_reps)
            n_angles = n_reps

            if options.phi_in == "all":
                # get relevant phis
                phis_in = np.tile(options.phi_in, n_reps)
            else:
                if options.phi_in == 0:
                    phis_in = np.tile(0.0001, n_reps)

                else:
                    phis_in = np.tile(options.phi_in, n_reps)

        else:
            if options.random_ray_angles:
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
            options.x_limits
            if "x_limits" in options
            else [
                surfaces[0].x_min + 0.01 * surfaces[0].Lx,
                surfaces[0].x_max - 0.01 * surfaces[0].Lx,
            ]
        )
        y_limits = (
            options.y_limits
            if "y_limits" in options
            else [
                surfaces[0].y_min + 0.01 * surfaces[0].Ly,
                surfaces[0].y_max - 0.01 * surfaces[0].Ly,
            ]
        )

        if options.random_ray_position:
            xs = np.random.uniform(x_limits[0], x_limits[1], nx)
            ys = np.random.uniform(y_limits[0], y_limits[1], ny)

        else:
            xs = np.linspace(x_limits[0], x_limits[1], nx)
            ys = np.linspace(y_limits[0], y_limits[1], ny)

        if analytical:
            pass

        else:

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
                    (np.pi / 2) / options.lookuptable_angles,
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
    d_theta,
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

    if lookuptable is not None:
        lookuptable_wl_sp = lookuptable.sel(pol=["s", "p"]).sel(wl=wl * 1e9).load()
        lookuptable_wl = lookuptable.sel(wl=wl * 1e9).load()

    else:
        lookuptable_wl = None
        lookuptable_wl_sp = None

    logger.info(f"RT calculation for wavelength = {wl * 1e9} nm")

    theta_out = np.zeros((n_angles, nx * ny))
    phi_out = np.zeros((n_angles, nx * ny))
    A_surface_layers = np.zeros((n_angles, nx * ny, n_abs_layers))
    theta_local_incidence = np.zeros((n_angles, nx * ny))
    pol_local_incidence = np.zeros((n_angles, nx * ny, 2))

    for i2 in range(n_angles):

        theta = thetas_in[i2]
        phi = phis_in[i2]
        r = abs((h + 1e-8) / cos(theta))
        r_a_0 = np.real(
            np.array(
                [r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)]
            )
        )
        pol, pol_vectors = make_pol_vectors(pol, theta, phi)

        for c, vals in enumerate(product(xs, ys)):
            ray, th_o, phi_o, surface_A = single_ray_interface(
                vals[0],
                vals[1],
                nks[:, i1],
                r_a_0,
                theta,
                phi,
                surfaces,
                pol,
                pol_vectors,
                d_theta,
                wl,
                Fr_or_TMM,
                lookuptable_wl_sp,
            )

            if th_o < 0:  # can do outside loup with np.where
                th_o = -th_o
                phi_o = phi_o + np.pi
            theta_out[i2, c] = th_o
            phi_out[i2, c] = phi_o
            A_surface_layers[i2, c] = surface_A[0]
            theta_local_incidence[i2, c] = np.real(surface_A[1])
            pol_local_incidence[i2, c] = ray.pol

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

    # for some reason, sometimes get really small -ve values like -1e-300 instead of 0.
    # this causes issues and produces nans in the matrix multiplication (dot_wl) later, although
    # only on Ubuntu for some reason?
    out_mat[out_mat < 0] = 0
    A_mat[A_mat < 0] = 0

    out_mat = COO.from_numpy(out_mat)  # sparse matrix
    A_mat = COO.from_numpy(A_mat)

    # out_mat[out_mat < 0] = 0

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
                lookuptable_wl,
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


def make_profiles_wl(
    unique_thetas,
    n_a_in,
    side,
    widths,
    angle_distmat,
    lookuptable,
    pol,
    depth_spacing,
    prof_layers,
):
    # widths and depth_spacing are passed in nm!

    def profile_per_layer(xx, z, offset, side, non_zero):
        layer_index = xx.coords["layer"].item(0) - 1
        x = xx.squeeze()
        x = x[non_zero]
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
        by_layer = x.groupby("layer", squeeze=False).map(
            profile_per_layer, z=z, offset=offset, side=side, non_zero=non_zero
        )
        return by_layer

    def scale_func(x, scale_params):
        xx = x.squeeze()
        return xx.data[:, None, None] * scale_params

    def select_func(x, const_params):
        xx = x.squeeze()
        return (xx.data[:, None, None] != 0) * const_params

    pr = xr.DataArray(
        angle_distmat.todense(),
        dims=["local_theta", "global_index"],
        coords={"local_theta": unique_thetas, "global_index": np.arange(0, n_a_in)},
    )

    # lookuptable layers are 1-indexed
    if pol[0] > 0.999:
        pol_s = "s"

    elif pol[1] > 0.999:
        pol_s = "p"

    else:
        pol_s = "u"

    data = lookuptable.loc[dict(side=1, pol=pol_s)].interp(
        angle=pr.coords["local_theta"], #wl=wl * 1e9
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

    scale_res = pr.groupby("global_index", squeeze=False).map(scale_func, scale_params=s_params)
    const_res = pr.groupby("global_index", squeeze=False).map(select_func, const_params=c_params)

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


def single_ray_interface(
    x, y, nks, r_a_0, theta, phi, surfaces, pol, pol_vec, d_theta, wl, Fr_or_TMM, lookuptable
):
    direction = 1  # start travelling downwards; 1 = down, -1 = up
    mat_index = 0  # start in first medium
    surf_index = 0
    stop = False

    # could be done before to avoid recalculating every time
    r_a = r_a_0 + np.array([x, y, 0])
    r_b = np.array(
        [x, y, 0]
    )  # set r_a and r_b so that ray has correct angle & intersects with first surface
    d = (r_b - r_a) / np.linalg.norm(r_b - r_a)  # direction (unit vector) of ray

    ray = Ray(1, d, r_a, pol_vec[0], pol_vec[1], pol)

    while not stop:

        surf = surfaces[surf_index]

        ray.r_a[0] = ray.r_a[0] - surf.Lx * (
            (ray.r_a[0] + ray.d[0] * (surf.zcov - ray.r_a[2]) / ray.d[2]) // surf.Lx
        )
        ray.r_a[1] = ray.r_a[1] - surf.Ly * (
            (ray.r_a[1] + ray.d[1] * (surf.zcov - ray.r_a[2]) / ray.d[2]) // surf.Ly
        )

        res, theta, phi, theta_loc, _, _ = single_interface_check(
            ray,
            nks[mat_index],
            nks[mat_index + 1],
            surf,
            surf.Lx,
            surf.Ly,
            direction,
            surf.zcov,
            d_theta,
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

    return ray, theta, phi, surface_A