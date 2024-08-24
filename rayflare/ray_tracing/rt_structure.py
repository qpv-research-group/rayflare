# Copyright (C) 2021-2024 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU Lesser General Public License (LGPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: p.pearce@unsw.edu.au

import numpy as np
import os
from cmath import sin, cos, acos
from math import atan2
from itertools import product
import xarray as xr
from joblib import Parallel, delayed
from copy import deepcopy
from solcore.state import State
from scipy.spatial.transform import Rotation
from numba import jit

from rayflare.utilities import get_savepath, get_wavelength, process_pol
from rayflare.transfer_matrix_method.lookup_table import make_TMM_lookuptable
from rayflare import logger
from .analytical_rt import (lambertian_scattering, calculate_lambertian_profile,
                            analytical_start, dummy_prop_rays)
from .rt_common import Ray, single_cell_check, single_interface_check, normalize, norm


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

    # direction = 1: travelling down
    # direction = -1: travelling up

    #      incidence medium (0)
    # surface 0  ---------
    #           material 1
    # surface 1  ---------
    #           material 2
    # surface 2  ---------

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

        if isinstance(options, dict):
            options = State(options)

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

        self.lambertian_results = None
        if hasattr(options, "lambertian_approximation"):
            if options.lambertian_approximation:
                self.lambertian_results = lambertian_scattering(self, save_location, options)

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
           - pol: Polarisation of the light: 's', 'p' or 'u', or a list/tuple of length 2 with the fraction
                  of ['s', 'p'] polarized light.
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
        periodic = options.periodic if "periodic" in options else 1
        depth_spacing_interfaces = (
            options.depth_spacing * 1e9 if "depth_spacing" in options else 1
        )
        lambertian_approximation = options.lambertian_approximation
        analytical_rt = options.analytical_ray_tracing

        if not options.parallel:
            n_jobs = 1

        else:
            n_jobs = options.n_jobs if "n_jobs" in options else -1

        widths = self.widths[:]
        widths.insert(0, 0)
        widths.append(0)
        widths = 1e6 * np.array(widths)  # convert to um

        z_space = 1e6 * options.depth_spacing_bulk  # convert from m to um
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

        nx = options.nx
        ny = options.ny

        if options.random_ray_position:
            xs = np.random.uniform(x_limits[0], x_limits[1], nx)
            ys = np.random.uniform(y_limits[0], y_limits[1], ny)

        else:
            xs = np.linspace(x_limits[0], x_limits[1], nx)
            ys = np.linspace(y_limits[0], y_limits[1], ny)

        # need to calculate r_a and r_b
        # a total of n_rays will be traced; this is divided by the number of x and y points to scan so we know
        # how many times we need to repeat
        n_reps = int(np.ceil(options.n_rays / (nx * ny)))

        pol = process_pol(options.pol)
        pol = np.array(pol)/np.sum(pol)

        r_a_0_norm = normalize(r_a_0)

        rot_90 = Rotation.from_euler('y', np.pi/2)
        initial_p_dir = rot_90.apply(r_a_0_norm)
        initial_s_dir = np.cross(r_a_0_norm, initial_p_dir)

        initial_pol_vectors = [initial_s_dir, initial_p_dir]

        randomize = options.randomize_surface

        initial_mat = (
            options.initial_material if "initial_material" in options else 0
        )
        initial_dir = (
            options.initial_direction if "initial_direction" in options else 1
        )

        cum_width = np.cumsum([0] + widths)

        depths = []
        depth_indices = []

        for i1 in range(len(widths)):
            depth_indices.append(
                (z_pos < np.cumsum(widths)[i1]) & (z_pos >= np.cumsum(widths)[i1 - 1])
            )
            depths.append(z_pos[depth_indices[i1]] - np.cumsum(widths)[i1 - 1])

        if analytical_rt > 0:
            prop_rays, result_per_wl, result_detail = analytical_start(
                nks,
                alphas,
                theta,
                r_a_0,
                phi,
                surfaces,
                widths,
                z_pos,
                depths,
                depth_indices,
                pol,
                initial_mat,
                initial_dir,
                tmm_args,
                options.analytical_ray_tracing,
                wavelengths*1e9
            )
            A_interface_to_add = result_per_wl["A_per_interface"]

        else:
            prop_rays = dummy_prop_rays()
            A_interface_to_add = None

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
                initial_pol_vectors,
                nx,
                ny,
                n_reps,
                xs,
                ys,
                randomize,
                initial_mat,
                initial_dir,
                periodic,
                (np.pi/2)/options.lookuptable_angles,
                lambertian_approximation,
                tmm_args + [wavelengths[i1]],
                prop_rays.isel(wl=i1),
            )
            for i1 in range(len(wavelengths)))

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
        n_rem = np.stack([item[9] for item in allres])

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

            if A_interface_to_add is not None:
                for i1, add in enumerate(A_interface_to_add):
                    A_per_interface[i1] = A_per_interface[i1] + add


        else:
            A_per_interface = 0
            interface_profiles = 0

        non_abs = np.logical_and(~np.isnan(thetas), np.abs(thetas) < 10)

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

        absorption_profiles[absorption_profiles < 0] = 0

        # process A_interfaces

        if np.any(np.abs(thetas) >= 10):  # Lambertian scattering
            # TODO: make this a separate function
            # +ve if travelling down, about to hit rear surface.
            # stop normal RT right before next interaction with surface, but AFTER taking into account bulk
            # absorption on this pass

            # per wavelength, find how many rays need to be accounted for with the Lambertian model:
            # which direction it's travelling in, which material it is in,
            # and the total intensity in those rays

            # indexing: wavelengths, material, direction
            I_transmitted = np.zeros(len(wavelengths))

            for i1 in range(len(mats)-2):

                direction_in_mat = np.sign(thetas[np.abs(thetas/10) == (i1+1)])[0]
                I_per_mat = [np.sum(Is[l][np.all((np.abs(thetas[l]/10) == (i1+1), np.sign(thetas[l]) == direction_in_mat), axis=0)]) for l in range(len(wavelengths))]
                I_lambertian = np.array(I_per_mat)/(n_reps*nx*ny) + I_transmitted

                lambertian_RAT = self.lambertian_results[i1][0].sel(direction=direction_in_mat)
                lambertian_A1 = self.lambertian_results[i1][1].sel(direction=direction_in_mat)
                lambertian_A2 = self.lambertian_results[i1][2].sel(direction=direction_in_mat)

                # where np.abs(thetas) == 10, want to divide remaining intensity in Is at the same location correctly between
                # reflection, interface absorption, and transmission

                I_RAT = lambertian_RAT * I_lambertian

                R += I_RAT.sel(event='R').data

                if i1 == len(mats) - 3: # final non-transmission bulk
                    T += I_RAT.sel(event='T').data
                else:
                    I_transmitted = I_RAT.sel(event='T').data

                A_layer[:, i1+1] += I_RAT.sel(event='A_bulk').data

                add_profile = calculate_lambertian_profile(self, I_lambertian, options,
                                                           direction_in_mat,
                                                           self.lambertian_results[i1][3], alphas[i1+1],
                                                           depths[i1+1],
                                                           I_RAT.sel(event='A_bulk').data)

                absorption_profiles[:, depth_indices[i1+1]] += add_profile

                if sum(self.tmm_or_fresnel) > 0:
                    add_frontsurf = (I_lambertian * lambertian_A1).data.T
                    add_backsurf = (I_lambertian * lambertian_A2).data.T

                    A_per_interface[i1] = A_per_interface[i1] + add_frontsurf
                    A_per_interface[i1+1] = A_per_interface[i1+1] + add_backsurf

        # add results from analytical ray tracing:
        if analytical_rt > 0:

            # TODO: make this a separate function
            R += result_per_wl["R"]
            T += result_per_wl["T"]
            A_layer += result_per_wl["A_per_layer"]
            absorption_profiles += result_per_wl["profile"]

            max_rays = len(xs) * len(ys) * n_reps

            thetas, phis, Is, n_passes, n_interactions = update_ray_tracing_results(
                max_rays, n_rem, result_detail, thetas, phis, Is, n_passes, n_interactions
            )

        refl_0 = (
            non_abs
            * np.less_equal(np.real(thetas), np.pi / 2, where=non_abs)
            * (n_passes == 0)
        )

        R0 = np.real(Is * refl_0).T / (n_reps * nx * ny)
        R0 = np.sum(R0, 0)

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
            "interface_profiles": interface_profiles,
            "Is": Is,
            "xy": [xs, ys],
        }

    def calculate_profile(self, options):
        prof_results = self.calculate(options)
        return prof_results


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

    savepath = get_savepath(save_location, options.project_name)

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



def update_ray_tracing_results(
    max_rays, n_rem, analytical_result, thetas, phis, Is, n_passes, n_interactions
):
    n_analytical = max_rays - n_rem
    wl_ind_anlt = np.where(n_analytical > 0)[0]

    overall_R = analytical_result["overall_R"]
    overall_T = analytical_result["overall_T"]
    A_details = analytical_result["A_details"]

    n_abs_anlt = np.round(np.sum(A_details.A.data, 0) * max_rays, 0).astype(int)
    R_per_wl = np.sum(overall_R.I, 0)
    T_per_wl = np.sum(overall_T.I, 0)

    for wl_ind in wl_ind_anlt:
        thetas[wl_ind][n_rem[wl_ind]:(n_rem[wl_ind] + n_abs_anlt[wl_ind])] = np.nan
        phis[wl_ind][n_rem[wl_ind]:(n_rem[wl_ind] + n_abs_anlt[wl_ind])] = np.nan
        Is[wl_ind][n_rem[wl_ind]:(n_rem[wl_ind] + n_abs_anlt[wl_ind])] = 0

        A_details_wl = A_details.isel(wl=wl_ind)
        n_per_entry = max_rays * A_details_wl.A.data
        n_per_entry_int = np.round(n_per_entry).astype(int)

        while np.sum(n_per_entry_int) < n_abs_anlt[wl_ind]:
            n_per_entry_int[np.argmax(n_per_entry - n_per_entry_int)] += 1

        while np.sum(n_per_entry_int) > n_abs_anlt[wl_ind]:
            n_per_entry_int[np.argmin(n_per_entry - n_per_entry_int)] -= 1

        A_details_wl = A_details_wl.sel(unique_direction=n_per_entry_int > 0)
        n_per_entry_int = n_per_entry_int[n_per_entry_int > 0]

        offset_n = 0
        for i1, n_entries in enumerate(n_per_entry_int):
            n_passes[wl_ind][(n_rem[wl_ind] + offset_n):(n_rem[wl_ind] + n_entries)] = A_details_wl.n_passes[i1].data
            n_interactions[wl_ind][(n_rem[wl_ind] + offset_n):(n_rem[wl_ind] + n_entries)] = A_details_wl.n_interactions[i1].data
            offset_n += n_entries

        n_other = n_analytical[wl_ind] - n_abs_anlt[wl_ind]

        if T_per_wl[wl_ind] > 0:
            n_R = int(np.round(R_per_wl[wl_ind] * n_other, 0))
            n_T = n_other - n_R
        else:
            n_R = n_other
            n_T = 0

        n_cat = [n_R, n_T]
        data_x = [overall_R, overall_T]

        current_start = n_rem[wl_ind] + n_abs_anlt[wl_ind]

        for anlt_data, n_in_cat in zip(data_x, n_cat):
            if n_in_cat > 0:
                weights = anlt_data.I.isel(wl=wl_ind).data
                rays_per_dir = weights / np.sum(weights) * n_in_cat
                n_rays_per_direction = np.round(rays_per_dir).astype(int)

                while sum(n_rays_per_direction) < n_in_cat:
                    n_rays_per_direction[np.argmax(rays_per_dir - n_rays_per_direction)] += 1

                while sum(n_rays_per_direction) > n_in_cat:
                    n_rays_per_direction[np.argmin(rays_per_dir - n_rays_per_direction)] -= 1

                # scale_I = weights * max_rays / n_rays_per_direction
                # avoid warning if n_rays_per_direction is 0 anywhere:
                scale_I = np.divide(weights * max_rays, n_rays_per_direction,
                                    out=np.zeros_like(weights),
                                    where=n_rays_per_direction != 0)

                if 'wl' in anlt_data.direction.dims:
                    theta_R = np.arccos(anlt_data.direction.isel(wl=wl_ind, xyz=2))
                    phi_R = np.arctan(anlt_data.direction.isel(wl=wl_ind, xyz=1) / anlt_data.direction.isel(wl=wl_ind, xyz=0))
                else:
                    theta_R = np.arccos(anlt_data.direction.isel(xyz=2))
                    phi_R = np.arctan(anlt_data.direction.isel(xyz=1) / anlt_data.direction.isel(xyz=0))

                for i1, n_unique_R in enumerate(n_rays_per_direction):
                    thetas[wl_ind][current_start:(current_start + n_unique_R)] = theta_R[i1]
                    phis[wl_ind][current_start:(current_start + n_unique_R)] = phi_R[i1]
                    Is[wl_ind][current_start:(current_start + n_unique_R)] = scale_I[i1]
                    n_interactions[wl_ind][current_start:(current_start + n_unique_R)] = anlt_data.n_interactions[i1]
                    current_start += n_unique_R

    return thetas, phis, Is, n_passes, n_interactions

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
    initial_pol_vec,
    nx,
    ny,
    n_reps,
    xs,
    ys,
    randomize,
    initial_mat,
    initial_dir,
    periodic,
    d_theta,
    lambertian_approximation=0,
    tmm_args=None,
    existing_rays=None,
):

    # analytical front surface ray-tracing should take place outside the loop,
    # since it can be done for all wavelengths at the same time, and modify
    # the variables passed to single_ray_stack accordingly.

    # Will need to modify:
    # - x/y (randomize)
    # - r_a_0
    # - randomize should be forced True if using analytical front surface
    # - initial_mat
    # - number of rays should be modified to account for those which have already been
    #   reflected or absorbed in the front surface

    # # generally same across wavelengths, but can be changed by analytical
    # # ray tracing happening first
    # decide whether ray-tracing needs to happen at all:

    # thetas and phis divided into
    thetas = np.zeros(n_reps * nx * ny)
    phis = np.zeros(n_reps * nx * ny)
    n_passes = np.zeros(n_reps * nx * ny)
    n_interactions = np.zeros(n_reps * nx * ny)
    A_layer = np.zeros(len(widths))
    Is = np.zeros(n_reps * nx * ny)

    A_interfaces = [[] for _ in range(len(surfaces) + 1)]
    local_thetas = [[] for _ in range(len(surfaces) + 1)]
    local_pols = [[] for _ in range(len(surfaces) + 1)]
    directions = [[] for _ in range(len(surfaces) + 1)]

    profiles = np.zeros(len(z_pos))

    profile_arrays = [[] for _ in range(len(surfaces))]

    if existing_rays is None:
        continue_wl = True
        prop_rays_analytical = False

    else:
        continue_wl = np.sum(existing_rays.I) > I_thresh
        n_remaining = 0
        prop_rays_analytical = True

        if tmm_args[0] > 0:

            A_in_interfaces = [np.zeros(n_l) for n_l in tmm_args[4]]
        else:
            A_in_interfaces = 0

        # need to populate the arrays with results from the rays which were already
        # absorbed:



    # print('continue_wl', continue_wl)

    if continue_wl:

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

        d = -normalize(r_a_0)

        if initial_dir != 1: # upwards initial direction
            d[2] = -d[2]

        if tmm_args is None:
            tmm_args = [0]


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

        # existing_rays is an xarray DataSet which contains 3 DataArrays:
        # - I: intensity of the rays travelling in each unique direction, at the wavelength of interest
        # - direction: the x/y/z directions of the rays
        # - mat_i: the index of the material the ray has just traversed.


        end_ind = nx*ny*np.ones(n_reps, dtype=int)

        if prop_rays_analytical:

            phong_params = np.array([x.phong for x in surfaces])
            phong_options = [x.phong_options for x in surfaces]
            # TODO: pol will also change if already interacted with a surface!
            # And polarization directions
            ds, pols, i_mats, i_dirs, surf_inds, n_remaining, I_in, n_inter_in, n_passes_in = (
                make_rt_args(existing_rays, xs, ys, n_reps, phong_params, phong_options))
            stop_before = int(np.ceil(n_remaining/(nx*ny)))
            # r_as need to be set so that z is somewhere within the current surface:
            z_offs = -cum_width[i_mats - 1] - 1e-8
            r_as = np.hstack((np.zeros((n_remaining,2)), z_offs[:,None]))
            # will probably need to end somewhere halfway through the x/y loop:
            end_ind[stop_before:] = 0
            end_ind[stop_before - 1] = n_remaining - (stop_before - 1)*nx*ny
            # print('remaining', n_remaining)

        else:

            ds = np.array([d for _ in range(n_reps * nx * ny)])
            pols = np.array([pol for _ in range(n_reps * nx * ny)])
            i_mats = np.array([initial_mat for _ in range(n_reps * nx * ny)])
            i_dirs = np.array([initial_dir for _ in range(n_reps * nx * ny)])
            surf_inds = np.array([surf_index for _ in range(n_reps * nx * ny)])

            x_y_combs = np.array(list(product(xs, ys)))

            r_as = np.array([[r_a_0[0] + vals[0], r_a_0[1] + vals[1], z_offset] for vals in x_y_combs]) # ?
            # stack this n_reps times:
            r_as = np.tile(r_as, (n_reps, 1))

            I_in = np.ones(n_reps * nx * ny)
            n_passes_in = np.zeros(n_reps * nx * ny)
            n_inter_in = np.zeros(n_reps * nx * ny)
            n_remaining = n_reps*nx*ny

        overall_i = 0
        for j1 in range(n_reps):
            offset = j1 * nx * ny

            for c in range(end_ind[j1]):
                (
                    ray,
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
                    nks,
                    alphas,
                    r_as[overall_i],
                    ds[overall_i],
                    surfaces,
                    additional_tmm_args,
                    widths,
                    z_pos,
                    depths,
                    depth_indices,
                    I_thresh,
                    pols[overall_i],
                    initial_pol_vec,
                    d_theta,
                    randomize,
                    i_mats[overall_i],
                    i_dirs[overall_i],
                    surf_inds[overall_i],
                    periodic,
                    lambertian_approximation,
                    n_passes_in[overall_i],
                    n_inter_in[overall_i],
                    I_in[overall_i],
                )

                overall_i += 1

                A_interfaces[A_interface_index].append(A_interface_array)
                profiles += profile / (n_reps * nx * ny)
                thetas[c + offset] = th_o
                phis[c + offset] = phi_o
                Is[c + offset] = np.real(ray.I)
                A_layer += A_per_layer / (n_reps * nx * ny)
                n_passes[c + offset] = n_pass
                n_interactions[c + offset] = n_interact
                local_thetas[A_interface_index].append(np.real(th_local))
                local_pols[A_interface_index].append(ray.pol)
                directions[A_interface_index].append(direction)

        A_interfaces = A_interfaces[1:]
        # index 0 are all entries for non-interface-absorption events.
        local_thetas = local_thetas[1:]
        local_pols = local_pols[1:]
        directions = directions[1:]

        phis[np.isnan(thetas)] = np.nan

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
                            local_pols[i1],
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
        n_remaining,
    )

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
    local_pols_i,
    depth_spacing,
):

    def profile_per_layer(xx, z, offset, side):

        layer_index = xx.coords["layer"].item(0) - 1
        x = xx.squeeze(dim="layer")
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

    # lookuptable should only have s and p entries (in that order) at this point
    th_array = np.abs(local_thetas_i)
    front_incidence = np.where(directions_i == 1)[0]
    rear_incidence = np.where(directions_i == -1)[0]

    local_pols_i = np.array(local_pols_i).T
    # need to scale absorption profile for each ray depending on
    # how much intensity was left in it when that ray was absorbed (this is done for total absorption inside
    # single_ray_stack)

    if len(front_incidence) > 0:

        A_lookup_front = lookuptable.Alayer.loc[
            dict(side=1, layer=prof_layer_list_i)
        ].interp(angle=th_array[front_incidence] #, wl=wl * 1e9
                 )# )

        local_pols_dir = local_pols_i[:, front_incidence]

        A_lookup_front = np.sum(local_pols_dir[:,:, None] * A_lookup_front, 0)

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

        params = lookuptable.Aprof.loc[
            dict(side=1, layer=prof_layer_list_i)
        ].interp(angle=th_array[front_incidence])

        ans_list = []

        for i1, pol in enumerate(['s', 'p']):

            params_pol = params.sel(pol=pol)

            s_params_pol = params_pol.loc[
                dict(coeff=["A1", "A2", "A3_r", "A3_i"])
            ]  # have to scale these to make sure integrated absorption is correct
            c_params_pol = params_pol.loc[
                dict(coeff=["a1", "a3"])
            ]  # these should not be scaled

            scale_res = local_pols_dir[i1][:,None, None]*s_params_pol * scale_factor[:, None, None]

            if np.sum(scale_res) > 0:
                params_pol = xr.concat((scale_res, c_params_pol), dim="coeff")

                ans_pol = (
                    params_pol.groupby("angle", squeeze=False)
                    .map(profile_per_angle, z=z_list, offset=offsets, side=1)
                    .drop_vars("coeff")
                )

                ans_list.append(ans_pol.reduce(np.sum, ["angle"]).fillna(0))

            else:
                ans_list.append(0)

        profile_front = ans_list[0] + ans_list[1]

    else:
        profile_front = 0

    if len(rear_incidence) > 0:

        A_lookup_back = lookuptable.Alayer.loc[
            dict(side=-1, layer=prof_layer_list_i)
        ].interp(angle=th_array[rear_incidence],
                 # wl=wl * 1e9,
                 )

        local_pols_dir = local_pols_i[:, rear_incidence]

        A_lookup_back = np.sum(local_pols_dir[:, :, None] * A_lookup_back, 0)

        data_back = data_prof_layers[rear_incidence]

        non_zero = xr.where(A_lookup_back > 1e-10, A_lookup_back, np.nan)

        scale_factor = (
            np.divide(data_back, non_zero).mean(dim="layer", skipna=True).data
        )  # can get slight differences in values between layers

        params = lookuptable.Aprof.loc[
            dict(side=-1, layer=prof_layer_list_i)
        ].interp(angle=th_array[rear_incidence])

        ans_list = []

        for i1, pol in enumerate(['s', 'p']):
            params_pol = params.sel(pol=pol)

            s_params_pol = params_pol.loc[
                dict(coeff=["A1", "A2", "A3_r", "A3_i"])
            ]  # have to scale these to make sure integrated absorption is correct
            c_params_pol = params_pol.loc[
                dict(coeff=["a1", "a3"])
            ]  # these should not be scaled

            scale_res = local_pols_dir[i1][:, None, None] * s_params_pol * scale_factor[:, None,
                                                                           None]

            if np.sum(scale_res) > 0:
                params_pol = xr.concat((scale_res, c_params_pol), dim="coeff")

                ans_pol = (
                    params_pol.groupby("angle", squeeze=False)
                    .map(profile_per_angle, z=z_list, offset=offsets, side=-1)
                    .drop_vars("coeff")
                )

                ans_list.append(ans_pol.reduce(np.sum, ["angle"]).fillna(0))

            else:
                ans_list.append(0)

        profile_back = ans_list[0] + ans_list[1]


    else:

        profile_back = 0

    profile = profile_front + profile_back

    if np.sum(profile.data) > 0:
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

    else:
        return []

def single_ray_stack(
    nks,
    alphas,
    r_a,
    d,
    surfaces,
    tmm_kwargs_list,
    widths,
    z_pos,
    depths,
    depth_indices,
    I_thresh,
    pol,
    pol_vec,
    d_theta,
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

    ray = Ray(I_in, d, r_a, pol_vec[0], pol_vec[1], pol)

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

    profile = np.zeros(len(z_pos))
    # do everything in microns in here
    A_per_layer = np.zeros(len(widths))

    A_interface_array = 0
    A_interface_index = 0

    stop = False

    while not stop:

        surf = surfaces[surf_index]

        if randomize and (n_passes > 0):
            h = surf.z_max - surf.z_min + 0.1
            r_b = [
                np.random.rand() * surf.Lx,
                np.random.rand() * surf.Ly,
                surf.zcov,
            ]

            if ray.d[2] == 0:
                # ray travelling exactly  parallel to surface
                # print("parallel ray")
                ray.d[2] = -direction * 1e-3  # make it not parallel in the right direction

            n_z = np.ceil(abs(h / ray.d[2]))
            ray.r_a = r_b - n_z * ray.d

        if periodic:

            ray.r_a[0] = ray.r_a[0] - surf.Lx * (
                (ray.r_a[0] + ray.d[0] * (surf.zcov - ray.r_a[2]) / ray.d[2]) // surf.Lx
            )
            ray.r_a[1] = ray.r_a[1] - surf.Ly * (
                (ray.r_a[1] + ray.d[1] * (surf.zcov - ray.r_a[2]) / ray.d[2]) // surf.Ly
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

        res, theta, phi, th_local, n_interactions, _ = single_surface[periodic](
            ray,
            ni,
            nj,
            surf,
            surf.Lx,
            surf.Ly,
            direction,
            surf.zcov,
            d_theta,
            n_interactions,
            **tmm_kwargs_list[surf_index]
        )

        if res == 0:  # reflection
            direction = -direction  # changing direction due to reflection

            # staying in the same material, so mat_i does not change, but surf_index does
            surf_index = surf_index + direction

            if surf.phong:
                # do phong scattering
                theta, phi = ray_update_phong(ray, theta, phi, surf.phong_options)


        elif res == 1:  # transmission
            surf_index = surf_index + direction
            mat_i = mat_i + direction

            if surf.phong:
                # do phong scattering
                theta, phi = ray_update_phong(ray, theta, phi, surf.phong_options)

        elif res == 2:  # absorption
            stop = True  # absorption in an interface (NOT a bulk layer!)
            A_interface_array = (
                ray.I * theta[:] / np.sum(theta)
            )  # if absorbed, theta contains information about A_per_layer
            A_interface_index = surf_index + 1
            theta = None
            ray.I = 0

        if direction == 1 and mat_i == (len(widths) - 1):
            stop = True  # have ended with transmission

        elif direction == -1 and mat_i == 0:
            stop = True  # have ended with reflection

        # print("phi", np.real(atan2(d[1], d[0])))

        if not stop:
            I_b = ray.I

            ray.I, DA, stop, theta = traverse(
                ray.I,
                widths[mat_i],
                theta,
                alphas[mat_i],
                depths[mat_i],
                I_thresh,
                direction,
            )

            DA[np.isnan(DA)] = 0

            # traverse bulk layer. Possibility of absorption; in this case will return stop = True
            # and theta = None

            A_per_layer[mat_i] = np.real(A_per_layer[mat_i] + I_b - ray.I)
            profile[depth_indices[mat_i]] = np.real(
                profile[depth_indices[mat_i]] + DA
            )

            n_passes = n_passes + 1

            if lambertian_approximation and n_passes >= lambertian_approximation:
                # choose a direction randomly, with probability determined by Lambertian scattering
                stop = True
                theta = 10*direction*mat_i # +ve if travelling down, about to hit rear surface.
                # stop right before next interaction with surface, but AFTER taking into account bulk
                # absorption on this pass

    # print("Ray ending with pol:", norm(ray.s_vector)**2, norm(ray.p_vector)**2)
    return (
        ray,
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



@jit(nopython=True)
def traverse(ray_I, width, theta, alpha, positions, I_thresh, direction):
    stop = False
    ratio = alpha / np.real(np.abs(cos(theta)))
    DA_u = ray_I * ratio * np.exp((-ratio * positions))
    I_back =  ray_I * np.exp(-ratio * width)

    if I_back < I_thresh:
        stop = True
        theta = None

    if direction == -1:
        DA_u = np.flip(DA_u)

    intgr = np.trapz(DA_u, positions)

    DA = (ray_I - I_back) * DA_u / intgr

    return I_back, DA, stop, theta


def rotate_vector(rot_obj, delta_theta, delta_phi):
 # in coordinate system of d, generate a vector which has these angles:

    xy_mag = np.sin(delta_theta)
    p_d_coord = np.array((xy_mag * np.cos(delta_phi),
                          xy_mag * np.sin(delta_phi),
                          np.cos(delta_theta))).T

    rotate_to_xyz = rot_obj.apply(p_d_coord)

    return rotate_to_xyz

def ray_update_phong(ray, theta, phi, phong_options):

    delta_theta = np.arccos(np.random.rand() ** (1 / (phong_options[0] + 1)))
    # delta_theta = np.random.normal(0, phong_options[0])
    delta_phi = 2 * np.pi * np.random.rand()

    rot_to_d = Rotation.from_euler('yz', [theta,
                                          phi])

    new_dir = rotate_vector(rot_to_d, delta_theta, delta_phi)

    if np.sign(ray.d[2]) != np.sign(new_dir[2]):
        new_dir = rotate_vector(rot_to_d, delta_theta, delta_phi - np.pi)

    ray.d = new_dir

    theta = np.real(acos(ray.d[2]))
    phi = np.real(atan2(ray.d[1], ray.d[0]))

    if np.sign(ray.d[2]) != np.sign(new_dir[2]):
        raise ValueError("Direction not flipped correctly")

    if phong_options[1]:
        # randomise the polarization
        ray.pol = np.random.rand(2)
        ray.pol = ray.pol / np.sum(ray.pol)

    return theta, phi


def ray_update_phong_vec(ray_ds, pols, phong_options, n_rays):

    total_rays = np.sum(n_rays)

    delta_theta = np.arccos(np.random.rand(total_rays) ** (1 / (phong_options[0] + 1)))
    delta_phi = 2 * np.pi * np.random.rand(total_rays)

    total_ind = 0
    all_dirs = np.empty((total_rays, 3))

    for i1, ray_d in enumerate(ray_ds):
        theta = np.arccos(ray_d[2])
        phi = atan2(ray_d[1], ray_d[0])
        rot_to_d = Rotation.from_euler('yz', [theta,
                                              phi])

        #new_dir = rotate_vector(rot_to_d, delta_theta, -delta_phi)
        new_dirs = rotate_vector(rot_to_d, delta_theta[total_ind:total_ind + n_rays[i1]],
                                 delta_phi[total_ind:total_ind + n_rays[i1]])


        wrong_direction = np.sign(ray_d[2]) != np.sign(new_dirs[:,2])
        new_dirs[wrong_direction] = rotate_vector(rot_to_d, delta_theta[total_ind:total_ind + n_rays[i1]][wrong_direction],
                                                  delta_phi[total_ind:total_ind + n_rays[i1]][wrong_direction] - np.pi)
        all_dirs[total_ind:total_ind + n_rays[i1]] = new_dirs

        if np.any(np.sign(new_dirs[:,2]) != np.sign(ray_d[2])):
            raise ValueError("Direction not flipped correctly")
        total_ind += n_rays[i1]

    if phong_options[1]:
        # randomise the polarization
        all_pols = np.random.rand(total_rays, 2)
        all_pols = all_pols / np.sum(all_pols, 1)[:, None]

    else:
        all_pols = np.vstack([np.tile(pols[i], (n_rays[i], 1)) for i in range(len(pols))])

    return all_dirs, all_pols

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
            lookuptable = lookuptable.sel(pol=['s', 'p'])
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

def make_rt_args(existing_rays, xs, ys, n_reps, phong_params, phong_options):
    """Make arguments for single_ray_stack with existing rays from analytical calculation."""
    max_rays = len(xs) * len(ys) * n_reps  # shouldn't really need to calculate this here

    # TODO: move this filtering outside?
    Is = existing_rays.I.data
    dirs = existing_rays.direction.data[Is > 1e-9]
    pols = existing_rays.pol.data[Is > 1e-9]
    current_mat = existing_rays.mat_i.data[Is > 1e-9]
    n_interactions = existing_rays.n_interactions[Is > 1e-9]
    n_passes = existing_rays.n_passes[Is > 1e-9]
    Is = Is[Is > 1e-9]

    previous_surface = current_mat - (dirs[:,2] < 0) # for rays travelling up, the last surface
    # interacted with has surf_index = current_mat. For rays travelling down, they were transmitted
    # through that same surface, so the surf_index is current_mat -1. These should all be the same
    # surface, otherwise something has gone wrong.
    previous_surface = np.unique(previous_surface)[0]

    rays_per_direction = np.floor(Is*max_rays).astype(int)
    rays_per_direction[rays_per_direction == 0] = 1
    # TODO: keep track of which face the rays came from, then generate random position within that face?
    # print(dirs)
    scale_factor = Is/(rays_per_direction/max_rays)

    if phong_params[previous_surface]:

        ds, pols = ray_update_phong_vec(dirs, pols, phong_options[previous_surface], rays_per_direction)

    else:

        ds = np.vstack([np.tile(dirs[i], (rays_per_direction[i], 1)) for i in range(len(dirs))])
        pols = np.vstack([np.tile(pols[i], (rays_per_direction[i], 1)) for i in range(len(pols))])

    i_mats = np.concatenate([[current_mat[i]]*rays_per_direction[i] for i in range(len(current_mat))])
    i_dirs = np.ones_like(i_mats)
    i_dirs[ds[:,2] > 0] = -1

    surf_inds = np.zeros_like(i_mats)
    surf_inds[i_dirs == 1] = i_mats[i_dirs == 1]
    surf_inds[i_dirs == -1] = i_mats[i_dirs == -1] - 1

    n_inters = np.concatenate([[n_interactions[i]]*rays_per_direction[i] for i in range(len(current_mat))])
    n_passes = np.concatenate([[n_passes[i]]*rays_per_direction[i] for i in range(len(current_mat))])

    n_remaining = np.sum(rays_per_direction)

    scale_I = np.concatenate([[scale_factor[i]]*rays_per_direction[i] for i in range(len(current_mat))])

    return ds, pols, i_mats, i_dirs, surf_inds, n_remaining, scale_I, n_inters, n_passes
