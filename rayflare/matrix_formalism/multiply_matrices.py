# Copyright (C) 2021-2024 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU Lesser General Public License (LGPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: p.pearce@unsw.edu.au

import numpy as np
from sparse import load_npz, COO, stack, einsum
from rayflare.angles import make_angle_vector, fold_phi, overall_bin
import os
import xarray as xr
from solcore.state import State

from rayflare.structure import Interface, BulkLayer
from rayflare.utilities import get_savepath

from rayflare import logger

def calculate_RAT(SC, options, save_location="default"):
    """
    After the list of Interface and BulkLayers has been processed by process_structure,
    this function calculates the R, A and T by calling matrix_multiplication.

    :param SC: list of Interface and BulkLayer objects. Order is [Interface, BulkLayer, Interface]
    :param options: options for the matrix calculations (State object or dictionary)
    :param save_location: location from which to load the redistribution matrices. Current options:

              - 'default', which stores the results in folder in your home directory called 'RayFlare_results'
              - 'current', which stores the results in the current working directory
              - or you can specify the full path location for wherever you want the results to be stored.

            This should match what was specified for process_structure.

    :return: The number of returned values depends on whether absorption profiles were calculated or not. The first two
            are always returned, the final two are only returned if a calculation of absorption profiles was done.

            - RAT - an xarray with coordinates bulk_index and wl (wavelength), and 3 data variables: R (reflection),
              T (transmission) and A_bulk (absorption in the bulk medium. Currently, the bulk index can only be 0.
            - results_per_pass - a dictionary with entries 'r' (reflection), 't' (transmission), 'a' (absorption in the
              surfaces) and 'A' (bulk absorption), which store these quantities per each pass of the bulk or interaction
              with the relevant surface during matrix multiplication. 'r', 't' and 'A' are lists of length 1, corresponding
              to one set of values for each bulk material; the list entry is an array which is indexed as
              (pass number, wavelength). 'a' is a list of length two, corresponding to absorption in the front and back
              interface respectively. Each entry in the list is an array indexed as (pass number, wavelength, layer index).
            - profile - a list of xarrays, one for each surface. These store the absorption profiles and have coordinates
              wavelength and z (depth) position.
            - bulk_profile - a list of arrays, one for each bulk (currently, always only one). Indices are
              (wavelength, position)

    """

    if isinstance(options, dict):
        options = State(options)

    bulk_mats = []
    bulk_widths = []
    layer_widths = []
    layer_names = []
    calc_prof_list = []

    for struct in SC:
        if isinstance(struct, BulkLayer):
            bulk_mats.append(struct.material)
            bulk_widths.append(struct.width)

        if isinstance(struct, Interface):
            layer_names.append(struct.name)
            layer_widths.append((np.array(struct.widths) * 1e9).tolist())
            calc_prof_list.append(struct.prof_layers)

    results = matrix_multiplication(
        bulk_mats, bulk_widths, options, layer_names, calc_prof_list, save_location
    )

    return results


def make_v0(
    th_in, phi_in, num_wl, n_theta_bins, c_azimuth, phi_sym, theta_spacing="sin"
):
    """
    This function makes the v0 array, corresponding to the input power per angular channel
    at each wavelength, of size (num_wl, n_angle_bins_in) where n_angle_bins in = len(angle_vector)/2

    :param th_in: Polar angle of the incoming light (in radians)
    :param phi_in: Azimuthal angle of the incoming light (in radians), or can be set as 'all' \
    in which case the power is spread equally over all the phi bins for the relevant theta.
    :param num_wl: Number of wavelengths
    :param n_theta_bins: Number of theta bins in the matrix multiplication
    :param c_azimuth: c_azimuth used to generate the matrices being multiplied
    :param phi_sym: Defines symmetry element [0, phi_sym] into which values of phi can be collapsed (in radians)

    :return: v0, an array of size (num_wl, n_angle_bins_in)
    """

    theta_intv, phi_intv, angle_vector = make_angle_vector(
        n_theta_bins, phi_sym, c_azimuth, theta_spacing
    )
    n_a_in = int(len(angle_vector) / 2)
    v0 = np.zeros((num_wl, n_a_in))
    th_bin = np.digitize(th_in, theta_intv) - 1
    phi_intv = phi_intv[th_bin]
    ov_bin = np.argmin(abs(angle_vector[:, 0] - th_bin))
    if phi_in == "all":
        n_phis = len(phi_intv) - 1
        v0[:, ov_bin : (ov_bin + n_phis)] = 1 / n_phis
    else:
        phi_ind = np.digitize(phi_in, phi_intv) - 1
        ov_bin = ov_bin + phi_ind
        v0[:, ov_bin] = 1
    return v0


def out_to_in_matrix(phi_sym, angle_vector, theta_intv, phi_intv):

    if phi_sym == 2 * np.pi:
        phi_sym = phi_sym - 0.0001
    out_to_in = np.zeros((len(angle_vector), len(angle_vector)))
    binned_theta_out = (
        np.digitize(np.pi - angle_vector[:, 1], theta_intv, right=True) - 1
    )

    phi_rebin = fold_phi(angle_vector[:, 2] + np.pi, phi_sym)

    phi_out = xr.DataArray(
        phi_rebin,
        coords={"theta_bin": (["angle_in"], binned_theta_out)},
        dims=["angle_in"],
    )

    bin_out = (
        phi_out.groupby("theta_bin")
        .map(overall_bin, args=(phi_intv, angle_vector[:, 0]))
        .data
    )

    out_to_in[bin_out, np.arange(len(angle_vector))] = 1

    up_to_down = out_to_in[int(len(angle_vector) / 2) :, : int(len(angle_vector) / 2)]
    down_to_up = out_to_in[: int(len(angle_vector) / 2), int(len(angle_vector) / 2) :]

    return COO.from_numpy(up_to_down), COO.from_numpy(down_to_up)


def make_D(alphas, thick, thetas):
    """
    Makes the bulk absorption vector for the bulk material.

    :param alphas: absorption coefficient (m^{-1})
    :param thick: thickness of the slab in m
    :param thetas: incident thetas in angle_vector (second column)

    :return:
    """
    diag = np.exp(-alphas[:, None] * thick / abs(np.cos(thetas[None, :])))
    D_1 = stack([COO.from_numpy(np.diag(x)) for x in diag])
    return D_1

# dot_wl and dot_wl_u2d now use einsum method from sparse, which is significantly
# faster than the previous for loop implementation. Thanks to Johnson Wong
# (GitHub: arsonwong)
def dot_wl(mat, vec):
    # note: sometimes get nans here in result, even when there are no nans
    # in mat or vec.
    if len(mat.shape) == 3:
        result = einsum('ijk,ik->ij', mat, COO(vec)).todense()

    if len(mat.shape) == 2:
        result = einsum('jk,ik->ij', mat, COO(vec)).todense()

    return result


def dot_wl_u2d(mat, vec):

    result = einsum('jk,ik->ij', mat, COO(vec)).todense()
    return result


def bulk_profile_calc(v_1, v_2, alphas, thetas, d, depths, A):

    per_bin = v_1 - v_2  # total absorption per bin
    abscos = np.abs(np.cos(thetas))
    denom = 1 - np.exp(-alphas[:, None] * d / abscos[None, :])
    norm = np.divide(per_bin, denom, where=denom != 0)

    result = np.empty((v_1.shape[0], len(depths)))

    for i1 in range(v_1.shape[0]):
        a_x = ((alphas[i1] * norm[i1]) / (abscos))[None, :] * np.exp(
            -alphas[i1] * depths[:, None] / abscos[None, :]
        )
        result[i1, :] = np.sum(a_x, 1)

    check = np.trapz(result, depths, axis=1)
    # the bulk layer is often thick so you don't want the depth spacing too fine,
    # but at short wavelengths this causes an issue where the integrated depth profile
    # is not equal to the total absorption. Scale the profile to fix this and make things
    # consistent.
    scale = np.divide(A, check, where=check != 0)

    corrected = scale[:, None] * result

    return corrected


def load_redistribution_matrices(
    results_path, n_a_in, n_interfaces, layer_names, front_or_rear, calc_prof_list=None
):

    R = []
    T = []
    A = []
    P = []
    I = []

    if front_or_rear == "front":
        n_max = n_interfaces

    else:
        n_max = n_interfaces - 1

    for i1 in range(n_max):

        mat_path = os.path.join(
            results_path, layer_names[i1] + front_or_rear + "RT.npz"
        )
        absmat_path = os.path.join(
            results_path, layer_names[i1] + front_or_rear + "A.npz"
        )

        fullmat = load_npz(mat_path)
        absmat = load_npz(absmat_path)

        if front_or_rear == "front":  # matrices for front incidence

            if len(fullmat.shape) == 3:
                R.append(fullmat[:, :n_a_in, :])
                T.append(fullmat[:, n_a_in:, :])
                A.append(absmat)

            else:

                R.append(fullmat[:n_a_in, :])
                T.append(fullmat[n_a_in:, :])
                A.append(absmat)

        else:  # matrices for rear incidence
            if len(fullmat.shape) == 3:
                R.append(fullmat[:, n_a_in:, :])
                T.append(fullmat[:, :n_a_in, :])
                A.append(absmat)

            else:
                R.append(fullmat[n_a_in:, :])
                T.append(fullmat[:n_a_in, :])
                A.append(absmat)

        if calc_prof_list[i1] is not None:
            profmat_path = os.path.join(
                results_path, layer_names[i1] + front_or_rear + "profmat.nc"
            )
            prof_int = xr.load_dataset(profmat_path)
            profile = prof_int["profile"]
            intgr = prof_int["intgr"]
            P.append(profile)
            I.append(intgr)

        else:
            P.append([])
            I.append([])

    return R, T, A, P, I


def append_per_pass_info(i1, vr, vt, a, vf_2, vb_1, Tb, Tf, Af, Ab):

    vr[i1].append(
        dot_wl(Tb[i1], vf_2[i1])
    )  # matrix travelling up in medium 0, i.e. reflected overall by being transmitted through front surface
    vt[i1].append(
        dot_wl(Tf[i1 + 1], vb_1[i1])
    )  # transmitted into medium below through back surface

    a[i1 + 1].append(dot_wl(Af[i1 + 1], vb_1[i1]))  # absorbed in 2nd surface
    a[i1].append(dot_wl(Ab[i1], vf_2[i1]))  # absorbed in 1st surface (from the back)

    return vr, vt, a


def matrix_multiplication(
    bulk_mats, bulk_thick, options, layer_names, calc_prof_list, save_location
):
    """

    :param bulk_mats: list of bulk materials
    :param bulk_thick: list of bulk thicknesses (in m)
    :param options: user options (State object)
    :param layer_names: list of names of the Interface layers, to load the redistribution matrices
    :param calc_prof_list: list of lists - for each interface, which layers should be included in profile calculations
           (can be empty)
    :param save_location: string, location of saved matrices
    :return:
    :rtype:
    """

    results_path = get_savepath(save_location, options.project_name)

    n_bulks = len(bulk_mats)
    n_interfaces = n_bulks + 1

    theta_spacing = (
        options.theta_spacing if "theta_spacing" in options else "sin"
    )

    theta_intv, phi_intv, angle_vector = make_angle_vector(
        options.n_theta_bins,
        options.phi_symmetry,
        options.c_azimuth,
        theta_spacing,
    )
    n_a_in = int(len(angle_vector) / 2)

    num_wl = len(options["wavelength"])

    # bulk thickness in m

    thetas = angle_vector[:n_a_in, 1]

    if options.phi_in != "all" and options.phi_in > options.phi_symmetry:
        # fold phi_in back into phi_symmetry
        phi_in = fold_phi(options["phi_in"], options["phi_symmetry"])

    else:
        phi_in = options["phi_in"]

    v0 = make_v0(
        options["theta_in"],
        phi_in,
        num_wl,
        options["n_theta_bins"],
        options["c_azimuth"],
        options["phi_symmetry"],
        theta_spacing,
    )

    up2down, down2up = out_to_in_matrix(
        options["phi_symmetry"], angle_vector, theta_intv, phi_intv
    )

    D = []
    depths_bulk = []
    for i1 in range(n_bulks):
        D.append(
            make_D(bulk_mats[i1].alpha(options["wavelength"]), bulk_thick[i1], thetas)
        )

        if options["bulk_profile"]:
            depths_bulk.append(
                np.arange(0, bulk_thick[i1], options["depth_spacing_bulk"])
            )

    # load redistribution matrices

    Rf, Tf, Af, Pf, If = load_redistribution_matrices(
        results_path, n_a_in, n_interfaces, layer_names, "front", calc_prof_list
    )

    Rb, Tb, Ab, Pb, Ib = load_redistribution_matrices(
        results_path, n_a_in, n_interfaces, layer_names, "rear", calc_prof_list
    )

    len_calcs = np.array([len(x) if x is not None else 0 for x in calc_prof_list])

    a = [[] for _ in range(n_interfaces)]
    vr = [[] for _ in range(n_bulks)]
    vt = [[] for _ in range(n_bulks)]
    A = [[] for _ in range(n_bulks)]

    vf_1 = [[] for _ in range(n_interfaces)]
    vb_1 = [[] for _ in range(n_interfaces)]
    vf_2 = [[] for _ in range(n_interfaces)]
    vb_2 = [[] for _ in range(n_interfaces)]

    if np.any(len_calcs > 0) or options.bulk_profile:
        # need to calculate profiles in either the bulk or the interfaces

        a_prof = [[] for _ in range(n_interfaces)]
        A_prof = [[] for _ in range(n_bulks)]
        logger.debug(f"Initial intensity: {np.sum(v0, axis=1)}")

        for i1 in range(n_bulks):

            # v0 is actually travelling down, but no reason to start in 'outgoing' ray format.
            vf_1[i1] = dot_wl(Tf[i1], v0)  # pass through front surface
            # print(vf_1[i1])
            # vf_1: incoming to outgoing
            # print("Transmitted through front", np.sum(vf_1[i1], axis=1))

            vr[i1].append(dot_wl(Rf[i1], v0))  # reflected from front surface
            # print("Reflected from front", np.sum(vr[i1][-1], axis=1))
            a[i1].append(
                dot_wl(Af[i1], v0)
            )  # absorbed in front surface at first interaction
            # print("Absorbed in front", np.sum(a[i1][-1], axis=1))

            if len(If[i1]) > 0:
                scale = ((np.sum(Af[i1].todense(), 1) * v0) / If[i1]).fillna(0)
                # print(((np.sum(Af[i1].todense(), 1) * v0) / If[i1]))
                # print("Af", Af[i1].todense())
                # print("v0", v0)
                # print("last few points of Pf:", Pf[i1][:, -5:, :])
                scaled_prof = scale * Pf[i1]
                a_prof[i1].append(np.sum(scaled_prof, 1))
                # print("SHAPE:", a_prof[i1][-1].shape)
                # print("Integrated absorbed:", np.trapz(a_prof[i1][-1], dx=options.depth_spacing*1e9, axis=1))

            power = np.sum(vf_1[i1], axis=1)
            # print("Power remaining", power)

            # rep
            i2 = 1

            while np.any(power > options["I_thresh"]):
                vf_1[i1] = dot_wl_u2d(down2up, vf_1[i1])  # outgoing to incoming
                # print("Travelling down int, before", np.sum(vf_1[i1], axis=1))
                # vb_1: incoming (just absorption through bulk)
                vb_1[i1] = dot_wl(D[i1], vf_1[i1])  # pass through bulk, downwards
                # vb_1 already an incoming ray
                # print("Travelling down int, after ", np.sum(vb_1[i1], axis=1))

                if len(If[i1 + 1]) > 0:

                    scale = (
                        (np.sum(Af[i1 + 1].todense(), 1) * vb_1[i1]) / If[i1 + 1]
                    ).fillna(0)
                    scaled_prof = scale * Pf[i1 + 1]
                    # print("Pf, back surface", Pf[i1+1])
                    a_prof[i1 + 1].append(np.sum(scaled_prof, 1))
                    # print("Integrated absorbed (back):", np.trapz(a_prof[i1 + 1][-1], dx=options.depth_spacing * 1e9, axis=1))

                A[i1].append(np.sum(vf_1[i1], 1) - np.sum(vb_1[i1], 1))
                # print("Total absorbed, back", A[i1][-1])

                if options.bulk_profile:
                    A_prof[i1].append(
                        bulk_profile_calc(
                            vf_1[i1],
                            vb_1[i1],
                            bulk_mats[i1].alpha(options["wavelength"]),
                            thetas,
                            bulk_thick[i1],
                            depths_bulk[i1],
                            A[i1][-1],
                        )
                    )
                    # print("bulk profile (down) integrated", np.trapz(A_prof[i1][-1], dx=options.depth_spacing_bulk, axis=1))

                vb_2[i1] = dot_wl(
                    Rf[i1 + 1], vb_1[i1]
                )  # reflect from back surface. incoming -> up
                # print("Reflected from back", np.sum(vb_2[i1], axis=1))
                # vb_2: outgoing
                vf_2[i1] = dot_wl(D[i1], vb_2[i1])  # pass through bulk, upwards
                # print("Travelling up int, after", np.sum(vf_2[i1], axis=1))

                A[i1].append(
                    np.sum(vb_2[i1], 1) - np.sum(vf_2[i1], 1)
                )  # binning doesn't matter here because summing
                # print("Bulk A (total):", A[i1][-1])

                if options.bulk_profile:
                    # vb_2 needs to be transformed to incoming ray format
                    A_prof[i1].append(
                        np.flip(
                            bulk_profile_calc(
                                vb_2[i1],
                                vf_2[i1],
                                bulk_mats[i1].alpha(options["wavelength"]),
                                thetas,
                                bulk_thick[i1],
                                depths_bulk[i1],
                                A[i1][-1],
                            ),
                            1,
                        )
                    )
                    # print("bulk profile (up) integrated", np.trapz(A_prof[i1][-1], dx=options.depth_spacing_bulk, axis=1))

                vf_2[i1] = dot_wl_u2d(up2down, vf_2[i1])  # prepare for rear incidence
                # vf_2: incoming
                # print("Travelling up, before R", np.sum(vf_2[i1], axis=1))

                if len(Ib[i1]) > 0:
                    scale = ((np.sum(Ab[i1].todense(), 1) * vf_2[i1]) / Ib[i1]).fillna(
                        0
                    )
                    scaled_prof = scale * Pb[i1]
                    # print("Pb", Pb[i1])
                    a_prof[i1].append(np.sum(scaled_prof, 1))
                    # print("SHAPE:", a_prof[i1][-1].shape)
                    # print("Integrated absorbed (front surface, rear inc):",
                    #       np.trapz(a_prof[i1][-1], dx=options.depth_spacing * 1e9, axis=1))

                vf_1[i1] = dot_wl(Rb[i1], vf_2[i1])  # reflect from front surface
                # print("Reflected from front", np.sum(vf_1[i1], axis=1))
                # vf_1 will be outgoing, gets fixed at start of next loop
                power = np.sum(vf_1[i1], axis=1)

                vr, vt, a = append_per_pass_info(
                    i1, vr, vt, a, vf_2, vb_1, Tb, Tf, Af, Ab
                )

                # print("Absorbed from back", np.sum(a[i1][-1], axis=1))

                # rewrite as f string:

                logger.info(f"After iteration {i2}: maximum power fraction remaining = {np.max(power)}")

                i2 += 1

    else:  # no profile calculation in bulk or interfaces

        for i1 in range(n_bulks):

            vf_1[i1] = dot_wl(Tf[i1], v0)  # pass through front surface
            vr[i1].append(dot_wl(Rf[i1], v0))  # reflected from front surface
            a[i1].append(
                dot_wl(Af[i1], v0)
            )  # absorbed in front surface at first interaction
            power = np.sum(vf_1[i1], axis=1)

            # vf_1[i1] = dot_wl_u2d(up2down, vf_1[i1])

            # rep
            i2 = 1

            while np.any(power > options["I_thresh"]):
                vf_1[i1] = dot_wl_u2d(down2up, vf_1[i1])  # outgoing to incoming
                vb_1[i1] = dot_wl(D[i1], vf_1[i1])  # pass through bulk, downwards
                A[i1].append(np.sum(vf_1[i1], 1) - np.sum(vb_1[i1], 1))

                vb_2[i1] = dot_wl(Rf[i1 + 1], vb_1[i1])  # reflect from back surface
                vf_2[i1] = dot_wl(D[i1], vb_2[i1])  # pass through bulk, upwards
                vf_2[i1] = dot_wl_u2d(up2down, vf_2[i1])  # prepare for rear incidence
                vf_1[i1] = dot_wl(Rb[i1], vf_2[i1])  # reflect from front surface
                A[i1].append(np.sum(vb_2[i1], 1) - np.sum(vf_2[i1], 1))
                power = np.sum(vf_1[i1], axis=1)
                logger.info(f"After iteration {i2}: maximum power fraction remaining = {np.max(power)}")

                vr, vt, a = append_per_pass_info(
                    i1, vr, vt, a, vf_2, vb_1, Tb, Tf, Af, Ab
                )

                i2 += 1

    vr = [np.array(item) for item in vr]
    vt = [np.array(item) for item in vt]
    a = [np.array(item) for item in a]
    A = [np.array(item) for item in A]

    sum_dims = ["bulk_index", "wl"]
    sum_coords = {"bulk_index": np.arange(0, n_bulks), "wl": options["wavelength"]}

    R = xr.DataArray(
        np.array([np.sum(item, (0, 2)) for item in vr]),
        dims=sum_dims,
        coords=sum_coords,
        name="R",
    )

    if i2 > 1:

        A_bulk = xr.DataArray(
            np.array([np.sum(item, 0) for item in A]),
            dims=sum_dims,
            coords=sum_coords,
            name="A_bulk",
        )

        T = xr.DataArray(
            np.array([np.sum(item, (0, 2)) for item in vt]),
            dims=sum_dims,
            coords=sum_coords,
            name="T",
        )

        if np.any(len_calcs > 0) or options.bulk_profile:

            A_prof = [np.array(item) for item in A_prof]

            a_prof = [np.array(item) for item in a_prof]

            results_per_pass = {
                "r": vr,
                "t": vt,
                "a": a,
                "A": A,
                "a_prof": a_prof,
                "A_prof": A_prof,
            }

            A_interface = xr.DataArray(
                np.array([np.sum(item, (0, 2)) for item in a]),
                dims=["surf_index", "wl"],
                coords={
                    "surf_index": np.arange(0, n_interfaces),
                    "wl": options["wavelength"],
                },
                name="A_interface",
            )
            profile = []
            for j1, item in enumerate(a_prof):
                if len(item) > 0:
                    item[item < 0] = 0
                    profile.append(
                        xr.DataArray(
                            np.sum(item, 0),
                            dims=["wl", "z"],
                            coords={"wl": options["wavelength"]},
                            name="A_profile" + str(j1),
                        )
                    )  # not necessarily same number of z coords per layer stack

            bulk_profile = [np.sum(prof_el, 0) for prof_el in A_prof]
            RAT = xr.merge([R, A_bulk, A_interface, T])

            return RAT, results_per_pass, profile, bulk_profile

        else:

            results_per_pass = {"r": vr, "t": vt, "a": a, "A": A}

            RAT = xr.merge([R, A_bulk, T])

            return RAT, results_per_pass

    else:
        RAT = xr.merge([R])
        results_per_pass = {"r": vr, "t": vt, "a": a, "A": A}

        return RAT, results_per_pass
