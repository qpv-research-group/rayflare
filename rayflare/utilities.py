# Copyright (C) 2021-2024 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU Lesser General Public License (LGPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: p.pearce@unsw.edu.au

import numpy as np
from solcore.interpolate import interp1d
import os
from sparse import load_npz
import xarray as xr

from rayflare import logger

def make_absorption_function(result, structure, options):
    """
    :param result: the result of an optical calculation; this can be from using the angular redistribution matrix method,
         or from using rt_structure().calculate, tmm_structure.calculate() or rcwa_structure.calculate(). This should be
         the full result of the calculation, not just the absorption profile.
    :param structure: the structure object used to calculate the result; this can be a Structure object, a tmm_structure
            object, a rcwa_structure object or a rt_structure object.
    :param options: the options used to calculate the optical result.

    :return: all_positions, diff_absorb_fn. a list of positions along the structure (in units of m),
            and a function that can be used to
            calculate the absorption at any position along the structure (in units of m-1). This includes all the layers
            which were present
            in structure, not just those where absorption profiles were requested (if relevant).
    """

    # do these imports in here to prevent 'circular import' errors, because these files import from this file
    from rayflare.transfer_matrix_method import tmm_structure
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
    from rayflare.ray_tracing import rt_structure
    from rayflare.structure import Structure

    # determine the type of structure to figure out how to process the result:

    if isinstance(structure, Structure):
        matrix_method = True

    elif isinstance(structure, tmm_structure):
        matrix_method = False
        interfaces_and_bulks = False
        depth_spacing = options.depth_spacing

    elif isinstance(structure, rcwa_structure):
        matrix_method = False
        interfaces_and_bulks = False
        depth_spacing = options.depth_spacing

    elif isinstance(structure, rt_structure):
        matrix_method = False

        if sum(structure.tmm_or_fresnel) > 0:
            interfaces_and_bulks = True

        else:
            interfaces_and_bulks = False
            depth_spacing = options.depth_spacing_bulk

    else:
        raise ValueError(
            "Structure type not recognised; should be a Structure, tmm_structure, rcwa_structure or rt_structure object."
        )

    if matrix_method:

        profile_front = result[2][0]
        profile_bulk = result[3][0]
        profile_back = result[2][1]

        widths_front = np.array(structure[0].widths) * 1e9  # in nm
        width_bulk = np.array(structure[1].width) * 1e6  # in um
        widths_back = np.array(structure[2].widths) * 1e9  # in nm

        d_ints = options.depth_spacing * 1e9  # m to nm
        d_bulk = options.depth_spacing_bulk * 1e6  # m to um

        front_positions = np.arange(0, np.sum(widths_front) - 1e-12, d_ints)  # in nm
        bulk_positions = np.arange(0, width_bulk - 1e-12, d_bulk)  # in um
        back_positions = np.arange(0, np.sum(widths_back) - 1e-12, d_ints)  # in nm

        layer_start = np.insert(np.cumsum(np.insert(widths_front, 0, 0)), 0, 0)
        layer_end = np.cumsum(np.insert(widths_front, 0, 0))

        prof_front = np.zeros((len(options.wavelength), len(front_positions)))

        if structure[0].prof_layers is not None:

            index_offset = 0

            for i1 in structure[0].prof_layers:
                indices = np.logical_and(
                    front_positions >= layer_start[i1], front_positions < layer_end[i1]
                )
                ind_diff = np.sum(indices)
                prof_front[:, indices] = (
                    1e9
                    * profile_front[
                        :, index_offset : (index_offset + ind_diff)
                    ]  # convert from nm-1 to m-1
                )
                index_offset += ind_diff

        layer_start = np.insert(np.cumsum(np.insert(widths_back, 0, 0)), 0, 0)
        layer_end = np.cumsum(np.insert(widths_back, 0, 0))

        prof_back = np.zeros((len(options.wavelength), len(back_positions)))

        if structure[2].prof_layers is not None:

            index_offset = 0

            for i1 in structure[2].prof_layers:
                indices = np.logical_and(
                    back_positions >= layer_start[i1], back_positions < layer_end[i1]
                )
                ind_diff = np.sum(indices)
                prof_back[:, indices] = (
                    1e9
                    * profile_back[
                        :, index_offset : (index_offset + ind_diff)
                    ]  # convert from nm-1 to m-1
                )
                index_offset += ind_diff

        if options.bulk_profile:

            prof_bulk = profile_bulk[:, : len(bulk_positions)]  # already in m-1

        else:
            prof_bulk = np.zeros_like(bulk_positions)

        all_positions = np.hstack(
            (
                front_positions / 1e9,
                bulk_positions / 1e6 + np.sum(widths_front) / 1e9,
                back_positions / 1e9 + np.sum(widths_front) / 1e9 + width_bulk / 1e6,
            )
        )  # all units get converted back to m

        all_absorption = np.hstack((prof_front, prof_bulk, prof_back))

        diff_absorb_fn = interp1d(all_positions, all_absorption)

        return all_positions, diff_absorb_fn

    else:

        if interfaces_and_bulks:
            # rt_structure with tmm (and possibly also Fresnel!) interfaces

            all_data = []
            all_positions = []

            cumulative_width = 0  # cumulative width in nm
            index_offset_bulk = 0

            layer_start_bulk = np.insert(np.cumsum(structure.widths, 0), 0, 0)
            layer_end_bulk = np.cumsum(structure.widths, 0)

            bulk_positions = (
                np.arange(
                    0, np.sum(structure.widths) * 1e6, options.depth_spacing_bulk * 1e6
                )
                / 1e6
            )  # widths and spacing are in m

            for i1, surf in enumerate(structure.surfaces):

                index_offset_surf = 0  # data for each interface is saved separately in a list in result["interface_profiles"]

                if structure.tmm_or_fresnel[i1] == 1:  # tmm interface

                    surf_positions = np.arange(
                        0,
                        np.sum(structure.interface_layer_widths[i1]),
                        options.depth_spacing * 1e9,
                    )
                    surf_prof = np.zeros(
                        (len(options.wavelength), len(surf_positions))
                    )

                    if hasattr(surf, "prof_layers"):

                        data = result["interface_profiles"][i1]

                        layer_start = np.insert(
                            np.cumsum(
                                np.insert(structure.interface_layer_widths[i1], 0, 0)
                            ),
                            0,
                            0,
                        )
                        # insert 0 because prof_layers in 1-indexed
                        layer_end = np.cumsum(
                            np.insert(structure.interface_layer_widths[i1], 0, 0)
                        )

                        for j1 in surf.prof_layers:
                            indices = np.logical_and(
                                surf_positions >= layer_start[j1],
                                surf_positions < layer_end[j1],
                            )
                            ind_diff = np.sum(indices)

                            surf_prof[:, indices] = (
                                1e9
                                * data[
                                    :,
                                    index_offset_surf : (index_offset_surf + ind_diff),
                                ]  # convert from nm-1 to m-1
                            )
                            index_offset_surf += ind_diff

                    all_data.append(surf_prof)
                    all_positions.append(
                        surf_positions / 1e9 + cumulative_width
                    )  # convert to m

                    cumulative_width += (
                        np.sum(structure.interface_layer_widths[i1]) * 1e-9
                    )  # cumulative in m

                # bulk underneath the interface:
                if (
                    i1 < len(structure.surfaces) - 1
                ):  # no bulk underneath final interface!

                    data = result["profile"]

                    indices = np.logical_and(
                        bulk_positions >= layer_start_bulk[i1],
                        bulk_positions < layer_end_bulk[i1],
                    )
                    # print(layer_start_bulk[i1], layer_end_bulk[i1])
                    # print(indices)
                    ind_diff = np.sum(indices)
                    bulk_prof = (
                        1e9
                        * data[:, index_offset_bulk : (index_offset_bulk + ind_diff)]
                    )  # convert from nm-1 to m-1
                    pos_from_zero = bulk_positions[indices] - np.min(
                        bulk_positions[indices]
                    )
                    # print(bulk_prof)
                    index_offset_bulk += ind_diff

                    all_data.append(bulk_prof)
                    all_positions.append(pos_from_zero + cumulative_width)

                    cumulative_width += structure.widths[i1]

            positions = np.hstack(all_positions)
            diff_absorb_fn = interp1d(positions, np.hstack(all_data))

        else:

            positions = np.arange(0, structure.width*1e6, depth_spacing*1e6)/1e6
            diff_absorb_fn = interp1d(positions, 1e9 * result["profile"])

        return positions, diff_absorb_fn


def get_matrices_or_paths(
    structpath, surf_name, front_or_rear, prof_layers=None, overwrite=False
):

    savepath_RT = os.path.join(structpath, surf_name + front_or_rear + "RT.npz")
    savepath_A = os.path.join(structpath, surf_name + front_or_rear + "A.npz")

    if prof_layers is not None:
        prof_mat_path = os.path.join(
            structpath, surf_name + front_or_rear + "profmat.nc"
        )

    if os.path.isfile(savepath_RT) and os.path.isfile(savepath_A) and not overwrite:
        logger.info("Existing angular redistribution matrices found")
        existing = True
        full_mat = load_npz(savepath_RT)
        A_mat = load_npz(savepath_A)

        if prof_layers is not None:
            if os.path.isfile(prof_mat_path):
                prof_dataset = xr.load_dataset(prof_mat_path)
                return [existing, [full_mat, A_mat, prof_dataset]]
            else:
                logger.info("Recalculating with absorption profile information")
                existing = False
                return [existing, [savepath_RT, savepath_A, prof_mat_path]]

        else:
            return [existing, [full_mat, A_mat]]

    else:
        if prof_layers is not None:
            return [False, [savepath_RT, savepath_A, prof_mat_path]]

        else:
            return [False, [savepath_RT, savepath_A]]


def get_savepath(save_location, project_name):
    """
    Returns the full path where matrices will be stored.

    :param save_location: string - location where the calculated redistribution matrices should be stored. Currently recognized are:

        - 'default', which stores the results in folder in your home directory called 'RayFlare_results'
        - 'current', which stores the results in the current working directory
        - or you can specify the full path location for wherever you want the results to be stored.

        In each case, the results will be stored in a subfolder with the name of the project (options.project_name)
    :param project_name: the project name (string)

    :return: full file path where matrices are stored (string)
    """
    if save_location == "current":
        cwd = os.getcwd()
        structpath = os.path.join(cwd, project_name)

    elif save_location == "default":
        home = os.path.expanduser("~")
        structpath = os.path.join(home, "RayFlare_results", project_name)
        if not os.path.isdir(os.path.join(home, "RayFlare_results")):
            os.mkdir(os.path.join(home, "RayFlare_results"))

    else:
        structpath = os.path.join(save_location, project_name)
        if not os.path.isdir(save_location):
            os.mkdir(save_location)

    if not os.path.isdir(structpath):
        os.mkdir(structpath)

    return structpath


def process_pol(input_pol):
    if len(input_pol) == 2:
        pol = input_pol

    else:
        if input_pol in "sp":
            pol = (int(input_pol == "s"), int(input_pol == "p"))

        elif input_pol == "u":
            pol = (np.sqrt(2) / 2, np.sqrt(2) / 2)

        else:
            raise ValueError(
                "Polarization must be 's', 'p', 'u', or a tuple with the s and p components."
            )

    return pol

