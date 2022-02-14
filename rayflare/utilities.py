# Copyright (C) 2021 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU General Public License (GPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: p.pearce@unsw.edu.au

import numpy as np
import os
from sparse import load_npz
import xarray as xr
from solcore.interpolate import interp1d

def get_matrices_or_paths(structpath, surf_name, front_or_rear, prof_layers=None):

    savepath_RT = os.path.join(structpath, surf_name + front_or_rear + 'RT.npz')
    savepath_A = os.path.join(structpath, surf_name + front_or_rear + 'A.npz')

    if prof_layers is not None:
        prof_mat_path = os.path.join(structpath, surf_name + front_or_rear + 'profmat.nc')

    if os.path.isfile(savepath_RT) and os.path.isfile(savepath_A):
        print('Existing angular redistribution matrices found')
        existing = True
        full_mat = load_npz(savepath_RT)
        A_mat = load_npz(savepath_A)

        if prof_layers is not None:
            if os.path.isfile(prof_mat_path):
                prof_dataset = xr.load_dataset(prof_mat_path)
                return [existing, [full_mat, A_mat, prof_dataset]]
            else:
                print('Recalculating with absorption profile information')
                existing = False
                return [existing, [savepath_RT, savepath_A, prof_mat_path]]

        else:
            return [existing, [full_mat, A_mat]]

    else:
        if prof_layers is not None:
            return [False, [savepath_RT, savepath_A, prof_mat_path]]

        else:
            return [False, [savepath_RT, savepath_A]]


def make_absorption_function(profile_result, structure, options, matrix_method=False):
    """
    :param profile:
    :param options:

    :return diff_absorb_fn:
    """

    if matrix_method:

        widths_front = np.array(structure[0].widths)*1e9 # in nm
        width_bulk = np.array(structure[1].width)*1e6 # in um
        widths_back = np.array(structure[2].widths)*1e9 # in nm

        d_ints = options.depth_spacing*1e9 # in nm
        d_bulk = options.depth_spacing_bulk*1e6 # in um

        front_positions = np.arange(0, np.sum(widths_front)-1e-12, d_ints) # in nm
        bulk_positions = np.arange(0, width_bulk-1e-12, d_bulk) # in um
        back_positions = np.arange(0, np.sum(widths_back)-1e-12, d_ints) # in nm

        layer_start = np.insert(np.cumsum(np.insert(widths_front, 0, 0)), 0, 0)
        layer_end = np.cumsum(np.insert(widths_front, 0, 0))

        prof_front = np.zeros((len(options.wavelengths), len(front_positions)))

        if structure[0].prof_layers is not None:

            index_offset = 0

            for i1 in structure[0].prof_layers:
                indices = np.logical_and(front_positions >= layer_start[i1], front_positions < layer_end[i1])
                ind_diff = np.sum(indices)
                prof_front[:, indices] = 1e9*profile_result[0][:, index_offset:(index_offset+ind_diff)]
                index_offset = index_offset + ind_diff

        layer_start = np.insert(np.cumsum(np.insert(widths_back, 0, 0)), 0, 0)
        layer_end = np.cumsum(np.insert(widths_back, 0, 0))

        prof_back = np.zeros((len(options.wavelengths), len(back_positions)))

        if structure[2].prof_layers is not None:

            index_offset = 0

            for i1 in structure[2].prof_layers:
                indices = np.logical_and(back_positions >= layer_start[i1], back_positions < layer_end[i1])
                ind_diff = np.sum(indices)
                prof_back[:, indices] = 1e9*profile_result[2][:, index_offset:(index_offset+ind_diff)]
                index_offset = index_offset + ind_diff

        if options.bulk_profile:

            prof_bulk = profile_result[1][:, :len(bulk_positions)]

        else:
            prof_bulk = np.zeros_like(bulk_positions)

        all_positions = np.hstack((front_positions/1e9,
                                   bulk_positions/1e6 + np.sum(widths_front)/1e9,
                                   back_positions/1e9 + np.sum(widths_front)/1e9 + width_bulk/1e6))
        all_absorption = np.hstack((prof_front, prof_bulk, prof_back))

        diff_absorb_fn = interp1d(all_positions, all_absorption)

        return all_positions, diff_absorb_fn

    else:

        positions = np.arange(0, structure.width, options.depth_spacing)
        diff_absorb_fn = interp1d(positions, 1e9 * profile_result)

        return positions, diff_absorb_fn

