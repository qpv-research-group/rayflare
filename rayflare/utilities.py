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


def make_absorption_function(profile_result, cell_width, depth_spacing):
    """
    :param profile:
    :param options:

    :return diff_absorb_fn:
    """

    positions = np.arange(0, cell_width, depth_spacing)
    diff_absorb_fn = interp1d(positions, 1e9 * profile_result)

    return diff_absorb_fn