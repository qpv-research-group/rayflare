import os
from sparse import load_npz
import xarray as xr

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