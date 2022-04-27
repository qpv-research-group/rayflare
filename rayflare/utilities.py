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
from solcore import si
from solcore.interpolate import interp1d as interp1d_SC
from solcore.absorption_calculator import OptiStack as OptiStack_SC
from solcore.absorption_calculator.transfer_matrix import np_cache


class interp1d(interp1d_SC):
    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        """Find interpolated y_new = f(x_new).

        Parameters
        ----------
        x_new : number or array
            New independent variable(s).

        Returns
        -------
        y_new : ndarray
            Interpolated value(s) corresponding to x_new.

        """

        # 1. Handle values in x_new that are outside of x.  Throw error,
        #    or return a list of mask array indicating the outofbounds values.
        #    The behavior is set by the bounds_error variable.
        x_new = np.asarray(x_new)
        out_of_bounds = self._check_bounds(x_new)

        y_new = self._call(x_new)

        # Rotate the values of y_new back so that they correspond to the
        # correct x_new values. For N-D x_new, take the last (for linear)
        # or first (for other splines) N axes
        # from y_new and insert them where self.axis was in the list of axes.
        nx = x_new.ndim
        ny = y_new.ndim

        # 6. Fill any values that were out of bounds with fill_value.
        # and
        # 7. Rotate the values back to their proper place.

        if nx == 0:
            # special case: x is a scalar
            if out_of_bounds:
                if ny == 0:
                    return np.asarray(self.fill_value)
                else:
                    y_new[...] = self.fill_value
            return np.asarray(y_new)
        elif self._kind in ('linear', 'nearest'):
            if ny == 1:
                y_new[..., out_of_bounds] = self.fill_value
                axes = list(range(ny - nx))
                axes[self.axis:self.axis] = list(range(ny - nx, ny))
                return y_new.transpose(axes)
            else:
                y_new[..., out_of_bounds] = self.fill_value[..., None]
                axes = list(range(ny - nx))
                axes[self.axis:self.axis] = list(range(ny - nx, ny))
                return y_new.transpose(axes)
        else: # pragma: no cover
            y_new[out_of_bounds] = self.fill_value
            axes = list(range(nx, ny))
            axes[self.axis:self.axis] = list(range(nx))
            return y_new.transpose(axes)


class OptiStack(OptiStack_SC): # pragma: no cover

    def _add_raw_nk_layer(self, layer):
        """ Adds a layer to the end (bottom) of the stack. The layer must be defined as a list containing the layer
        thickness in nm, the wavelength, the n and the k data as array-like objects.

        :param layer: The new layer to add as [thickness, wavelength, n, k]
        :return: None
        """
        # We make sure that the wavelengths are increasing, reversing the arrays otherwise.

        if len(layer[1]) > 1:

            if layer[1][0] > layer[1][-1]:
                layer[1] = layer[1][::-1]

                layer[2] = layer[2][::-1]
                layer[3] = layer[3][::-1]

        self.widths.append(layer[0])

        if len(layer) >= 5:
            self.models.append(layer[4])
            c = si(layer[5][0], 'nm')
            w = si(layer[5][1], 'nm')
            d = layer[5][2]  # = 0 for increasing, =1 for decreasing

            def mix(x):

                out = 1 + np.exp(-(x - c) / w)
                out = d + (-1) ** d * 1 / out

                return out

            n_data = np_cache(lambda x: self.models[-1].n_and_k(x) * mix(x) + (1 - mix(x)) * interp1d(
                x=si(layer[1], 'nm'), y=layer[2], fill_value=layer[2][-1])(x))
            k_data = np_cache(lambda x: interp1d(x=si(layer[1], 'nm'), y=layer[3], fill_value=layer[3][-1])(x))

            self.n_data.append(n_data)
            self.k_data.append(k_data)

        else:
            self.models.append([])

            self.n_data.append(np_cache(interp1d(x=si(layer[1], 'nm'), y=layer[2], fill_value=layer[2][-1], axis=0)))
            self.k_data.append(np_cache(interp1d(x=si(layer[1], 'nm'), y=layer[3], fill_value=layer[3][-1], axis=0)))



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

        diff_absorb_fn = interp1d_SC(all_positions, all_absorption)

        return all_positions, diff_absorb_fn

    else:

        positions = np.arange(0, structure.width, options.depth_spacing)
        diff_absorb_fn = interp1d_SC(positions, 1e9 * profile_result)

        return positions, diff_absorb_fn

