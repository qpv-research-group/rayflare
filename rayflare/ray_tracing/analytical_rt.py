import numpy as np
import xarray as xr
import os
from solcore.state import State
from .rt_common import make_pol_vectors

from rayflare import logger

theta_lamb = np.linspace(0, 0.999 * np.pi / 2, 100)
def traverse_vectorised(width, theta, alpha, I_i, positions, direction):

    ratio = alpha[None, :] / np.real(np.abs(np.cos(theta)))
    DA_u = I_i[:, :, None] * ratio[:, :, None] * np.exp((-ratio[:, :, None] * positions[None, None, :]))
    # DA_u dimensions: (directions, wavelength, position)

    I_back = I_i * np.exp(-ratio * width)

    # stop = np.where(I_back < I_thresh)[0]

    if direction == -1:
        DA_u = np.flip(DA_u)

    intgr = np.trapz(DA_u, positions, axis=2)

    DA = np.divide(
        ((I_i[:, :, None] - I_back[:, :, None]) * DA_u), intgr[:, :, None], where=intgr[:, :, None] != 0,
        out=np.zeros_like(DA_u),
    ).T

    return DA, I_back

def calc_RAT_Fresnel(theta, pol, *args):
    n1 = args[0]
    n2 = args[1]
    theta_t = np.arcsin((n1 / n2) * np.sin(theta))

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

    return Rs, Rp, np.array([0]), 1 - Rs, 1 - Rp

def calc_RAT_TMM(theta, pol, *args):
    lookuptable = args[0]

    angles = xr.DataArray(theta, dims=['unique_direction', 'wl_angle'])
    wls = xr.DataArray(lookuptable.wl.data, dims='wl_angle')

    data = lookuptable.sel(
        angle=angles, wl=wls, method="nearest"
    )

    # rearrange coordinates:
    Rs = data.R.sel(pol="s").transpose("unique_direction", "wl_angle")
    Rp = data.R.sel(pol="p").transpose("unique_direction", "wl_angle")

    Ts = data.T.sel(pol="s").transpose("unique_direction", "wl_angle")
    Tp = data.T.sel(pol="p").transpose("unique_direction", "wl_angle")

    A_per_layer = np.sum(data.Alayer.transpose("layer", "unique_direction", "wl_angle", "pol")*pol, -1)
    A_per_layer = A_per_layer.transpose("unique_direction", "layer", "wl_angle")
    return (np.real(Rs.data), np.real(Rp.data),
            np.real(A_per_layer.data),
            np.real(Ts.data), np.real(Tp.data))

class dummy_prop_rays:

    def __init__(self):
        # the only thing this needs to do is return None when used with .isel(wl=i)
        pass

    def isel(self, wl):
        return None

class zero_intensity_rays:

    def __init__(self):
        # the only thing this needs to do is return I=0 when used with .isel(wl=i)
        self.I = 0

    def isel(self, wl):
        return State(I = 0)

def update_absorbed_details(x, A, n_interactions, n_passes):

    if A.ndim == 1:
        A = A[None, :]

    if n_passes.ndim == 0:
        n_passes = np.array([n_passes]*A.shape[0])

    if n_interactions.ndim == 0:
        n_interactions = np.array([n_interactions]*A.shape[0])

    x.append(xr.Dataset({
                "A": xr.DataArray(A,
                                  dims=["unique_direction", "wl"]),
                "n_interactions": xr.DataArray(n_interactions,
                                               dims=["unique_direction"]),
                "n_passes": xr.DataArray(n_passes, dims=["unique_direction"]),
            }))


def analytical_start(nks,
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
                max_interactions,
                wls
                     ):

    # if light is incident on a planar surface at an off-normal angle, different wavelengths will
    # have different angles of incidence! This should be fine but not currently expected by
    # analytical_per_facet

    # # generally same across wavelengths, but can be changed by analytical
    # # ray tracing happening first

    if initial_dir == -1:
        raise ValueError("Incidence from below (direction = -1) not implemented for analytical "
                         "ray-tracing")

    if np.sum(tmm_args[0]) == 0:
        # only Fresnel surfaces and tmm_args is just a list of zeroes.
        tmm_args = [0] + [len(surfaces)*[0]]
        # need tmm_args to have a second element, so that inner functions can correctly
        # identify that they need to use Fresnel and not TMM

    n_wl = len(wls)
    wls = xr.DataArray(wls, dims='wl_angle')

    current_pol = np.tile(pol, (n_wl, 1)) # "length"**2 of E-field vectors in s and p directions.
    # sum is 1.

    _, current_pol_vectors = make_pol_vectors(pol, theta, phi)

    # want the dimensions for current_pol_vectors to be: (pol, xyz, wl), need to repeat n_wl times
    # along new axis which will be the final index:
    current_pol_vectors = np.repeat(current_pol_vectors[:, :, None], n_wl, axis=2)

    mat_i = initial_mat

    # if initial_dir == 1: # travelling down
    surf_index = initial_mat
    # next_mat = initial_mat + initial_dir

    # else: # travelling up
    #     surf_index = initial_mat - 1

    profile = np.zeros((len(z_pos), n_wl))
    # do everything in microns
    A_per_layer = np.zeros((len(widths), n_wl))

    A_per_interface = [[] for _ in range(len(surfaces))]

    # make xarrays with attribute I = 0:
    overall_R = xr.Dataset({"I": xr.DataArray(np.zeros((1, n_wl)),
                                              dims=["unique_direction", "wl"])})
    overall_T = xr.Dataset({"I": xr.DataArray(np.zeros((1, n_wl)),
                                              dims=["unique_direction", "wl"])})

    theta = theta*np.ones(n_wl)
    angles = xr.DataArray(theta, dims='wl_angle')

    single_direction = True

    d = -r_a_0 / np.linalg.norm(r_a_0)

    I_remaining = xr.DataArray(np.ones((1, n_wl)), dims=["unique_direction", "wl"])

    n_interactions = 0
    n_passes = 0
    a_details = []
    prop_rays = []

    while single_direction:

        I_rem_data = I_remaining.data[0]

        R_data, A_data, T_data, R_pol, T_pol, A_int_detail = analytical_per_face(surfaces[surf_index],
                                      surf_index,
                                      d,
                                      tmm_args,
                                      nks,
                                      initial_dir,
                                      current_pol,
                                      current_pol_vectors,
                                      max_interactions[surf_index],
                                      )

        # Note: the polarization vector direction will not be updated here. In current
        # implementation with rt_structure, this is ok, because the calculation ends here.
        # If continuing the calculation, will need to calculate s and p vectors from the
        # ray direction; do need to store the current_pol


        A_per_interface[surf_index] = A_data*I_rem_data[None, None, :]

        R_data['n_interactions'] = R_data["n_interactions"] + n_interactions
        T_data['n_interactions'] = T_data["n_interactions"] + n_interactions

        n_int_detail = np.arange(1, A_int_detail.shape[0] + 1) + n_interactions

        A_int_detail = A_int_detail*I_rem_data[None, :]

        for i1, Ada in enumerate(A_int_detail):
            update_absorbed_details(a_details, Ada,   np.array([n_int_detail[i1]]), np.array([n_passes]))

        # scale R_data and T_data by I_remaining:
        R_data['I'] = R_data.I * I_rem_data[None, :]
        T_data['I'] = T_data.I * I_rem_data[None, :]

        if mat_i == initial_mat:
            n_passes_R = 0

        else:
            n_passes_R = n_passes + 1

        n_passes += 1

        R_data = xr.merge([R_data, xr.DataArray(n_passes_R).rename("n_passes")])
        T_data = xr.merge([T_data, xr.DataArray(n_passes).rename("n_passes")])

        current_pol = T_pol

        # surf_index only right for incidence from above
        DA, I = traverse_vectorised(
            widths[surf_index + initial_dir], # units?
            T_data.theta_t.data,
            alphas[surf_index + initial_dir], # units?
            np.ones_like(T_data.theta_t),
            depths[surf_index + initial_dir],
            initial_dir,
        )

        if surf_index == 0 and initial_dir == 1:
            # any rays that were reflected here are reflected overall into the incidence medium
            overall_R = R_data
            # do not want to include reflected rays in propagating rays, they are
            # accounted for in overall_R

        else:
            # could have multiple planar surfaces (though should really just make them
            # all part of the same surface in that case, with incoherent layers if necessary!),
            # and in that case we want to record rays travelling upwards here but continue with
            # analytical ray tracing

            # need to propagate these rays through the bulk and account for attenuation
            # of these rays and absorption in the bulk:
            theta_R = np.arccos(
                R_data.direction[:, 2] / np.linalg.norm(R_data.direction, axis=1))

            DA_R, I_R = traverse_vectorised(
                widths[mat_i],  # units?
                theta_R.data,
                alphas[mat_i],  # units?
                np.ones_like(theta_R.data),
                depths[mat_i],
                initial_dir,
            )

            I_abs_R = 1 - I_R

            I_out_per_direction_R = R_data.I.data * I_abs_R
            update_absorbed_details(a_details, I_out_per_direction_R, R_data.n_interactions, R_data.n_passes)

            A_actual_R = np.sum(I_out_per_direction_R, axis=0)
            DA_actual_R = np.sum(R_data.I.data.T * DA_R, axis=2)

            A_per_layer[mat_i] += np.real(A_actual_R)
            profile[depth_indices[mat_i]] += np.real(+ DA_actual_R)

            R_remaining = R_data.I * I_R
            prop_rays.append(xr.Dataset(
                {
                    "I": R_remaining,
                    "direction": R_data.direction,
                    "mat_i": mat_i,
                    "n_interactions": R_data.n_interactions,
                    "n_passes": R_data.n_passes,
                    "pol": xr.DataArray(R_pol, dims=["unique_direction", "wl", "sp"]),
                }
            )
            )

        if surf_index == len(surfaces) - 1 and initial_dir == 1:
            # any rays that were transmitted here are transmitted overall into the transmission medium
            overall_T = T_data
            include_T = False  # do not want to include transmitted rays in propagating rays, they are
            # accounted for in overall_T
            # TODO: does this need to be scaled?

        else:
            include_T = True

        I_abs = 1 - I

        I_out_per_direction = T_data.I.data * I_abs
        update_absorbed_details(a_details, I_out_per_direction, T_data.n_interactions,
                                            T_data.n_passes)
        A_actual = np.sum(I_out_per_direction, axis=0)

        DA_actual = np.sum(T_data.I.data.T*DA, axis=2)

        surf_index += initial_dir
        mat_i += initial_dir

        A_per_layer[mat_i] += np.real(A_actual)
        profile[depth_indices[mat_i]] += np.real(+ DA_actual)
        remaining_after_bulk = np.real(T_data.I * I)

        in_structure = surf_index < len(surfaces) and surf_index >= 0

        # if all rays are still travelling in the same direction, continue with analytical RT. Otherwise continue on to
        # 'normal' ray tracing. Otherwise, or if we have check the last surface in the structure,
        # end and return results.

        if (np.unique(T_data.direction, axis=0).shape[0] > 1 or
                not in_structure or surfaces[surf_index-1].phong):

            # single_direction = False
            # end, need to save/return final results here
            # absorption (profile and total) in bulk have been tracked as we went along
            # Need to save results for of remaining intensities and directions for each wavelength,
            # and where these rays are - note that we have already accounted for traversal of the
            # bulk layer, so these rays should start right before the next surface in 'normal'
            # ray-tracing.

            # Also need to save overall reflection and transmission into semi-infinite surrounding
            # which has happened so far (this should have been done above)

            # array with dimensions: (face, wl)
            # dataarrays for: direction (xyz), intensity, number of interactions,

            if include_T:

                if T_pol.ndim == 2:
                    T_pol = T_pol[None, :, :]

                prop_rays.append(xr.Dataset(
                    {
                        "I": remaining_after_bulk,
                        "direction": T_data.direction,
                        "mat_i": mat_i,
                        "n_interactions": T_data.n_interactions,
                        "n_passes": T_data.n_passes,
                        "pol": xr.DataArray(T_pol, dims=["unique_direction", "wl", "sp"]),
                    }
                )
                )

            # stack prop_rays along the unique_direction axis:
            if len(prop_rays) > 0:
                prop_rays = xr.concat(prop_rays, dim="unique_direction")

            else:
                prop_rays = zero_intensity_rays()

            a_details = xr.concat(a_details, dim="unique_direction")
            a_details["A"] = np.real(a_details["A"])

            # set nan values to 0:
            a_details["A"] = a_details["A"].fillna(0)

            R_to_add = np.sum(overall_R.I, axis=0).data
            T_to_add = np.sum(overall_T.I, axis=0).data

            A_interface_to_add = []
            for darray in A_per_interface:
                if np.sum(darray) > 0:
                    A_interface_to_add.append(np.sum(darray.data, axis=0).T)

                else:
                    A_interface_to_add.append(0)

            # return profile.T, A_per_layer.T, a_details, A_per_interface, overall_R, overall_T, prop_rays
            result_per_wl = {"R": R_to_add, "T": T_to_add, "A_per_layer": A_per_layer.T,
                             "A_per_interface": A_interface_to_add, "profile": profile.T}
            result_detail = {"overall_R": overall_R, "overall_T": overall_T, "A_details": a_details}

            return prop_rays, result_per_wl, result_detail

        else:
            # continue, but need to update inputs: transmitted rays at each wavelength become new
            # incident rays.
            angles.data = T_data.theta_t.data[0]
            theta = T_data.theta_t.data[0]
            I_remaining = remaining_after_bulk

            # can only reach here if surfaces so far have been planar; end up with
            # two directions because planar surface is made of two triangles, but they
            # contain the same information
            # s vector does not need to be changed if previous surface was planar
            I_remaining = I_remaining.sum(dim='unique_direction').expand_dims('unique_direction')

            d = T_data.direction[0].expand_dims('unique_direction').data
            new_p_vector = [np.cos(np.real(angles.data)), np.zeros_like(np.real(angles.data)), -np.sin(np.real(angles.data))]
            current_pol_vectors[1] = new_p_vector
            # current_pol = current_pol # if analytical_planar only
            current_pol =  current_pol[0]


def analytical_per_face(current_surf,
                         surf_index,
                         r_in,
                         tmm_args,
                         nks,
                         direction,
                         current_pol,
                         current_pol_vectors,
                         max_interactions,
                        ):

    n_wavelengths = nks.shape[1]
    how_many_faces = len(current_surf.N)
    normals = current_surf.N

    if current_pol.ndim == 2:
        current_pol = current_pol[None, :, :]

    if current_pol_vectors.ndim == 3:
        current_pol_vectors = np.tile(current_pol_vectors, (how_many_faces, 1, 1, 1))
        current_pol_vectors = np.moveaxis(current_pol_vectors, 0, 1)
        # indexing is (pol, face, xyz, wl)

    if tmm_args[0] > 0:
        n_layers = tmm_args[4][surf_index]

    else:
        n_layers = 0

    # TODO: only correct for downwards
    n0 = nks[surf_index]
    n1 = nks[surf_index + direction]

    opposite_faces = np.where(np.dot(normals, normals.T) < 0)[1]

    if tmm_args[1][surf_index] == 0:
        calc_RAT = calc_RAT_Fresnel
        R_args = [n0, n1]
        # TODO: above only correct for downwards

    else:
        calc_RAT = calc_RAT_TMM
        structpath = tmm_args[2]
        surf_name = tmm_args[3][surf_index] + "int_{}".format(surf_index)
        lookuptable = xr.open_dataset(os.path.join(structpath, surf_name + ".nc")).loc[dict(pol=['s', 'p'], side=direction)]
        # do I want to load this?
        R_args = [lookuptable]

    if len(r_in.flatten()) == 3:

        r_inc = np.tile(r_in[:,None], (how_many_faces, 1, n_wavelengths))  # (4, 3) array
        # r_inc = r_inc[:, :, None]

    else:
        # (4, 3, n_wavelengths array):
        r_inc = np.tile(r_in, (how_many_faces, 1, 1))


    area = np.sqrt(
        np.sum(np.cross(current_surf.P_0s - current_surf.P_1s, current_surf.P_2s - current_surf.P_1s, axis=1) ** 2, 1)
        ) / 2

    relevant_face = np.arange(how_many_faces)

    R_per_it = np.zeros((how_many_faces, max_interactions, n_wavelengths))
    T_per_it = np.zeros((how_many_faces, max_interactions, n_wavelengths))
    T_dir_per_it = np.zeros((how_many_faces, max_interactions, n_wavelengths))
    T_pol_per_it = np.zeros((how_many_faces, max_interactions, n_wavelengths, 2))
    A_per_it = np.zeros((how_many_faces, n_layers, max_interactions, n_wavelengths))

    stop_it = np.ones(how_many_faces, dtype=int) * max_interactions

    cos_inc = -np.sum(normals[relevant_face, :, None] * r_inc, 1)  # dot product

    hit_prob = area[relevant_face, None] * cos_inc  # scale by area of each triangle
    hit_prob[
        cos_inc < 0] = 0  # if negative, then the ray is shaded from that pyramid face and will never hit it
    hit_prob = hit_prob / np.sum(hit_prob, axis=0)  # initial probability of hitting each face

    reflected_ray_directions = np.zeros((how_many_faces, 3, max_interactions, n_wavelengths))
    transmitted_ray_directions = np.zeros((how_many_faces, 3, max_interactions, n_wavelengths))

    N_interaction = 0

    while np.sum(cos_inc) > 0:

        reflected_direction = r_inc - 2 * np.sum(r_inc*normals[relevant_face, :, None], axis=1)[:, None] * normals[relevant_face, :, None]
        reflected_direction = reflected_direction / np.linalg.norm(reflected_direction, axis=1)[:, None]

        reflected_ray_directions[:, :, N_interaction] = reflected_direction

        # if negative, then the ray is shaded from that pyramid face and will never hit it

        tr_par = (n0 / n1) * (r_inc - np.sum(r_inc * normals[relevant_face, :, None], axis=1)[:,
                                      None] * normals[relevant_face, :, None])
        tr_par_norm = np.linalg.norm(tr_par, axis=1)
        tr_par_norm[tr_par_norm > 1] = 1
        tr_perp = -np.sqrt(1 - tr_par_norm ** 2)[:, None, :] * normals[relevant_face, :, None]

        refracted_rays = np.real(tr_par + tr_perp)
        refracted_rays  = refracted_rays / np.linalg.norm(refracted_rays, axis=1)[:,None, :]
        transmitted_ray_directions[:, :,  N_interaction] = refracted_rays

        normals_relevant = normals[relevant_face][:, :, None]

        ray_plane_s_direction = np.cross(r_inc, normals_relevant, axis=1)

        # this gives 0s if the ray is parallel to the normal
        norm_s_dir = np.linalg.norm(ray_plane_s_direction, axis=1)

        # find indices of places where this happens:
        is_parallel = np.abs(norm_s_dir) < 1e-10
        # this has dimensions (face, wl)

        if np.any(is_parallel):
            ray_plane_s_temp = np.moveaxis(ray_plane_s_direction, 1, -1)
            current_s_vector = np.moveaxis(current_pol_vectors[0], 1, -1)
            # if the ray is parallel to the normal, then the s direction is the same as the
            # previous s direction

            ray_plane_s_temp[is_parallel] = current_s_vector[is_parallel]

            ray_plane_s_direction = np.moveaxis(ray_plane_s_temp, -1, 1)


        ray_plane_s_direction = ray_plane_s_direction / np.linalg.norm(ray_plane_s_direction,
                                                                       axis=1)[:,None,:]

        s_s = np.einsum('ijk,ijk->ik', current_pol_vectors[0], ray_plane_s_direction)
        p_s = np.einsum('ijk,ijk->ik', current_pol_vectors[1], ray_plane_s_direction)
        # dimensions: (faces, n_wl)

        s_component_sq = (current_pol[:, :, 0] * s_s ** 2 + current_pol[:, :, 1] * p_s ** 2)
        p_component_sq = 1 - s_component_sq

        pol_weight = np.moveaxis(np.array([s_component_sq, p_component_sq]), 0, -1)
        Rs_prob, Rp_prob, A_prob, Ts_prob, Tp_prob = calc_RAT(np.arccos(cos_inc), pol_weight, *R_args)
        # these are for light which is s and p polarized with respect to each face; need to
        # find components of each ray which are s and p polarized with respect to the
        # existing ray's s and p directions
        reflected_p_direction = np.cross(r_inc, ray_plane_s_direction, axis=1)

        current_pol_vectors = np.stack([ray_plane_s_direction, reflected_p_direction])
        # dimensions of Rs_prob etc.: (n_faces, n_wls)
        # dimensions of normals[relevant_face]: (n_faces, 3)
        # dimensions of current_pol_vectors: (sp, xyz, n_wl)
        # dimensions of r_inc: (n_faces, 3, n_wls)

        R_stack = np.stack((Rs_prob*s_component_sq, Rp_prob*p_component_sq), axis=-1)
        R_prob = np.sum(R_stack, -1)

        # stack Ts_prob and Tp_prob so that the final index of the array is the new one:
        T_stack = np.stack((Ts_prob*s_component_sq, Tp_prob*p_component_sq), axis=-1)

        # fixed (? not checked) up to here for vector polarization directions
        current_pol = R_stack/R_prob[:,:,None]
        # current_pol = current_pol / (np.sum(current_pol, -1)[:, :, None])

        if np.sum(A_prob) > 0:
            A_prob_sum = np.sum(A_prob, axis=1)

        else:
            A_prob_sum = 0

        T_prob = 1 - R_prob - A_prob_sum

        T_per_it[:, N_interaction] = T_prob

        A_per_it[:, :, N_interaction] = A_prob

        T_dir_per_it[:, N_interaction] = np.abs(
            refracted_rays[:, 2] / np.linalg.norm(refracted_rays,
                                                  axis=1))  # cos (global) of refracted ray

        # T_pol_per_it[:, N_interaction] = T_stack/T_prob[:,:,None]
        T_pol_per_it[:, N_interaction] = np.divide(T_stack, T_prob[:, :, None], out=T_pol_per_it[:, N_interaction],
                  where=T_prob[:, :, None] > 1e-15)

        cos_inc[reflected_direction[:, 2] > 0] = 0
        stop_it[
            np.all((np.all(reflected_direction[:, 2] > 0, axis=1), stop_it > N_interaction),
                   axis=0)] = N_interaction
         # want to end for this surface, since rays are travelling upwards -> no intersection

        R_per_it[:,N_interaction] = R_prob  # intensity reflected from each face, relative to incident total intensity 1

        # once ray travels upwards once, want to end calculation for that plane; don't want to
        # double count

        if len(opposite_faces) > 0:
            relevant_face = opposite_faces[relevant_face]

        r_inc = reflected_direction

        if np.sum(cos_inc) == 0:
            # no more interactions with any of the faces
            break

        cos_inc = -np.sum(normals[relevant_face, :, None] * r_inc, 1)  # dot product
        cos_inc[cos_inc < 0] = 0

        N_interaction += 1

    remaining_intensity = np.insert(np.cumprod(R_per_it, axis=1), 0, np.ones((how_many_faces, n_wavelengths)),
                                    axis=1)[:, :-1]

    R_total = np.array([hit_prob[j1] * np.prod(R_per_it[j1, :stop_it[j1] + 1], axis=0) for j1 in
               range(how_many_faces)])
    final_R_directions = np.array([reflected_ray_directions[j1, :, stop_it[j1]] for j1 in
                          range(how_many_faces)])
    final_R_pol = current_pol

    # the weight of each of these directions is R_total

    # loop through faces and interactions:
    final_T_directions = []
    final_T_weights = []
    final_T_n_interactions = []
    final_T_pol = []

    for j1 in range(how_many_faces):
        for j2 in range(stop_it[j1] + 1):
            final_T_directions.append(transmitted_ray_directions[j1, :, j2])
            final_T_weights.append(hit_prob[j1]*remaining_intensity[j1, j2]*T_per_it[j1, j2])
            final_T_n_interactions.append(j2 + 1)
            final_T_pol.append(T_pol_per_it[j1, j2])

    final_T_weights = np.array(final_T_weights) # is this a function of wavelength?
    final_T_weights[final_T_weights < 0] = 0
    final_T_directions = np.array(final_T_directions)
    final_T_pol = np.array(final_T_pol)
    final_T_pol = np.divide(final_T_pol, np.sum(final_T_pol, -1)[:, :, None],
                            out=final_T_pol, where=np.sum(final_T_pol, -1)[:, :, None] > 1e-15)

    # A_per_it has dimensions: (face, layer, interaction, wavelength)
    # sum over layers:
    A_per_it_sum = np.sum(hit_prob[:,None]*remaining_intensity*np.sum(A_per_it, axis=1), 0)
    # A_per_it_sum is the total absorption in the interface (all faces) per interaction

    A_total = hit_prob[:, None] * np.sum(remaining_intensity[:, None, :, :] * A_per_it, axis=2)

    # theta_out_R = np.arccos(final_R_directions[:, 2] / np.linalg.norm(final_R_directions, axis=1))
    # phi_out_R = np.arctan2(final_R_directions[:, 1], final_R_directions[:, 0])
    # number of reps of each theta value for the angular distribution:

    theta_out_T = np.arccos(final_T_directions[:, 2] / np.linalg.norm(final_T_directions, axis=1))
    # phi_out_T = np.arctan2(final_T_directions[:, 1], final_T_directions[:, 0])

    # list of results and their dimensions:
    # R_total: (face, wavelength)
    # final_R_directions: (face, 3, wavelength)
    # theta_out_R: (face, wavelength)
    # phi_out_R: (face, wavelength)

    # A_total: (face, layer, wavelength)

    # final_T_weights: (number of outgoing directions, wavelength)
    # final_T_directions: (number of outgoing directions, 3, wavelength)
    # final_T_n_interactions: (number of outgoing directions)
    # theta_out_T: (number of outgoing directions, wavelength)
    # phi_out_T: (number of outgoing directions, wavelength)

    # make xarrays for each of these:
    total_I = np.sum(R_total, 0) + np.sum(final_T_weights, 0) + np.sum(np.sum(A_total, 1), 0)

    if not np.allclose(total_I, 1, rtol=5e-3, atol=5e-3):
        logger.warning(f"Total intensity not conserved in analytical ray tracing, indicating "
                      f"the number of interactions is not constant across the unit cell. Maximum "
                      f"deviation: {100*np.max(np.abs(total_I - 1)):.1f} %")

    R_total = xr.DataArray(R_total, dims=["unique_direction", "wl"])
    final_R_directions = xr.DataArray(final_R_directions, dims=["unique_direction", "xyz", "wl"])
    n_interactions = xr.DataArray(stop_it + 1, dims=["unique_direction"])
    # theta_out_R = xr.DataArray(theta_out_R, dims=["unique_direction"])
    # phi_out_R = xr.DataArray(phi_out_R, dims=["unique_direction"])

    # if all rays originating from a surface had the same number of interactions, this should sum
    # to:

    R_data = xr.Dataset(
        {
            "I": R_total/total_I,
            "direction": final_R_directions,
            "n_interactions": n_interactions,
        }
    )

    A_data = xr.DataArray(A_total/total_I[None, None, :], dims=["unique_direction", "layer", "wl"])

    final_T_weights = xr.DataArray(final_T_weights, dims=["unique_direction", "wl"])
    final_T_directions = xr.DataArray(final_T_directions, dims=["unique_direction", "xyz", "wl"])
    final_T_n_interactions = xr.DataArray(final_T_n_interactions, dims=["unique_direction"])
    theta_out_T = xr.DataArray(theta_out_T, dims=["unique_direction", "wl"])

    T_data = xr.Dataset(
        {
            "I": final_T_weights/total_I,
            "direction": final_T_directions,
            "n_interactions": final_T_n_interactions,
            "theta_t": theta_out_T,
        }
    )

    return R_data, A_data, T_data, final_R_pol, final_T_pol, A_per_it_sum
