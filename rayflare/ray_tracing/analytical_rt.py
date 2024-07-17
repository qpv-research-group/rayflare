import numpy as np
import xarray as xr
import os
from rayflare.utilities import get_savepath
from copy import deepcopy

theta_lamb = np.linspace(0, 0.999 * np.pi / 2, 100)
def traverse_vectorised(width, theta, alpha, I_i, positions, I_thresh, direction):

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
    if pol == "s":
        Rs = (
                np.abs(
                    (n1 * np.cos(theta) - n2 * np.cos(theta_t))
                    / (n1 * np.cos(theta) + n2 * np.cos(theta_t))
                )
                ** 2
        )
        return Rs, [0]

    if pol == "p":
        Rp = (
                np.abs(
                    (n1 * np.cos(theta_t) - n2 * np.cos(theta))
                    / (n1 * np.cos(theta_t) + n2 * np.cos(theta))
                )
                ** 2
        )
        return Rp, [0]

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
        return (Rs + Rp) / 2, np.array([0])

def calc_RAT_Fresnel_vec(theta, pol, *args):

    n1 = args[0]
    n2 = args[1]
    ratio = np.clip((n1[None, :] / n2[None, :]) * np.sin(theta[:, None]), -1, 1)
    theta_t = np.arcsin(ratio)

    if pol == "s":
        Rs = (
                np.abs(
                    (n1[None, :] * np.cos(theta[:,None]) - n2[None, :] * np.cos(theta_t))
                    / (n1[None, :] * np.cos(theta[:, None]) + n2[None, :] * np.cos(theta_t))
                )
                ** 2
        )

        # Rs[np.isnan(Rs)] = 1

        return Rs, [0]

    if pol == "p":
        Rp = (
                np.abs(
                    (n1[None, :] * np.cos(theta_t) - n2[None, :] * np.cos(theta[:,None]))
                    / (n1[None, :] * np.cos(theta_t) + n2[None, :] * np.cos(theta[:,None]))
                )
                ** 2
        )

        # Rp[np.isnan(Rp)] = 1

        return Rp, [0]

    else:
        Rs = (
                np.abs(
                    (n1[None, :] * np.cos(theta[:,None]) - n2[None, :] * np.cos(theta_t))
                    / (n1[None, :] * np.cos(theta[:,None]) + n2[None, :] * np.cos(theta_t))
                )
                ** 2
        )
        Rp = (
                np.abs(
                    (n1[None, :] * np.cos(theta_t) - n2[None, :] * np.cos(theta[:,None]))
                    / (n1[None, :] * np.cos(theta_t) + n2[None, :] * np.cos(theta[:,None]))
                )
                ** 2
        )
        # Rs[np.isnan(Rs)] = 1
        # Rp[np.isnan(Rp)] = 1

        return (Rs + Rp) / 2, np.array([0])

def calc_RAT_TMM(theta, pol, *args):
    lookuptable = args[0]
    side = args[1]

    angles = xr.DataArray(theta, dims=['face', 'wl_angle'])
    wls = xr.DataArray(lookuptable.wl.data, dims='wl_angle')

    data = lookuptable.sel(
        angle=angles, wl=wls, method="nearest"
    )

    # rearrange coordinates:


    R = data["R"].transpose("face", "wl_angle")
    A_per_layer = data["Alayer"].transpose("face", "layer", "wl_angle")

    return np.real(R.data), np.real(A_per_layer.data)


def analytical_start(nks,
                alphas,
                theta,
                r_a_0,
                phi,
                surfaces,
                widths,
                cum_width,
                z_pos,
                depths,
                depth_indices,
                I_thresh,
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

    n_wl = len(wls)
    wls = xr.DataArray(wls, dims='wl_angle')

    mat_i = initial_mat

    n_passes = 0

    if initial_dir == 1: # travelling down
        surf_index = initial_mat

    else: # travelling up
        surf_index = initial_mat - 1

    profile = np.zeros((len(z_pos), n_wl))
    # do everything in microns
    A_per_layer = np.zeros((len(widths), n_wl))

    A_per_interface = [[] for _ in range(len(surfaces))]

    overall_R = 0
    overall_T = 0

    next_mat = initial_mat + initial_dir

    theta = theta*np.ones(n_wl)
    angles = xr.DataArray(theta, dims='wl_angle')

    single_direction = True

    d = -r_a_0 / np.linalg.norm(r_a_0)
    # TODO: check d for travelling upwards

    if initial_dir != 1:
        d[2] = -d[2]

    I_remaining = xr.DataArray(np.ones((1, n_wl)), dims=["face", "wl"])

    n_interactions = xr.DataArray(np.zeros((1, n_wl)), dims=["face", "wl"])

    data_lists = []

    while single_direction:

        normals = surfaces[surf_index].N
        # snell's law for direction of transmitted rays:
        n0 = nks[mat_i]
        n1 = nks[next_mat]

        if np.all(np.abs(normals[:,2]) > 0.99):
            # if the surface is planar, can just use TMM or Fresnel equations directly
            # should already have a lookuptable, if necessary

            theta_t = np.real(np.arcsin(np.real(n0) / n1 * np.sin(angles.data)))

            x = np.sin(theta_t) * np.cos(phi)
            y = np.sin(theta_t) * np.sin(phi)
            z = np.cos(theta_t)

            # make sign of x, y, z the same as d:
            x = np.abs(x) * np.sign(d[0])
            y = np.abs(y) * np.sign(d[1])
            z = np.abs(z) * np.sign(d[2])

            final_T_directions = xr.DataArray(np.stack((x, y, z))[None, :, :],
                                              dims=["face", "xyz", "wl"])

            theta_t = xr.DataArray(theta_t[None, :], dims=["face", "wl"])

            final_R_directions = xr.DataArray(deepcopy(d)[None, :], dims=["face", "xyz"])

            # reflection: z -> -z, no other changes to ray direction
            final_R_directions[:, 2] = -final_R_directions[:, 2]

            if tmm_args[1][surf_index] == 1:
                structpath = tmm_args[2]
                surf_name = tmm_args[3][surf_index] + "int_{}".format(surf_index)

                lookuptable = xr.open_dataset(os.path.join(structpath, surf_name + ".nc")).loc[
                    dict(pol=pol, side=initial_dir)].interp(angle=angles, wl=wls).load()

                R = lookuptable.R.data
                A_per_layer = lookuptable.Alayer.data
                T = lookuptable.T.data

                R_total = xr.DataArray(I_remaining*R[None, :], dims=["face", "wl"])

                # INTERFACE (not bulk!) absorption
                A_per_interface[surf_index] = xr.DataArray(I_remaining*A_per_layer[None, :, :], dims=["face", "wl", "layer"])


            else:
                # Fresnel equations
                R = calc_RAT_Fresnel(theta, pol, n0, n1)[0]

                R_total = xr.DataArray(I_remaining*R[None, :], dims=["face", "wl"])

                T = 1 - R

                # no interface absorption:
                A_per_interface[surf_index] = xr.DataArray(np.zeros((1, n_wl, 1)), dims=["face", "wl", "layer"])

            R_data = xr.Dataset(
                {
                    "R_total": R_total,
                    "final_R_directions": final_R_directions,
                }
            )

            n_interactions += 1  # can only have one interaction with planar surface

            T_data = xr.Dataset(
                {
                    "final_T_weights": I_remaining * T,
                    "final_T_directions": final_T_directions,
                    "final_T_n_interactions": n_interactions,
                    "theta_out_T": theta_t,
                }
            )


        else:
            # do analytical RT for non-planar surface with multiple faces

            R_data, A_data, T_data = analytical_per_face(surfaces[surf_index],
                                          surf_index,
                                          d,
                                          tmm_args,
                                          nks,
                                          initial_dir,
                                          pol,
                                          max_interactions,
                                          )

            # TODO: I think these are not being scaled at all?
            I_rem_data = I_remaining.data[0]

            A_per_interface[surf_index] = A_data*I_rem_data[None, None, :]


        # surf_index only right for incidence from above
        DA, I = traverse_vectorised(
            widths[surf_index + 1], # units?
            T_data.theta_out_T.data,
            alphas[surf_index + 1], # units?
            np.ones_like(T_data.theta_out_T),
            depths[surf_index + 1],
            I_thresh,
            initial_dir,
        )

        # expand I_remaining along the face axis using xarray:
        I_rem_data = I_remaining.data[0]
        DA = DA * I_rem_data[None, :, None] # scaled by intensity remaining BEFORE this surface
        I = I * I_rem_data[None, :]

        I_abs = I_rem_data - I

        if surf_index == 0 and initial_dir == 1:
            # any rays that were reflected here are reflected overall into the incidence medium
            overall_R = R_data
            include_R = False # do not want to include reflected rays in propagating rays, they are
            # accounted for here

        else:
            include_R = True

        if surf_index == len(surfaces) - 1 and initial_dir == 1:
            # any rays that were transmitted here are transmitted overall into the transmission medium
            overall_T = T_data
            include_T = False  # do not want to include transmitted rays in propagating rays, they are
            # accounted for here
            # TODO: this needs to be scaled (?)

        else:
            include_T = True

        # TODO: the above if statements are only for incidence from above.

        # data_lists.append({
        #     "R_data": R_data,
        #     "A_data": A_data,
        #     "T_data": T_data,
        #     "DA": DA,
        #     "I": I
        # }
        # )

        I_out_actual = np.sum(T_data.final_T_weights.data * I_abs, axis=0)
        # A_bulk_actual = np.sum(T_data.final_T_weights.data - I_out_actual)
        DA_actual = np.sum(T_data.final_T_weights.data.T*DA, axis=2)
        # theta_out_T[stop] = np.nan
        # phi_out_T[stop] = np.nan

        surf_index += initial_dir
        mat_i += initial_dir

        A_per_layer[mat_i] = np.real(A_per_layer[mat_i] + I_out_actual)
        profile[depth_indices[mat_i]] = np.real(
            profile[depth_indices[mat_i]] + DA_actual
        )

        n_passes = n_passes + 1

        # if all rays are still travelling in the same direction, continue with analytical RT. Otherwise continue on to
        # 'normal' ray tracing.

        if np.unique(T_data.final_T_directions, axis=0).shape[0] > 1:
            single_direction = False
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

            prop_rays = []

            if include_R:

                # TODO: add n_interactions

                prop_rays.append(xr.Dataset(
                    {
                        "I": R_data.R_total.rename({"face": "unique_direction"}),
                        "direction": R_data.final_R_directions.rename({"face": "unique_direction"})
                    }
                )
                )

            if include_T:
                remaining_after_bulk = T_data.final_T_weights*I
                prop_rays.append(xr.Dataset(
                    {
                        "I": remaining_after_bulk.rename({"outgoing": "unique_direction"}),
                        "direction": T_data.final_T_directions.rename({"outgoing": "unique_direction"})
                    }
                )
                )

            # stack prop_rays along the unique_direction axis:

            prop_rays = xr.concat(prop_rays, dim="unique_direction")

            return profile, A_per_layer, A_per_interface, overall_R, overall_T, prop_rays


        else:
            # continue, but need to update inputs: transmitted rays at each wavelength become new
            # incident rays.
            angles.data = T_data.theta_out_T.data[0]
            theta = T_data.theta_out_T.data[0]
            #I_remaining.data = 0
            # losses from:
            # - interface absorption
            # - bulk absorption
            # - reflection
            I_new = I_rem_data - np.sum(R_data.R_total, 0) - np.sum(A_per_layer, 0) - I_out_actual
            I_remaining[0] = I_new
            # TODO: I think this only works for downwards

            d = T_data.final_T_directions.data[0]

            # need to construct d for each wavelength:


    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(wls, data_lists[0]['R_data'].R_total[0])
    plt.plot(wls, data_lists[0]['A_data'][0])
    plt.show()

    print('done')


    # should first check if surface is planar; if it is, can just use TMM directly.

    # what information do we need at the end of this?
    # - information about the rays which need to be propagated forward to the normal ray-tracing procedure:
    #    - intensities
    #    - directions
    #    - distribution of these directions (same as intensities)
    #    - number of interactions of these rays
    #    - all of these are as a function of wavelength
    #    - need information on what has already happened to the rays: absorption per layer (bulk
    #      and interface) so far, overall reflection/transmission into semi-infinite surrounding media,
    #      and bulk absorption profiles
    #    - can implement interface absorption profiles later


def analytical_per_face(current_surf,
                         surf_index,
                         r_in,
                         tmm_args,
                         nks,
                         direction,
                         pol,
                         max_interactions,
                         ):

    n_wavelengths = nks.shape[1]
    how_many_faces = len(current_surf.N)
    normals = current_surf.N

    if tmm_args[0] > 0:
        n_layers = tmm_args[4][surf_index]

    else:
        n_layers = 0


    # TODO: only correct for downwards
    n0 = np.real(nks[surf_index])
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
        lookuptable = xr.open_dataset(os.path.join(structpath, surf_name + ".nc")).loc[dict(pol=pol, side=direction)]
        # do I want to load this?
        R_args = [lookuptable, 1]

    if len(r_in.flatten()) == 3:

        r_inc = np.tile(r_in, (how_many_faces, 1))  # (4, 3) array
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

    while N_interaction < max_interactions:

        cos_inc = -np.sum(normals[relevant_face, :, None] * r_inc, 1)  # dot product

        reflected_direction = r_inc - 2 * np.sum(r_inc*normals[relevant_face, :, None], axis=1)[:, None] * normals[relevant_face, :, None]
        reflected_direction = reflected_direction / np.linalg.norm(reflected_direction, axis=1)[:, None]

        reflected_ray_directions[:, :, N_interaction] = reflected_direction

        cos_inc[cos_inc < 0] = 0
        # if negative, then the ray is shaded from that pyramid face and will never hit it

        tr_par = (n0 / n1) * (r_inc - np.sum(r_inc*normals[relevant_face, :, None], axis=1)[:,None] * normals[relevant_face, :, None])
        tr_perp = -np.sqrt(1 - np.linalg.norm(tr_par,axis=1) ** 2)[:, None, :] * normals[relevant_face, :, None]

        refracted_rays = np.real(tr_par + tr_perp)
        refracted_rays  = refracted_rays / np.linalg.norm(refracted_rays, axis=1)[:,None, :]
        transmitted_ray_directions[:, :,  N_interaction] = refracted_rays

        R_prob, A_prob = calc_RAT(np.arccos(cos_inc), pol, *R_args)

        if np.sum(A_prob) > 0:
            A_prob_sum = np.sum(A_prob, axis=1)

        else:
            A_prob_sum = 0

        T_per_it[:, N_interaction] = 1 - R_prob - A_prob_sum

        A_per_it[:, :, N_interaction] = A_prob

        T_dir_per_it[:, N_interaction] = np.abs(
            refracted_rays[:, 2] / np.linalg.norm(refracted_rays,
                                                  axis=1))  # cos (global) of refracted ray

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

        N_interaction += 1

    remaining_intensity = np.insert(np.cumprod(R_per_it, axis=1), 0, np.ones((how_many_faces, n_wavelengths)),
                                    axis=1)[:, :-1]

    R_total = np.array([hit_prob[j1] * np.prod(R_per_it[j1, :stop_it[j1] + 1], axis=0) for j1 in
               range(how_many_faces)])
    final_R_directions = np.array([reflected_ray_directions[j1, :, stop_it[j1]] for j1 in
                          range(how_many_faces)])
    # the weight of each of these directions is R_total

    # loop through faces and interactions:
    final_T_directions = []
    final_T_weights = []
    final_T_n_interactions = []

    for j1 in range(how_many_faces):
        for j2 in range(stop_it[j1] + 1):
            final_T_directions.append(transmitted_ray_directions[j1, :, j2])
            final_T_weights.append(hit_prob[j1]*remaining_intensity[j1, j2]*T_per_it[j1, j2])
            final_T_n_interactions.append(j2 + 1)

    final_T_weights = np.array(final_T_weights) # is this a function of wavelength?
    final_T_weights[final_T_weights < 0] = 0
    final_T_directions = np.array(final_T_directions)

    # checked up to here

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

    R_total = xr.DataArray(R_total, dims=["face", "wl"])
    final_R_directions = xr.DataArray(final_R_directions, dims=["face", "xyz", "wl"])
    # theta_out_R = xr.DataArray(theta_out_R, dims=["face"])
    # phi_out_R = xr.DataArray(phi_out_R, dims=["face"])

    R_data = xr.Dataset(
        {
            "R_total": R_total,
            "final_R_directions": final_R_directions,
            # "theta_out_R": theta_out_R,
            # "phi_out_R": phi_out_R,
        }
    )

    A_data = xr.DataArray(A_total, dims=["face", "layer", "wl"])

    final_T_weights = xr.DataArray(final_T_weights, dims=["outgoing", "wl"])
    final_T_directions = xr.DataArray(final_T_directions, dims=["outgoing", "xyz", "wl"])
    final_T_n_interactions = xr.DataArray(final_T_n_interactions, dims=["outgoing"])
    theta_out_T = xr.DataArray(theta_out_T, dims=["outgoing", "wl"])
    # phi_out_T = xr.DataArray(phi_out_T, dims=["outgoing", "wl"])

    T_data = xr.Dataset(
        {
            "final_T_weights": final_T_weights,
            "final_T_directions": final_T_directions,
            "final_T_n_interactions": final_T_n_interactions,
            "theta_out_T": theta_out_T,
            # "phi_out_T": phi_out_T,
        }
    )


    return R_data, A_data, T_data



def analytical_front_surface(front, r_in, n0, n1, pol, max_interactions, n_layers, direction,
                             n_reps,
                             positions,
                             bulk_width,
                             alpha_bulk,
                             I_thresh,
                             Fr_or_TMM=0,
                             lookuptable=None,
                             ):

    # n0 should be real
    # n1 can be complex

    # TODO:
    # reflectance (not intensity but directions) will always be the same, regardless of wavelength,
    # so this could be calculated once and then used for all wavelengths. Currently, because the
    # analytical calculation divides the rays into categories at the end, the accuracy of the R/A/T value
    # will be limited to 1/n_rays. This will be addressed in a future release.

    how_many_faces = len(front.N)
    normals = front.N
    opposite_faces = np.where(np.dot(normals, normals.T) < 0)[1]

    if len(opposite_faces) == 0:
        max_interactions =  1

    if Fr_or_TMM == 0:
        calc_RAT = calc_RAT_Fresnel
        R_args = [n0, n1]

    else:
        calc_RAT = calc_RAT_TMM
        R_args = [lookuptable, 1]

    r_inc = np.tile(r_in, (how_many_faces, 1))  # (4, 3) array

    area = np.sqrt(
        np.sum(np.cross(front.P_0s - front.P_1s, front.P_2s - front.P_1s, axis=1) ** 2, 1)
        ) / 2

    relevant_face = np.arange(how_many_faces)

    R_per_it = np.zeros((how_many_faces, max_interactions))
    T_per_it = np.zeros((how_many_faces, max_interactions))
    T_dir_per_it = np.zeros((how_many_faces, max_interactions))
    A_per_it = np.zeros((how_many_faces, n_layers, max_interactions))

    stop_it = np.ones(how_many_faces, dtype=int) * max_interactions

    cos_inc = -np.sum(normals[relevant_face] * r_inc, 1)  # dot product

    hit_prob = area[relevant_face] * cos_inc  # scale by area of each triangle
    hit_prob[
        cos_inc < 0] = 0  # if negative, then the ray is shaded from that pyramid face and will never hit it
    hit_prob = hit_prob / np.sum(hit_prob)  # initial probability of hitting each face

    reflected_ray_directions = np.zeros((how_many_faces, 3, max_interactions))
    transmitted_ray_directions = np.zeros((how_many_faces, 3, max_interactions))

    N_interaction = 0

    while N_interaction < max_interactions:

        cos_inc = -np.sum(normals[relevant_face] * r_inc, 1)  # dot product

        reflected_direction = r_inc - 2 * np.sum(r_inc*normals[relevant_face], axis=1)[:,None] * normals[relevant_face]
        reflected_direction = reflected_direction / np.linalg.norm(reflected_direction, axis=1)[:, None]

        reflected_ray_directions[:, :, N_interaction] = reflected_direction

        cos_inc[cos_inc < 0] = 0
        # if negative, then the ray is shaded from that pyramid face and will never hit it

        tr_par = (n0 / n1) * (r_inc - np.sum(r_inc*normals[relevant_face], axis=1)[:,None] * normals[relevant_face])
        tr_perp = -np.sqrt(1 - np.linalg.norm(tr_par,axis=1) ** 2)[:, None] * normals[relevant_face]

        refracted_rays = np.real(tr_par + tr_perp)
        refracted_rays  = refracted_rays / np.linalg.norm(refracted_rays, axis=1)[:,None]
        transmitted_ray_directions[:, :,  N_interaction] = refracted_rays

        R_prob, A_prob = calc_RAT(np.arccos(cos_inc), pol, *R_args)

        if np.sum(A_prob) > 0:
            A_prob_sum = np.sum(A_prob, axis=1)

        else:
            A_prob_sum = 0

        T_per_it[:, N_interaction] = 1 - R_prob - A_prob_sum

        A_per_it[:, :, N_interaction] = A_prob

        T_dir_per_it[:, N_interaction] = np.abs(
            refracted_rays[:, 2] / np.linalg.norm(refracted_rays,
                                                  axis=1))  # cos (global) of refracted ray

        cos_inc[reflected_direction[:, 2] > 0] = 0
        stop_it[
            np.all((reflected_direction[:, 2] > 0, stop_it > N_interaction),
                   axis=0)] = N_interaction
         # want to end for this surface, since rays are travelling upwards -> no intersection

        R_per_it[:,
        N_interaction] = R_prob  # intensity reflected from each face, relative to incident total intensity 1

        # once ray travels upwards once, want to end calculation for that plane; don't want to
        # double count

        if len(opposite_faces) > 0:
            relevant_face = opposite_faces[relevant_face]

        r_inc = reflected_direction

        if np.sum(cos_inc) == 0:
            # no more interactions with any of the faces
            break

        N_interaction += 1

    remaining_intensity = np.insert(np.cumprod(R_per_it, axis=1), 0, np.ones(how_many_faces),
                                    axis=1)[:, :-1]

    R_total = np.array([hit_prob[j1] * np.prod(R_per_it[j1, :stop_it[j1] + 1]) for j1 in
               range(how_many_faces)])
    final_R_directions = np.array([reflected_ray_directions[j1, :, stop_it[j1]] for j1 in
                          range(how_many_faces)])
    # the weight of each of these directions is R_total

    # loop through faces and interactions:
    final_T_directions = []
    final_T_weights = []
    final_T_n_interactions = []

    for j1 in range(how_many_faces):
        for j2 in range(stop_it[j1] + 1):
            final_T_directions.append(transmitted_ray_directions[j1, :, j2])
            final_T_weights.append(hit_prob[j1]*remaining_intensity[j1, j2]*T_per_it[j1, j2])
            final_T_n_interactions.append(j2 + 1)

    final_T_weights = np.array(final_T_weights)
    final_T_weights[final_T_weights < 0] = 0
    final_T_directions = np.array(final_T_directions)

    A_total = hit_prob[:, None] * np.sum(remaining_intensity[:, None, :] * A_per_it, axis=2)

    theta_out_R = np.arccos(final_R_directions[:, 2] / np.linalg.norm(final_R_directions, axis=1))
    phi_out_R = np.arctan2(final_R_directions[:, 1], final_R_directions[:, 0])
    # number of reps of each theta value for the angular distribution:
    n_reps_R = n_reps * R_total

    theta_out_T = np.arccos(final_T_directions[:, 2] / np.linalg.norm(final_T_directions, axis=1))
    phi_out_T = np.arctan2(final_T_directions[:, 1], final_T_directions[:, 0])

    n_reps_T = n_reps * final_T_weights

    n_reps_A_surf = np.sum(A_total) * n_reps

    # now make sure n_reps_R, n_reps_T and n_reps_A_surf add to n_reps, remained is divided fairly:
    n_reps_R_int = np.floor(n_reps_R).astype(int)
    n_reps_T_int = np.floor(n_reps_T).astype(int)
    n_reps_A_surf_int = np.floor(n_reps_A_surf).astype(int)

    n_reps_R_remainder = np.sum(n_reps_R - n_reps_R_int)
    n_reps_T_remainder = np.sum(n_reps_T - n_reps_T_int)
    n_reps_A_surf_remainder = n_reps_A_surf - n_reps_A_surf_int

    rays_to_divide = n_reps - np.sum(n_reps_R_int) - np.sum(n_reps_T_int) - n_reps_A_surf_int

    # add these rays to the ones with the highest remainders:
    extra_rays_R = np.round(n_reps_R_remainder / (
                n_reps_R_remainder + n_reps_T_remainder + n_reps_A_surf_remainder) * rays_to_divide).astype(
        int)
    extra_rays_T = np.round(n_reps_T_remainder / (n_reps_T_remainder + n_reps_A_surf_remainder) * (
                rays_to_divide - extra_rays_R)).astype(int)

    extra_rays_A = rays_to_divide - extra_rays_R - extra_rays_T

    n_reps_R_int[np.argmax(n_reps_R_remainder)] += extra_rays_R
    n_reps_T_int[np.argmax(n_reps_T_remainder)] += extra_rays_T
    n_reps_A_surf_int += extra_rays_A

    # see which of the transmitted rays reach the back of the Si before falling below
    # I_thresh

    DA, stop, I = traverse_vectorised(
        bulk_width,
        theta_out_T,
        alpha_bulk,
        np.ones_like(theta_out_T),
        positions,
        I_thresh,
        direction,
    )

    I_out_actual = final_T_weights*I
    A_bulk_actual = np.sum(final_T_weights - I_out_actual)

    theta_out_T[stop] = np.nan
    phi_out_T[stop] = np.nan

    # make the list of theta_out values

    theta_R_reps = np.concatenate(
        [np.tile(theta_out_R[j], n_reps_R_int[j]) for j in range(how_many_faces)])
    phi_R_reps = np.concatenate(
        [np.tile(phi_out_R[j], n_reps_R_int[j]) for j in range(how_many_faces)])
    n_interactions_R_reps = np.concatenate(
        [np.tile(stop_it[j] + 1, n_reps_R_int[j]) for j in range(how_many_faces)])
    I_R_reps = np.ones_like(theta_R_reps)
    n_passes_R_reps = np.zeros_like(theta_R_reps)

    theta_A_surf_reps = np.ones(n_reps_A_surf_int) * np.nan
    phi_A_surf_reps = np.ones(n_reps_A_surf_int) * np.nan
    n_interactions_A_surf_reps = np.ones(n_reps_A_surf_int)
    I_A_surf_reps = np.zeros_like(theta_A_surf_reps)
    n_passes_A_surf_reps = np.zeros_like(theta_A_surf_reps)

    theta_T_reps = np.concatenate(
        [np.tile(theta_out_T[j], n_reps_T_int[j]) for j in range(len(theta_out_T))])
    phi_T_reps = np.concatenate(
        [np.tile(phi_out_T[j], n_reps_T_int[j]) for j in range(len(phi_out_T))])
    n_interactions_T_reps = np.concatenate(
        [np.tile(final_T_n_interactions[j], n_reps_T_int[j]) for j in
         range(len(final_T_n_interactions))])
    I_T_reps = np.concatenate([np.tile(I[j], n_reps_T_int[j]) for j in range(len(I))])

    n_passes_T_reps = np.ones_like(theta_T_reps)

    theta_out = np.concatenate([theta_R_reps, theta_A_surf_reps, theta_T_reps])
    phi_out = np.concatenate([phi_R_reps, phi_A_surf_reps, phi_T_reps])
    n_interactions = np.concatenate(
        [n_interactions_R_reps, n_interactions_A_surf_reps, n_interactions_T_reps])
    I_out = np.concatenate([I_R_reps, I_A_surf_reps, I_T_reps])

    n_passes = np.concatenate(
        [n_passes_R_reps, n_passes_A_surf_reps, n_passes_T_reps])

    profile = np.sum(final_T_weights[:, None] * DA, axis=0)

    return theta_out, phi_out, I_out, n_interactions, n_passes, A_bulk_actual, profile, np.sum(A_total, axis=0)


def lambertian_scattering(strt, save_location, options):

    structpath = get_savepath(save_location, options.project_name)

    I_theta = np.cos(theta_lamb)
    I_theta = I_theta/np.sum(I_theta)

    phi = np.linspace(0, options.phi_symmetry, 40)

    # make a grid of rays with these thetas and phis

    theta_grid, phi_grid = np.meshgrid(theta_lamb, phi)
    theta_grid = theta_grid.flatten()
    phi_grid = phi_grid.flatten()

    r_a_0 = np.real(
        np.array(
            [np.sin(theta_grid) * np.cos(phi_grid), np.sin(theta_grid) * np.sin(phi_grid),
             np.cos(theta_grid)]
        )
    )

    r_a_0_rear = np.copy(r_a_0)
    r_a_0_rear[2, :] = -r_a_0_rear[2, :]

    result_list = []

    for mat_index in range(1, len(strt.widths) + 1):

        front_inside = strt.textures[mat_index - 1][1]
        rear_inside = strt.textures[mat_index][0]

        n_triangles_front = len(front_inside.P_0s)
        n_triangles_rear = len(rear_inside.P_0s)

        hit_prob_front = np.matmul(front_inside.N, r_a_0)

        theta_local_front = np.arccos(hit_prob_front)

        theta_local_front[theta_local_front > np.pi / 2] = 0

        hit_prob_rear = -np.matmul(rear_inside.N, r_a_0_rear)
        theta_local_rear = np.arccos(hit_prob_rear)

        theta_local_rear[theta_local_rear > np.pi / 2] = 0

        n_front_layers = len(strt.textures[mat_index - 1][0].interface_layers) if hasattr(strt.textures[mat_index - 1][0], 'interface_layers') else 0
        n_rear_layers = len(strt.textures[mat_index][0].interface_layers) if hasattr(strt.textures[mat_index][0], 'interface_layers') else 0

        unique_angles_front, inverse_indices_front = np.unique(theta_local_front, return_inverse=True)
        unique_angles_rear, inverse_indices_rear = np.unique(theta_local_rear, return_inverse=True)

        if n_front_layers > 0:
            lookuptable_front = xr.open_dataset(os.path.join(structpath, front_inside.name + f"int_{mat_index - 1}.nc"))

            data_front = lookuptable_front.loc[dict(side=-1, pol=options.pol)].sel(
                angle=abs(unique_angles_front), wl=options.wavelength * 1e9, method="nearest"
            ).load()
            R_front = np.real(data_front["R"].data).T
            A_per_layer_front = np.real(data_front["Alayer"].data).T
            A_all_front = A_per_layer_front[:, inverse_indices_front].reshape(
                (n_front_layers,) + theta_local_front.shape + (len(options.wavelength),))

            A_reshape_front = A_all_front.reshape(
                (n_front_layers, n_triangles_front, len(phi), len(theta_lamb), len(options.wavelength)))


        else:
            R_front = \
                calc_RAT_Fresnel_vec(unique_angles_front, options.pol,
                                     strt.mats[mat_index].n(options.wavelength),
                                     strt.mats[mat_index - 1].n(options.wavelength))[0]
            A_reshape_front = 0

        R_all_front = R_front[inverse_indices_front].reshape(
            theta_local_front.shape + (len(options.wavelength),))

        if n_rear_layers > 0:
            lookuptable_rear = xr.open_dataset(os.path.join(structpath, rear_inside.name + f"int_{mat_index}.nc"))
            data_rear = lookuptable_rear.loc[dict(side=1, pol=options.pol)].sel(
                angle=abs(unique_angles_rear), wl=options.wavelength * 1e9, method="nearest"
            )
            R_rear = np.real(data_rear["R"].data).T
            A_per_layer_rear = np.real(data_rear["Alayer"].data).T
            A_all_rear = A_per_layer_rear[:, inverse_indices_rear].reshape(
                (n_rear_layers,) + theta_local_rear.shape + (len(options.wavelength),))

            A_reshape_rear = A_all_rear.reshape(
                (n_rear_layers, n_triangles_rear, len(phi), len(theta_lamb), len(options.wavelength)))


        else:
            R_rear = \
            calc_RAT_Fresnel_vec(unique_angles_rear, options.pol, strt.mats[mat_index].n(options.wavelength),
                                 strt.mats[mat_index + 1].n(options.wavelength))[0]
            A_reshape_rear = 0

        R_all_rear = R_rear[inverse_indices_rear].reshape(
            theta_local_rear.shape + (len(options.wavelength),))

        # now populate matrix of local angles based on these probabilities

        # identify allowed angles:

        # surface normals:

        hit_prob_front[hit_prob_front < 0] = 0
        hit_prob_rear[hit_prob_rear < 0] = 0

        # calculate area of each triangle
        area_front = np.sqrt(
            np.sum(np.cross(front_inside.P_0s - front_inside.P_1s, front_inside.P_2s - front_inside.P_1s, axis=1) ** 2, 1)
            ) / 2

        area_front = area_front / max(area_front)

        hit_prob_front = area_front[:, None] * hit_prob_front / np.sum(hit_prob_front, axis=0)

        hit_prob_reshape_front = hit_prob_front.reshape((n_triangles_front, len(phi), len(theta_lamb)))
        # now take the average over all the faces and azimuthal angles
        R_reshape_front = R_all_front.reshape((n_triangles_front, len(phi), len(theta_lamb), len(options.wavelength)))

        R_weighted_front = R_reshape_front * hit_prob_reshape_front[:, :, :, None]
        R_polar_front = np.sum(np.mean(R_weighted_front, 1), 0)

        A_surf_weighted_front = A_reshape_front * hit_prob_reshape_front[None, :, :, :, None]
        A_polar_front = np.sum(np.mean(A_surf_weighted_front, 2), 1)

        area_rear = np.sqrt(
            np.sum(np.cross(rear_inside.P_0s - rear_inside.P_1s, rear_inside.P_2s - rear_inside.P_1s, axis=1) ** 2, 1)
            ) / 2

        area_rear = area_rear / max(area_rear)

        hit_prob_rear = area_rear[:, None] * hit_prob_rear / np.sum(hit_prob_rear, axis=0)

        hit_prob_reshape_rear = hit_prob_rear.reshape((n_triangles_rear, len(phi), len(theta_lamb)))
        # now take the average over all the faces and azimuthal angles
        R_reshape_rear = R_all_rear.reshape((n_triangles_rear, len(phi), len(theta_lamb), len(options.wavelength)))

        R_weighted_rear = R_reshape_rear * hit_prob_reshape_rear[:, :, :, None]
        R_polar_rear = np.sum(np.mean(R_weighted_rear, 1), 0)

        A_surf_weighted_rear = A_reshape_rear * hit_prob_reshape_rear[None, :, :, :, None]
        A_polar_rear = np.sum(np.mean(A_surf_weighted_rear, 2), 1)

        # calculate travel distance for each ray
        I_rear = I_theta[:, None] * np.exp(-strt.widths[0] * strt.mats[1].alpha(options.wavelength[None, :]) / np.cos(theta_lamb)[:, None])

        R_1 = np.sum(I_theta[:, None]*R_polar_front, axis=0)
        R_2 = np.sum(I_theta[:, None]*R_polar_rear, axis=0)

        A_1 = np.sum(I_theta[:, None]*A_polar_front, axis=1)
        A_2 = np.sum(I_theta[:, None]*A_polar_rear, axis=1)
        # total probability of absorption in bulk:

        # infinite series:

        A_bulk = 1 - np.sum(I_rear, axis=0)

        T_1 = 1 - R_1 - np.sum(A_1, axis=0)
        T_2 = 1 - R_2 - np.sum(A_2, axis=0)

        r = (1 - R_1 * R_2 * (1 - A_bulk) ** 2)
        # if starting after reflection from front:
        # P_escape_front_down = (1 - A_bulk) ** 2 * T_1 * R_2 / r
        # P_escape_back_down = (1 - A_bulk) * T_2 / r
        # P_absorb_down = (A_bulk + (1 - A_bulk) * R_2 * A_bulk) / r
        # P_front_surf_down = (1 - A_bulk) ** 2 * R_2 * A_1 / r
        # P_rear_surf_down = (1 - A_bulk) * A_2 / r
        P_escape_front_down = (1 - A_bulk) * T_1 * R_2 / r
        P_escape_back_down = T_2 / r
        P_absorb_down = R_2 * A_bulk * (1 - A_bulk * R_1 + R_1)/ r
        P_front_surf_down = (1 - A_bulk) * R_2 * A_1 / r
        P_rear_surf_down = A_2 / r

        # if starting after reflection from rear:
        P_escape_front_up = T_1 / r
        P_escape_back_up = (1 - A_bulk) * T_2 * R_1 / r
        P_absorb_up = R_1 * A_bulk * (1 - A_bulk * R_2 + R_2)/ r
        P_front_surf_up = A_1 / r
        P_rear_surf_up = (1 - A_bulk) * R_1 * A_2 / r

        initial_down = xr.DataArray(np.stack((P_escape_front_down, P_absorb_down, P_escape_back_down)),
                     dims=['event', 'wavelength'],
                     coords={'event': ['R', 'A_bulk', 'T'], 'wavelength': options.wavelength})

        initial_up = xr.DataArray(np.stack((P_escape_front_up, P_absorb_up, P_escape_back_up)),
                     dims=['event', 'wavelength'],
                     coords={'event': ['R', 'A_bulk', 'T'], 'wavelength': options.wavelength})

        # does layer order need tp be flipped?
        front_surf_P = xr.DataArray(np.stack((P_front_surf_down, P_front_surf_up)),
                        dims=['direction', 'layer', 'wavelength'],
                        coords={'direction': [1, -1], 'wavelength': options.wavelength})

        rear_surf_P = xr.DataArray(np.stack((P_rear_surf_down, P_rear_surf_up)),
                        dims=['direction', 'layer', 'wavelength'],
                        coords={'direction': [1, -1], 'wavelength': options.wavelength})


        # Add a new dimension for the initial direction
        initial_down = initial_down.expand_dims({"direction": [1]})
        initial_up = initial_up.expand_dims({"direction": [-1]})

        # Concatenate the two xarrays along the new dimension
        merged = xr.concat([initial_down, initial_up], dim="direction")

    return merged, front_surf_P, rear_surf_P, [R_1, R_2]


def calculate_lambertian_profile(strt, I_wl, options, initial_direction,
                                 lambertian_results, alphas, position):

    I_theta = np.cos(theta_lamb)
    I_theta = I_theta / np.sum(I_theta)

    profile_wl = np.zeros((len(I_wl), len(position)))

    [R_top, R_bot] = lambertian_results

    if initial_direction == 1:
        R1 = R_bot # CHECK
        R2 = R_top

    else:
        R1 = R_top
        R2 = R_bot

    for i1, I0 in enumerate(I_wl):

        I = I0
        I_angular = I * I_theta
        direction = -initial_direction # first thing that happens is reflection!
        DA = np.zeros((len(theta_lamb), len(position)))

        while I > options.I_thresh:

            # 1st surf interaction
            I_angular = I_angular * R1[i1]

            # absorption

            DA_pass, _, I_angular = traverse_vectorised(
                strt.widths[0]*1e6,
                theta_lamb,
                alphas[i1],
                I_angular,
                position,
                options.I_thresh,
                direction,
            )

            DA += DA_pass

            I_angular = I_angular * R2[i1]

            direction = -direction

            DA_pass, _, I_angular = traverse_vectorised(
                strt.widths[0],
                theta_lamb,
                alphas[i1],
                I_angular,
                position,
                options.I_thresh,
                direction,
            )
            DA += DA_pass

            direction = -direction

            I = np.sum(I_angular)

        profile_wl[i1] = np.sum(DA, axis=0)

    return profile_wl
