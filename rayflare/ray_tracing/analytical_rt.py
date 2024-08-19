import numpy as np
import xarray as xr
import os
from rayflare.utilities import get_savepath
from copy import deepcopy
from solcore.state import State

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
        pass

    def isel(self, wl):
        return State(I = 0)

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

    if np.sum(tmm_args[0]) == 0:
        # only Fresnel surfaces and tmm_args is just a list of zeroes.
        tmm_args = [0] + [len(surfaces)*[0]]
        # need tmm_args to have a second element, so that inner functions can correctly
        # identify that they need to use Fresnel and not TMM

    n_wl = len(wls)
    wls = xr.DataArray(wls, dims='wl_angle')

    current_pol = np.tile(pol, (n_wl, 1))

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

    # make xarrays with attribute I = 0:
    overall_R = xr.Dataset({"I": xr.DataArray(np.zeros((1, n_wl)), dims=["unique_direction", "wl"])})
    overall_T = xr.Dataset({"I": xr.DataArray(np.zeros((1, n_wl)), dims=["unique_direction", "wl"])})

    next_mat = initial_mat + initial_dir

    theta = theta*np.ones(n_wl)
    angles = xr.DataArray(theta, dims='wl_angle')

    single_direction = True

    d = -r_a_0 / np.linalg.norm(r_a_0)
    # TODO: check d for travelling upwards

    if initial_dir != 1:
        d[2] = -d[2]

    I_remaining = xr.DataArray(np.ones((1, n_wl)), dims=["unique_direction", "wl"])

    n_interactions = 0

    # TODO: absorbed_details should include interface absorption

    absorbed_details = [[] for _ in range(len(widths))]

    prop_rays = []

    while single_direction:

        normals = surfaces[surf_index].N
        # snell's law for direction of transmitted rays:
        n0 = nks[mat_i]
        n1 = nks[next_mat]

        I_rem_data = I_remaining.data[0]

        if np.all(np.abs(normals[:,2]) > 0.999) and d.ndim == 1:
            # if the surface is planar, can just use TMM or Fresnel equations directly
            # should already have a lookuptable, if necessary

            theta_t = np.arcsin(n0) / n1 * np.sin(angles.data)

            x = np.sin(theta_t) * np.cos(phi)
            y = np.sin(theta_t) * np.sin(phi)
            z = np.cos(theta_t)

            # make sign of x, y, z the same as d:
            x = np.abs(x) * np.sign(d[0])
            y = np.abs(y) * np.sign(d[1])
            z = np.abs(z) * np.sign(d[2])

            final_T_directions = xr.DataArray(np.real(np.stack((x, y, z)))[None, :, :],
                                              dims=["unique_direction", "xyz", "wl"])

            theta_t = xr.DataArray(theta_t[None, :], dims=["unique_direction", "wl"])

            # if d.ndim == 1:
            final_R_directions = xr.DataArray(deepcopy(d)[None, :], dims=["unique_direction", "xyz"])

            # else:
            #     final_R_directions = xr.DataArray(deepcopy(d)[None, :, :], dims=["unique_direction", "xyz", "wl"])

            # reflection: z -> -z, no other changes to ray direction
            final_R_directions[:, 2] = -final_R_directions[:, 2]

            if tmm_args[1][surf_index] == 1:
                structpath = tmm_args[2]
                surf_name = tmm_args[3][surf_index] + "int_{}".format(surf_index)

                lookuptable = xr.open_dataset(os.path.join(structpath, surf_name + ".nc")).loc[
                    dict(side=initial_dir, pol=['s', 'p'])].interp(angle=angles, wl=wls).load()

                [Rs, Rp] = lookuptable.R.data
                [Ts, Tp] = lookuptable.T.data

                A_per_int_layer = np.sum(lookuptable.Alayer.transpose("layer", "wl_angle", "pol")*current_pol, -1)

                R = np.sum(np.stack((Rs, Rp), -1) * current_pol, -1)
                T = np.sum(np.stack((Ts, Tp), -1) * current_pol, -1)

                # INTERFACE (not bulk!) absorption
                A_per_interface[surf_index] = xr.DataArray((I_rem_data[None, :]*A_per_int_layer.data)[None, :, :], dims=["unique_direction", "layer", "wl"])


            else:
                # Fresnel equations
                Rs, Rp, _, Ts, Tp = calc_RAT_Fresnel(theta, pol, n0, n1)

                R = np.sum(np.stack((Rs, Rp),-1) * current_pol, -1)

                T = 1 - R

                # no interface absorption:
                A_per_interface[surf_index] = xr.DataArray(np.zeros((1, 1, n_wl)), dims=["unique_direction", "layer", "wl"])

            R_pol = np.stack((Rs, Rp), -1) * current_pol
            R_pol = R_pol / (np.sum(R_pol, -1)[:, None])

            T_pol = np.stack((Ts, Tp), -1) * current_pol
            T_pol = T_pol / (np.sum(T_pol, -1)[:, None])

            R_total = xr.DataArray(I_remaining * R[None, :], dims=["unique_direction", "wl"])

            n_interactions += 1  # can only have one interaction with planar surface regardless of
            # angle of incidence

            R_data = xr.Dataset(
                {
                    "I": R_total,
                    "direction": final_R_directions,
                    "n_interactions": np.array([n_interactions]),
                }
            )

            T_data = xr.Dataset(
                {
                    "I": I_remaining * T,
                    "direction": final_T_directions,
                    "n_interactions": np.array([n_interactions]),
                    "theta_t": theta_t,
                }
            )

        else:
            # do analytical RT for non-planar surface with multiple faces

            R_data, A_data, T_data, R_pol, T_pol = analytical_per_face(surfaces[surf_index],
                                          surf_index,
                                          d,
                                          tmm_args,
                                          nks,
                                          initial_dir,
                                          current_pol,
                                          max_interactions,
                                          )


            A_per_interface[surf_index] = A_data*I_rem_data[None, None, :]

            R_data['n_interactions'] = R_data["n_interactions"] + n_interactions
            T_data['n_interactions'] = T_data["n_interactions"] + n_interactions

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

        # expand I_remaining along the face axis using xarray:
        # I_rem_after_int = I_remaining.data[0] # updated with interface absorption
        # DA = DA * I_rem_data[None, :, None] # scaled by intensity remaining BEFORE this surface
        # I = I * I_rem_data[None, :]

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
            absorbed_details[mat_i].append(
                xr.Dataset({
                    "A": xr.DataArray(I_out_per_direction_R, dims=["unique_direction", "wl"]),
                    "n_interactions": R_data.n_interactions,
                    "n_passes": R_data.n_interactions
                })
            )

            A_actual_R = np.sum(I_out_per_direction_R, axis=0)
            # A_bulk_actual = np.sum(T_data.I.data - I_out_actual)
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
        absorbed_details[mat_i + initial_dir].append(
            xr.Dataset({
                "A": xr.DataArray(I_out_per_direction, dims=["unique_direction", "wl"]),
                "n_interactions": T_data.n_interactions,
                "n_passes": T_data.n_interactions
            })
        )
        A_actual = np.sum(I_out_per_direction, axis=0)
        # A_bulk_actual = np.sum(T_data.I.data - I_out_actual)
        DA_actual = np.sum(T_data.I.data.T*DA, axis=2)
        # theta_out_T[stop] = np.nan
        # phi_out_T[stop] = np.nan

        surf_index += initial_dir
        mat_i += initial_dir

        A_per_layer[mat_i] += np.real(A_actual)
        profile[depth_indices[mat_i]] += np.real(+ DA_actual)
        remaining_after_bulk = np.real(T_data.I * I)

        in_structure = surf_index < len(surfaces) and surf_index >= 0

        # if all rays are still travelling in the same direction, continue with analytical RT. Otherwise continue on to
        # 'normal' ray tracing. Otherwise, or if we have check the last surface in the structure,
        # end and return results.

        if np.unique(T_data.direction, axis=0).shape[0] > 1 or not in_structure:

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

            return profile.T, A_per_layer.T, absorbed_details, A_per_interface, overall_R, overall_T, prop_rays

        else:
            # continue, but need to update inputs: transmitted rays at each wavelength become new
            # incident rays.
            angles.data = T_data.theta_t.data[0]
            theta = T_data.theta_t.data[0]
            #I_remaining.data = 0
            # losses from:
            # - interface absorption
            # - bulk absorption
            # - reflection
            # I_new = I_rem_data - np.sum(R_data.I, 0) - np.sum(A_per_layer, 0)
            I_remaining = remaining_after_bulk
            # TODO: I think this only works for downwards
            # can only reach here if surfaces so far have been planar; end up with
            # two directions because planar surface is made of two triangles, but they
            # contain the same information
            I_remaining = I_remaining.sum(dim='unique_direction').expand_dims('unique_direction')

            d = T_data.direction[0].expand_dims('unique_direction').data


            # need to construct d for each wavelength:

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.plot(wls, data_lists[0]['R_data'].R_total[0])
    # plt.plot(wls, data_lists[0]['A_data'][0])
    # plt.show()
    #
    # print('done')


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
                         current_pol,
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

        Rs_prob, Rp_prob, A_prob, Ts_prob, Tp_prob = calc_RAT(np.arccos(cos_inc), current_pol, *R_args)

        R_stack = np.stack((Rs_prob, Rp_prob), axis=-1)
        R_prob = np.sum(R_stack*current_pol, -1)

        # stack Ts_prob and Tp_prob so that the final index of the array is the new one:
        T_stack = np.stack((Ts_prob, Tp_prob), axis=-1)

        current_pol = R_stack * current_pol
        current_pol = current_pol / (np.sum(current_pol, -1)[:, :, None])

        # nor

        if np.sum(A_prob) > 0:
            A_prob_sum = np.sum(A_prob, axis=1)

        else:
            A_prob_sum = 0

        T_per_it[:, N_interaction] = 1 - R_prob - A_prob_sum

        A_per_it[:, :, N_interaction] = A_prob

        T_dir_per_it[:, N_interaction] = np.abs(
            refracted_rays[:, 2] / np.linalg.norm(refracted_rays,
                                                  axis=1))  # cos (global) of refracted ray

        T_pol_per_it[:, N_interaction] = T_stack * current_pol
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
    final_T_pol = final_T_pol / np.sum(final_T_pol, -1)[:, :, None]

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

    R_total = xr.DataArray(R_total, dims=["unique_direction", "wl"])
    final_R_directions = xr.DataArray(final_R_directions, dims=["unique_direction", "xyz", "wl"])
    n_interactions = xr.DataArray(stop_it + 1, dims=["unique_direction"])
    # theta_out_R = xr.DataArray(theta_out_R, dims=["unique_direction"])
    # phi_out_R = xr.DataArray(phi_out_R, dims=["unique_direction"])

    R_data = xr.Dataset(
        {
            "I": R_total,
            "direction": final_R_directions,
            "n_interactions": n_interactions,
        }
    )

    A_data = xr.DataArray(A_total, dims=["unique_direction", "layer", "wl"])

    final_T_weights = xr.DataArray(final_T_weights, dims=["unique_direction", "wl"])
    final_T_directions = xr.DataArray(final_T_directions, dims=["unique_direction", "xyz", "wl"])
    final_T_n_interactions = xr.DataArray(final_T_n_interactions, dims=["unique_direction"])
    theta_out_T = xr.DataArray(theta_out_T, dims=["unique_direction", "wl"])

    T_data = xr.Dataset(
        {
            "I": final_T_weights,
            "direction": final_T_directions,
            "n_interactions": final_T_n_interactions,
            "theta_t": theta_out_T,
        }
    )

    return R_data, A_data, T_data, final_R_pol, final_T_pol


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
        I_rear = I_theta[:, None] * np.exp(-strt.widths[mat_index - 1] * strt.mats[mat_index].alpha(options.wavelength[None, :]) / np.cos(theta_lamb)[:, None])

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

        result_list.append([merged, front_surf_P, rear_surf_P, [R_1, R_2]])

    # stack merged, front_surf_P and rear_surf_P arrays along a nr

    return result_list


def calculate_lambertian_profile(strt, I_wl, options, initial_direction,
                                 lambertian_R_results, alphas, position,
                                 total_A):
    def traverse_lambertian(width, theta, alpha, I_i, positions, direction):

        ratio = alpha / np.real(np.abs(np.cos(theta)))
        DA_u = I_i[:, None] * ratio[:, None] * np.exp((-ratio[:, None] * positions[None, :]))
        I_back = I_i * np.exp(-ratio * width)

        if direction == -1:
            DA_u = np.flip(DA_u)

        intgr = np.trapz(DA_u, positions, axis=1)

        DA = np.divide(
            ((I_i[:, None] - I_back[:, None]) * DA_u).T, intgr,
        ).T

        DA[intgr == 0] = 0

        return DA, I_back


    I_theta = np.cos(theta_lamb)
    I_theta = I_theta / np.sum(I_theta)

    profile_wl = np.zeros((len(I_wl), len(position)))

    [R_top, R_bot] = lambertian_R_results

    if initial_direction == 1:
        R1 = R_bot # CHECK
        R2 = R_top

    else:
        R1 = R_top
        R2 = R_bot

    cont_wl_ind = np.where(I_wl > options.I_thresh)[0]

    for i1 in cont_wl_ind:

        I = I_wl[i1]
        I_angular = I * I_theta
        direction = -initial_direction # first thing that happens is reflection!
        DA = np.zeros((len(theta_lamb), len(position)))

        while I > options.I_thresh:

            # 1st surf interaction
            I_angular = I_angular * R1[i1]

            # absorption

            DA_pass, I_angular = traverse_lambertian(
                strt.widths[0]*1e6,
                theta_lamb,
                alphas[i1],
                I_angular,
                position,
                direction,
            )

            DA += DA_pass

            I_angular = I_angular * R2[i1]

            direction = -direction

            DA_pass, I_angular = traverse_lambertian(
                strt.widths[0],
                theta_lamb,
                alphas[i1],
                I_angular,
                position,
                direction,
            )

            DA += DA_pass

            direction = -direction

            I = np.sum(I_angular)

        sum_over_angles = np.sum(DA, axis=0)
        int_I = np.trapz(sum_over_angles, position)
        profile_wl[i1] = (total_A[i1]/int_I)*np.sum(DA, axis=0)

    return profile_wl
