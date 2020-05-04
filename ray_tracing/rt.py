import numpy as np
import numpy.matlib
import os
from scipy.spatial import Delaunay
from cmath import sin, cos, sqrt, acos, atan
from math import atan2
from random import random
from itertools import product
import xarray as xr
from angles import fold_phi, make_angle_vector
from sparse import COO, save_npz, load_npz, stack
from config import results_path
from joblib import Parallel, delayed
from time import time
from copy import deepcopy


def RT(group, incidence, transmission, surf_name, options, Fr_or_TMM = 0, front_or_rear = 'front',
       n_absorbing_layers=0, calc_profile=[], only_incidence_angle=False, widths=[], save=True):
    """Calculates the reflection/transmission and absorption redistribution matrices for an interface using
    either a previously calculated TMM lookup table or the Fresnel equations.
    :param group: an RTgroup object containing the surface textures
    :param incidence: incidence medium
    :param: transmission: transmission medium
    :param surf_name: name of the surface (to save matrices)
    :param options: dictionary of options
    :param Fr_or_TMM: whether to use the Fresnel equations (0) or a TMM lookup table (1)
    :param front_or_rear: whether light is incident from the front or rear
    :param for a structure with multiple interface layers, where a TMM lookuptable is used, the number of layers in
    the interface
    :param calc_profile: whether to save the relevant information to calculate the depth-dependent absorption
    profile
    :param only_incidence_angle: if True, the ray-tracing will only be performed for the incidence theta and phi
    specified in the options.
    :return out_mat: the R/T redistribution matrix at each wavelength, indexed as (wavelength, angle_bin_out, angle_bin_in)
    :return A_mat: the absorption redistribution matrix (total absorption per layer), indexed as (wavelength, layer_out, angle_bin_in)
    :return local_angle_mat: only if calc_profile = True. A matrix storing the local incidence angles for rays which were absorbed.
    This is used to calculate absorption profiles using TMM.
    """

    if Fr_or_TMM > 0 or save:
        structpath = os.path.join(results_path, options['project_name']) # also need this to get lookup table
    if save:

        if not os.path.isdir(structpath):
            os.mkdir(structpath)

        savepath_RT = os.path.join(structpath, surf_name + front_or_rear + 'RT.npz')
        savepath_A = os.path.join(structpath, surf_name + front_or_rear + 'A.npz')
        prof_mat_path = os.path.join(results_path, options['project_name'],
                                     surf_name + front_or_rear + 'profmat.nc')

        if Fr_or_TMM > 0:
            savepath_prof = os.path.join(structpath, surf_name + front_or_rear + 'Aprof.npz')

    if os.path.isfile(savepath_RT) and save:
        print('Existing angular redistribution matrices found')
        allArrays = load_npz(savepath_RT)
        absArrays = load_npz(savepath_A)
        if os.path.isfile(savepath_prof):
            local_angles = load_npz(savepath_prof)

            if os.path.isfile(prof_mat_path):
                prof_int = xr.load_dataset(prof_mat_path)
                profile = prof_int['profile']
                intgr = prof_int['intgr']
                return allArrays, absArrays, local_angles, profile, intgr

            else:
                return allArrays, absArrays, local_angles


        else:
            return allArrays, absArrays

    else:
        wavelengths = options['wavelengths']
        n_rays = options['n_rays']
        nx = options['nx']
        ny = options['ny']
        n_angles = int(np.ceil(n_rays/(nx*ny)))

        phi_sym = options['phi_symmetry']
        n_theta_bins = options['n_theta_bins']
        c_az = options['c_azimuth']
        pol = options['pol']
        nm_spacing = options['nm_spacing']

        if front_or_rear == 'front':
            side = 1
        else:
            side = -1

        if Fr_or_TMM == 1:
            lookuptable = xr.open_dataset(os.path.join(structpath, surf_name + '.nc'))
            if front_or_rear == 'rear':
                lookuptable = lookuptable.assign_coords(side=np.flip(lookuptable.side))
        else:
            lookuptable = None

        theta_intv, phi_intv, angle_vector = make_angle_vector(n_theta_bins, phi_sym, c_az)

        if only_incidence_angle:
            print('Calculating matrix only for incidence theta/phi')
            if options['theta_in'] == 0:
                th_in = 0.0001
            else:
                th_in = options['theta_in']

            angles_in = angle_vector[:int(len(angle_vector) / 2), :]
            n_reps = int(np.ceil(n_angles / len(angles_in)))
            thetas_in = np.tile(th_in, n_reps)
            print('only inc angle' , thetas_in)
            n_angles = n_reps

            if options['phi_in'] == 'all':
                # get relevant phis
                phis_in = np.tile(options['phi_in'], n_reps)
            else:
                if options['phi_in'] == 0:
                    phis_in = np.tile(0.0001, n_reps)

                else:
                    phis_in = np.tile(options['phi_in'], n_reps)

        else:
            if options['random_angles']:
                thetas_in = np.random.random(n_angles)*np.pi/2
                phis_in = np.random.random(n_angles)*2*np.pi
            else:
                angles_in = angle_vector[:int(len(angle_vector)/2),:]
                n_reps = int(np.ceil(n_angles/len(angles_in)))
                thetas_in = np.tile(angles_in[:,1], n_reps)[:n_angles]
                phis_in = np.tile(angles_in[:,2], n_reps)[:n_angles]

        if front_or_rear == 'front':
            mats = [incidence]
        else:
            mats = [transmission]

        for i1 in range(len(group.materials)):
            mats.append(group.materials[i1])

        if front_or_rear == 'front':
            mats.append(transmission)
        else:
            mats.append(incidence)

        # list of lists: first in tuple is front incidence
        if front_or_rear == 'front':
            surfaces = [x[0] for x in group.textures]

        else:
            surfaces = [x[1] for x in group.textures]

        nks = np.empty((len(mats), len(wavelengths)), dtype=complex)

        for i1, mat in enumerate(mats):
            nks[i1] = mat.n(wavelengths) + 1j*mat.k(wavelengths)

        h = max(surfaces[0].Points[:, 2])
        x_lim = surfaces[0].Lx
        y_lim = surfaces[0].Ly

        xs = np.linspace(x_lim/100, x_lim-(x_lim/100), nx)
        ys = np.linspace(y_lim/100, y_lim-(y_lim/100), ny)

        if options['parallel']:
            allres = Parallel(n_jobs=options['n_jobs'])(delayed(RT_wl)
                                           (i1, wavelengths[i1], n_angles, nx, ny,
                                            widths, thetas_in, phis_in, h,
                                            xs, ys, nks, surfaces,
                                            pol, phi_sym, theta_intv,
                                            phi_intv, angle_vector, Fr_or_TMM, n_absorbing_layers,
                                            lookuptable, calc_profile, nm_spacing, side)
                                       for i1 in range(len(wavelengths)))

        else:
            allres = [RT_wl(i1, wavelengths[i1], n_angles, nx, ny, widths,
                            thetas_in, phis_in, h, xs, ys, nks, surfaces,
                                     pol, phi_sym, theta_intv, phi_intv,
                            angle_vector, Fr_or_TMM, n_absorbing_layers, lookuptable, calc_profile, nm_spacing, side)
                      for i1 in range(len(wavelengths))]

        allArrays = stack([item[0] for item in allres])
        absArrays = stack([item[1] for item in allres])

        if save:
            save_npz(savepath_RT, allArrays)
            save_npz(savepath_A, absArrays)

        if Fr_or_TMM > 0:
            local_angles = stack([item[2] for item in allres])
            if save:
                save_npz(savepath_prof, local_angles)
            #make_profile_data(options, np.unique(angle_vector[:,1]), int(len(angle_vector) / 2),
            #                  front_or_rear, surf_name, n_absorbing_layers, widths)

            if len(calc_profile) > 0:
                profile = xr.concat([item[3] for item in allres], 'wl')
                intgr = xr.concat([item[4] for item in allres], 'wl')
                intgr.name = 'intgr'
                profile.name = 'profile'
                allres = xr.merge([intgr, profile])

                if save:
                    allres.to_netcdf(prof_mat_path)
                    save_npz(savepath_prof, local_angles)

                return allArrays, absArrays, local_angles, profile, intgr

            else:
                return allArrays, absArrays, local_angles

        else:
            return allArrays, absArrays


def RT_wl(i1, wl, n_angles, nx, ny, widths, thetas_in, phis_in, h, xs, ys, nks, surfaces,
          pol, phi_sym, theta_intv, phi_intv, angle_vector, Fr_or_TMM, n_abs_layers, lookuptable, calc_profile, nm_spacing, side):
    print('wavelength = ', wl)

    theta_out = np.zeros((n_angles, nx * ny))
    phi_out = np.zeros((n_angles, nx * ny))
    A_surface_layers = np.zeros((n_angles, nx*ny, n_abs_layers))
    theta_local_incidence = np.zeros((n_angles, nx*ny))

    for i2 in range(n_angles):

        theta = thetas_in[i2]
        phi = phis_in[i2]
        r = abs((h + 1) / cos(theta))
        r_a_0 = np.real(np.array([r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)]))
        for c, vals in enumerate(product(xs, ys)):
            I, th_o, phi_o, surface_A = \
                single_ray_interface(vals[0], vals[1], nks[:, i1],
                           r_a_0, theta, phi, surfaces, pol, wl, Fr_or_TMM, lookuptable)

            if th_o < 0: # can do outside loup with np.where
                th_o = -th_o
                phi_o = phi_o + np.pi
            theta_out[i2, c] = th_o
            phi_out[i2, c] = phi_o
            A_surface_layers[i2, c] = surface_A[0]
            theta_local_incidence[i2, c] = np.real(surface_A[1])


    #phi_out[theta_out < 0] = phi_out + np.pi
    #theta_out = abs(theta_out) # discards info about phi!
    phi_out = fold_phi(phi_out, phi_sym)
    phis_in = fold_phi(phis_in, phi_sym)

    if side == -1:
        not_absorbed = np.where(theta_out < (np.pi+0.1))
        thetas_in = np.pi-thetas_in
        #phis_in = np.pi-phis_in # unsure about this part

        theta_out[not_absorbed] = np.pi-theta_out[not_absorbed]
        #phi_out = np.pi-phi_out # unsure about this part

    #phi_out = fold_phi(phi_out, phi_sym)
    #phis_in = fold_phi(phis_in, phi_sym)

    #theta_out = abs(theta_out) # discards info about phi!



    theta_local_incidence = np.abs(theta_local_incidence)
    n_thetas = len(theta_intv) - 1

    if Fr_or_TMM > 0:
        # now we need to make bins for the absorption
        theta_intv = np.append(theta_intv, 11)
        phi_intv = phi_intv + [np.array([0])]

    # xarray: can use coordinates in calculations using apply!
    binned_theta_in = np.digitize(thetas_in, theta_intv, right=True) - 1

    binned_theta_out = np.digitize(theta_out, theta_intv, right=True) - 1
    #print(binned_theta_out, theta_out, theta_intv)


    #print(binned_theta_in)
    #print(binned_theta_out)
    # -1 to give the correct index for the bins in phi_intv

    phi_in = xr.DataArray(phis_in,
                          coords={'theta_bin': (['angle_in'], binned_theta_in)},
                          dims=['angle_in'])

    bin_in = phi_in.groupby('theta_bin').apply(overall_bin,
                                               args=(phi_intv, angle_vector[:, 0])).data

    phi_out = xr.DataArray(phi_out,
                           coords={'theta_bin': (['angle_in', 'position'], binned_theta_out)},
                           dims=['angle_in', 'position'])

    bin_out = phi_out.groupby('theta_bin').apply(overall_bin,
                                                 args=(phi_intv, angle_vector[:, 0])).data


    out_mat = np.zeros((len(angle_vector), int(len(angle_vector) / 2)))
    # everything is coming in from above so we don't need 90 -> 180 in incoming bins
    A_mat = np.zeros((n_abs_layers, int(len(angle_vector)/2)))

    n_rays_in_bin = np.zeros(int(len(angle_vector) / 2))
    n_rays_in_bin_abs = np.zeros(int(len(angle_vector) / 2))

    binned_local_angles = np.digitize(theta_local_incidence, theta_intv, right=True) - 1
    local_angle_mat = np.zeros((int((len(theta_intv) -1 )/ 2), int(len(angle_vector) / 2)))

    if side == 1:
        offset = 0
    else:
        offset = int(len(angle_vector) / 2)

    for l1 in range(len(thetas_in)):
        for l2 in range(nx * ny):
            n_rays_in_bin[bin_in[l1]-offset] += 1
            if binned_theta_out[l1, l2] <= (n_thetas-1):
                # reflected or transmitted
                out_mat[bin_out[l1, l2], bin_in[l1]-offset] += 1
                #print('RT bin in-offset', bin_in[l1]-offset)
                #print(thetas_in[l1], binned_theta_out[l1, l2])

            else:
                # absorbed in one of the surface layers
                n_rays_in_bin_abs[bin_in[l1]-offset] += 1
                #print('A bin in', bin_in[l1]-offset, l1, l2)
                per_layer = A_surface_layers[l1, l2]
                A_mat[:, bin_in[l1]-offset] += per_layer
                local_angle_mat[binned_local_angles[l1, l2], bin_in[l1]-offset] += 1

    # normalize
    out_mat = out_mat/n_rays_in_bin
    overall_abs_frac = n_rays_in_bin_abs/n_rays_in_bin
    abs_scale = overall_abs_frac/np.sum(A_mat, 0)
    #print('A_mat', np.sum(A_mat, 0)/n_rays_in_bin_abs)
    intgr = np.sum(A_mat, 0)/n_rays_in_bin_abs
    A_mat = abs_scale*A_mat
    out_mat[np.isnan(out_mat)] = 0
    A_mat[np.isnan(A_mat)] = 0

    out_mat = COO(out_mat)  # sparse matrix
    A_mat = COO(A_mat)

    if Fr_or_TMM > 0:
        local_angle_mat = local_angle_mat/np.sum(local_angle_mat, 0)
        local_angle_mat[np.isnan(local_angle_mat)] = 0
        local_angle_mat = COO(local_angle_mat)

        #print(calc_profile)

        if len(calc_profile) > 0:
            n_a_in = int(len(angle_vector)/2)
            thetas = angle_vector[:n_a_in, 1]
            unique_thetas = np.unique(thetas)

            #from profiles need: project name, pol, nm_spacing
            #print(local_angle_mat.todense())
            #print('going into making profiles')

            profile = make_profiles_wl(unique_thetas, n_a_in, side, widths,
                         local_angle_mat, wl, lookuptable, pol, nm_spacing, calc_profile)

            intgr = xr.DataArray(intgr, dims=['global_index'],
                                 coords={'global_index': np.arange(0, n_a_in)}).fillna(0)

            return out_mat, A_mat, local_angle_mat, profile, intgr

        else:
            return out_mat, A_mat, local_angle_mat

    else:
        return out_mat, A_mat

class rt_structure:
    def __init__(self, textures, materials, widths, incidence, transmission):

        self.textures = textures
        self.widths = widths

        mats = [incidence]
        for i1 in range(len(materials)):
            mats.append(materials[i1])
        mats.append(transmission)

        self.mats = mats

        surfs_no_offset = [x[0] for x in textures]

        cum_width = np.cumsum([0] + widths) * 1e6

        surfaces = []

        for i1, text in enumerate(surfs_no_offset):
            points_loop = deepcopy(text.Points)
            points_loop[:, 2] = points_loop[:, 2] - cum_width[i1]
            surfaces.append(RTSurface(points_loop))

        self.surfaces = surfaces
        self.surfs_no_offset= surfs_no_offset
        self.cum_width = cum_width

    def calculate(self, options):
        wavelengths = options['wavelengths']
        theta = options['theta']
        phi = options['phi']
        I_thresh = options['I_thresh']
    
        widths =  self.widths
        widths.insert(0, 0)
        widths.append(0)
        widths = 1e6*np.array(widths)  # convert to um
    
        z_space = 1e6*options['depth_spacing']
        z_pos = np.arange(0, sum(widths), z_space)

        mats = self.mats
        surfaces = self.surfaces
    
        nks = np.empty((len(mats), len(wavelengths)), dtype=complex)
        alphas = np.empty((len(mats), len(wavelengths)), dtype=complex)
        R = np.zeros(len(wavelengths))
        T = np.zeros(len(wavelengths))
    
        absorption_profiles = np.zeros((len(wavelengths), len(z_pos)))
        A_layer = np.zeros((len(wavelengths), len(widths)))
    
        for i1, mat in enumerate(mats):
            nks[i1] = mat.n(wavelengths) + 1j*mat.k(wavelengths)
            alphas[i1] = mat.k(wavelengths)*4*np.pi/(wavelengths*1e6)
    
        h = max(surfaces[0].Points[:, 2])
        r = abs((h + 1) / cos(theta))
        r_a_0 = np.real(np.array([r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)]))
    
        x_lim = surfaces[0].Lx
        y_lim = surfaces[0].Ly
    
        nx = options['nx']
        ny = options['ny']
    
        xs = np.linspace(x_lim/100, x_lim-(x_lim/100), nx)
        ys = np.linspace(y_lim/100, y_lim-(y_lim/100), ny)
    
        # need to calculate r_a and r_b
        # a total of n_rays will be traced; this is divided by the number of x and y points to scan so we know
        # how many times we need to repeat
        n_reps = np.int(np.ceil(options['n_rays']/(nx*ny)))

        # thetas and phis divided into
        thetas = np.zeros((n_reps*nx*ny, len(wavelengths)))
        phis = np.zeros((n_reps*nx*ny, len(wavelengths)))

        Is = np.zeros((n_reps*nx*ny, len(wavelengths)))
    
        pol = options['pol']

        if not options['parallel']:
            for j1 in range(n_reps):

                offset = j1*nx*ny
                #print(offset, n_reps)
                for c, vals in enumerate(product(xs, ys)):

                    for i1, wl in enumerate(wavelengths):
                        I, profile, A_per_layer, th_o, phi_o = single_ray_stack(vals[0], vals[1], nks[:, i1],
                                                                                          alphas[:, i1], r_a_0, theta, phi,
                                                                                          surfaces, widths, z_pos, I_thresh, pol)
                        absorption_profiles[i1] = absorption_profiles[i1] + profile/(n_reps*nx*ny)
                        thetas[c+offset, i1] = th_o
                        phis[c+offset, i1] = phi_o
                        A_layer[i1] = A_layer[i1] + A_per_layer/(n_reps*nx*ny)
                        if th_o is not None:
                            if np.real(th_o) < np.pi/2:
                                R[i1] = np.real(R[i1] + I/(n_reps*nx*ny))
                            else:
                                T[i1] = np.real(T[i1] + I/(n_reps*nx*ny))

            return {'R': R, 'T': T, 'A_per_layer': A_layer[:, 1:-1], 'profile': absorption_profiles/1e3,
                    'thetas': thetas, 'phis': phis}


        else:

            allres = Parallel(n_jobs=-1)(delayed(parallel_inner)(nks[:, i1], alphas[:, i1], r_a_0, theta, phi,
                                                                  surfaces, widths, z_pos, I_thresh, pol, nx, ny, n_reps, xs, ys) for
                                        i1 in range(len(wavelengths)))

            #allres = [parallel_inner(nks[:, i1], alphas[:, i1], r_a_0, theta, phi,
            #                         surfaces, widths, z_pos, I_thresh, pol, nx, ny, n_reps, xs, ys) for
            #          i1 in range(len(wavelengths))]

            I = np.stack(item[0] for item in allres)
            absorption_profiles = np.stack([item[1] for item in allres])
            A_layer = np.stack([item[2] for item in allres])
            thetas = np.stack([item[3] for item in allres])
            phis = np.stack([item[4] for item in allres])

            #                absorption_profiles[c+offset, :] = absorption_profiles_c

            # thetas[c + offset, :] = thetas_c

            # phis[c + offset, :] = phis_c

            # Is[c + offset, :] = I_c

            # A_layer = A_layer + A_layer_c / (nx * ny * n_reps)

            non_abs = ~np.isnan(thetas)
            refl = np.logical_and(non_abs, np.real(thetas) < np.pi / 2)
            trns = np.logical_and(non_abs, np.real(thetas) > np.pi / 2)
            R = np.real(I * refl).T / (n_reps*nx * ny)
            T = np.real(I * trns).T / (n_reps*nx * ny)
            R = np.sum(R, 0)
            T = np.sum(T, 0)

            return {'R': R, 'T': T, 'A_per_layer': A_layer[:, 1:-1], 'profile': absorption_profiles/1e3,
                    'thetas': thetas, 'phis': phis}
    #
    #             # TODO: maybe faster to put wavelength loop outside and then parallelize everything inside that - less overhead, more happening in each loop
    #             for j1 in range(n_reps):
    #
    #                 offset = j1 * nx * ny
    #                 #print(offset, n_reps)
    #                 for c, vals in enumerate(product(xs, ys)):
    #
    #
    #                     allres = Parallel(n_jobs=-1)(delayed(single_ray_stack)(vals[0], vals[1], nks[:, i1],
    #                                                                             alphas[:, i1], r_a_0, theta, phi,
    #                                                                             surfaces, widths, z_pos, I_thresh, pol) for
    #                               i1 in range(len(wavelengths)))
    #
    #                     I_c = np.stack(item[0] for item in allres)
    #                     absorption_profiles_c = np.stack([item[1] for item in allres])
    #                     A_layer_c = np.stack([item[2] for item in allres])
    #                     thetas_c = np.stack([item[3] for item in allres])
    #                     phis_c = np.stack([item[4] for item in allres])
    #
    # #                absorption_profiles[c+offset, :] = absorption_profiles_c
    #                     thetas[c+offset, :] = thetas_c
    #                     phis[c+offset, :] = phis_c
    #                     Is[c+offset, :] = I_c
    #                     A_layer = A_layer + A_layer_c/(nx*ny*n_reps)
    #
    #             non_abs = ~np.isnan(thetas)
    #
    #             refl = np.logical_and(non_abs, np.real(thetas) < np.pi/2)
    #             trns = np.logical_and(non_abs, np.real(thetas) > np.pi / 2)
    #
    #             R = np.real(Is*refl)/(nx*ny*n_reps)
    #             T = np.real(Is*trns)/(nx*ny*n_reps)
    #
    #             R = np.sum(R, 0)
    #             T = np.sum(T, 0)
    #
    #
    #             return {'A_per_layer': A_layer[:, 1:-1], 'absorption_profiles': absorption_profiles,
    #                     'thetas': thetas, 'phis': phis_c, 'fullres': allres, 'Is': Is,
    #                     'R': R, 'T': T}



def parallel_inner(nks, alphas, r_a_0, theta, phi, surfaces, widths, z_pos, I_thresh, pol, nx, ny, n_reps, xs, ys):
    # thetas and phis divided into
    thetas = np.zeros(n_reps * nx * ny)
    phis = np.zeros(n_reps * nx * ny)
    absorption_profiles = np.zeros(len(z_pos))
    A_layer = np.zeros(len(widths))
    Is = np.zeros(n_reps*nx*ny)

    profiles = np.zeros(len(z_pos))

    for j1 in range(n_reps):
        offset = j1 * nx * ny
        # print(offset, n_reps)
        for c, vals in enumerate(product(xs, ys)):
            I, profile, A_per_layer, th_o, phi_o = single_ray_stack(vals[0], vals[1], nks, alphas, r_a_0, theta, phi,
            surfaces, widths, z_pos, I_thresh, pol)

            phi_o[th_o < 0] = phi_o + np.pi
            th_o = abs(th_o)

            profiles = profiles + profile/(n_reps*nx*ny)
            thetas[c+offset] = th_o
            phis[c+offset] = phi_o
            Is[c+offset] = I
            A_layer = A_layer+A_per_layer/(n_reps*nx*ny)

    return Is, profiles, A_layer, thetas, phis








def normalize(x):
    if sum(x > 0):
        x = x/sum(x) - x.coords['A']
    return x

def overall_bin(x, phi_intv, angle_vector_0):
    phi_ind = np.digitize(x, phi_intv[x.coords['theta_bin'].data[0]], right=True) - 1
    bin = np.argmin(abs(angle_vector_0 - x.coords['theta_bin'].data[0])) + phi_ind
    return bin

def make_profiles_wl(unique_thetas, n_a_in, side, widths,
                     angle_distmat, wl, lookuptable, pol, nm_spacing, prof_layers):

    def profile_per_layer(xx, z, offset, side, non_zero):
        layer_index = xx.coords['layer'].item(0) - 1
        x = xx[non_zero]
        part1 = x[:,0] * np.exp(x[:,4] * z[layer_index])
        part2 = x[:,1] * np.exp(-x[:,4] * z[layer_index])
        part3 = (x[:,2] + 1j * x[:,3]) * np.exp(1j * x[:,5] * z[layer_index])
        part4 = (x[:,2] - 1j * x[:,3]) * np.exp(-1j * x[:,5] * z[layer_index])
        result = np.real(part1 + part2 + part3 + part4)
        if side == -1:
            result = np.flip(result, 1)
        return result.reduce(np.sum, axis=0).assign_coords(dim_0=z[layer_index] + offset[layer_index])

    def profile_per_angle(x, z, offset, side, nz):
        i2 = x.coords['global_index'].item(0)
        non_zero=np.where(nz[:, i2])[0]
        by_layer = x.groupby('layer').apply(profile_per_layer, z=z, offset=offset, side=side, non_zero=non_zero)
        return by_layer

    #def scaled_profile(x, z, offset, side):
    #    print('wl1')
    #    xloc = x.loc[dict(coeff='A1')].reduce(np.sum,'layer')
    #    nz = xloc != 0
    #    by_angle = x.groupby('global_index').apply(profile_per_angle, z=z, offset=offset, side=side, nz=nz)
    #    return by_angle

    def scale_func(x, scale_params):
        return x.data[:, None, None] * scale_params

    def select_func(x, const_params):
        return (x.data[:, None, None] != 0) * const_params

    #num_wl = len(options['wavelengths'])

    pr = xr.DataArray(angle_distmat.todense(), dims=['local_theta', 'global_index'],
                      coords={'local_theta': unique_thetas, 'global_index': np.arange(0, n_a_in)})
    #lookuptable layers are 1-indexed
    data = lookuptable.loc[dict(side=side, pol=pol)].interp(angle=pr.coords['local_theta'], wl = wl*1e9)

    params = data['Aprof'].drop(['layer', 'side', 'angle', 'pol']).transpose('local_theta', 'layer', 'coeff')

    s_params = params.loc[dict(coeff=['A1', 'A2', 'A3_r',
                    'A3_i'])]  # have to scale these to make sure integrated absorption is correct
    c_params = params.loc[dict(coeff=['a1', 'a3'])]  # these should not be scaled

    scale_res = pr.groupby('global_index').apply(scale_func, scale_params=s_params)
    const_res = pr.groupby('global_index').apply(select_func, const_params=c_params)


    params = xr.concat((scale_res, const_res), dim='coeff').assign_coords(layer=np.arange(1, len(widths)+1))
    params = params.transpose('local_theta', 'global_index', 'layer', 'coeff')

    z_list = []

    for l_w in widths:
        z_list.append(xr.DataArray(np.arange(0, l_w, nm_spacing)))

    offsets = np.cumsum([0] + widths)[:-1]
    #start = time()

    xloc = params.loc[dict(coeff='A1')].reduce(np.sum, 'layer')
    nz = xloc != 0

    ans = params.loc[dict(layer=prof_layers)].groupby('global_index').apply(profile_per_angle,
                                                                            z=z_list, offset=offsets, side=side, nz=nz).drop('coeff')
    ans = ans.fillna(0)

    #print('Took ' + str(time() - start) + ' seconds')

    profile = ans.reduce(np.sum, 'layer')
    # start = time()
    # # there is no reason to do this again; this should be the same as the absorption per
    # # layer before scaling (currently not saved)
    # intgr = xr.DataArray(np.zeros((len(widths), len(params.global_index))),
    #                      dims=['layer', 'global_index'],
    #                      coords={'global_index': params.global_index})
    #
    # for i2, width in enumerate(widths):
    #     A1 = params.loc[dict(coeff='A1', layer=i2+1)]
    #     A2 = params.loc[dict(coeff='A2', layer=i2+1)]
    #     A3_r = params.loc[dict(coeff='A3_r', layer=i2+1)]
    #     A3_i = params.loc[dict(coeff='A3_i', layer=i2+1)]
    #     a1 = params.loc[dict(coeff='a1', layer=i2+1)]
    #     a3 = params.loc[dict(coeff='a3', layer=i2+1)]
    #
    #     intgr_width = ((A1 / a1) * (np.exp(a1 * width) - 1) - (A2 / a1) * (np.exp(-a1 * width) - 1) - \
    #                    1j * ((A3_r + 1j * A3_i) / a3) * (np.exp(1j * a3 * width) - 1) + 1j * (
    #                            (A3_r - 1j * A3_i) / a3) * (
    #                            np.exp(-1j * a3 * width) - 1)).fillna(0)
    #
    #     intgr[i2] = intgr_width.reduce(np.sum, 'local_theta')
    #
    # intgr = intgr.reduce(np.sum, 'layer')
    # print('Took ' + str(time() - start) + ' seconds')

    return profile


class RTSurface:
    def __init__(self, Points):

        tri = Delaunay(Points[:, [0, 1]])
        self.simplices = tri.simplices
        self.Points = Points
        self.P_0s = Points[tri.simplices[:, 0]]
        self.P_1s = Points[tri.simplices[:, 1]]
        self.P_2s = Points[tri.simplices[:, 2]]
        self.crossP = np.cross(self.P_1s - self.P_0s, self.P_2s - self.P_0s)
        self.size = self.P_0s.shape[0]
        self.Lx = abs(min(Points[:, 0])-max(Points[:, 0]))
        self.Ly = abs(min(Points[:, 1])-max(Points[:, 1]))
        self.z_min = min(Points[:, 2])
        self.z_max = max(Points[:, 2])

        self.find_area()

        self.zcov= Points[:,2][np.all(np.array([Points[:,0] == min(Points[:,0]), Points[:,1] == min(Points[:,1])]), axis = 0)]


    def find_area(self):
        xyz = np.stack((self.P_0s, self.P_1s, self.P_2s))
        cos_theta = np.sum((xyz[0] - xyz[1])*(xyz[2] - xyz[1]), 1)

        theta = np.arccos(cos_theta)
        self.area = np.sum((0.5*np.linalg.norm(xyz[0] - xyz[1], axis =1)*np.linalg.norm(xyz[2] - xyz[1], axis=1)*np.sin(theta)))/(self.Lx*self.Ly)



def calc_R(n1, n2, theta, pol):
    theta_t = np.arcsin((n1/n2)*np.sin(theta))
    Rs = np.abs((n1*np.cos(theta)-n2*np.cos(theta_t))/(n1*np.cos(theta)+n2*np.cos(theta_t)))**2
    Rp = np.abs((n1 * np.cos(theta_t) - n2 * np.cos(theta)) / (n1 * np.cos(theta_t) + n2 * np.cos(theta)))**2
    if pol == 's':
        return Rs
    if pol == 'p':
        return Rp
    else:
        return (Rs+Rp)/2

def exit_side(r_a, d, Lx, Ly):
    n = np.array([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]])    # surface normals: top, right, bottom, left
    p_0 = np.array([[0, Ly, 0], [Lx, 0, 0], [0, 0, 0], [0, 0, 0]])  # points on each plane
    t = np.sum((p_0 - r_a) * n, axis=1) / np.sum(d * n, axis=1)     # r_intersect = r_a + t*d
    which_intersect = t > 0                                         # only want intersections of forward-travelling ray
    t[~which_intersect] = float('inf')                              # set others to inf to avoid finding when doing min
    which_side = np.argmin(t)                                       # find closest plane

    return which_side, t[which_side]


def calc_angle(x):

    v1 = np.array([0, 1])
    return np.math.atan2(np.linalg.det([x, v1]), np.dot(x, v1))  # - 180 to 180

def single_ray_stack(x, y,  nks, alphas, r_a_0, theta, phi, surfaces, widths, z_pos, I_thresh, pol):
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
    # do everything in microns
    A_per_layer = np.zeros(len(widths))

    direction = 1   # start travelling downwards; 1 = down, -1 = up
    mat_index = 0   # start in first medium
    surf_index = 0
    stop = False
    I = 1

    r_a = r_a_0 + np.array([x, y, 0])
    r_b = np.array([x, y, 0])           # set r_a and r_b so that ray has correct angle & intersects with first surface

    d = (r_b - r_a) / np.linalg.norm(r_b - r_a) # direction (unit vector) of ray
    n_passes = 0

    depths = []
    depth_indices = []
    for i1 in range(len(widths)):
        depth_indices.append((z_pos < np.cumsum(widths)[i1]) & (z_pos >= np.cumsum(widths)[i1 - 1]))
        depths.append(z_pos[depth_indices[i1]] - \
             np.cumsum(widths)[i1 - 1])


    while not stop:

        surf = surfaces[surf_index]

        r_a[0] = r_a[0]-surf.Lx*((r_a[0]+d[0]*(surf.zcov-r_a[2])/d[2])//surf.Lx)
        r_a[1] = r_a[1]-surf.Ly*((r_a[1]+d[1]*(surf.zcov-r_a[2])/d[2])//surf.Ly)

        if direction == 1:
            ni = nks[mat_index]
            nj = nks[mat_index + 1]

        else:
            ni = nks[mat_index-1]
            nj = nks[mat_index]

        res, theta, phi, r_a, d, _ = single_interface_check(r_a, d, ni,
                                     nj, surf, surf.Lx, surf.Ly, direction,
                                     surf.zcov, pol)

        if res == 0:  # reflection
            direction = -direction  # changing direction due to reflection

            # staying in the same material, so mat_index does not change, but surf_index does
            surf_index = surf_index + direction

        if res == 1:  # transmission
            surf_index = surf_index + direction
            mat_index = mat_index + direction  # is this right?


        I_b = I
        DA, stop, I, theta = traverse(widths[mat_index], theta, alphas[mat_index], x, y, I,
                                      depths[mat_index], I_thresh, direction)
        A_per_layer[mat_index] = np.real(A_per_layer[mat_index] + I_b - I)
        profile[depth_indices[mat_index]] = np.real(profile[depth_indices[mat_index]] + DA)

        n_passes = n_passes + 1

        if direction == 1 and mat_index == (len(widths) - 1):
            stop = True
            # have ended with transmission

        elif direction == -1 and mat_index == 0:
            stop = True
            # have ended with reflection

    return I, profile, A_per_layer, theta, phi

def single_ray_interface(x, y,  nks, r_a_0, theta, phi, surfaces, pol, wl, Fr_or_TMM, lookuptable):

    direction = 1   # start travelling downwards; 1 = down, -1 = up
    mat_index = 0   # start in first medium
    surf_index = 0
    stop = False
    I = 1

    # could be done before to avoid recalculating every time
    r_a = r_a_0 + np.array([x, y, 0])
    r_b = np.array([x, y, 0])           # set r_a and r_b so that ray has correct angle & intersects with first surface
    d = (r_b - r_a) / np.linalg.norm(r_b - r_a) # direction (unit vector) of ray


    while not stop:

        surf = surfaces[surf_index]

        r_a[0] = r_a[0]-surf.Lx*((r_a[0]+d[0]*(surf.zcov-r_a[2])/d[2])//surf.Lx)
        r_a[1] = r_a[1]-surf.Ly*((r_a[1]+d[1]*(surf.zcov-r_a[2])/d[2])//surf.Ly)

        res, theta, phi, r_a, d, theta_loc = single_interface_check(r_a, d, nks[mat_index],
                                     nks[mat_index+1], surf, surf.Lx, surf.Ly, direction,
                                     surf.zcov, pol, wl, Fr_or_TMM, lookuptable)

        if res == 0:  # reflection
            direction = -direction # changing direction due to reflection

            # staying in the same material, so mat_index does not change, but surf_index does
            surf_index = surf_index + direction

            surface_A = [0, 10]

        if res == 1:  # transmission
            surf_index = surf_index + direction
            mat_index = mat_index + direction # is this right?

            surface_A = [0, 10]

        if res == 2:
            surface_A = [theta, theta_loc] # passed a list of absorption per layer in theta
            stop = True
            theta = 10 # theta is actually list of absorption per layer

        if direction == 1 and mat_index == 1:

            stop = True
            # have ended with transmission

        elif direction == -1 and mat_index == 0:
            stop = True

    return I, theta, phi, surface_A


def traverse(width, theta, alpha, x, y, I_i, positions, I_thresh, direction):

    stop = False
    DA = (alpha/abs(cos(theta)))*I_i*np.exp((-alpha*positions/abs(cos(theta))))
    I_back = I_i*np.exp(-alpha*width/abs(cos(theta)))
    if I_back < I_thresh:
        stop = True
        theta = None

    if direction == -1:
        DA = np.flip(DA)

    return DA, stop, I_back, theta


def decide_RT_Fresnel(n0, n1, theta, d, N, side, pol, rnd, wl = None, lookuptable = None):
    R = calc_R(n0, n1, theta, pol)

    if rnd <= R:  # REFLECTION
        d = np.real(d - 2 * np.dot(d, N) * N)
        d = d / np.linalg.norm(d)

    else:  # TRANSMISSION)
        # transmission, refraction
        # for now, ignore effect of k on refraction

        tr_par = np.real((n0 / n1) ** side) * (d - np.dot(d, N) * N)
        tr_perp = -sqrt(1 - np.linalg.norm(tr_par) ** 2) * N

        side = -side
        d = np.real(tr_par + tr_perp)
        d = d / np.linalg.norm(d)

    return d, side, None # never absorbed, A = False

def decide_RT_TMM(n0, n1, theta, d, N, side, pol, rnd, wl, lookuptable):
    data = lookuptable.loc[dict(side=side, pol=pol)].sel(angle=abs(theta), wl=wl*1e9, method='nearest')
    R = np.real(data['R'].data.item(0))
    T = np.real(data['T'].data.item(0))
    A_per_layer = np.real(data['Alayer'].data)


    if rnd <= R:  # REFLECTION

        d = np.real(d - 2 * np.dot(d, N) * N)
        d = d / np.linalg.norm(d)
        A = None

    elif (rnd > R) & (rnd <= (R+T)):   # TRANSMISSION
        # transmission, refraction
        # for now, ignore effect of k on refraction

        tr_par = ((np.real(n0) / np.real(n1)) ** side) * (d - np.dot(d, N) * N)

        tr_perp = -sqrt(1 - np.linalg.norm(tr_par) ** 2) * N

        side = -side
        d = np.real(tr_par + tr_perp)
        d = d / np.linalg.norm(d)
        A = None

    else:
        # absorption
        A = A_per_layer

    return d, side, A


def single_interface_check(r_a, d, ni, nj, tri, Lx, Ly, side, z_cov, pol, wl=None, Fr_or_TMM=0, lookuptable=None):
    decide = {0: decide_RT_Fresnel, 1: decide_RT_TMM}

    # weird stuff happens around edges; can get transmission counted as reflection
    d0 = d
    intersect = True
    checked_translation = False

    # [top, right, bottom, left]
    translation = np.array([[0, -Ly, 0], [-Lx, 0, 0], [0, Ly, 0], [Lx, 0, 0]])
    i1 = 0
    while intersect:
        i1 = i1+1

        result = check_intersect(r_a, d, tri)
        if result == False and not checked_translation:
            which_side, tt = exit_side(r_a, d, Lx, Ly)

            r_a = r_a + translation[which_side]
            checked_translation = True

        elif result == False and checked_translation:

            o_t = np.real(acos(d[2] / (np.linalg.norm(d) ** 2)))
            o_p = np.real(atan2(d[1], d[0]))

            if np.sign(d0[2]) == np.sign(d[2]):
                intersect = False
                final_res = 1

            else:
                intersect = False
                final_res = 0


        else:
            # there has been an intersection

            intersn = result[0] # coordinate of the intersection (3D)

            theta =  result[1]
            N = result[2]*side # so angles get worked out correctly, relative to incident face normal

            if side == 1:
                n0 = ni
                n1 = nj

            else:
                n0 = nj
                n1 = ni

            rnd = random()

            d, side, A = decide[Fr_or_TMM](n0, n1, theta, d, N, side, pol, rnd, wl, lookuptable)

            r_a = np.real(intersn + d / 1e9) # this is to make sure the raytracer doesn't immediately just find the same intersection again

            checked_translation = False # reset, need to be able to translate the ray back into the unit cell again if necessary

            if A is not None:
                intersect = False
                final_res = 2
                o_t = A
                o_p = 0

        # if we are above surface, going downwards, or below surface, going upwards, and we have not yet
        # reached z_cov, we should keep trying to translate, so want to set check_translation = false
        # we also need to update r_a so that check_translation can correctly identify the next unit cell
        # the ray will move into

        if (side == 1 and d[2] < 0 and r_a[2] > z_cov) or \
                (side == -1 and d[2] > 0 and r_a[2] < z_cov) and checked_translation:  # going down but above surface

            exit, t = exit_side(r_a, d, Lx, Ly)

            r_a = r_a + t*d + translation[exit]

            checked_translation = False

    return final_res, o_t, o_p, r_a, d, theta # theta is LOCAL incidence angle (relative to texture)

def check_intersect(r_a, d, tri):

    # all the stuff which is only surface-dependent (and not dependent on incoming direction) is
    # in the surface object tri.
    pref = 1 / np.sum(np.matlib.repmat(np.transpose([-d]), 1, tri.size).T * tri.crossP, axis=1)
    corner = r_a - tri.P_0s
    t = pref * np.sum(tri.crossP * corner, axis=1)
    u = pref * np.sum(np.cross(tri.P_2s - tri.P_0s, np.matlib.repmat(np.transpose([-d]), 1, tri.size).T) * corner, axis=1)
    v = pref * np.sum(np.cross(np.matlib.repmat(np.transpose([-d]), 1, tri.size).T, tri.P_1s - tri.P_0s) * corner, axis=1)
    As = np.vstack((t, u, v))

    which_intersect = (As[1, :] + As[2, :] <= 1) & (np.all(As[[1, 2],] >= -1e-10, axis=0)) & (As[0, :] > 0) # get errors if set exactly to zero.
    if sum(which_intersect) > 0:

        t = t[which_intersect]
        P0 = tri.P_0s[which_intersect]
        P1 = tri.P_1s[which_intersect]
        P2 = tri.P_2s[which_intersect]
        ind = np.argmin(t)
        t = min(t)

        intersn = r_a + t * d
        N = np.cross(P1[ind] - P0[ind], P2[ind] - P0[ind])
        N = N / np.linalg.norm(N)

        theta = atan(np.linalg.norm(np.cross(N, -d))/np.dot(N, -d))  # in radians, angle relative to plane
        return [intersn, theta, N]
    else:
        return False
