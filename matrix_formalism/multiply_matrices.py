import numpy as np
import xarray as xr
from sparse import load_npz, dot, tensordot, COO, stack
from config import results_path
from angles import make_angle_vector
from solcore import material
import os
import matplotlib.pyplot as plt
import xarray as xr
from time import time
from solcore import material
from solcore.structure import Layer

def make_profile_data(options, unique_thetas, n_a_in, side, layer_name, n_layers, widths):

    if side == 1:
        side_str = 'front'
    else:
        side_str = 'rear'

    prof_mat_path = os.path.join(results_path, options['project_name'], layer_name + side_str + 'profmat.nc')
    #int_mat_path = os.path.join(results_path, options['project_name'], layer_name + side_str + 'intmat.nc')

    if os.path.isfile(prof_mat_path):
        print('Existing absorption profile calculation data found')
        #print(prof_mat_path)
        allres = xr.open_dataset(prof_mat_path)
        #print(allres)
        intgr = allres['intgr']
        profile = allres['profile']

    else:
        angle_dist_path = os.path.join(results_path, options['project_name'], layer_name + side_str + 'Aprof.npz')

        angle_distmat = load_npz(angle_dist_path)
        num_wl = len(options['wavelengths'])
        pr = xr.DataArray(angle_distmat.todense(), dims=['wl', 'local_theta', 'global_index'],
                          coords={'wl': options['wavelengths'] * 1e9, 'local_theta': unique_thetas,
                                  'global_index': np.arange(0, n_a_in)})

        lookuptable = xr.open_dataset(
            os.path.join(results_path, options['project_name'], layer_name + '.nc'))
        data = lookuptable.loc[dict(side=side, pol=options['pol'])].interp(angle=pr.coords['local_theta'],
                                                                        wl=pr.coords['wl'])

        params = data['Aprof'].drop(['layer', 'side', 'angle', 'pol']).transpose('wl', 'local_theta', 'layer', 'coeff')

        s_params = params.loc[
            dict(coeff=['A1', 'A2', 'A3_r',
                        'A3_i'])]  # have to scale these to make sure integrated absorption is correct
        c_params = params.loc[dict(coeff=['a1', 'a3'])]  # these should not be scaled

        scale_res = pr.groupby('global_index').apply(scale_func, scale_params=s_params)
        const_res = pr.groupby('global_index').apply(select_func, const_params=c_params)

        params = xr.concat((scale_res, const_res), dim='coeff').assign_coords(layer=np.arange(0, n_layers))

        #total_width = np.sum(layer_widths[i1])

        z_list = []
        for l_w in widths:
            z_list.append(xr.DataArray(np.arange(0, l_w, options['nm_spacing'])))

        offsets = np.cumsum([0] + widths)[:-1]
        start = time()
        #print(params)
        ans = params.groupby('wl').apply(scaled_profile, z=z_list, offset=offsets, side=side).drop('coeff')
        ans = ans.fillna(0)

        #print('Took ' + str(time() - start) + ' seconds')

        profile = ans.reduce(np.sum, 'layer')


        # there is no reason to do this again; this should be the same as the absorption per
        # layer before scaling (currently not saved)
        intgr = xr.DataArray(np.zeros((n_layers, num_wl, len(params.global_index))),
                           dims=['layer', 'wl', 'global_index'],
                           coords={'wl': params.wl, 'global_index': params.global_index})

        for i2, width in enumerate(widths):
            A1 = params.loc[dict(coeff='A1', layer=i2)]
            A2 = params.loc[dict(coeff='A2', layer=i2)]
            A3_r = params.loc[dict(coeff='A3_r', layer=i2)]
            A3_i = params.loc[dict(coeff='A3_i', layer=i2)]
            a1 = params.loc[dict(coeff='a1', layer=i2)]
            a3 = params.loc[dict(coeff='a3', layer=i2)]

            intgr_width = ((A1 / a1) * (np.exp(a1 * width) - 1) - (A2 / a1) * (np.exp(-a1 * width) - 1) - \
                         1j * ((A3_r + 1j * A3_i) / a3) * (np.exp(1j * a3 * width) - 1) + 1j * (
                                 (A3_r - 1j * A3_i) / a3) * (
                                 np.exp(-1j * a3 * width) - 1)).fillna(0)

            intgr[i2] = intgr_width.reduce(np.sum, 'local_theta')

        intgr = intgr.reduce(np.sum, 'layer')
        intgr.name = 'intgr'
        profile.name = 'profile'
        allres = xr.merge([intgr, profile])
        allres.to_netcdf(prof_mat_path)

    return profile, intgr

def make_v0(th_in, phi_in, num_wl, n_theta_bins, c_azimuth):
    """
    This function makes the v0 array, corresponding to the input power per angular channel
    at each wavelength, of size (num_wl, n_angle_bins_in) where n_angle_bins in = len(angle_vector)/2

    :param th_in: Polar angle of the incoming light (in radians)
    :param phi_in: Azimuthal angle of the incoming light (in radians), or can be set as 'all'
    in which case the power is spread equally over all the phi bins for the relevant theta.
    :param num_wl: Number of wavelengths
    :param n_theta_bins: Number of theta bins in the matrix multiplication
    :param c_azimuth: c_azimuth used to generate the matrices being multiplied
    :return: v0, an array of size (num_wl, n_angle_bins_in)
    """
    theta_intv, phi_intv, angle_vector = make_angle_vector(n_theta_bins, np.pi / 2, c_azimuth)
    n_a_in = int(len(angle_vector) / 2)
    v0 = np.zeros((num_wl, n_a_in))
    th_bin = np.digitize(th_in, theta_intv) - 1
    phi_intv = phi_intv[th_bin]
    bin = np.argmin(abs(angle_vector[:,0] - th_bin))
    if phi_in == 'all':
        n_phis = len(phi_intv) - 1
        v0[:, bin:(bin+n_phis)] = 1/n_phis
    else:
        phi_ind = np.digitize(phi_in, phi_intv) -1
        bin = bin + phi_ind
        v0[:, bin] = 1
    return v0

def make_D(alphas, thick, thetas):
    """
    Makes the bulk absorption vector for the bulk material
    :param alphas: absorption coefficient (m^{-1})
    :param thick: thickness of the slab in m
    :param thetas: incident thetas in angle_vector (second column)
    :return:
    """
    diag = np.exp(-alphas[:, None] * thick / np.cos(thetas[None, :]))
    D_1 = stack([COO(np.diag(x)) for x in diag])
    return D_1

def dot_wl(mat, vec):
    result = np.empty((vec.shape[0], mat.shape[1]))
    for i1 in range(vec.shape[0]):
        result[i1, :] = dot(mat[i1], vec[i1])
    return result

def dot_wl_prof(mat, vec):
    result = np.empty((vec.shape[0], mat.shape[1], mat.shape[3]))
    for i1 in range(vec.shape[0]):
        result[i1, :, :] = dot(mat[i1], vec[i1])
    return result

def scale_func(x, scale_params):
    return x.data[:,:, None, None]*scale_params

def select_func(x, const_params):
    return (x.data[:,:,None, None] != 0)*const_params

def profile_per_layer(x, z, offset, side):
    layer_index = x.coords['layer'].item(0)
    #print(x)
    non_zero = x[np.all(x, axis=1)]
    A1 = non_zero.loc[dict(coeff='A1')]
    A2 = non_zero.loc[dict(coeff='A2')]
    A3_r = non_zero.loc[dict(coeff='A3_r')]
    A3_i = non_zero.loc[dict(coeff='A3_i')]
    a1 = non_zero.loc[dict(coeff='a1')]
    a3 = non_zero.loc[dict(coeff='a3')]

    part1 = A1* np.exp(a1 * z[layer_index])
    part2 = A2* np.exp(-a1 * z[layer_index])
    part3 = (A3_r + 1j*A3_i)* np.exp(1j * a3 * z[layer_index])
    part4 = (A3_r - 1j*A3_i) * np.exp(-1j * a3 * z[layer_index])
    result = np.real(part1 + part2 + part3 + part4)
    if side == -1:
        result = np.flip(result, 1)

    return result.reduce(np.sum, axis=0).assign_coords(dim_0=z[layer_index]+offset[layer_index])

def profile_per_angle(x, z, offset, side):
    by_layer = x.groupby('layer').apply(profile_per_layer, z=z, offset=offset, side=side)
    return by_layer

def scaled_profile(x, z, offset, side):
    print('wl1')
    by_angle = x.groupby('global_index').apply(profile_per_angle, z=z, offset=offset, side=side)
    return by_angle

def bulk_profile(x, ths):
    print('wl2')
    return np.exp(-x/ths)



# def profile_per_layer(x, z, offset, side):
#
#     layer_index = x.coords['layer'].item(0)
#     nz = x.where(x != 0, drop=True)
#     # print(z[layer_index])
#     part1 = nz.loc[dict(coeff='A1')] * np.exp(nz.loc[dict(coeff='a1')] * z[layer_index])
#     part2 = nz.loc[dict(coeff='A2')] * np.exp(-nz.loc[dict(coeff='a1')] * z[layer_index])
#     part3 = (nz.loc[dict(coeff='A3_r')] + 1j * nz.loc[dict(coeff='A3_i')]) * np.exp(
#         1j * nz.loc[dict(coeff='a3')] * z[layer_index])
#     part4 = (nz.loc[dict(coeff='A3_r')] - 1j * nz.loc[dict(coeff='A3_i')]) * np.exp(
#         -1j * nz.loc[dict(coeff='a3')] * z[layer_index])
#     result = np.real(part1 + part2 + part3 + part4)
#     if side == -1:
#          result = np.flip(result, 1)
#         # print(result)
#     return result.reduce(np.sum, 'local_theta').assign_coords(dim_0=z[layer_index]+offset[layer_index])
#
# def profile_per_angle(x, z, offset, side):
#     by_layer = x.groupby('layer').apply(profile_per_layer, z=z, offset=offset, side=side)
#     return by_layer
#
# def scaled_profile(x, z, offset, side):
#     print('wl')
#     by_angle = x.groupby('layer').apply(profile_per_layer, z=z, offset=offset, side=side)
#     return by_angle

def matrix_multiplication(bulk_mats, bulk_thick, options, layer_widths=[], n_layers=[], layer_names=[]):
    n_bulks = len(bulk_mats)
    n_interfaces = n_bulks + 1

    theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'], np.pi/2, options['c_azimuth'])
    n_a_in = int(len(angle_vector)/2)

    num_wl = len(options['wavelengths'])

    #wls = np.linspace(600, 1100, num_wl)*1e-9
    #pol = 's'

    # bulk thickness in m

    thetas = angle_vector[:n_a_in, 1]

    v0 = make_v0(options['theta_in'], options['phi_in'], num_wl,
                 options['n_theta_bins'], options['c_azimuth'])

    D = []
    for i1 in range(n_bulks):
        D.append(make_D(bulk_mats[i1].alpha(options['wavelengths']), bulk_thick[i1], thetas))

    unique_thetas = np.unique(thetas)

    # Adown = []
    #
    # for i1 in range(n_bulks):
    #     prof_mat_path = os.path.join(results_path, options['project_name'], 'bulk' + str(i1)+ 'profmat.nc')
    #     # int_mat_path = os.path.join(results_path, options['project_name'], layer_name + side_str + 'intmat.nc')
    #
    #     if os.path.isfile(prof_mat_path):
    #         print('Existing bulk absorption profile calculation data found')
    #         bulk_prof = xr.open_dataset(prof_mat_path)['bulk_prof']
    #
    #     else:
    #         alphas = xr.DataArray(bulk_mats[i1].alpha(options['wavelengths']),
    #                               dims=['wl'], coords={'wl': options['wavelengths']})
    #         thetas = xr.DataArray(thetas, dims=['global_index'],
    #                               coords={'global_index': np.arange(0, n_a_in)})
    #         z = xr.DataArray(np.arange(0, bulk_thick[i1], options['nm_spacing'] * 1e-9),
    #                          dims=['z'],
    #                          coords={'z': np.arange(0, bulk_thick[i1], options['nm_spacing'] * 1e-9)})
    #
    #         OD = alphas * z
    #         print(OD)
    #         bulk_prof = OD.groupby('wl').apply(bulk_profile, ths=thetas)
    #         bulk_prof = bulk_prof.rename('bulk_prof')
    #         bulk_prof.to_netcdf(prof_mat_path)
    #         print(bulk_prof.shape)
    #
    #     Adown.append(bulk_prof)


    # front incidence matrices
    Rf = []
    Tf = []
    Af = []
    Pf = []
    If = []
    side = 1

    for i1 in range(n_interfaces):
        mat_path = os.path.join(results_path, options['project_name'], layer_names[i1] + 'frontRT.npz')
        absmat_path = os.path.join(results_path, options['project_name'], layer_names[i1] + 'frontA.npz')

        fullmat = load_npz(mat_path)
        absmat = load_npz(absmat_path)

        Rf.append(fullmat[:, :n_a_in, :])
        Tf.append(np.flip(fullmat[:, n_a_in:, :],1 ))
        Af.append(absmat)


        if options['calc_profile']:
            profile, intgr = make_profile_data(options, unique_thetas, n_a_in, side,
                                               layer_names[i1], n_layers[i1], layer_widths[i1])
            Pf.append(profile)
            If.append(intgr)


    # rear incidence matrices
    Rb = []
    Tb = []
    Ab = []
    Pb = []
    Ib = []
    paramsb = []
    side = -1

    for i1 in range(n_interfaces-1):
        mat_path = os.path.join(results_path, options['project_name'], layer_names[i1] + 'rearRT.npz')
        absmat_path = os.path.join(results_path, options['project_name'], layer_names[i1] + 'rearA.npz')

        fullmat = load_npz(mat_path)
        absmat = load_npz(absmat_path)

        Rb.append(fullmat[:, :n_a_in, :])
        Tb.append(np.flip(fullmat[:, n_a_in:, :],1 ))
        Ab.append(absmat)

        if options['calc_profile']:
            profile, intgr = make_profile_data(options, unique_thetas, n_a_in, side,
                                               layer_names[i1], n_layers[i1], layer_widths[i1])

            Pb.append(profile)
            Ib.append(intgr)


    if options['calc_profile']:
        a = [[] for _ in range(n_interfaces)]
        a_prof = [[] for _ in range(n_interfaces)]
        vr = [[] for _ in range(n_bulks)]
        vt = [[] for _ in range(n_bulks)]
        A = [[] for _ in range(n_bulks)]
        A_prof = [[] for _ in range(n_bulks)]

        vf_1 = [[] for _ in range(n_interfaces)]
        vb_1 = [[] for _ in range(n_interfaces)]
        vf_2 = [[] for _ in range(n_interfaces)]
        vb_2 = [[] for _ in range(n_interfaces)]


        for i1 in range(n_bulks):

            z = xr.DataArray(np.arange(0, bulk_thick[i1], options['nm_spacing']*1e-9), dims='z').chunk({'z': 10000})

            vf_1[i1] = dot_wl(Tf[i1], v0) # pass through front surface
            vr[i1].append(dot_wl(Rf[i1], v0)) # reflected from front surface
            a[i1].append(dot_wl(Af[i1], v0)) # absorbed in front surface at first interaction
            #print(v0)
            #print(If[i1])
            v_xr = xr.DataArray(v0, dims = ['wl', 'global_index'],
                                               coords = {'wl': If[i1].coords['wl'],
                                                         'global_index': np.arange(0, n_a_in)})
            int_power = xr.dot(v_xr, If[i1], dims = 'global_index')
            scale = (np.sum(dot_wl(Af[i1],v0), 1)/int_power).fillna(0)

            a_prof[i1].append((scale*xr.dot(v_xr, Pf[i1], dims = 'global_index')).data)
            power = np.sum(vf_1[i1], axis=1)

            # rep
            i2=1

            while np.any(power > options['I_thresh']):
                print(i2)
                #print(power)

                vb_1[i1] = dot_wl(D[i1], vf_1[i1]) # pass through bulk, downwards
                scale = (np.sum(dot_wl(Af[i1+1], vb_1[i1]), 1) / int_power).fillna(0)
                #('front profile')
                v_xr = xr.DataArray(vb_1[i1], dims=['wl', 'global_index'],
                                    coords={'wl': If[i1+1].coords['wl'],
                                            'global_index': np.arange(0, n_a_in)})

                a_prof[i1+1].append((scale * xr.dot(v_xr, Pf[i1+1], dims='global_index')).data)


                #remaining_power.append(np.sum(vb_1, axis=1))
                A[i1].append(np.sum(vf_1[i1], 1) - np.sum(vb_1[i1], 1))

                nz_thetas = vf_1[i1] != 0
                #print('bulk profile')
                # Aprof_loop = np.zeros((num_wl, len(z)))
                # for k1, wl in enumerate(options['wavelengths']):
                #     th_wl = thetas[nz_thetas[k1]]
                #     cs = xr.DataArray(np.cos(th_wl), dims='theta')
                #     #print(cs)
                #     pref_wl = xr.DataArray(vf_1[i1][k1][nz_thetas[k1]], dims='theta')
                #     alpha = bulk_mats[i1].alpha(wl)
                #     Aprof_loop[k1, :] = np.sum(pref_wl*(alpha/cs)*np.exp(-alpha*z/cs), 0)

                # cs = xr.DataArray(np.cos(thetas), dims = 'theta')
                # v_xr= xr.DataArray(vf_1[i1], dims = ['wl', 'theta'])
                # v_xr = v_xr.chunk({'wl': 10})
                #
                # alpha = xr.DataArray(bulk_mats[i1].alpha(options['wavelengths']), dims=['wl'])
                # alpha = alpha.chunk({'wl': 10})
                # print(alpha)
                # res=np.sum(v_xr*(alpha/cs)*np.exp(-alpha*z/cs),1).compute().visualize()
                #
                # print(res)
                #
                # A_prof[i1].append(res)

                vb_2[i1] = dot_wl(Rf[i1+1], vb_1[i1]) # reflect from back surface
                vf_2[i1] = dot_wl(D[i1], vb_2[i1]) # pass through bulk, upwards

                #print('rear profile')
                v_xr = xr.DataArray(vf_2[i1], dims=['wl', 'global_index'],
                                    coords={'wl': Ib[i1].coords['wl'],
                                            'global_index': np.arange(0, n_a_in)})
                int_power = xr.dot(v_xr, Ib[i1], dims='global_index')
                scale = (np.sum(dot_wl(Ab[i1], vf_2[i1]), 1) / int_power).fillna(0)
                a_prof[i1].append((scale * xr.dot(v_xr, Pb[i1], dims='global_index')).data)

                #remaining_power.append(np.sum(vf_2, axis=1))
                A[i1].append(np.sum(vb_2[i1], 1) - np.sum(vf_2[i1], 1))
                vf_1[i1] = dot_wl(Rb[i1], vf_2[i1]) # reflect from front surface
                power = np.sum(vf_1[i1], axis=1)

                # nz_thetas = vb_2[i1] != 0

                # Aprof_loop = np.zeros((num_wl, len(z)))
                # for k1, wl in enumerate(options['wavelengths']):
                #     th_wl = thetas[nz_thetas[k1]]
                #     cs = xr.DataArray(np.cos(th_wl), dims='theta')
                #     #print(cs)
                #     pref_wl = xr.DataArray(vb_2[i1][k1][nz_thetas[k1]], dims='theta')
                #     alpha = bulk_mats[i1].alpha(wl)
                #     Aprof_loop[k1, :] = np.flip(np.sum(pref_wl*alpha*np.exp(-alpha*z/cs), 0))
                # cs = xr.DataArray(np.cos(thetas), dims = 'theta')
                # v_xr= xr.DataArray(vb_2[i1], dims = ['wl', 'theta'])
                # v_xr = v_xr.chunk({'wl': 2})
                #
                # alpha = xr.DataArray(bulk_mats[i1].alpha(options['wavelengths']), dims=['wl'])
                # alpha = alpha.chunk({'wl': 2})
                # res=np.sum(v_xr*(alpha/cs)*np.exp(-alpha*z/cs),1).compute()
                # print(res)
                #
                # A_prof[i1].append(np.flip(res,1))

                #A_prof[i1].append(Aprof_loop)

                vr[i1].append(dot_wl(Tb[i1], vf_2[i1]))  # matrix travelling up in medium 0, i.e. reflected overall by being transmitted through front surface
                vt[i1].append(dot_wl(Tf[i1+1], vb_1[i1]))  # transmitted into medium below through back surface
                a[i1+1].append(dot_wl(Af[i1+1], vb_1[i1]))  # absorbed in 2nd surface
                a[i1].append(dot_wl(Ab[i1], vf_2[i1]))  # absorbed in 1st surface (from the back)

                i2+=1

        vr = [np.array(item) for item in vr]
        vt = [np.array(item) for item in vt]
        a = [np.array(item) for item in a]
        A = [np.array(item) for item in A]
        a_prof = [np.array(item) for item in a_prof]

        results_per_pass = {'r': vr, 't': vt, 'a': a, 'A': A, 'a_prof': a_prof}

        # for i2 in range(3):
        #     for i1 in range(n_interfaces):
        #         plt.figure()
        #         z = np.arange(0, np.sum(layer_widths[i1]), options['nm_spacing'])
        #         plt.plot(z, a_prof[i1][i2, 0, :].T)
        #         plt.title(str(i2) + 'interface ' + str(i1))
        #         plt.show()
        sum_dims = ['bulk_index', 'wl']
        sum_coords = {'bulk_index': np.arange(0, n_bulks), 'wl': options['wavelengths']}
        R = xr.DataArray(np.array([np.sum(item, (0,2)) for item in vr]),
                           dims=sum_dims, coords=sum_coords, name = 'R')
        T = xr.DataArray(np.array([np.sum(item, (0,2)) for item in vt]),
                           dims=sum_dims, coords=sum_coords, name = 'T')
        A_bulk = xr.DataArray(np.array([np.sum(item, 0) for item in A]),
                           dims=sum_dims, coords=sum_coords, name = 'A_bulk')

        A_interface = xr.DataArray(np.array([np.sum(item, (0,2)) for item in a]),
                                   dims=['surf_index', 'wl'],
                                   coords = {'surf_index': np.arange(0, n_interfaces),
                                             'wl': options['wavelengths']}, name = 'A_interface')
        profile = []
        for j1, item in enumerate(a_prof):
            profile.append(xr.DataArray(np.sum(item, 0),
                   dims=['wl', 'z'], coords = {'wl': options['wavelengths']},
                                        name = 'A_profile' + str(j1))) # not necessarily same number of z coords per layer stack

        bulk_profile = np.array(A_prof)

        RAT = xr.merge([R, A_bulk, A_interface, T])
        # for i2 in range(num_wl):
        #     plt.figure()
        #     for i1 in range(n_interfaces):
        #         z = np.arange(0, np.sum(layer_widths[i1]), options['nm_spacing'])
        #         plt.plot(z, profile[i1][i2].T)
        #
        #     plt.show()
        #
        # plt.figure()
        # for i2 in range(5):
        #     i1= 0
        #     z = np.arange(0, np.sum(layer_widths[i1]), options['nm_spacing'])
        #     plt.plot(z, a_prof[i1][i2, 0, :])
        #
        # plt.figure()
        # plt.plot(options['wavelengths'], R.T)
        # plt.plot(options['wavelengths'], T.T)
        # plt.plot(options['wavelengths'], A_interface.T)
        # plt.plot(options['wavelengths'], A_bulk.T)
        # plt.plot(options['wavelengths'], R[0] + T[0] + A_interface[0] + A_interface[1]+A_bulk[0])
        # plt.legend(['R', 'T', 'front', 'back', 'bulk'])
        # plt.show()

        #return R, T, A_bulk, A_interface, profile
        return RAT, results_per_pass, profile, bulk_profile

    else:
        a = [[] for _ in range(n_interfaces)]
        vr = [[] for _ in range(n_bulks)]
        vt = [[] for _ in range(n_bulks)]
        A = [[] for _ in range(n_bulks)]

        vf_1 = [[] for _ in range(n_interfaces)]
        vb_1 = [[] for _ in range(n_interfaces)]
        vf_2 = [[] for _ in range(n_interfaces)]
        vb_2 = [[] for _ in range(n_interfaces)]

        for i1 in range(n_bulks):

            vf_1[i1] = dot_wl(Tf[i1], v0)  # pass through front surface
            vr[i1].append(dot_wl(Rf[i1], v0))  # reflected from front surface
            a[i1].append(dot_wl(Af[i1], v0))  # absorbed in front surface at first interaction
            power = np.sum(vf_1[i1], axis=1)

            # rep
            i2 = 1

            while np.any(power > options['I_thresh']):
                print(i2)
                vb_1[i1] = dot_wl(D[i1], vf_1[i1])  # pass through bulk, downwards

                # remaining_power.append(np.sum(vb_1, axis=1))
                A[i1].append(np.sum(vf_1[i1], 1) - np.sum(vb_1[i1], 1))
                vb_2[i1] = dot_wl(Rf[i1 + 1], vb_1[i1])  # reflect from back surface
                vf_2[i1] = dot_wl(D[i1], vb_2[i1])  # pass through bulk, upwards

                # remaining_power.append(np.sum(vf_2, axis=1))
                A[i1].append(np.sum(vb_2[i1], 1) - np.sum(vf_2[i1], 1))
                vf_1[i1] = dot_wl(Rb[i1], vf_2[i1])  # reflect from front surface
                power = np.sum(vf_1[i1], axis=1)

                vr[i1].append(dot_wl(Tb[i1], vf_2[i1]))  # matrix travelling up in medium 0, i.e. reflected overall by being transmitted through front surface
                vt[i1].append(dot_wl(Tf[i1 + 1], vb_1[i1]))  # transmitted into medium below through back surface
                a[i1 + 1].append(dot_wl(Af[i1 + 1], vb_1[i1]))  # absorbed in 2nd surface
                a[i1].append(dot_wl(Ab[i1], vf_2[i1]))  # absorbed in 1st surface (from the back)

                i2 += 1

        vr = [np.array(item) for item in vr]
        vt = [np.array(item) for item in vt]
        a = [np.array(item) for item in a]
        A = [np.array(item) for item in A]

        results_per_pass = {'r': vr, 't': vt, 'a': a, 'A': A}

        sum_dims = ['bulk_index', 'wl']
        sum_coords = {'bulk_index': np.arange(0, n_bulks), 'wl': options['wavelengths']}
        R = xr.DataArray(np.array([np.sum(item, (0,2)) for item in vr]),
                           dims=sum_dims, coords=sum_coords, name = 'R')
        if i1 >1 :
            A_bulk = xr.DataArray(np.array([np.sum(item, 0) for item in A]),
                               dims=sum_dims, coords=sum_coords, name = 'A_bulk')

            T = xr.DataArray(np.array([np.sum(item, (0,2)) for item in vt]),
                               dims=sum_dims, coords=sum_coords, name = 'T')

            RAT = xr.merge([R, A_bulk, T])

        else:
            RAT = xr.merge([R])

        return RAT, results_per_pass



