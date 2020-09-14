import numpy as np
from sparse import load_npz, dot, COO, stack
from config import results_path
from angles import make_angle_vector, fold_phi
import os
import xarray as xr
from structure import Interface, BulkLayer


def calculate_RAT(SC, options):
    """
    After the list of Interface and BulkLayers has been processed by process_structure,
    this function calculates the R, A and T by calling matrix_multiplication.
    :param SC: list of Interface and BulkLayer objects. Order is [Interface, BulkLayer, Interface]
    :param options: options for the matrix calculations
    """

    bulk_mats = []
    bulk_widths = []
    layer_widths = []
    n_layers = []
    layer_names = []
    calc_prof_list = []

    for i1, struct in enumerate(SC):
        if type(struct) == BulkLayer:
            bulk_mats.append(struct.material)
            bulk_widths.append(struct.width)
        if type(struct) == Interface:
            layer_names.append(struct.name)

            n_layers.append(len(struct.layers))
            layer_widths.append((np.array(struct.widths)*1e9).tolist())
            calc_prof_list.append(struct.prof_layers)



    results = matrix_multiplication(bulk_mats, bulk_widths, options,
                                                               layer_widths, n_layers, layer_names, calc_prof_list)

    return results


def make_v0(th_in, phi_in, num_wl, n_theta_bins, c_azimuth, phi_sym):
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

    theta_intv, phi_intv, angle_vector = make_angle_vector(n_theta_bins, phi_sym, c_azimuth)
    n_a_in = int(len(angle_vector)/2)
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


def out_to_in_matrix(phi_sym, angle_vector, theta_intv, phi_intv):

    if phi_sym == 2*np.pi:
        phi_sym = phi_sym - 0.0001
    out_to_in = np.zeros((len(angle_vector), len(angle_vector)))
    binned_theta_out = np.digitize(np.pi-angle_vector[:,1], theta_intv, right=True) - 1

    phi_rebin = fold_phi(angle_vector[:,2] + np.pi, phi_sym)

    phi_out = xr.DataArray(phi_rebin,
                           coords={'theta_bin': (['angle_in'], binned_theta_out)},
                           dims=['angle_in'])

    bin_out = phi_out.groupby('theta_bin').apply(overall_bin,
                                                 args=(phi_intv, angle_vector[:, 0])).data

    out_to_in[bin_out, np.arange(len(angle_vector))] = 1

    up_to_down = out_to_in[int(len(angle_vector)/2):, :int(len(angle_vector)/2)]
    down_to_up = out_to_in[:int(len(angle_vector)/2), int(len(angle_vector)/2):]

    return COO(up_to_down), COO(down_to_up)


def overall_bin(x, phi_intv, angle_vector_0):
    phi_ind = np.digitize(x, phi_intv[x.coords['theta_bin'].data[0]], right=True) - 1
    bin = np.argmin(abs(angle_vector_0 - x.coords['theta_bin'].data[0])) + phi_ind
    return bin



def make_D(alphas, thick, thetas):
    """
    Makes the bulk absorption vector for the bulk material
    :param alphas: absorption coefficient (m^{-1})
    :param thick: thickness of the slab in m
    :param thetas: incident thetas in angle_vector (second column)
    :return:
    """
    #print(alphas, abs(np.cos(thetas[None, :])))
    diag = np.exp(-alphas[:, None] * thick / abs(np.cos(thetas[None, :])))
    #print(diag)
    D_1 = stack([COO(np.diag(x)) for x in diag])
    return D_1

def dot_wl(mat, vec):
    print(mat.shape)
    result = np.empty((vec.shape[0], mat.shape[1]))

    if len(mat.shape) == 3:
        for i1 in range(vec.shape[0]):  # loop over wavelengths
            result[i1, :] = dot(mat[i1], vec[i1])

    if len(mat.shape) == 2:
        for i1 in range(vec.shape[0]):  # loop over wavelengths
            result[i1, :] = dot(mat, vec[i1])

    return result

def dot_wl_u2d(mat, vec):
    result = np.empty((vec.shape[0], vec.shape[1]))
    for i1 in range(vec.shape[0]):  # loop over wavelengths
        result[i1, :] = dot(mat, vec[i1])
    return result

def dot_wl_prof(mat, vec):
    result = np.empty((vec.shape[0], mat.shape[1], mat.shape[3]))
    for i1 in range(vec.shape[0]): # loop over wavelengths
        result[i1, :, :] = dot(mat[i1], vec[i1])
    return result


def bulk_profile(x, ths):
    print('wl2')
    return np.exp(-x/ths)


def matrix_multiplication(bulk_mats, bulk_thick, options,
                          layer_widths=[], n_layers=[], layer_names=[], calc_prof_list=[]):
    n_bulks = len(bulk_mats)
    n_interfaces = n_bulks + 1

    theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'], options['c_azimuth'])
    n_a_in = int(len(angle_vector)/2)

    num_wl = len(options['wavelengths'])

    #wls = np.linspace(600, 1100, num_wl)*1e-9
    #pol = 's'

    # bulk thickness in m

    thetas = angle_vector[:n_a_in, 1]

    v0 = make_v0(options['theta_in'], options['phi_in'], num_wl,
                 options['n_theta_bins'], options['c_azimuth'], options['phi_symmetry'])

    up2down, down2up = out_to_in_matrix(options['phi_symmetry'], angle_vector, theta_intv, phi_intv)

    D = []
    for i1 in range(n_bulks):
        D.append(make_D(bulk_mats[i1].alpha(options['wavelengths']), bulk_thick[i1], thetas))

    #unique_thetas = np.unique(thetas)

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

        if len(fullmat.shape) == 3:
            Rf.append(fullmat[:, :n_a_in, :])
            Tf.append(fullmat[:, n_a_in:, :])
            Af.append(absmat)

        else:
            print(fullmat.shape)
            Rf.append(fullmat[:n_a_in, :])
            Tf.append(fullmat[n_a_in:, :])
            Af.append(absmat)


        if len(calc_prof_list[i1]) > 0:
            #profile, intgr = make_profile_data(options, unique_thetas, n_a_in, side,
            #                                   layer_names[i1], n_layers[i1], layer_widths[i1])
            profmat_path = os.path.join(results_path, options['project_name'], layer_names[i1] + 'frontprofmat.nc')
            prof_int = xr.load_dataset(profmat_path)
            profile = prof_int['profile']
            intgr = prof_int['intgr']
            Pf.append(profile)
            If.append(intgr)

        else:
            Pf.append([])
            If.append([])


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

        if len(fullmat.shape) == 3:
            Rb.append(fullmat[:, n_a_in:, :])
            Tb.append(fullmat[:, :n_a_in, :])
            Ab.append(absmat)

        else:
            Rb.append(fullmat[n_a_in:, :])
            Tb.append(fullmat[:n_a_in, :])
            Ab.append(absmat)


        if len(calc_prof_list[i1]) > 0:
            #profile, intgr = make_profile_data(options, unique_thetas, n_a_in, side,
            #                                   layer_names[i1], n_layers[i1], layer_widths[i1])
            profmat_path = os.path.join(results_path, options['project_name'], layer_names[i1] + 'rearprofmat.nc')
            prof_int = xr.load_dataset(profmat_path)
            profile = prof_int['profile']
            intgr = prof_int['intgr']
            Pb.append(profile)
            Ib.append(intgr)

        else:
            Pb.append([])
            Ib.append([])

    len_calcs = np.array([len(x) for x in calc_prof_list])
    print(len_calcs)
    print(np.any(len_calcs > 0))

    if np.any(len_calcs > 0):
        print('a')
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

            #z = xr.DataArray(np.arange(0, bulk_thick[i1], options['nm_spacing']*1e-9), dims='z')
            # v0 is actually travelling down, but no reason to start in 'outgoing' ray format.
            vf_1[i1] = dot_wl(Tf[i1], v0) # pass through front surface
            vr[i1].append(dot_wl(Rf[i1], v0)) # reflected from front surface
            a[i1].append(dot_wl(Af[i1], v0)) # absorbed in front surface at first interaction
            #print(v0)
            #print(If[i1])

            if len(If[i1] > 0):
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
                vf_1[i1] = dot_wl_u2d(down2up, vf_1[i1]) # outgoing to incoming
                vb_1[i1] = dot_wl(D[i1], vf_1[i1])  # pass through bulk, downwards
                # vb_1 already an incoming ray


                if len(If[i1+1]) > 0:

                    v_xr = xr.DataArray(vb_1[i1], dims=['wl', 'global_index'],
                                        coords={'wl': If[i1+1].coords['wl'],
                                                'global_index': np.arange(0, n_a_in)})
                    int_power = xr.dot(v_xr, If[i1+1], dims='global_index')
                    scale = (np.sum(dot_wl(Af[i1+1], vb_1[i1]), 1) / int_power).fillna(0)
                    #('front profile')


                    a_prof[i1+1].append((scale * xr.dot(v_xr, Pf[i1+1], dims='global_index')).data)

                #remaining_power.append(np.sum(vb_1, axis=1))
                A[i1].append(np.sum(vf_1[i1], 1) - np.sum(vb_1[i1], 1))

                nz_thetas = vf_1[i1] != 0

                vb_2[i1] = dot_wl(Rf[i1+1], vb_1[i1]) # reflect from back surface. incoming -> up

                vf_2[i1] = dot_wl(D[i1], vb_2[i1]) # pass through bulk, upwards

                #print('rear profile')
                if len(Ib[i1]) > 0:
                    v_xr = xr.DataArray(vf_2[i1], dims=['wl', 'global_index'],
                                        coords={'wl': Ib[i1].coords['wl'],
                                                'global_index': np.arange(0, n_a_in)})
                    int_power = xr.dot(v_xr, Ib[i1], dims='global_index')
                    scale = (np.sum(dot_wl(Ab[i1], vf_2[i1]), 1) / int_power).fillna(0)
                    a_prof[i1].append((scale * xr.dot(v_xr, Pb[i1], dims='global_index')).data)

                #remaining_power.append(np.sum(vf_2, axis=1))

                A[i1].append(np.sum(vb_2[i1], 1) - np.sum(vf_2[i1], 1))

                vf_2[i1] = dot_wl_u2d(up2down, vf_2[i1]) # prepare for rear incidence
                vf_1[i1] = dot_wl(Rb[i1], vf_2[i1]) # reflect from front surface
                power = np.sum(vf_1[i1], axis=1)

                # nz_thetas = vb_2[i1] != 0

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
            if len(item) > 0:
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
        print('b')
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

                print('before d2u', np.sum(vf_1[i1]))
                vf_1[i1] = dot_wl_u2d(down2up, vf_1[i1]) # outgoing to incoming
                print('after 2du', np.sum(vf_1[i1]))
                #print('vf_1 after', vf_1[i1])
                vb_1[i1] = dot_wl(D[i1], vf_1[i1])  # pass through bulk, downwards
                print('before back ref', np.sum(vb_1[i1]))
                # remaining_power.append(np.sum(vb_1, axis=1))
                A[i1].append(np.sum(vf_1[i1], 1) - np.sum(vb_1[i1], 1))

                vb_2[i1] = dot_wl(Rf[i1 + 1], vb_1[i1])  # reflect from back surface
                print('after back ref', np.sum(vb_2[i1]))
                vf_2[i1] = dot_wl(D[i1], vb_2[i1]) # pass through bulk, upwards
                #print('vb_2', vb_2[i1])
                print('after u2d', np.sum(vf_2[i1]))
                vf_2[i1] = dot_wl_u2d(up2down, vf_2[i1]) # prepare for rear incidence
                print('after u2d/before front ref', np.sum(vf_2[i1]))
                vf_1[i1] = dot_wl(Rb[i1], vf_2[i1]) # reflect from front surface
                print('after front ref', np.sum(vf_1[i1]))
                #print('Rf, Rb, and vf2', Rf[i1][20].todense(), Rb[i1][20].todense(), vf_2[i1][20])
                #print('powersrem', np.sum(vb_2[i1], 1), np.sum(vf_2[i1], 1), np.sum(vf_1[i1], 1))
                # remaining_power.append(np.sum(vf_2, axis=1))
                A[i1].append(np.sum(vb_2[i1], 1) - np.sum(vf_2[i1], 1))
                power = np.sum(vf_1[i1], axis=1)
                #print('power', power)

                vr[i1].append(dot_wl(Tb[i1], vf_2[i1]))  # matrix travelling up in medium 0, i.e. reflected overall by being transmitted through front surface
                print('lost in front ref', np.sum(vr[i1]))
                #print('Tf, vb1', Tf[i1 + 1][20].todense(), vb_1[i1][20])
                vt[i1].append(dot_wl(Tf[i1 + 1], vb_1[i1]))  # transmitted into medium below through back surface
                print('lost in back ref', np.sum(vt[i1]))
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
        if i2 > 1 :
            A_bulk = xr.DataArray(np.array([np.sum(item, 0) for item in A]),
                               dims=sum_dims, coords=sum_coords, name = 'A_bulk')

            T = xr.DataArray(np.array([np.sum(item, (0,2)) for item in vt]),
                               dims=sum_dims, coords=sum_coords, name = 'T')

            RAT = xr.merge([R, A_bulk, T])

        else:
            RAT = xr.merge([R])

        return RAT, results_per_pass



