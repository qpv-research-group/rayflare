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

def make_D(alphas, thick, thetas):
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
    print('wl')
    by_angle = x.groupby('global_index').apply(profile_per_angle, z=z_list, offset=offset, side=side)
    return by_angle

options = {'nm_spacing': 1, 'project_name': 'testing', 'layer_names': ['GaInPGaAsonSi', 'GaAsonSi'],
           'layer_widths': [[500, 700], [500]], 'n_layers': [2,1], 'calc_profile': True}

nm_spacing = 1

Si = material('Si')()

project_name = 'testing'

layer_names = ['GaInPGaAsonSi', 'GaAsonSi']

layer_widths = [[500, 700], [500]]

n_layers = [2, 1]

bulk_mats = [Si]

bulk_thick = [0.5*1e-6]

calc_profile = True

n_bulks = len(bulk_mats)
n_interfaces = n_bulks + 1

theta_intv, phi_intv, angle_vector = make_angle_vector(100, np.pi/2, 0.25)
n_a_in = int(len(angle_vector)/2)

num_wl = 4

wls = np.linspace(600, 1100, num_wl)*1e-9
pol = 's'

# bulk thickness in m

thetas = angle_vector[:n_a_in, 1]

# Need a function to make v0 automatically depending on the incidence angle
# can specify theta and phi, if only theta is specified distribute uniformly over phis
v0 = np.zeros((num_wl, n_a_in))
v0[:,0] = 1

D = []
for i1 in range(n_bulks):
    D.append(make_D(bulk_mats[i1].alpha(wls), bulk_thick[i1], thetas))

unique_thetas = np.unique(thetas)
# front incidence matrices
Rf = []
Tf = []
Af = []
Pf = []
If = []
side = 1

for i1 in range(n_interfaces):
    mat_path = os.path.join(results_path, project_name, layer_names[i1] + 'frontRT.npz')
    absmat_path = os.path.join(results_path, project_name, layer_names[i1] + 'frontA.npz')

    fullmat = load_npz(mat_path)
    absmat = load_npz(absmat_path)

    Rf.append(fullmat[:, :n_a_in, :])
    Tf.append(fullmat[:, n_a_in:, :])
    Af.append(absmat)

    if calc_profile:
        angle_dist_path = os.path.join(results_path, project_name, layer_names[i1] + 'frontAprof.npz')
        angle_distmat = load_npz(angle_dist_path)

        pr = xr.DataArray(angle_distmat.todense(), dims=['wl', 'local_theta', 'global_index'],
                          coords={'wl': wls * 1e9, 'local_theta': unique_thetas,
                                  'global_index': np.arange(0, len(angle_vector) / 2)})

        lookuptable = xr.open_dataset(os.path.join(results_path, project_name, layer_names[i1] + '.nc'))
        data = lookuptable.loc[dict(side=side, pol=pol)].sel(angle=pr.coords['local_theta'],
                                                             wl=pr.coords['wl'], method='nearest')

        params = data['Aprof'].drop(['layer', 'side', 'angle', 'pol']).transpose('wl', 'local_theta', 'layer', 'coeff')

        s_params = params.loc[
            dict(coeff=['A1', 'A2', 'A3_r',
                        'A3_i'])]  # have to scale these to make sure integrated absorption is correct
        c_params = params.loc[dict(coeff=['a1', 'a3'])]  # these should not be scaled

        scale_res = pr.groupby('global_index').apply(scale_func, scale_params=s_params)
        const_res = pr.groupby('global_index').apply(select_func, const_params=c_params)

        params = xr.concat((scale_res, const_res), dim='coeff').assign_coords(layer=np.arange(0, n_layers[i1]))

        total_width = np.sum(layer_widths[i1])

        z_list = []
        for l_w in layer_widths[i1]:
            z_list.append(xr.DataArray(np.arange(0, l_w, nm_spacing)))

        offsets = np.cumsum([0] + layer_widths[i1])[:-1]
        start = time()
        ans = params.groupby('wl').apply(scaled_profile, z=z_list, offset=offsets, side=side).drop('coeff')
        ans = ans.fillna(0)

        print('Took ' + str(time() - start) + ' seconds')

        profile = ans.reduce(np.sum, 'layer')

        int = xr.DataArray(np.zeros((n_layers[i1], num_wl, len(params.global_index))),
                           dims=['layer', 'wl', 'global_index'],
                           coords={'wl': params.wl, 'global_index': params.global_index})

        for i2, width in enumerate(layer_widths[i1]):
            A1 = params.loc[dict(coeff='A1', layer=i2)]
            A2 = params.loc[dict(coeff='A2', layer=i2)]
            A3_r = params.loc[dict(coeff='A3_r', layer=i2)]
            A3_i = params.loc[dict(coeff='A3_i', layer=i2)]
            a1 = params.loc[dict(coeff='a1', layer=i2)]
            a3 = params.loc[dict(coeff='a3', layer=i2)]

            int_width = ((A1 / a1) * (np.exp(a1 * width) - 1) - (A2 / a1) * (np.exp(-a1 * width) - 1) - \
                         1j * ((A3_r + 1j * A3_i) / a3) * (np.exp(1j * a3 * width) - 1) + 1j * (
                                     (A3_r - 1j * A3_i) / a3) * (
                                 np.exp(-1j * a3 * width) - 1)).fillna(0)

            int[i2] = int_width.reduce(np.sum, 'local_theta')

        int = int.reduce(np.sum, 'layer')
        Pf.append(profile)
        If.append(int)


# rear incidence matrices
Rb = []
Tb = []
Ab = []
Pb = []
Ib = []
paramsb = []
side = -1

for i1 in range(n_interfaces):
    mat_path = os.path.join(results_path, project_name, layer_names[i1] + 'rearRT.npz')
    absmat_path = os.path.join(results_path, project_name, layer_names[i1] + 'rearA.npz')

    fullmat = load_npz(mat_path)
    absmat = load_npz(absmat_path)

    Rb.append(fullmat[:, :n_a_in, :])
    Tb.append(fullmat[:, n_a_in:, :])
    Ab.append(absmat)

    if calc_profile:
        angle_dist_path = os.path.join(results_path, project_name, layer_names[i1] + 'rearAprof.npz')
        angle_distmat = load_npz(angle_dist_path)

        pr = xr.DataArray(angle_distmat.todense(), dims=['wl', 'local_theta', 'global_index'],
                          coords={'wl': wls * 1e9, 'local_theta': unique_thetas,
                                  'global_index': np.arange(0, len(angle_vector) / 2)})

        lookuptable = xr.open_dataset(os.path.join(results_path, project_name, layer_names[i1] + '.nc'))
        data = lookuptable.loc[dict(side=side, pol=pol)].sel(angle=pr.coords['local_theta'],
                                                             wl=pr.coords['wl'], method='nearest')

        params = data['Aprof'].drop(['layer', 'side', 'angle', 'pol']).transpose('wl', 'local_theta', 'layer', 'coeff')

        s_params = params.loc[
            dict(coeff=['A1', 'A2', 'A3_r',
                        'A3_i'])]  # have to scale these to make sure integrated absorption is correct
        c_params = params.loc[dict(coeff=['a1', 'a3'])]  # these should not be scaled

        scale_res = pr.groupby('global_index').apply(scale_func, scale_params=s_params)
        const_res = pr.groupby('global_index').apply(select_func, const_params=c_params)

        params = xr.concat((scale_res, const_res), dim='coeff').assign_coords(layer=np.arange(0, n_layers[i1]))

        total_width = np.sum(layer_widths[i1])

        # z = xr.DataArray(np.arange(0, total_width, 0.5))
        z_list = []
        for l_w in layer_widths[i1]:
            z_list.append(xr.DataArray(np.arange(0, l_w, nm_spacing)))

        offsets = np.cumsum([0] + layer_widths[i1])[:-1]
        start = time()
        ans = params.groupby('wl').apply(scaled_profile, z=z_list, offset=offsets, side=side).drop('coeff')
        ans = ans.fillna(0)

        print('Took ' + str(time() - start) + ' seconds')

        ans = np.flip(ans, 2)
        profile = ans.reduce(np.sum, 'layer')

        int = xr.DataArray(np.zeros((n_layers[i1], num_wl, len(params.global_index))),
                           dims=['layer', 'wl', 'global_index'],
                           coords={'wl': params.wl, 'global_index': params.global_index})

        for i2, width in enumerate(layer_widths[i1]):
            A1 = params.loc[dict(coeff='A1', layer=i2)]
            A2 = params.loc[dict(coeff='A2', layer=i2)]
            A3_r = params.loc[dict(coeff='A3_r', layer=i2)]
            A3_i = params.loc[dict(coeff='A3_i', layer=i2)]
            a1 = params.loc[dict(coeff='a1', layer=i2)]
            a3 = params.loc[dict(coeff='a3', layer=i2)]

            int_width = ((A1 / a1) * (np.exp(a1 * width) - 1) - (A2 / a1) * (np.exp(-a1 * width) - 1) - \
                         1j * ((A3_r + 1j * A3_i) / a3) * (np.exp(1j * a3 * width) - 1) + 1j * (
                                     (A3_r - 1j * A3_i) / a3) * (
                                 np.exp(-1j * a3 * width) - 1)).fillna(0)

            int[i2] = int_width.reduce(np.sum, 'local_theta')

        int = int.reduce(np.sum, 'layer')

        Pb.append(profile)
        Ib.append(int)


if calc_profile:
    a = [[] for _ in range(n_interfaces)]
    a_prof = [[] for _ in range(n_interfaces)]
    vr = [[] for _ in range(n_bulks)]
    vt = [[] for _ in range(n_bulks)]
    A = [[] for _ in range(n_bulks)]

    vf_1 = [[] for _ in range(n_interfaces)]
    vb_1 = [[] for _ in range(n_interfaces)]
    vf_2 = [[] for _ in range(n_interfaces)]
    vb_2 = [[] for _ in range(n_interfaces)]


    for i1 in range(n_bulks):

        vf_1[i1] = dot_wl(Tf[i1], v0) # pass through front surface
        vr[i1].append(dot_wl(Rf[i1], v0)) # reflected from front surface
        a[i1].append(dot_wl(Af[i1], v0)) # absorbed in front surface at first interaction
        v_xr = xr.DataArray(v0, dims = ['wl', 'global_index'],
                                           coords = {'wl': If[i1].coords['wl'],
                                                     'global_index': np.arange(0, len(angle_vector)/2)})
        int_power = xr.dot(v_xr, If[i1], dims = 'global_index')
        scale = (np.sum(dot_wl(Af[i1],v0), 1)/int_power).fillna(0)
        a_prof[i1].append((scale*xr.dot(v_xr, Pf[i1], dims = 'global_index')).data)
        power = np.sum(vf_1[i1], axis=1)

        # rep
        i2=1

        while np.any(power > 1e-5):
            print(i2)

            vb_1[i1] = dot_wl(D[i1], vf_1[i1]) # pass through bulk, downwards

            v_xr = xr.DataArray(vb_1[i1], dims=['wl', 'global_index'],
                                coords={'wl': If[i1+1].coords['wl'],
                                        'global_index': np.arange(0, len(angle_vector) / 2)})
            int_power = xr.dot(v_xr, If[i1+1], dims='global_index')
            scale = (np.sum(dot_wl(Af[i1+1], vb_1[i1]), 1) / int_power).fillna(0)
            a_prof[i1+1].append((scale * xr.dot(v_xr, Pf[i1+1], dims='global_index')).data)


            #remaining_power.append(np.sum(vb_1, axis=1))
            A[i1].append(np.sum(vf_1[i1], 1) - np.sum(vb_1[i1], 1))
            vb_2[i1] = dot_wl(Rf[i1+1], vb_1[i1]) # reflect from back surface
            vf_2[i1] = dot_wl(D[i1], vb_2[i1]) # pass through bulk, upwards

            v_xr = xr.DataArray(vf_2[i1], dims=['wl', 'global_index'],
                                coords={'wl': Ib[i1].coords['wl'],
                                        'global_index': np.arange(0, len(angle_vector) / 2)})
            int_power = xr.dot(v_xr, Ib[i1], dims='global_index')
            scale = (np.sum(dot_wl(Ab[i1], vf_2[i1]), 1) / int_power).fillna(0)
            a_prof[i1].append((scale * xr.dot(v_xr, Pb[i1], dims='global_index')).data)

            #remaining_power.append(np.sum(vf_2, axis=1))
            A[i1].append(np.sum(vb_2[i1], 1) - np.sum(vf_2[i1], 1))
            vf_1[i1] = dot_wl(Rb[i1], vf_2[i1]) # reflect from front surface
            power = np.sum(vf_1[i1], axis=1)

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

    for i2 in range(3):
        for i1 in range(n_interfaces):
            plt.figure()
            z = np.arange(0, np.sum(layer_widths[i1]), nm_spacing)
            plt.plot(z, a_prof[i1][i2, 0, :].T)
            plt.title(str(i2) + 'interface ' + str(i1))
            plt.show()

    R = np.array([np.sum(item, (0,2)) for item in vr])
    T = np.array([np.sum(item, (0,2)) for item in vt])
    A_bulk = np.array([np.sum(item, 0) for item in A])
    A_interface = np.array([np.sum(item, (0,2)) for item in a])
    profile = [np.sum(item, 0) for item in a_prof] # not necessarily same number of z coords per layer stack

    for i2 in range(num_wl):
        plt.figure()
        for i1 in range(n_interfaces):
            z = np.arange(0, np.sum(layer_widths[i1]), nm_spacing)
            plt.plot(z, profile[i1][i2].T)

        plt.show()

    plt.figure()
    for i2 in range(5):
        i1= 0
        z = np.arange(0, np.sum(layer_widths[i1]), nm_spacing)
        plt.plot(z, a_prof[i1][i2, 0, :])

    plt.figure()
    plt.plot(wls, R.T)
    plt.plot(wls, T.T)
    plt.plot(wls, A_interface.T)
    plt.plot(wls, A_bulk.T)
    plt.plot(wls, R[0] + T[0] + A_interface[0] + A_interface[1]+A_bulk[0])
    plt.legend(['R', 'T', 'front', 'back', 'bulk'])
    plt.show()

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

        while np.any(power > 1e-5):
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

    R = np.array([np.sum(item, (0, 2)) for item in vr])
    T = np.array([np.sum(item, (0, 2)) for item in vt])
    A_bulk = np.array([np.sum(item, 0) for item in A])
    A_interface = np.array([np.sum(item, (0, 2)) for item in a])


    plt.figure()
    plt.plot(wls, R.T)
    plt.plot(wls, T.T)
    plt.plot(wls, A_interface.T)
    plt.plot(wls, A_bulk.T)
    plt.plot(wls, R[0] + T[0] + A_interface[0] + A_interface[1] + A_bulk[0])
    plt.legend(['R', 'T', 'front', 'back', 'bulk'])
    plt.show()



