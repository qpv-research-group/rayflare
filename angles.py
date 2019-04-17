import numpy as np
import xarray as xr

def make_angle_vector(n_angle_bins, phi_sym, c_azimuth):
    sin_a_b = np.linspace(0, 1, n_angle_bins+1) # number of bins is between 0 and 90 degrees
    # even spacing in terms of sin(theta) rather than theta
    # will have the same number of bins between 90 and 180 degrees

    theta_intv= np.concatenate([np.arcsin(sin_a_b), np.pi-np.flip(np.arcsin(sin_a_b[:-1]))])

    theta_middle = (theta_intv[:-1] + theta_intv[1:])/2
    phi_intv = []
    angle_vector = np.empty((0,3))

    for i1, theta in enumerate(theta_middle):
        if theta > np.pi/2:
            ind = len(theta_intv)-(i1 + 1) # + 1 because Python is zero-indexed
        else:
            ind = i1 + 1

        phi_intv.append(np.linspace(0, phi_sym, np.ceil(c_azimuth*ind)+1))
        phi_middle = (phi_intv[i1][:-1] + phi_intv[i1][1:])/2

        angle_vector = np.append(angle_vector, np.array([np.array(len(phi_middle)*[i1]),
                                      np.array(len(phi_middle)*[theta]),
                                          phi_middle]).T, axis = 0)

    return theta_intv, phi_intv, angle_vector


    # folding phi back into first quadrant: need to make sure it is positive, so 0 to 360
    # not -180 to + 180. Then just do phi % phi_sym

    # -180 to +180 -> 0 to 360 degrees:
    # abs(angle//180)*360 + angle
    # full transformation: (abs(angle//np.pi)*2*np.pi + angle) % phi_sym

def fold_phi(phis, phi_sym):
    return (abs(phis//np.pi)*2*np.pi + phis) % phi_sym

def theta_summary(out_mat, angle_vector):
    out_mat = xr.DataArray(out_mat, dims=['index_out', 'index_in'],
                           coords={'theta_in': (['index_in'], angle_vector[:out_mat.shape[1],1]),
                                   'theta_out': (['index_out'], angle_vector[:out_mat.shape[0],1])})

    sum_mat = out_mat.groupby('theta_in').apply(np.sum, args=(1, None))
    sum_mat = sum_mat.groupby('theta_out').apply(np.sum, args=(0, None))

    R = sum_mat[sum_mat.coords['theta_out'] < np.pi / 2, :].reduce(np.sum, 'theta_out').data
    T = sum_mat[sum_mat.coords['theta_out'] > np.pi / 2, :].reduce(np.sum, 'theta_out').data

    return sum_mat.data, R, T

def theta_summary_A(A_mat, angle_vector):
    A_mat = xr.DataArray(A_mat, dims=['layer_out', 'index_in'],
                           coords={'theta_in': (['index_in'], angle_vector[:A_mat.shape[1],1]),
                                   'layer_out': 1+np.arange(A_mat.shape[0])})
    sum_mat = A_mat.groupby('theta_in').apply(np.sum, args=(1, None))

    return sum_mat.data
