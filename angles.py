import numpy as np
import xarray as xr

def make_angle_vector(n_angle_bins, phi_sym, c_azimuth):
    """Makes the binning intervals & angle vector depending on the relevant options.
    :param n_angle_bins: number of bins per 90 degrees in the polar direction (theta)
    :param phi_sym: phi angle (in radians) for the rotational symmetry of the unit cell; e.g. for a square-based pyramid,
    pi/4
    :param c_azimuth: a number between 0 and 1 which determines how finely the space is discretized in the
    azimuthal direction. N_azimuth = c_azimuth*r where r is the index of the polar angle bin. N_azimuth is
    rounded up to the nearest intergest if c_azimuth is not an integer.
    :return theta_intv: edges of the theta (polar angle) bins
    :return phi_intv: list with the edges of the phi bins for every theta bin
    :return angle_vector: array where the first column is the r index (theta bin), the second column in
    the mean theta for that bin, and the third column is the mean phi for that bin.
    """
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

def fold_phi(phis, phi_sym):
    """'Folds' phi angles back into symmetry element from 0 -> phi_sym radians"""
    return (abs(phis//np.pi)*2*np.pi + phis) % phi_sym

def theta_summary(out_mat, angle_vector):
    """Accepts an RT redistribution matrix and sums it over all the azimuthal angle bins to create an output
    in terms of theta_in and theta_out.
    :param out_mat: an RT (or just R or T) redistribution matrix
    :param angle_vector: corresponding angle_vector array (output from make_angle_vector)
    :return sum_mat: the theta summary matrix
    :return R: the overall reflection probability for every incidence theta
    :return T: the overall transmission probaility for every incidence theta"""
    out_mat = xr.DataArray(out_mat, dims=['index_out', 'index_in'],
                           coords={'theta_in': (['index_in'], angle_vector[:out_mat.shape[1],1]),
                                   'theta_out': (['index_out'], angle_vector[:out_mat.shape[0],1])})

    sum_mat = out_mat.groupby('theta_in').apply(np.mean, args=(1, None))
    sum_mat = sum_mat.groupby('theta_out').apply(np.mean, args=(0, None))

    R = sum_mat[sum_mat.coords['theta_out'] < np.pi / 2, :].reduce(np.sum, 'theta_out').data
    T = sum_mat[sum_mat.coords['theta_out'] > np.pi / 2, :].reduce(np.sum, 'theta_out').data

    return sum_mat.data, R, T

def theta_summary_A(A_mat, angle_vector):
    """Accepts an absorption per layer redistribution matrix and sums it over all the azimuthal angle bins to create an output
    in terms of theta_in and theta_out.
    :param out_mat: an absorption redistribution matrix
    :param angle_vector: corresponding angle_vector array (output from make_angle_vector)
    :return sum_mat: the theta summary matrix
    """
    A_mat = xr.DataArray(A_mat, dims=['layer_out', 'index_in'],
                           coords={'theta_in': (['index_in'], angle_vector[:A_mat.shape[1],1]),
                                   'layer_out': 1+np.arange(A_mat.shape[0])})
    sum_mat = A_mat.groupby('theta_in').apply(np.sum, args=(1, None))

    return sum_mat.data
