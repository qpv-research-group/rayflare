import numpy as np
from sparse import COO, save_npz
from rayflare.angles import fold_phi
from rayflare.utilities import get_matrices_or_paths

def lambertian_matrix(angle_vector, theta_intv, surf_name, structpath,
                      front_or_rear='front', save=True):
    """
    Generates a redistribution matrix for perfect Lambertian reflection.

    :param angle_vector: an angle_vector in the standard format
    :param theta_intv: the theta (polar) intervals (edges of the bins) to be used
    :param surf_name: surface name (for saving the matrix)
    :param options: dictionary of user options
    :param front_or_rear: generate matrix for 'front' or 'rear' incidence?
    :param save: Boolean, whether to save the resulting matrix (True) or only return it (False). Default True

    :return: the redistribution matrix in sparse COO format.
    """

    existing_mats, path_or_mats = get_matrices_or_paths(structpath, surf_name, front_or_rear)

    if existing_mats:
        return path_or_mats

    else:
        theta_values = np.unique(angle_vector[angle_vector[:,1] < np.pi/2,1])
        dtheta = np.diff(theta_intv[theta_intv <= np.pi/2])

        dP = np.cos(theta_values)*dtheta

        # matrix has indexing (out, in): row picks out 'out' entry, column picks out which v0 element

        # since it doesn't matter what the incidence angle is for Lambertian scattering, all the columns rows be identical!

        # how many phi entries are there for each theta?

        n_phis = [np.sum(angle_vector[:,1] == theta) for theta in theta_values]

        column = [x for sublist in [[dP[i1]/n]*n for i1, n in enumerate(n_phis)] for x in sublist]

        whole_matrix = np.vstack([column]*int(len(angle_vector)/2)).T

        # renormalize (rounding errors)

        whole_matrix_R = whole_matrix/np.sum(whole_matrix,0)

        whole_matrix_T = np.zeros_like(whole_matrix_R)

        whole_matrix = np.vstack([whole_matrix_R, whole_matrix_T])

        A_matrix = np.zeros((1,int(len(angle_vector)/2)))

        allArray = COO(whole_matrix)
        absArray = COO(A_matrix)

        if save:
            save_npz(path_or_mats[0], allArray)
            save_npz(path_or_mats[1], absArray)

        return allArray, absArray


def mirror_matrix(angle_vector, theta_intv, phi_intv, surf_name, options, structpath,
                  front_or_rear='front', save=True):
    """
    Generates a redistribution matrix for a perfect mirror (100% reflection).

    :param angle_vector: an angle_vector in the standard format
    :param theta_intv: the theta (polar) intervals (edges of the bins) to be used
    :param phi_intv: the phi (azimuthal) intervals (edges of the bins)
    :param surf_name: surface name (for saving the matrix)
    :param options: dictionary of user options
    :param front_or_rear: generate matrix for 'front' or 'rear' incidence?
    :param save: Boolean, whether to save the resulting matrix (True) or only return it (False). Default True

    :return: the redistribution matrix in sparse COO format.
    """

    existing_mats, path_or_mats = get_matrices_or_paths(structpath, surf_name, front_or_rear)

    if existing_mats:
        return path_or_mats

    else:
        if front_or_rear == "front":

            angle_vector_th = angle_vector[:int(len(angle_vector)/2),1]
            angle_vector_phi = angle_vector[:int(len(angle_vector)/2),2]

            phis_out = fold_phi(angle_vector_phi + np.pi, options['phi_symmetry'])


        else:
            angle_vector_th = angle_vector[int(len(angle_vector) / 2):, 1]
            angle_vector_phi = angle_vector[int(len(angle_vector) / 2):, 2]

            phis_out = fold_phi(angle_vector_phi + np.pi, options['phi_symmetry'])

        # matrix will be all zeros with just one '1' in each column/row. Just need to determine where it goes

        binned_theta = np.digitize(angle_vector_th, theta_intv, right=True) - 1

        bin_in = np.arange(len(angle_vector_phi))

        phi_ind = [np.digitize(phi, phi_intv[binned_theta[i1]], right=True) - 1 for i1, phi in enumerate(phis_out)]
        overall_bin = [np.argmin(abs(angle_vector[:,0] - binned_theta[i1])) + phi_i for i1, phi_i in enumerate(phi_ind)]

        whole_matrix = np.zeros((len(overall_bin)*2, len(overall_bin)))

        whole_matrix[overall_bin, bin_in] = 1


        A_matrix = np.zeros((1, len(overall_bin)))

        allArray = COO(whole_matrix)
        absArray = COO(A_matrix)

        if save:
            save_npz(path_or_mats[0], allArray)
            save_npz(path_or_mats[1], absArray)

        return allArray, absArray
