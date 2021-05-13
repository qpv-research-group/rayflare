from rayflare.rigorous_coupled_wave_analysis.rcwa import fold_phi, get_reciprocal_lattice
import numpy as np

try:
    import S4
except Exception as err:
    print('WARNING: The RCWA solver will not be available because an S4 installation has not been found.')

def get_order_directions(wl, size, m_max, incidence, transmission, theta, phi, phi_sym=np.pi/2):
    """
    Returns analytically-calculated order directions for given incidence and transmission media, incidence
    angles (polar and azimuthal)
    :param wl: wavelength in nm (array)
    :param size: tuple of lattice vectors
    :param m_max: maximum index of the diffraction orders, results are returned for all combinations up to this value
    :param incidence: incidence medium. Can be a Solcore material, constant, or an array/list matching the length of the wavelength array
    :param transmission: transmission medium. Can be a Solcore material, constant, or an array/list matching the length of the wavelength array
    :param theta: polar incidence angle in radians
    :param phi: azimuthal incidence angle in radians
    :param phi_sym: phi values for which the grating is unique
    :return: dictionary containing the diffraction orders (order_index), and the corresponding azimuthal angle (phi),
    polar angle in reflection (theta_r) and transmission (theta_t), real space in-plane magnitudes (fr_x and fr_y)
    and reciprocal space in-plane magnitude (kxy)
    """

    rl = get_reciprocal_lattice(size, 9)

    if isinstance(incidence, (list, np.ndarray)):
        if len(incidence) == len(wl):
            n_in = np.real(incidence)
            k_in = np.imag(incidence)
        else:
            raise ValueError("Optical constant array of incidence medium has different length to wavelength array")

    elif isinstance(incidence, (int, float)):
        n_in = np.real(incidence)
        k_in = np.imag(incidence)

    else:
        n_in = incidence.n(wl * 1e-9)
        k_in = incidence.k(wl * 1e-9)

    if isinstance(transmission, (list, np.ndarray)):
        if len(transmission) == len(wl):
            n_out = np.real(transmission)
            k_out = np.imag(transmission)
        else:
            raise ValueError("Optical constant array of incidence medium has different length to wavelength array")

    elif isinstance(transmission, (int, float)):
        n_out = np.real(transmission)
        k_out = np.imag(transmission)

    else:
        n_out = transmission.n(wl * 1e-9)
        k_out = transmission.k(wl * 1e-9)

    fi_x = np.real((n_in / wl) * np.sin(theta) *
                   np.sin(phi))
    fi_y = np.real((n_in / wl) * np.sin(theta) *
                   np.cos(phi))

    xy = np.arange(-m_max, m_max+1, 1)

    xv, yv = np.meshgrid(xy, xy)

    fr_x = np.add(fi_x[:, None], xv.flatten() * rl[0][0] + yv.flatten() * rl[1][0])
    fr_y = np.add(fi_y[:, None], xv.flatten() * rl[0][1] + yv.flatten() * rl[1][1])

    eps_inc = (n_in + 1j * k_in) ** 2

    eps_sub = (n_out + 1j * k_out) ** 2

    fr_z = np.sqrt((eps_inc / (wl ** 2))[:, None] - fr_x ** 2 - fr_y ** 2)

    ft_z = np.sqrt((eps_sub / (wl ** 2))[:, None] - fr_x ** 2 - fr_y ** 2)

    phi_rt = np.nan_to_num(np.arctan(fr_x / fr_y))
    phi_rt = fold_phi(phi_rt, phi_sym)
    theta_r = np.real(np.arccos(fr_z / np.sqrt(fr_x ** 2 + fr_y ** 2 + fr_z ** 2)))
    theta_t = np.pi - np.real(np.arccos(ft_z / np.sqrt(fr_x ** 2 + fr_y ** 2 + ft_z ** 2)))

    kxy = np.sqrt(np.sum((xv.flatten()[:, None] * 2*np.pi*rl[0] + yv.flatten()[:, None] * 2*np.pi*rl[1]) ** 2, 1))

    return {'order_index': [xv, yv], 'phi': phi_rt, 'theta_r': theta_r,
            'theta_t': theta_t, 'fr_x': fr_x, 'fr_y': fr_y, 'k_xy': kxy}


def group_diffraction_orders(size, basis_set, per_order=None):
    lv = np.array(size).T
    a1 = lv[:, 0]
    a2 = lv[:, 1]

    # reciprocal lattice vector
    R = np.array([[0, -1], [1, 0]])

    b1 = 2 * np.pi * np.dot(R, a2) / (np.dot(a1, np.dot(R, a2)))
    b2 = 2 * np.pi * np.dot(R, a1) / (np.dot(a2, np.dot(R, a1)))


    kxy = np.sqrt(np.sum((np.array(basis_set)[:, 0][:, None] * b1 + np.array(basis_set)[:, 1][:, None] * b2) ** 2, 1))

    unique_kxy = np.unique(np.round(kxy, 10),
                         return_index=True, return_counts=True, return_inverse=True)



    if per_order is not None:

        indices = unique_kxy[2]
        unique_power = np.zeros((per_order.shape[0], len(unique_kxy[0])))
        indiv_powers = [[] for _ in range(len(unique_kxy[0]))]
        for j1, ind in enumerate(indices):
            unique_power[:, ind] = unique_power[:, ind] + per_order[:, j1]
            indiv_powers[ind].append(per_order[0, j1])

        degen = [len(np.unique(np.round(x, 10))) for x in indiv_powers]

        return {'k_xy': unique_kxy[0], 'reps': unique_kxy[3], 'per_order': unique_power, 'degeneracy': degen}

    else:
        return {'k_xy': unique_kxy[0], 'reps': unique_kxy[3]}
