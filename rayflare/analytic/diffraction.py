from rigorous_coupled_wave_analysis.rcwa import  fold_phi
import numpy as np
try:
    import S4
except Exception as err:
    print('WARNING: The RCWA solver will not be available because an S4 installation has not been found.')


def get_reciprocal_lattice(size, orders):

    S = S4.New(size, orders)

    f_mat = S.GetReciprocalLattice()

    return 2*np.pi*np.array(f_mat)

def get_order_directions(wl, size, m_max, incidence, transmission, theta, phi, phi_sym=np.pi/2):
    # wl in nm
    rl = get_reciprocal_lattice(size, 9)/(2*np.pi)

    fi_x = np.real((incidence.n(wl * 1e-9) / wl) * np.sin(theta) *
                   np.sin(phi))
    fi_y = np.real((incidence.n(wl * 1e-9) / wl) * np.sin(theta) *
                   np.cos(phi))

    xy = np.arange(-m_max, m_max+1, 1)

    xv, yv = np.meshgrid(xy, xy)
    #print('inc', fi_x, fi_y)

    fr_x = np.add(fi_x[:, None], xv.flatten() * rl[0][0] + yv.flatten() * rl[1][0])
    fr_y = np.add(fi_y[:, None], xv.flatten() * rl[0][1] + yv.flatten() * rl[1][1])

    eps_inc = (incidence.n(wl * 1e-9) + 1j * incidence.k(wl * 1e-9)) ** 2

    eps_sub = (transmission.n(wl * 1e-9) + 1j * transmission.k(wl * 1e-9)) ** 2
    # print('eps/lambda', l_oc[0]/(wl**2))
    fr_z = np.sqrt((eps_inc / (wl ** 2))[:, None] - fr_x ** 2 - fr_y ** 2)

    ft_z = np.sqrt((eps_sub / (wl ** 2))[:, None] - fr_x ** 2 - fr_y ** 2)

    # print('ref', fr_x, fr_y, fr_z)

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

    # f_mat = RAT['reciprocal']
    # fg_1x = f_mat[0][0]
    # fg_1y = f_mat[0][1]
    # fg_2x = f_mat[1][0]
    # fg_2y = f_mat[1][1]
    #
    # fr_x = basis_set[:, 0] * fg_1x + basis_set[:, 1] * fg_2x
    # fr_y = basis_set[:, 0] * fg_1y + basis_set[:, 1] * fg_2y

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


def get_order_directions_nk(wl, size, m_max, incidence, transmission, theta, phi, phi_sym=np.pi/2):
    # wl in nm
    rl = get_reciprocal_lattice(size, 9)
    fi_x = np.real((incidence/ wl) * np.sin(theta) *
                   np.sin(phi))
    fi_y = np.real((incidence/ wl) * np.sin(theta) *
                   np.cos(phi))

    xy = np.arange(-m_max, m_max+1, 1)

    xv, yv = np.meshgrid(xy, xy)
    #print('inc', fi_x, fi_y)

    fr_x = np.add(fi_x[:, None], xv.flatten() * rl[0][0] + yv.flatten() * rl[1][0])
    fr_y = np.add(fi_y[:, None], xv.flatten() * rl[0][1] + yv.flatten() * rl[1][1])

    eps_inc = incidence ** 2

    eps_sub = transmission ** 2
    # print('eps/lambda', l_oc[0]/(wl**2))
    fr_z = np.sqrt((eps_inc / (wl ** 2))[:, None] - fr_x ** 2 - fr_y ** 2)

    ft_z = np.sqrt((eps_sub / (wl ** 2))[:, None] - fr_x ** 2 - fr_y ** 2)

    # print('ref', fr_x, fr_y, fr_z)

    phi_rt = np.nan_to_num(np.arctan(fr_x / fr_y))
    phi_rt = fold_phi(phi_rt, phi_sym)
    theta_r = np.real(np.arccos(fr_z / np.sqrt(fr_x ** 2 + fr_y ** 2 + fr_z ** 2)))
    theta_t = np.pi - np.real(np.arccos(ft_z / np.sqrt(fr_x ** 2 + fr_y ** 2 + ft_z ** 2)))

    return {'order_index': [xv, yv], 'phi': phi_rt, 'theta_r': theta_r, 'theta_t': theta_t, 'fr_x': fr_x, 'fr_y': fr_y}
