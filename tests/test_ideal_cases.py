import numpy as np
from pytest import approx

def test_lambertian_scattering():
    from rayflare.matrix_formalism.ideal_cases import lambertian_matrix
    from rayflare.angles import make_angle_vector

    theta_intv, phi_intv, angle_vector = make_angle_vector(20, np.pi/2, 0.25)

    mat = lambertian_matrix(angle_vector, theta_intv, 'test', 'test', 'front', False)

    R_ind = int(len(angle_vector)/2)

    assert np.sum(mat, 0).todense() == approx(1)
    assert np.sum(mat, 1)[R_ind:].todense() == approx(0)

def test_perfect_mirror():
    from rayflare.matrix_formalism.ideal_cases import mirror_matrix
    from rayflare.angles import make_angle_vector

    options = {'phi_symmetry': np.pi/4}

    theta_intv, phi_intv, angle_vector = make_angle_vector(20, options['phi_symmetry'], 0.25)

    mat = mirror_matrix(angle_vector, theta_intv, phi_intv, 'test', options, 'test', 'front', False)

    R_ind = int(len(angle_vector)/2)

    assert np.sum(mat, 0).todense() == approx(1)
    assert np.sum(mat, 1)[:R_ind].todense() == approx(1)
    assert np.sum(mat, 1)[R_ind:].todense() == approx(0)