import numpy as np
from pytest import approx

def test_lambertian_scattering():
    from rayflare.matrix_formalism.ideal_cases import lambertian_matrix
    from rayflare.angles import make_angle_vector

    theta_intv, _, angle_vector = make_angle_vector(20, np.pi/2, 0.25)

    mat, _ = lambertian_matrix(angle_vector, theta_intv, 'test', 'test', 'front', False)

    R_ind = int(len(angle_vector)/2)

    assert np.sum(mat, 0).todense() == approx(1)
    assert np.sum(mat, 1)[R_ind:].todense() == approx(0)


def test_perfect_mirror():
    from rayflare.matrix_formalism.ideal_cases import mirror_matrix
    from rayflare.angles import make_angle_vector

    options = {'phi_symmetry': np.pi/4}

    theta_intv, phi_intv, angle_vector = make_angle_vector(20, options['phi_symmetry'], 0.25)

    R_ind = int(len(angle_vector)/2)

    mat, _ = mirror_matrix(angle_vector, theta_intv, phi_intv, 'test', options, 'test', 'front', False)

    assert np.sum(mat, 0).todense() == approx(1)
    assert np.sum(mat, 1)[:R_ind].todense() == approx(1)
    assert np.sum(mat, 1)[R_ind:].todense() == approx(0)

    mat, _ = mirror_matrix(angle_vector, theta_intv, phi_intv, 'test', options, 'test', 'rear', False)

    assert np.sum(mat, 0).todense() == approx(1)
    assert np.sum(mat, 1)[:R_ind].todense() == approx(0)
    assert np.sum(mat, 1)[R_ind:].todense() == approx(1)


def test_lambertian_process():
    from solcore.structure import Layer
    from solcore import material

    # rayflare imports
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.options import default_options

    # Thickness of bulk Ge layer
    bulkthick = 200e-9

    wavelengths = np.linspace(640, 1700, 6) * 1e-9

    # set options
    options = default_options()
    options.wavelengths = wavelengths
    options.n_theta_bins = 6
    options.c_azimuth = 0.25
    options.project_name = 'Lambertian'


    Al2O3 = material('Al2O3')()
    Ge = material('Ge')()
    Air = material('Air')()

    front_materials = [Layer(100e-9, Al2O3)]

    front_surf = Interface('TMM', layers=front_materials, name='TMM', coherent=True)
    back_surf = Interface('Lambertian', name='lambert')
    back_surf_2 = Interface('Mirror', name='mirror')

    bulk_Ge = BulkLayer(bulkthick, Ge, name='Ge_bulk')  # bulk thickness in m

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ge)
    SC_mirror = Structure([front_surf, bulk_Ge, back_surf_2], incidence=Air, transmission=Ge)

    process_structure(SC, options)
    process_structure(SC_mirror, options)

    RAT = calculate_RAT(SC, options)
    RAT_mirror = calculate_RAT(SC_mirror, options)

    assert RAT[0]['T'].data == approx(0)
    assert RAT_mirror[0]['T'].data == approx(0)
    assert np.all(RAT_mirror[0]['A_bulk'][0].data < RAT[0]['A_bulk'][0].data)

    # repeat to check if loading works correctly
    RAT = calculate_RAT(SC, options)
    RAT_mirror = calculate_RAT(SC_mirror, options)

    assert RAT[0]['T'].data == approx(0)
    assert RAT_mirror[0]['T'].data == approx(0)
    assert np.all(RAT_mirror[0]['A_bulk'][0].data < RAT[0]['A_bulk'][0].data)

