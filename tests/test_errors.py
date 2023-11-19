import numpy as np
from pytest import raises

def pol_error():
    from rayflare.rigorous_coupled_wave_analysis.rcwa import process_pol

    pol_test = ["s", "p", "u", (np.sqrt(2) / 2, np.sqrt(2) / 2)]
    pol_output = [(1, 0), (0, 1), (np.sqrt(2) / 2, np.sqrt(2) / 2), (np.sqrt(2) / 2, np.sqrt(2) / 2)]

    for i1, pol in enumerate(pol_test):
        assert process_pol(pol) == pol_output[i1]

    # check error is raised for invalid polarisation
    with raises(ValueError):
        process_pol("invalid")





