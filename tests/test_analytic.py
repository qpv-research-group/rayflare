import numpy as np
from pytest import approx, mark
import sys


@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_get_order_directions():
    from rayflare.analytic.diffraction import get_order_directions
    from solcore import material

    Air = material("Air")()
    Si = material("Si")()

    x = 700

    size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

    wl = np.linspace(600, 1100, 50)

    res_ref = get_order_directions(wl, size, 3, Air, Si, 0.2, 0)

    in_oc = [1, 1.0, np.ones_like(wl), np.ones_like(wl).tolist()]
    out_oc = [Si.n(wl*1e-9), Si.n(wl*1e-9).tolist(), 4, 3.6]

    for ni in in_oc:
        for no in out_oc:
            res = get_order_directions(wl, size, 3, ni, no, 0.2, 0)

            assert np.all(np.array(res["order_index"]) == np.array(res_ref["order_index"]))
            assert np.min(res["theta_r"]) == approx(0.2)
            assert np.max(res["theta_r"]) == approx(np.pi/2)
            assert np.min(res["theta_t"]) == approx(np.pi/2)
            assert np.max(res["theta_t"]) < np.pi
            assert np.min(res["k_xy"]) == 0
            assert np.min(res["phi"]) >= 0
            assert np.max(res["phi"]) < np.pi/2
