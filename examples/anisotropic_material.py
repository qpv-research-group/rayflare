import numpy as np
from solcore import si, material
from solcore.structure import Layer
from solcore.constants import q, h, c
from solcore.interpolate import interp1d
from solcore.solar_cell import SolarCell

from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
from rayflare.transfer_matrix_method import tmm_structure
from rayflare.options import default_options

wavelengths = np.linspace(300, 1000, 6)*1e-9

options = default_options()

options.wavelengths = wavelengths
options.orders = 1


# [width of the layer in nm, wavelengths, n at these wavelengths, k at these wavelengths, geometry]

Air = material('Air')()

n_tensor = np.array([[[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
k_tensor = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

test_mat = [100, np.array([400, 900]), n_tensor, k_tensor, []]

rcwa_setup = rcwa_structure([test_mat], size=((100, 0), (0, 100)), options=options, incidence=Air, transmission=Air)