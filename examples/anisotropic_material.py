import numpy as np
from solcore import si, material
from solcore.structure import Layer
from solcore.constants import q, h, c
from solcore.interpolate import interp1d
from solcore.solar_cell import SolarCell

from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
from rayflare.transfer_matrix_method import tmm_structure
from rayflare.options import default_options

import matplotlib.pyplot as plt

wavelengths = np.linspace(300, 1000, 6)*1e-9

options = default_options()

options.wavelengths = wavelengths
options.orders = 1
options.parallel = False

# [width of the layer in nm, wavelengths, n at these wavelengths, k at these wavelengths, geometry]

Air = material('Air')()
Si = material("Si")()

n_tensor = np.array([[[2, 0, 0], [0, 2, 0], [0, 0, 2]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
k_tensor = np.array([[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]]])

test_mat = [100, np.array([400, 900]), n_tensor, k_tensor, []]
test_mat2 = [200, np.array([400, 900]), n_tensor, k_tensor, []]

rcwa_setup = rcwa_structure([test_mat, test_mat2], size=((100, 0), (0, 100)), options=options, incidence=Air, transmission=Si)

RAT = rcwa_setup.calculate(options)

plt.figure()
plt.plot(wavelengths*1e9, RAT["A_per_layer"])
plt.plot(wavelengths*1e9, RAT["R"] + RAT["T"] + np.sum(RAT["A_per_layer"], 1))
plt.show()

# import S4
#
# S = S4.New(Lattice=((1,0),(0,1)), NumBasis=100)
#
# S.SetMaterial(Name = 'Silicon', Epsilon = (
#         (12+0.01j, 0, 0),
#         (0, 12+0.01j, 0),
#         (0, 0, 12+0.01j)
#         ))
#
#
# S.SetMaterial(Name = 'Vacuum', Epsilon = (
#         (1, 0, 0),
#         (0, 1, 0),
#         (0, 0, 1)
#         ))
#
# S.SetMaterial(Name = 'mat2', Epsilon = (
#         (10+0.02j, 0, 0),
#         (0, 12+0.01j, 0),
#         (0, 0, 13+0.03j)
#         ))
#
#
# S.AddLayer('inc', 0, 'Vacuum')
# S.AddLayer(Name = 'slab', Thickness = 0.6, Material = 'Silicon')
# S.AddLayer(Name = 'slab2', Thickness = 0.6, Material = 'mat2')
# S.AddLayer('trn', 0, 'Vacuum')
#
# S.SetRegionCircle(
#         Layer = 'slab',
#         Material = 'Vacuum',
#         Center = (0,0),
#         Radius = 0.2
# )
#
# S.SetExcitationPlanewave(
#         IncidenceAngles=(
#                 10, # polar angle in [0,180)
#                 30  # azimuthal angle in [0,360)
#         ),
#         sAmplitude = 0.707+0.707j,
#         pAmplitude = 0.707-0.707j,
#         Order = 0
# )
#
# S.SetFrequency(1.2)
#
# from time import time
#
# start = time()
#
# (forw,back) = S.GetAmplitudes(Layer = 'trn', zOffset = 0)
#
# print(time()-start)