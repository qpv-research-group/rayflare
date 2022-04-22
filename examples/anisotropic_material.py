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

wavelengths = np.linspace(400, 800, 20)*1e-9

options = default_options()

options.wavelengths = wavelengths
options.orders = 1
options.parallel = False
options.A_per_order = True
options.pol = "p"
options.theta_in = 0
options.phi_in = 0

# [width of the layer in nm, wavelengths, n at these wavelengths, k at these wavelengths, geometry]

Air = material('Air')()
GaAs = material("GaAs")()
Si = material("Si")()
Ag = material("Ag")()

GaAs_n = GaAs.n(wavelengths)
GaAs_k = GaAs.k(wavelengths)

Si_n = Si.n(wavelengths)
Si_k = Si.k(wavelengths)

n_tensor = np.array([[[2, 0, 0], [0, 3, 0], [0, 0, 1]], [[1, 0, 0], [0, 2, 0], [0, 0, 2]]])
k_tensor = np.array([[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]]])

GaAs_n_tensor = np.array([np.diag([x,x,x]) for x in GaAs_n])
GaAs_k_tensor = np.array([np.diag([x,x,x]) for x in GaAs_k])

Si_n_tensor = np.array([np.diag([x,x,x]) for x in Si_n])
Si_k_tensor = np.array([np.diag([x,x,x]) for x in Si_k])

test_mat = [100, wavelengths*1e9, GaAs_n_tensor, GaAs_k_tensor, [{'type': 'rectangle', 'mat': Air,
                                                                  'center': (0, 0), 'angle': 0, 'halfwidths': (150, 150)}]]
test_mat2 = [1000, wavelengths*1e9, Si_n_tensor, Si_k_tensor, []]

test_mat = [100, np.array([400, 900]), n_tensor, k_tensor, []]
test_mat2 = [1000, np.array([400, 900]),  n_tensor, k_tensor, []]

rcwa_setup = rcwa_structure([Layer(100e-9, GaAs, geometry=[{'type': 'rectangle', 'mat': Air,
                                                                  'center': (0, 0), 'angle': 0, 'halfwidths': (150, 150)}]),
                             Layer(1e-6, Si)], size=((400, 0), (0, 400)), options=options, incidence=Air, transmission=Ag)

rcwa_setup_AS = rcwa_structure([test_mat, test_mat2], size=((400, 0), (0, 400)), options=options, incidence=Air, transmission=Ag)

options.pol = "s"
RAT_s_AS = rcwa_setup_AS.calculate(options)
# RAT_s = rcwa_setup.calculate(options)

options.pol = "p"
RAT_p_AS = rcwa_setup_AS.calculate(options)
# RAT_p = rcwa_setup.calculate(options)

plt.figure()
plt.plot(wavelengths*1e9, RAT_s_AS["A_per_layer"], '--')
plt.plot(wavelengths*1e9, RAT_s_AS["R"] + RAT_s_AS["T"] + np.sum(RAT_s_AS["A_per_layer"], 1))

plt.plot(wavelengths*1e9, RAT_p_AS["A_per_layer"])
plt.plot(wavelengths*1e9, RAT_p_AS["R"] + RAT_p_AS["T"] + np.sum(RAT_p_AS["A_per_layer"], 1))
plt.show()

# plt.figure()
# plt.plot(wavelengths*1e9, RAT_s["A_per_layer"])
# plt.plot(wavelengths*1e9, RAT_s["R"] + RAT_s["T"] + np.sum(RAT_s["A_per_layer"], 1))
#
# plt.plot(wavelengths*1e9, RAT_p["A_per_layer"])
# plt.plot(wavelengths*1e9, RAT_p["R"] + RAT_p["T"] + np.sum(RAT_p["A_per_layer"], 1))
# plt.show()

# prof = rcwa_setup.calculate_profile(options)
prof_AS = rcwa_setup_AS.calculate_profile(options)

plt.figure()
# plt.plot(prof['profile'][[0,20,50,100,249],:].T)
plt.plot(prof_AS['profile'][[0,5, 10],:].T, '--')
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