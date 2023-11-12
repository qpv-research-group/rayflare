from rayflare.rigorous_coupled_wave_analysis.rcwa import rcwa_structure
from rayflare.transfer_matrix_method import tmm_structure
from solcore import material, si
from solcore.solar_cell import Layer
from rayflare.options import default_options
import numpy as np
import matplotlib.pyplot as plt
from time import time

x = 1000
size = ((x, 0), (0, x))
hw = x / 4

Air = material("Air")()
Si = material("Si")()
SiO2 = material("SiO2")()
Ag = material("Ag")()

geom_1 = [{"type": "circle", "mat": SiO2, "radius": hw, "center": (0, 0)}]
geom_2 = [{"type": "circle", "mat": Air, "radius": hw, "center": (0, 0)}]

stack = [
    Layer(si("100nm"), SiO2),
    Layer(si("200nm"), Si, geometry=geom_1),
    Layer(si("3000nm"), Si),
    Layer(si("200nm"), Si, geometry=geom_1),
]

options = default_options()
options.wavelength = np.linspace(300, 1250, 120) * 1e-9
options.orders = 50
options.parallel = True
options.theta_in = 0 * np.pi / 180
options.pol = "u"

strt = rcwa_structure(stack, size, options, Air, Ag)
# IS = rcwa_structure_inkstone(stack, size, options, Air, Ag)

tmm_strt = tmm_structure(stack, Air, Ag)

RAT_planar = tmm_strt.calculate(options)

# options.RCWA_method = 'S4'
# strt.get_fields(3, 500, options, depth=100)
# options.RCWA_method = 'Inkstone'
# strt.get_fields(3, 500, options, depth=100)

start = time()
options.RCWA_method = "S4"
S4_res = strt.calculate(options)
S4_res = strt.calculate_profile(options)
print(time() - start)
#
# start = time()
# options.RCWA_method = 'Inkstone'
# IS_res = strt.calculate(options)
# IS_res = strt.calculate_profile(options)
# print(time() - start)

plt.figure()
plt.plot(options.wavelength * 1e9, S4_res["R"], "-r")
# plt.plot(options.wavelength*1e9, IS_res["R"], '--r')

plt.plot(options.wavelength * 1e9, S4_res["T"], "-b")
# plt.plot(options.wavelength*1e9, IS_res["T"], '--b')

# plt.plot(options.wavelength*1e9, S4_res["A_per_layer"], '-g')
# plt.plot(options.wavelength*1e9, IS_res["A_per_layer"], '--g')

plt.plot(options.wavelength * 1e9, np.sum(S4_res["A_per_layer"], 1), "-y")
# plt.plot(options.wavelength*1e9, np.sum(IS_res["A_per_layer"], 1), '--y')

plt.plot(options.wavelength * 1e9, np.sum(RAT_planar["A_per_layer"], 1))
plt.show()


plt.figure()
plt.plot(S4_res["profile"][15])
# plt.plot(IS_res["profile"][15], '--')
plt.show()
#
# stack = [
#     Layer(si("50nm"), SiO2),
#     Layer(si("500nm"), Si),
#     Layer(si("1000nm"), GaAs,
#           )
# ]
#
# options = default_options()
# options.wavelength = np.linspace(300, 1250, 80)*1e-9
# options.orders = 60
# options.parallel = True
# options.theta_in = 45*np.pi/180
# options.pol = 's'
#
# strt = rcwa_structure(stack, size, options, Air, Ag)
# # IS = rcwa_structure_inkstone(stack, size, options, Air, Ag)
#
# start = time()
# options.RCWA_method = 'S4'
# S4_res = strt.calculate(options)
# S4_res = strt.calculate_profile(options)
# print(time() - start)
#
# start = time()
# options.RCWA_method = 'Inkstone'
# IS_res = strt.calculate(options)
# IS_res = strt.calculate_profile(options)
# print(time() - start)
#
# plt.figure()
# plt.plot(options.wavelength*1e9, S4_res["R"], '-r')
# plt.plot(options.wavelength*1e9, IS_res["R"], '--r')
#
# plt.plot(options.wavelength*1e9, S4_res["T"], '-b')
# plt.plot(options.wavelength*1e9, IS_res["T"], '--b')
#
# plt.plot(options.wavelength*1e9, S4_res["A_per_layer"], '-g')
# plt.plot(options.wavelength*1e9, IS_res["A_per_layer"], '--g')
#
# plt.plot(options.wavelength*1e9, np.sum(S4_res["A_per_layer"], 1), '-y')
# plt.plot(options.wavelength*1e9, np.sum(IS_res["A_per_layer"], 1), '--y')
# plt.show()
#
# plt.figure()
# plt.plot(S4_res["profile"][5])
# plt.plot(IS_res["profile"][5], '--')
# plt.show()
#

#
# options.pol = 'u'
#
# S4 = rcwa_structure(stack, size, options, Air, Ag)
# IS = rcwa_structure_inkstone(stack, size, options, Air, Ag)
#
# start = time()
# S4_res = S4.calculate(options)
# print(time() - start)
# IS_res = IS.calculate(options)
#
# plt.figure()
# plt.plot(options.wavelength*1e9, S4_res["R"], '-r')
# plt.plot(options.wavelength*1e9, IS_res["R"], '--r')
#
# plt.plot(options.wavelength*1e9, S4_res["T"], '-b')
# plt.plot(options.wavelength*1e9, IS_res["T"], '--b')
#
# plt.plot(options.wavelength*1e9, S4_res["A_per_layer"], '-g')
# plt.plot(options.wavelength*1e9, IS_res["A_per_layer"], '--g')
#
# plt.plot(options.wavelength*1e9, np.sum(S4_res["A_per_layer"], 1), '-y')
# plt.plot(options.wavelength*1e9, np.sum(IS_res["A_per_layer"], 1), '--y')
# plt.show()
#
# S4.get_fields(1, 200, options)
