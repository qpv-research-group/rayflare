from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
from rayflare.transfer_matrix_method import tmm_structure
from solcore import material, si
from solcore.structure import Layer
from rayflare.options import default_options
import numpy as np
import matplotlib.pyplot as plt
from solcore.light_source import LightSource
from solcore.constants import q
import seaborn as sns
from matplotlib import rc
from joblib import Parallel, delayed
from time import time
import pygmo as pg
from rayflare.analytic.diffraction import get_order_directions
# Paper: https://doi.org/10.1002/adom.201700585
MgF2 = material("MgF2")()  # MgF2 (SOPRA database)
Air = material("Air")()
Si = material("Si")()

pillar_material = MgF2

wavelengths = np.array([5e-6])

# lattice vectors for the grating. Units are in nm!
options = default_options()
options.wavelength = wavelengths
options.orders = 43
options.A_per_order = True
# options.pol = (1/np.sqrt(2), 1j/np.sqrt(2))
options.pol = 's'

theta_air = 1.3

# what is the angle in Si?
theta_Si = np.arcsin(np.sin(theta_air) * 1 / Si.n(wavelengths))

wavelengths_2 = np.linspace(4000, 6000, 50) * 1e-9

p_necessary = 1e9*2*np.max(options.wavelength)/np.sqrt(3)

# options.wavelength = wavelengths_2
x = 1.1*p_necessary
size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

diffracted_directions = get_order_directions(options.wavelength*1e9, size, 3, Si, Air, 0, 0, np.pi/3)

grating_circles = [{
    "type": "circle",
    "mat": pillar_material,
    "center": (0, 0),
    "radius": 0.25*x,
}]

layers = [Layer(material=Air, width=si("2000nm"), geometry=grating_circles)]

inc_angle = np.linspace(0, np.pi-0.1, 10)

T_per_angle = np.zeros(len(inc_angle))
R_per_angle = np.zeros(len(inc_angle))
T_per_order_per_angle = np.zeros((len(inc_angle), options.orders))

T_per_angle_TMM = np.zeros(len(inc_angle))
R_per_angle_TMM = np.zeros(len(inc_angle))

diff_angles = []
diff_powers = []
diff_orders = []

for i1, th in enumerate(inc_angle):

    options.theta_in = th
    S4_setup = rcwa_structure(layers, size=size, options=options, incidence=Si, transmission=Air)
    RAT = S4_setup.calculate(options)

    T_per_angle[i1] = RAT["T"][0]
    R_per_angle[i1] = RAT["R"][0]
    T_per_order_per_angle[i1] = RAT["T_per_order"][0]

    # sort by how much power is in each order:
    order_power = np.argsort(RAT['T_per_order'][0])[::-1]
    orders = np.array(RAT['basis_set'])
    orders = orders[order_power]
    power_sorted = RAT['T_per_order'][0][order_power]

    orders = orders[power_sorted > 1e-7]
    power_sorted = power_sorted[power_sorted > 1e-7]

    diffracted_directions = get_order_directions(options.wavelength * 1e9, size, 3, Si, Air, th, 0,
                                                 np.pi / 3)
    orders_analytical = diffracted_directions['order_index']
    orders_analytical_ind = [np.where(np.all((orders_analytical[0].flatten() == x[0], orders_analytical[1].flatten() == x[1]), axis=0))[0] for x in orders]
    # find matching direction:

    if RAT['T'][0] > 1e-3:
        theta_t = diffracted_directions['theta_t'][0][orders_analytical_ind]

        diff_angles.append(theta_t)
        diff_powers.append(power_sorted)
        diff_orders.append(orders)

    else:
        diff_angles.append([])
        diff_powers.append([])
        diff_orders.append([])


    # options.theta_in = theta_air
    # S4_setup_abs = rcwa_structure(layers, size=size, options=options, incidence=Air, transmission=Si)
    # RAT_abs = S4_setup_abs.calculate(options)

    TMM_setup = tmm_structure(layer_stack=[], incidence=Si, transmission=Air)
    RAT_planar = TMM_setup.calculate(options)

    T_per_angle_TMM[i1] = RAT_planar["T"][0]
    R_per_angle_TMM[i1] = RAT_planar["R"][0]

# TMM calculation
plt.figure()
plt.plot(inc_angle * 180 / np.pi, T_per_angle, label="Emission")
plt.plot(inc_angle * 180 / np.pi, T_per_angle_TMM, '--', label="Emission planar")
plt.show()

S4_setup.get_fourier_epsilon(1, options.wavelength[0]*1e9, options)