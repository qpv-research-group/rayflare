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

# Paper: https://doi.org/10.1002/adom.201700585
MgF2 = material("MgF2")()  # MgF2 (SOPRA database)
Air = material("Air")()
Si = material("Si")()

pillar_material = MgF2

wavelengths = np.array([5e-6])

# lattice vectors for the grating. Units are in nm!
options = default_options()
options.wavelength = wavelengths
options.orders = 50
options.A_per_order = True
# options.pol = (1/np.sqrt(2), 1j/np.sqrt(2))
options.pol = 's'

theta_air = 1.3

# what is the angle in Si?
theta_Si = np.arcsin(np.sin(theta_air) * 1 / Si.n(wavelengths))

wavelengths_2 = np.linspace(4000, 6000, 50) * 1e-9

p_necessary = 1e9*2*np.max(wavelengths)/np.sqrt(3)

options.wavelength = wavelengths_2
x = 1.3*p_necessary
size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))
grating_circles = [{
    "type": "circle",
    "mat": pillar_material,
    "center": (0, 0),
    "radius": 0.16*x,
}]

layers = [Layer(material=Air, width=si("2000nm"), geometry=grating_circles)]

options.theta_in = 0.5
# options.theta_in = theta_Si
S4_setup = rcwa_structure(layers, size=size, options=options, incidence=Si, transmission=Air)
RAT = S4_setup.calculate(options)

# options.theta_in = theta_air
S4_setup_abs = rcwa_structure(layers, size=size, options=options, incidence=Air, transmission=Si)
RAT_abs = S4_setup_abs.calculate(options)

TMM_setup = tmm_structure(layer_stack=[], incidence=Si, transmission=Air)
RAT_planar = TMM_setup.calculate(options)

# TMM calculation


plt.figure()
plt.plot(wavelengths_2 * 1e9, RAT["T"], label="Emission")
plt.plot(wavelengths_2 * 1e9, RAT["R"], label="Reflection")

plt.plot(wavelengths_2 * 1e9, RAT_abs["T"], '--k', label="Absorption")
plt.plot(wavelengths_2 * 1e9, RAT_abs["R"], '--r', label="Reflection")

plt.plot(wavelengths_2 * 1e9, RAT_planar["T"], '-.k', label="Planar transmission")
plt.plot(wavelengths_2 * 1e9, RAT_planar["R"], '-.r', label="Planar reflection")
plt.xlabel("Wavelength (nm)")
plt.ylabel("R/T")
plt.legend()
plt.show()

rad_list = np.linspace(100, 1000, 21)
lattice_constant = np.linspace(500, 2000, 31)

options.theta_in = 0
options.wavelength = np.linspace(4.5, 5.5, 8)*1e-6
options.parallel = True
# 21 radii, 31 lattice constants: miniconda python takes 101 s
# "normal" python installed via pyenv takes 43 s

def fixed_lattice_constant(x, rad_list):

    size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

    print(x)

    T_result = np.zeros(len(rad_list))
    T_amp = np.zeros(len(rad_list), dtype=complex)
    inc_amp = np.zeros(len(rad_list), dtype=complex)

    for j1, rad in enumerate(rad_list):
        print(j1)

        grating_circles = [{"type": "circle", "mat": pillar_material, "center": (0, 0), "radius": rad}]

        layers = [Layer(material=Air, width=si("2000nm"), geometry=grating_circles)]

        S4_setup = rcwa_structure(layers, size=size, options=options, incidence=Si, transmission=Air)
        RAT = S4_setup.calculate(options)

        T_result[j1] = np.mean(RAT["T"])
        T_amp[j1] = RAT["T_amplitudes"][0][0][0]
        inc_amp[j1] = RAT['R_amplitudes'][0][0][0]

    return T_result, T_amp, inc_amp

# start = time()
# allres = Parallel(n_jobs=-1, prefer="threads")(delayed(fixed_lattice_constant)(x, rad_list) for x in lattice_constant)
# print("Time taken: ", time() - start)
#
# T_result = np.stack([item[0] for item in allres])
# phase_result = np.stack([item[1] for item in allres])
# inc_amp = np.stack([item[2] for item in allres])
#
# # calculate phase and put it in the range 0, 2pi
#
# # maximum T:
# max_T = np.nanmax(T_result)
# max_T_idx = np.unravel_index(np.nanargmax(T_result), T_result.shape)
#
# print("Maximal T: ", max_T, "radius:", rad_list[max_T_idx[1]], "lattice constant:", lattice_constant[max_T_idx[0]])
# plt.figure()
# plt.imshow(T_result, aspect="auto",
#            extent=[rad_list[0], rad_list[-1], lattice_constant[0], lattice_constant[-1]],
#            origin="lower", cmap="Spectral_r")
# plt.plot(rad_list[max_T_idx[1]], lattice_constant[max_T_idx[0]], 'kx')
# plt.colorbar()
# plt.xlabel("Radius (nm)")
# plt.ylabel("Lattice constant (nm)")
# plt.title("Transmission")
# plt.show()

class optimize_surface():

    def __init__(self, wavelengths, angles):
        self.wavelengths = wavelengths
        self.inc_angles = angles
        pass

    def calculate(self, x, angle):
        # wavelengths = np.array([1e-6])

        # lattice vectors for the grating. Units are in nm!
        options = default_options()
        options.wavelength = self.wavelengths
        options.orders = 50
        # options.A_per_order = True
        options.parallel = True
        # options.pol = (1/np.sqrt(2), 1j/np.sqrt(2))
        options.pol = 's'

        Air = material("Air")()
        Si = material("Si")()

        size = ((x[0], 0), (x[0] / 2, np.sin(np.pi / 3) * x[0]))

        grating_circles = [{"type": "circle", "mat": Si, "center": (0, 0), "radius": x[1]*x[0]}]

        layers = [Layer(material=Air, width=si(x[2], "nm"), geometry=grating_circles)]

        S4_setup = rcwa_structure(layers, size=size, options=options, incidence=Si,
                                  transmission=Air)


        options.theta_in = angle
        RAT = S4_setup.calculate(options)

        return RAT

    def fitness(self, x):

        T_angle = np.zeros(len(self.inc_angles))
        for i1, angle in enumerate(self.inc_angles):
            RAT = self.calculate(x, angle)
            T_angle[i1] = np.mean(RAT['T'])

        T_res = np.mean(T_angle)

        return [-T_res]

    def get_bounds(self):
        # [lower bounds for all parameters], [upper bounds for all parameters]
        return [(0, 0, 200), (2000, 1, 3000)]

crit_angle = np.arcsin(Air.n(5e-6)/Si.n(5e-6))
test_wl = np.linspace(4.5, 5.5, 8)*1e-6
angles = np.linspace(0, 0.99*crit_angle, 5)
prob = pg.problem(optimize_surface(test_wl, angles))

n_generations = 30
pop_size = 5*len(prob.get_bounds()[0])

# algo = pg.algorithm(pg.de(gen=n_generations))
# pop = pg.population(prob, pop_size)
# pop = algo.evolve(pop)

algo = pg.algorithm(pg.de(gen=1))
n_params = len(prob.get_bounds()[0])

best_f = np.zeros(n_generations)
mean_f = np.zeros(n_generations)
best_x = np.zeros((n_generations, n_params))

pop = pg.population(prob, pop_size)

for i1 in range(n_generations):

    pop = algo.evolve(pop)
    print(i1, pop.champion_f[0])
    best_f[i1] = pop.champion_f[0]
    mean_f[i1] = np.mean(pop.get_f())
    best_x[i1] = pop.champion_x

print(pop.champion_x, pop.champion_f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(np.arange(n_generations) + 1, best_f, 'ro-', label="Best fitness")
ax2.plot(np.arange(n_generations) + 1, mean_f, 'kx-',  label="Mean fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness")
ax2.set_xlabel("Generation")

ax1.set_title("Champion fitness")
ax2.set_title("Mean population fitness")
plt.show()

fig, axes = plt.subplots(1, n_params, figsize=(12, 6))
axes = axes.flatten()
for i1 in range(n_params):
    axes[i1].plot(np.arange(n_generations) + 1, best_x[:, i1], 'ro-', label="Best x")
    axes[i1].set_xlabel("Generation")
    axes[i1].set_ylabel("x")
    axes[i1].set_title("Parameter {}".format(i1))
    axes[i1].set_ylim(prob.get_bounds()[0][i1], prob.get_bounds()[1][i1])
plt.show()

ax1.set_title("Champion fitness")
ax2.set_title("Mean population fitness")
plt.show()

angles_2 = np.linspace(0, np.pi-0.1, 10)
cols = sns.cubehelix_palette(len(angles_2), start=.5, rot=-.75)

plt.figure()
for i1, th in enumerate(angles_2):
    RAT = optimize_surface(np.linspace(4.5, 5.5, 50)*1e-6, angles).calculate(pop.champion_x, th)

    plt.plot(wavelengths_2 * 1e9, RAT["T"], label=np.round(th*180/np.pi, 1), color=cols[i1])

plt.xlabel("Wavelength (nm)")
plt.ylabel("R/T")
plt.legend()
plt.show()