from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
from rayflare.transfer_matrix_method import tmm_structure
from solcore import material, si
from solcore.structure import Layer
from rayflare.options import default_options
import numpy as np
import matplotlib.pyplot as plt
import pygmo as pg
from solcore.light_source import LightSource
from solcore.constants import q
import seaborn as sns
from matplotlib import rc
from joblib import Parallel, delayed
from time import time

# Anaconda python takes around 10 x longer than "normal" Python?

# Paper: https://doi.org/10.1002/adom.201700585
# let's say we want a specific phase and the highest transmission possible at that phase change.
# Our objective function should include both of these values.

# make our objective function: (T - T_target)^2 + (phase/pi - phase_target/pi)^2

class optimize_surface():

    def __init__(self, target_phase=np.pi):

        self.target_phase = target_phase

    def fitness(self, x):
        wavelengths = np.array([1e-6])

        # lattice vectors for the grating. Units are in nm!
        options = default_options()
        options.wavelength = wavelengths
        options.orders = 50
        options.A_per_order = True
        options.parallel = False
        # options.pol = (1/np.sqrt(2), 1j/np.sqrt(2))
        options.pol = 's'

        MgF2 = material("MgF2")()  # MgF2 (SOPRA database)
        Air = material("Air")()
        aSi = material("Si")()

        size = ((x[0], 0), (x[0] / 2, np.sin(np.pi / 3) * x[0]))

        grating_circles = [{"type": "circle", "mat": aSi, "center": (0, 0), "radius": x[1]}]

        layers = [Layer(material=Air, width=si(x[2], "nm"), geometry=grating_circles)]

        S4_setup = rcwa_structure(layers, size=size, options=options, incidence=MgF2,
                                  transmission=Air)
        RAT = S4_setup.calculate(options)

        T_res = RAT["T"][0]
        phase_res = np.angle(RAT["T_amplitudes"][0][0][0])
        if phase_res < 0:
            phase_res = 2 * np.pi - np.abs(phase_res) # put in range 0 to 2pi

        return [(T_res - 1)**2 + (phase_res/np.pi - self.target_phase/np.pi)**2]

    def get_bounds(self):
        # [lower bounds for all parameters], [upper bounds for all parameters]
        return [(1000, 100, 1500), (3000, 1000, 2500)]


prob = pg.problem(optimize_surface(target_phase=np.pi))

n_generations = 50
pop_size = 5*len(prob.get_bounds()[0])

# algo = pg.algorithm(pg.de(gen=n_generations))
# pop = pg.population(prob, pop_size)
# pop = algo.evolve(pop)

algo = pg.algorithm(pg.de(gen=1))

best_f = np.zeros(n_generations)
mean_f = np.zeros(n_generations)
best_x = np.zeros((n_generations, len(prob.get_bounds()[0])))

pop = pg.population(prob, pop_size)
for i1 in range(n_generations):

    pop = algo.evolve(pop)

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


