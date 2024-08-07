from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
from solcore import material, si
from solcore.structure import Layer
from rayflare.options import default_options
import numpy as np
import matplotlib.pyplot as plt
import pygmo as pg
from rayflare.analytic.diffraction import get_order_directions
import seaborn as sns

# Paper: https://doi.org/10.1002/adom.201700585
MgF2 = material("MgF2")()  # MgF2 (SOPRA database)
Air = material("Air")()
Si = material("Si")()

class optimize_surface():

    def __init__(self,
                 wavelengths,
                 target_diffraction_angle=np.pi/4,
                 angle_tolerance=10*np.pi/180,
                 pol='s',
                 ):
        self.wavelengths = wavelengths
        # self.angle = target_diffraction_angle
        # self.tolerance = angle_tolerance
        self.target = target_diffraction_angle
        self.tolerance = angle_tolerance
        self.min_angle = np.pi - target_diffraction_angle - angle_tolerance
        self.max_angle = np.pi - target_diffraction_angle + angle_tolerance
        self.pol = pol

    def calculate(self, x):
        # wavelengths = np.array([1e-6])

        # lattice vectors for the grating. Units are in nm!
        # Note: annoyingly have to set options here internally every time, since the options object
        # doesn't like the parallel implementation for some reason. Hopefully will be fixed at some
        # point

        options = default_options()
        options.wavelength = self.wavelengths
        options.orders = 43
        options.detailed_rcwa = True
        options.parallel = True
        # options.pol = (1/np.sqrt(2), 1j/np.sqrt(2))
        options.pol = self.pol
        options.theta_in = 0

        Air = material("Air")()
        Si = material("Si")()

        size = ((x[0], 0), (x[0] / 2, np.sin(np.pi / 3) * x[0]))

        grating_circles = [{"type": "circle", "mat": Si, "center": (0, 0), "radius": x[1]*x[0]}]

        layers = [Layer(material=Air, width=si(x[2], "nm"), geometry=grating_circles)]

        S4_setup = rcwa_structure(layers, size=size, options=options, incidence=Air,
                                  transmission=Si)

        RAT = S4_setup.calculate(options)

        return RAT, options, size

    def fitness(self, x):

        RAT, options, size = self.calculate(x)

        # order_power = np.flip(np.argsort(RAT['T_per_order'], axis=1), axis=1) # dimensions: (wl, orders)
        orders = np.array(RAT['basis_set']) # dimensions: (orders, 2)
        # orders = orders[order_power] # dimensions: (wl, orders, 2)
        # power_sorted = np.sort(RAT['T_per_order'], axis=0) # dimensions: (wl, orders

        # orders = orders[power_sorted > 1e-7]
        # power_sorted = power_sorted[power_sorted > 1e-7]

        diffracted_directions = get_order_directions(self.wavelengths * 1e9, size, 4, Air, Si,
                                                     options.theta_in,
                                                     0,
                                                     np.pi / 3)
        orders_analytical = diffracted_directions['order_index']

        orders_analytical_ind = [np.where(
            np.all((orders_analytical[0].flatten() == x[0], orders_analytical[1].flatten() == x[1]),
                   axis=0))[0][0] for x in orders]

        theta_analytical = diffracted_directions['theta_t'][:, orders_analytical_ind]

        mask = np.all((theta_analytical > self.min_angle, theta_analytical < self.max_angle), axis=0)

        T_in_condition = np.sum(RAT['T_per_order'][mask])

        return [-T_in_condition/len(self.wavelengths)]

    def get_bounds(self):
        # [lower bounds for all parameters], [upper bounds for all parameters]
        return [(100, 0, 200), (10000, 0.5, 3000)]



crit_angle = np.arcsin(Air.n(5e-6)/Si.n(5e-6))
test_wl = np.linspace(4.5,  5.5,10)*1e-6

obj_class = optimize_surface(wavelengths=test_wl, target_diffraction_angle=45*np.pi/180,
                                   angle_tolerance=10*np.pi/180)
prob = pg.problem(obj_class)

n_generations = 40
pop_size = 5*len(prob.get_bounds()[0])

# algo = pg.algorithm(pg.de(gen=n_generations))
# pop = pg.population(prob, pop_size)
# pop = algo.evolve(pop)

algo = pg.algorithm(pg.de(gen=1)) # note: I am running this one generation at a time so I can get the
# best population at each step and plot how it evolves. If you just care about the final result,
# you can just set gen=n_generations here and not have the for loop below.

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


fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
ax1.plot(np.arange(n_generations) + 1, best_f, 'ro-', label="Champion fitness")
ax1.plot(np.arange(n_generations) + 1, mean_f, 'kx-',  label="Mean fitness")

ax1.axhline(-1, color='k', linestyle='--', label="Target")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness")
ax1.legend()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, n_params, figsize=(10, 4))
axes = axes.flatten()
for i1 in range(n_params):
    axes[i1].plot(np.arange(n_generations) + 1, best_x[:, i1], 'ro-', label="Best x")
    axes[i1].set_xlabel("Generation")
    axes[i1].set_ylabel("x")
    axes[i1].set_title("Parameter {}".format(i1))
    axes[i1].set_ylim(prob.get_bounds()[0][i1], prob.get_bounds()[1][i1])

plt.tight_layout()
plt.show()


RAT, options, size = obj_class.calculate(pop.champion_x)

diffracted_directions = get_order_directions(options.wavelength * 1e9, size, 4, Air, Si,
                                             options.theta_in,
                                             0,
                                             np.pi / 3)
orders_analytical = diffracted_directions['order_index']

orders = np.array(RAT['basis_set'])  # dimensions: (orders, 2)

# avg power per order over wl, determine which orders need to be included on x-axis
ind_power = np.mean(RAT['T_per_order'], axis=0) > 1e-7
all_labels = np.array([f'({x[0]}, {x[1]})' for x in RAT['basis_set']])[ind_power]

relevant_orders = np.array(RAT['basis_set'])[ind_power]

orders_analytical_ind = [np.where(
    np.all((orders_analytical[0].flatten() == x[0], orders_analytical[1].flatten() == x[1]),
           axis=0))[0][0] for x in relevant_orders]

theta_analytical = diffracted_directions['theta_t'][:, orders_analytical_ind]

# find angle for these orders:

colors = sns.color_palette("viridis", len(options.wavelength))
# plot power per order:

plt.figure(figsize=(5, 3.5))
for i1 in range(len(options.wavelength)):
    theta_increasing_order = np.argsort(theta_analytical[i1])
    theta_i1 = theta_analytical[i1][theta_increasing_order]
    T_i1 = RAT['T_per_order'][i1, ind_power][theta_increasing_order]
    # group together and add data points where theta is identical:
    values, indices, counts = np.unique(theta_i1, return_counts=True, return_index=True)
    subarrays = np.split(T_i1, indices)
    T_per_angle = np.array([np.sum(x) for x in subarrays])[1:]

    values = np.pi - values

    plt.scatter(180*values/np.pi, T_per_angle, color=colors[i1], facecolor='none',
                label=f'{options.wavelength[i1]*1e6:.2f} um')
# plt.scatter([f'({x[0]}, {x[1]})' for x in RAT['basis_set']], RAT['T_per_order'])
# rotate x axis labels 90 degrees:
# degree symbol:
plt.axvline(obj_class.target*180/np.pi, color='k', linestyle='-', alpha=0.5)
plt.axvline((obj_class.target + obj_class.tolerance)*180/np.pi, color='k', linestyle='--', alpha=0.5)
plt.axvline((obj_class.target - obj_class.tolerance)*180/np.pi, color='k', linestyle='--', alpha=0.5)
plt.xticks(rotation=90)
plt.xlabel("Direction in Si (°)")
plt.ylabel("Power")
plt.legend()
plt.tight_layout()
plt.show()


# orders_analytical_ind = [np.where(
#     np.all((orders_analytical[0].flatten() == x[0], orders_analytical[1].flatten() == x[1]),
#            axis=0))[0][0] for x in orders]
#
# theta_analytical = diffracted_directions['theta_t'][:, orders_analytical_ind]

mask = np.all((theta_analytical > obj_class.min_angle, theta_analytical < obj_class.max_angle), axis=0)

T_diff_per_wl = [np.sum(RAT['T_per_order'][i1, ind_power][mask[i1]]) for i1 in range(len(options.wavelength))]

plt.figure()
plt.plot(options.wavelength*1e6, RAT['T'], '--k', label='Total transmission')
plt.plot(options.wavelength*1e6, T_diff_per_wl, '-k', label='Diffracted into target angle')
plt.xlabel("Wavelength (um)")
plt.ylabel("Transmission")
plt.legend()
plt.tight_layout()
plt.show()

