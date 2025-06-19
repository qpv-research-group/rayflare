import numpy as np
import matplotlib.pyplot as plt
from rayflare.textures import regular_pyramids, planar_surface
from rayflare.ray_tracing import rt_structure
from solcore import material
from rayflare.options import default_options
from solcore.structure import Layer
from time import time

# structure:

# ---------------- (glass_text, interface 0)
# | glass          | (bulk 0)
# ---------------- (rear_glass_text, interface 1)
# | SiN + GaAs     | (interface 1 layers on rear_glass_text)
# ----------------
# | "glass"        | (bulk 1) - would be some kind of epoxy, modelled as glass since n ~ 1.5
# /\/\/\/\/\/\/\/\/\ (Si_front_text, interface 2)
# | Si            | (bulk 2)
# ---------------- (back_text, interface 3)

options = default_options()
wl = np.linspace(300, 1150, 40) * 1e-9

options.wavelength = wl

options.nx = 40
options.ny = 40
options.n_rays = 1 * options.nx ** 2
options.project_name = 'glass_GaAs_Si'
options.randomize_surface = True

options.theta_in = 0 * np.pi / 180
options.phi_in = 0 * np.pi / 180
options.I_thresh = 0.0002
options.parallel = True
options.maximum_passes = 50

SiOx = material("SiO2")()
Si = material("Si")()
Air = material("Air")()
SiN = material("Si3N4")()
GaAs = material("GaAs")()
glass = material("BK7")()

layers = [Layer(70e-9, SiN), Layer(100e-9, GaAs)]

glass_text = planar_surface()

rear_glass_text = planar_surface(interface_layers=layers)

# for opening angle of 50 degrees, there will be exactly two bounces for normally-incident rays
Si_front_text = regular_pyramids(50, True)

back_text = planar_surface()

rtstr_text = rt_structure(
    textures=[glass_text, rear_glass_text, Si_front_text, back_text],
    materials=[glass, glass, Si],
    widths=[10e-6, 10e-6, 70e-6],
    incidence=Air, transmission=Air,
    use_TMM=True, options=options,
    save_location="current",
    overwrite=True,
)

options.pol = 's'
for surf in rtstr_text.surfaces:
    surf.analytical = True

RAT_s_a = rtstr_text.calculate(options)

for surf in rtstr_text.surfaces:
    surf.analytical = False

RAT_s_f = rtstr_text.calculate(options)

options.pol = 'p'

for surf in rtstr_text.surfaces:
    surf.analytical = True

start = time()
RAT_p_a = rtstr_text.calculate(options)
print("analytical time taken: ", time() - start)

for surf in rtstr_text.surfaces:
    surf.analytical = False

start = time()
RAT_p_f = rtstr_text.calculate(options)
print("full time taken: ", time() - start)

options.pol = 'u'
for surf in rtstr_text.surfaces:
    surf.analytical = True
RAT_u_a = rtstr_text.calculate(options)

for surf in rtstr_text.surfaces:
    surf.analytical = False
RAT_u_f = rtstr_text.calculate(options)

titles = ['s-polarized', 'p-polarized', 'unpolarized']

# a = analytical, f = full

for i1, [rat_f, rat_a] in enumerate(
        zip([RAT_s_f, RAT_p_f, RAT_u_f], [RAT_s_a, RAT_p_a, RAT_u_a])):
    plt.figure()
    plt.plot(wl * 1e9, rat_a['R'], '-r', label='R a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['R'], '--r', label='R f', alpha=0.5)

    plt.plot(wl * 1e9, rat_a['R0'], '-c', label='R0 a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['R0'], '--c', label='R0 f', alpha=0.5)

    plt.plot(wl * 1e9, np.sum(rat_a['A_per_interface'][1], 1), '-b', label='A int 1 (SiN + GaAs) a', alpha=0.5)
    plt.plot(wl * 1e9, np.sum(rat_f['A_per_interface'][1], 1), '--b', label='A int 1 (SiN + GaAs) f', alpha=0.5)

    # plt.plot(wl * 1e9, rat_a['A_per_layer'][:, 0], '-g', label='A bulk 0 (glass) a', alpha=0.5)
    # plt.plot(wl * 1e9, rat_f['A_per_layer'][:, 0], '--g', label='A bulk 0 (glass) f', alpha=0.5)
    # # plt.plot(wl * 1e9, tmm['A_per_layer'][:, 1], '-.g')

    plt.plot(wl * 1e9, rat_a['A_per_layer'][:, 2], '-k', label='A bulk 2 (Si) a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['A_per_layer'][:, 2], '--k', label='A bulk 12 (Si) f', alpha=0.5)

    plt.plot(wl * 1e9, rat_a['T'], '-y', label='T a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['T'], '--y', label='T f', alpha=0.5)

    # plt.plot(wl * 1e9, np.sum(rat_a['A_per_interface'][2], 1), '-k', label='A int 2 a', alpha=0.5)
    # plt.plot(wl * 1e9, np.sum(rat_f['A_per_interface'][2], 1), '--k', label='A int 2 f', alpha=0.5)

    plt.title(titles[i1])

    sum_f = (rat_f['R'] + rat_f['T'] + np.sum(rat_f['A_per_layer'], 1) +
             np.sum(rat_f['A_per_interface'][1], 1) + np.sum(rat_f['A_per_interface'][2], 1))
    sum_a = (rat_a['R'] + rat_a['T'] + np.sum(rat_a['A_per_layer'], 1) +
             np.sum(rat_a['A_per_interface'][1], 1) + np.sum(rat_a['A_per_interface'][2], 1))

    plt.plot(wl * 1e9, sum_a, '-m', label='sum a')
    plt.plot(wl * 1e9, sum_f, '--m', label='sum f')

    # place legend outside plot:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
