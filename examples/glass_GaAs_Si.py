import numpy as np
import matplotlib.pyplot as plt
from rayflare.textures import xyz_texture, regular_pyramids, planar_surface
from rayflare.ray_tracing import rt_structure
from solcore import material
import seaborn as sns
from rayflare.options import default_options
from solcore.structure import Layer
from rayflare.transfer_matrix_method import tmm_structure

options = default_options()
wl = np.linspace(300, 1150, 40) * 1e-9

options.wavelength = wl

options.nx = 50
options.ny = 50
options.n_rays = 1 * options.nx ** 2
# options.x_limits = [2.5, 7.5]
# options.y_limits = [2.5, 7.5]
options.project_name = 'thin_textured_Si'
options.lambertian_approximation = 0
options.randomize_surface = True
options.analytical_ray_tracing = True
options.theta_in = 30 * np.pi / 180
options.phi_in = 0 * np.pi / 180
options.I_thresh = 0.0002
options.parallel = True

SiOx = material("SiO2")()
Si = material("Si")()
Air = material("Air")()
SiN = material("Si3N4")()
GaAs = material("GaAs")()
glass = material("BK7")()

options.analytical_ray_tracing = 0

layers = [Layer(70e-9, SiN), Layer(100e-9, GaAs)]
# layers = [Layer(70e-9, SiN)]
# textured front, absorbing layer: A int 0 looks the same, but R is too high

glass_text = planar_surface()

rear_glass_text = planar_surface(interface_layers=layers)

Si_front_text = regular_pyramids(50, True, interface_layers=layers)

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

options.pol = 'u'
options.analytical_ray_tracing = 2
RAT_u_a = rtstr_text.calculate(options)

options.analytical_ray_tracing = 0
RAT_u_f = rtstr_text.calculate(options)

titles = ['s', 'p', 'u']

for i1, [rat_f, rat_a] in enumerate(
        zip([RAT_u_f], [RAT_u_a])):
    plt.figure()
    plt.plot(wl * 1e9, rat_a['R'], '-r', label='R a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['R'], '--r', label='R f', alpha=0.5)

    plt.plot(wl * 1e9, rat_a['R0'], '-c', label='R a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['R0'], '--c', label='R f', alpha=0.5)

    plt.plot(wl * 1e9, rat_a['T'], '-y', label='T a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['T'], '--y', label='T f', alpha=0.5)

    plt.plot(wl * 1e9, rat_a['A_per_layer'][:, 0], '-g', label='A bulk 0 a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['A_per_layer'][:, 0], '--g', label='A bulk 0 f', alpha=0.5)
    # plt.plot(wl * 1e9, tmm['A_per_layer'][:, 1], '-.g')

    plt.plot(wl * 1e9, rat_a['A_per_layer'][:, 1], '-k', label='A bulk 1 a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['A_per_layer'][:, 1], '--k', label='A bulk 1 f', alpha=0.5)
    # plt.plot(wl*1e9, tmm['A_per_layer'][:,3], '-.k')

    plt.plot(wl * 1e9, np.sum(rat_a['A_per_interface'][1], 1), '-b', label='A int 1 a', alpha=0.5)
    plt.plot(wl * 1e9, np.sum(rat_f['A_per_interface'][1], 1), '--b', label='A int 1 f', alpha=0.5)

    plt.plot(wl * 1e9, np.sum(rat_a['A_per_interface'][2], 1), '-k', label='A int 2 a', alpha=0.5)
    plt.plot(wl * 1e9, np.sum(rat_f['A_per_interface'][2], 1), '--k', label='A int 2 f', alpha=0.5)

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
