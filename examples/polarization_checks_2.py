import numpy as np
import matplotlib.pyplot as plt
from rayflare.textures import xyz_texture, regular_pyramids, planar_surface
from rayflare.ray_tracing import rt_structure
from solcore import material
import seaborn as sns
from rayflare.options import default_options
from solcore.structure import Layer
from time import time
from rayflare.transfer_matrix_method import tmm_structure

method = 'newest'

options = default_options()
wl = np.linspace(400, 1050, 30) * 1e-9

options.wavelength = wl

linestyles = ['-', '--', '-.', ':']
options.nx = 30
options.ny = 30
options.n_rays = 2 * options.nx ** 2
# options.x_limits = [2.5, 7.5]
# options.y_limits = [2.5, 7.5]
options.project_name = 'polarization_comparison'
options.randomize_surface = True
options.I_thresh = 0.0002
options.parallel = True
options.analytical_ray_tracing = 2

Si = material("Si")()
GaAs = material("GaAs")()
Air = material("Air")()

tmm_strt = tmm_structure([Layer(300e-9, material("GaAs")())],
                         incidence=Air, transmission=Si)

pols = ['s', 'p', 'u']
surface_angles = [10]
incidence_angles = [35]
phis = [0, 45, 90]

for opening_angle in surface_angles:

    front_surf = regular_pyramids(opening_angle, True,
                                  interface_layers=[Layer(0e-9, GaAs)])


    rtstr = rt_structure(
        textures=[front_surf],
        materials=[],
        widths=[],
        incidence=Air, transmission=Si,
        use_TMM=False, options=options,
        save_location="current",
        overwrite=True,
    )

    for theta in incidence_angles:
        options.theta_in = theta * np.pi / 180

        for phi in phis:
            options.phi_in = phi * np.pi / 180

            plt.figure()
            for i1, pol in enumerate(pols):
                options.pol = pol

                rat = rtstr.calculate(options)
                # save_str = f'polarization_comparison_{method}_{pol}_{theta}_{phi}_{opening_angle}.npy'
                # np.save(save_str, rat)

                plt.plot(wl * 1e9, rat['R'], 'k', label='R', linestyle=linestyles[i1])
                plt.plot(wl * 1e9, rat['T'], 'r', label='T', linestyle=linestyles[i1])
                # plt.plot(wl * 1e9, rat['A_per_interface'][0], 'b', label='A', linestyle=linestyles[i1])
                sum_all =  rat['R'] + rat['T'] #+ rat['A_per_interface'][0][:,0]
                plt.plot(wl * 1e9, sum_all, '--g', label='R + T + A', alpha=0.5)
                plt.legend()
                plt.title(f'{method}, {pol} polarization, '
                          f'theta = {theta}, phi = {phi}, opening angle = {opening_angle}')

            plt.show()
