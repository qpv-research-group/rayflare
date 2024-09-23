# NOTE: this example requires RayFlare v2.0.0 or later

import numpy as np
import matplotlib.pyplot as plt
from rayflare.textures import regular_pyramids, planar_surface
from rayflare.ray_tracing import rt_structure
from solcore import material
from rayflare.options import default_options
from solcore.structure import Layer
from time import time

options = default_options()
wl = np.linspace(300, 1000, 30) * 1e-9

options.wavelength = wl

options.nx = 30
options.ny = 30
options.n_rays = 1 * options.nx ** 2
options.project_name = 'surface_comparison'
options.randomize_surface = True
options.phi_in = 20 * np.pi / 180
options.I_thresh = 0.0002
options.parallel = True
options.pol = 'u'

SiOx = material("SiO2")()
Si = material("Si")()
Air = material("Air")()
SiN = material("Si3N4")()
GaAs = material("GaAs")()
glass = material("BK7")()

layers = [Layer(70e-9, SiN), Layer(400e-9, GaAs)]

Si_front_text = regular_pyramids(45, True, interface_layers=layers)

back_text = planar_surface()

rtstr_text = rt_structure(
    textures=[Si_front_text],
    materials=[],
    widths=[],
    incidence=Air, transmission=Si,
    use_TMM=True, options=options,
    save_location="current",
    overwrite=True,
)

angle_in = np.linspace(0, 70, 6)

n_int_R_a = np.zeros((len(angle_in), len(wl)))
n_int_T_a = np.zeros((len(angle_in), len(wl)))

n_int_R_f = np.zeros((len(angle_in), len(wl)))
n_int_T_f = np.zeros((len(angle_in), len(wl)))

for i1, degrees_in in enumerate(angle_in):
    options.theta_in = degrees_in * np.pi / 180

    start = time()

    # allow analytical ray-tracing: set analytical = True for the texture
    rtstr_text.surfaces[0].analytical = True
    rat_a = rtstr_text.calculate(options)
    print("analytical time taken: ", time() - start)

    # average numer of interactions per wavelength for reflected and transmitted rays:
    n_int_R_a[i1] = [np.nanmean(rat_a.ray_data.n_interactions[i][rat_a.ray_data.thetas[i] < np.pi/2]) for i in range(len(wl))]
    n_int_T_a[i1] = [np.nanmean(rat_a.ray_data.n_interactions[i][rat_a.ray_data.thetas[i] > np.pi/2]) for i in range(len(wl))]

    start = time()
    rtstr_text.surfaces[0].analytical = False
    rtstr_text.surfaces[0].n_analytical_interactions = 4
    rat_f = rtstr_text.calculate(options)
    print("full time taken: ", time() - start)

    n_int_R_f[i1] = [np.nanmean(rat_f.ray_data.n_interactions[i][rat_f.ray_data.thetas[i] < np.pi/2]) for i in range(len(wl))]
    n_int_T_f[i1] = [np.nanmean(rat_f.ray_data.n_interactions[i][rat_f.ray_data.thetas[i] > np.pi/2]) for i in range(len(wl))]

    plt.figure()
    plt.plot(wl * 1e9, rat_a['R'], '-r', label='R analytical', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['R'], '--r', label='R full', alpha=0.5)

    plt.plot(wl * 1e9, rat_a['T'], '-y', label='T analytical', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['T'], '--y', label='T full', alpha=0.5)

    plt.plot(wl * 1e9, np.sum(rat_a['A_per_interface'][0], 1), '-k', label='A int 0 analytical', alpha=0.5)
    plt.plot(wl * 1e9, np.sum(rat_f['A_per_interface'][0], 1), '--k', label='A int 0 full', alpha=0.5)

    plt.title('Angle in: ' + str(degrees_in))
    # place legend outside plot:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('R/A/T')
    plt.tight_layout()
    plt.show()

# plot average number of interactions at 800 nm as a function of incidence angle:
wl_ind = np.argmin(np.abs(wl - 800e-9))

plt.show()
plt.plot(angle_in, n_int_R_a[:, wl_ind], '-r', label='R a', alpha=0.3)
plt.plot(angle_in, n_int_R_f[:, wl_ind], '--r', label='R f')
plt.plot(angle_in, n_int_T_a[:, wl_ind], '-k', label='T a', alpha=0.3)
plt.plot(angle_in, n_int_T_f[:, wl_ind], '--k', label='T f')
plt.legend()
plt.show()


# At 0 degrees, number of interactions for reflected rays is exactly 2 for all rays, and
# analytical and full ray-tracing agree exactly. At higher angles of incidence, some rays miss
# the opposite face. Analytical ray-tracing underestimates the number of interactions and therefore
# overestimates the reflectance.
