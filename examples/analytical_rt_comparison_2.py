import numpy as np
import matplotlib.pyplot as plt
from rayflare.textures import xyz_texture, regular_pyramids, planar_surface
from rayflare.ray_tracing import rt_structure
from solcore import material
import seaborn as sns
from rayflare.options import default_options
from solcore.structure import Layer
from time import time

options = default_options()
wl = np.linspace(300, 1130, 40) * 1e-9

options.wavelength = wl

options.nx = 50
options.ny = 50
options.n_rays = 1 * options.nx ** 2
options.project_name = 'surface_comparison'
options.lambertian_approximation = 0
options.randomize_surface = True
options.phi_in = 0 * np.pi / 180
options.I_thresh = 0.002
options.parallel = True
options.pol = 'u'

SiOx = material("SiO2")()
Si = material("Si")()
Air = material("Air")()
SiN = material("Si3N4")()
GaAs = material("GaAs")()
glass = material("BK7")()
pero = material("Perovskite_CsBr_1p6eV")()
MgF2 = material("MgF2")()
Ag = material("Ag_Jiang")()

ARC_layers = [Layer(100e-9, MgF2)]
Si_layers = [Layer(70e-9, SiN)]
# textured front, absorbing layer: A int 0 looks the same, but R is too high
planar_front = planar_surface(interface_layers=ARC_layers)

Si_front_text = regular_pyramids(50, True,
                                 interface_layers=Si_layers)

back_text = planar_surface()

rtstr_text = rt_structure(
    textures=[planar_front, Si_front_text, back_text],
    materials=[pero, Si],
    widths=[500e-9, 80e-6],
    incidence=Air, transmission=Ag,
    use_TMM=True, options=options,
    save_location="current",
    overwrite=True,
)

angle_in = np.linspace(0, 70, 8)

n_int_R_a = np.zeros((len(angle_in), len(wl)))
n_int_T_a = np.zeros((len(angle_in), len(wl)))

n_int_R_f = np.zeros((len(angle_in), len(wl)))
n_int_T_f = np.zeros((len(angle_in), len(wl)))

for i1, degrees_in in enumerate(angle_in):
    options.theta_in = degrees_in * np.pi / 180

    start = time()
    options.analytical_ray_tracing = 4
    rat_a = rtstr_text.calculate(options)
    print("analytical time taken: ", time() - start)

    n_int_R_a[i1] = [np.mean(rat_a['n_interactions'][i][rat_a['thetas'][i] < np.pi/2]) for i in range(len(wl))]
    n_int_T_a[i1] = [np.mean(rat_a['n_interactions'][i][rat_a['thetas'][i] > np.pi/2]) for i in range(len(wl))]

    start = time()
    options.analytical_ray_tracing = False
    rat_f = rtstr_text.calculate(options)
    print("full time taken: ", time() - start)

    n_int_R_f[i1] = [np.mean((rat_f['Is'][i]*rat_f['n_interactions'][i])[rat_f['thetas'][i] < np.pi/2]) for i in range(len(wl))]
    n_int_T_f[i1] = [np.mean((rat_f['Is'][i]*rat_f['n_interactions'][i])[rat_f['thetas'][i] > np.pi/2]) for i in range(len(wl))]

    plt.figure()
    plt.plot(wl * 1e9, rat_a['R'], '-r', label='R a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['R'], '--r', label='R f', alpha=0.5)

    plt.plot(wl * 1e9, rat_a['T'], '-y', label='T a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['T'], '--y', label='T f', alpha=0.5)

    # plt.plot(wl * 1e9, np.sum(rat_a['A_per_interface'][0], 1), '-k', label='A int 0 a', alpha=0.5)
    # plt.plot(wl * 1e9, np.sum(rat_f['A_per_interface'][0], 1), '--k', label='A int 0 f', alpha=0.5)

    plt.plot(wl * 1e9, rat_a['A_per_layer'][:, 0], '-g', label='A bulk 0 a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['A_per_layer'][:, 0], '--g', label='A bulk 0 f', alpha=0.5)

    plt.plot(wl * 1e9, rat_a['A_per_layer'][:, 1], '-k', label='A bulk 1 a', alpha=0.5)
    plt.plot(wl * 1e9, rat_f['A_per_layer'][:, 1], '--k', label='A bulk 1 f', alpha=0.5)

    plt.title('Angle in: ' + str(degrees_in))
    # place legend outside plot:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

wl_ind = np.argmin(np.abs(wl - 800e-9))

plt.show()
plt.plot(angle_in, n_int_R_a[:, wl_ind], '-r', label='R a', alpha=0.3)
plt.plot(angle_in, n_int_R_f[:, wl_ind], '--r', label='R f')
plt.plot(angle_in, n_int_T_a[:, wl_ind], '-k', label='T a', alpha=0.3)
plt.plot(angle_in, n_int_T_f[:, wl_ind], '--k', label='T f')
plt.legend()
plt.show()
