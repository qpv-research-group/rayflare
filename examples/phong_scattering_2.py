# NOTE: this example requires RayFlare v2.0.0 or later

import numpy as np

from rayflare.ray_tracing import rt_structure
from rayflare.textures import regular_pyramids, planar_surface
from rayflare.options import default_options

from solcore import material
from solcore import si
from solcore.structure import Layer

# imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# setting up some colours for plotting
pal = sns.color_palette("husl", 4)
sns.set_style("whitegrid")

d = 100e-6
# setting up Solcore materials
Air = material("Air")()

from solcore.absorption_calculator import download_db, search_db

Si = material("Si")()
SiN = material("Si3N4")()

# number of x and y points to scan across
nxy = 50

# setting options
options = default_options()
options.wavelength = np.linspace(900, 1180, 15) * 1e-9
options.nx = nxy
options.ny = nxy
options.n_rays = 2 * nxy**2
options.depth_spacing = si("0.1um")
options.parallel = True
options.analytical_ray_tracing = 0
options.I_thresh = 0.005
options.randomize_surface = True
options.project_name = "phong_scattering"
options.lookuptable_angles = 200

# analytical: phong and regular pyramids 15 degree rear look the same
front_opening_deg = 50

ARC = [Layer(70e-9, SiN)]

triangle_surf = regular_pyramids(elevation_angle=front_opening_deg, upright=True, interface_layers=ARC)

flat_surf = planar_surface(phong=False, interface_layers=ARC, analytical=True)  # pyramid size in microns
flat_triangles = regular_pyramids(15, upright=False, phong=False, interface_layers=ARC, analytical=True)
same_triangles = regular_pyramids(front_opening_deg, upright=False, phong=False, interface_layers=ARC, analytical=True)
flat_phong = regular_pyramids(elevation_angle=0, phong=True, phong_options=[1000, True], interface_layers=ARC, analytical=True)


# values for Phong scattering exponent. Higher = more specular, 1 = perfect Lambertian scattering
alpha_values = [1000, 100, 10, 1]


# keyword arguments which will be the same for all:
args_same = {"materials": [Si],
                "widths": [d],
                "incidence": Air,
                "transmission": Air,
              "use_TMM": False,
              "options": options,
              "overwrite": True}

# set up ray-tracing structures:
rtstr_planar = rt_structure(
    textures=[triangle_surf, flat_surf],
    **args_same
)

rtstr_flat_tri = rt_structure(
    textures=[triangle_surf, flat_triangles],
    **args_same
)

rtstr_same_tri = rt_structure(
    textures=[triangle_surf, same_triangles],
    **args_same
)

result_list_phong = []

start = time()

# Loop through the different alpha values
for alpha in alpha_values:
    flat_phong[0].phong_options[0] = alpha
    rtstr_phong = rt_structure(
        textures=[triangle_surf, flat_phong],
        **args_same
    )
    result_list_phong.append(rtstr_phong.calculate(options))

# Planar result:
result_planar = rtstr_planar.calculate(options)

# shallow pyramids:
result_flat_tri = rtstr_flat_tri.calculate(options)

# same opening angle pyramids:
result_same_tri = rtstr_same_tri.calculate(options)

print("total time:", time() - start)

plt.figure()

for i1, alpha in enumerate(alpha_values):
    plt.plot(options.wavelength*1e9, result_list_phong[i1]["A_per_layer"], '-', color=pal[i1],
             label=r"phong, $\alpha$ = " + str(alpha))

plt.plot(options.wavelength * 1e9, result_planar["A_per_layer"], '-.k',
         label="planar", alpha=0.6)
plt.plot(options.wavelength * 1e9, result_flat_tri["A_per_layer"], '-k',
         label="15 degree pyramid", alpha=0.6)
plt.plot(options.wavelength * 1e9, result_same_tri["A_per_layer"], '--k',
         label="same angle pyramid", alpha=0.6)

plt.ylim(0, 1)
plt.xlim(np.min(options.wavelength * 1e9), np.max(options.wavelength * 1e9))
plt.legend(title='Rear surface:')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorption")
plt.show()
