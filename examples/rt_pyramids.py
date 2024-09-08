import numpy as np

from rayflare.ray_tracing import rt_structure
from rayflare.textures import regular_pyramids, planar_surface
from rayflare.options import default_options

from solcore import material
from solcore import si

# imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# setting up some colours for plotting
pal = sns.color_palette("husl", 4)


# setting up Solcore materials
Air = material("Air")()

from solcore.absorption_calculator import download_db, search_db

# download_db()
Si_Green2008 = search_db("Green-2008")[0][0]
Si = material(str(Si_Green2008), nk_db=True)()

# number of x and y points to scan across
nxy = 30

# setting options
options = default_options()
options.wavelength = np.linspace(300, 1201, 100) * 1e-9
options.nx = nxy
options.ny = nxy
options.n_rays = 2 * nxy**2
options.depth_spacing_bulk = si("0.1um")
options.parallel = True

flat_surf = planar_surface(size=2)  # pyramid size in microns
triangle_surf = regular_pyramids(55, upright=False, size=2,
                                 analytical=True, phong=False,
                                 phong_options=[25, True])

# set up ray-tracing options
rtstr = rt_structure(
    textures=[triangle_surf, flat_surf],
    materials=[Si],
    widths=[si("100um")],
    incidence=Air,
    transmission=Air,
)

result = rtstr.calculate(options)

plt.figure()
plt.plot(options.wavelength*1e9, result['R'], label='R')
plt.plot(options.wavelength*1e9, result['A_per_layer'], label='A_bulk')
plt.plot(options.wavelength*1e9, result['T'], label='T')
plt.legend()
plt.show()

# make histogram of theta for transmitted rays only

transmitted_indices = [result['thetas'][i] > np.pi/2 for i in range(len(result))]

angle_bins = np.linspace(0, np.pi, 41)

theta_dist = np.array([np.histogram(result['thetas'][i1],
                                    bins=40, range=(0, np.pi), density=True)[0]
    for i1 in range(len(options.wavelength))])

theta_dist = theta_dist[options.wavelength*1e9 > 900]
theta_dist  = theta_dist[:, theta_dist.shape[1]//2:]

plt.figure()
plt.imshow(theta_dist, aspect='auto', cmap='viridis', extent=(np.pi/2, np.pi, 900, options.wavelength[-1]*1e9))
plt.xlabel('Theta (rad)')
plt.ylabel('Wavelength (nm)')
plt.colorbar()
plt.title('Theta distribution of transmitted rays')
plt.show()