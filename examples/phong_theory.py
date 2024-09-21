# NOTE: this example requires RayFlare v2.0.0 or later

import numpy as np

from rayflare.ray_tracing import rt_structure
from rayflare.textures import planar_surface
from rayflare.options import default_options

from solcore import material
import matplotlib.pyplot as plt
import seaborn as sns

pal = sns.color_palette("husl", 4)

Air = material("Air")()

# planar phong scattering surface with 100 % transmission:
options = default_options()

options.wavelength= np.array([300e-9])

alpha = 20 # alpha = 1 is Lambertian, alpha = infinity is specular

# we make a surface which is flat and does Phong scattering. We have no
# bulk materials, so our structure is just this surface, which has no surface layers; it thus
# acts only as a perfectly transmitting surface which redistributes light Lambertianly. This
# allows us to check the behaviour of the Phong scattering and compare it to the theoretical
# behaviour of cos^alpha(theta).
# reference: https://ieeexplore.ieee.org/document/8638770

flat_surf = planar_surface(phong=True, phong_options=[alpha, True])

rtstr = rt_structure(
    textures=[flat_surf],
    materials=[],
    widths=[],
    incidence=Air,
    transmission=Air,
)

options.n_rays = 5e4
rat_scat = rtstr.calculate(options)

# expected behaviour is a power law:
x = np.linspace(0, np.pi/2)
cos_x = np.cos(x)
power_law = cos_x**(alpha)

# transmitted angles are between pi/2 and pi. Move to 0-pi/2 (0-90 degrees):
thetas = np.pi-rat_scat.ray_data.thetas[0]

# bin the data:
n, bins = np.histogram(thetas, bins=70, density=True)

mean_theta_bin = np.mean([bins[0:-1], bins[1:]], 0)

# need to scale by the solid angle (higher theta -> emitting into a larger volume)
scaled_intensity = n/np.sin(mean_theta_bin)

fig, ax1 = plt.subplots()
plt.bar(mean_theta_bin*180/np.pi, scaled_intensity/np.max(scaled_intensity), width=180*np.diff(bins)[0]/np.pi,
        alpha=0.5, label='probability density/area')
plt.plot(x*180/np.pi, power_law, '-k', label=r'cos$^\alpha(\theta)$')
plt.legend()
plt.show()