from structure import Structure, Layer, Interface, Group
from ray_tracing.rt import RTSurface, RT
import numpy as np
import math
from solcore import material
from solcore import si
import matplotlib.pyplot as plt

Si = material('Si')()
Air = material('Air')()

char_angle = math.radians(55)
Lx = 1
Ly = 1
h = Lx*math.tan(char_angle)/2
x = np.array([0, Lx/2, Lx, 0, Lx])
y = np.array([0, Ly/2, 0, Ly, Ly])
z = np.array([0, -h, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf = RTSurface(Points)

z = np.array([0, 0, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf_back = RTSurface(Points)
wavelengths = si(np.linspace(900, 1200, 20), 'nm')
options =  {'wavelengths': wavelengths, 'I_thresh': 1e-5, 'theta': 0, 'phi': 0, 'n_rays': 100,
            'nx': 50, 'ny': 50}

theta = 0
phi = 0

a = [Interface(texture = surf), Layer(material = Si, width = si('200um')), \
     Interface(texture = surf_back)]

struct = Group(a, 'RT', depth_spacing = si('1nm'))

z_pos = np.arange(0, 200, 1e-3)

incidence = Air
transmission = Air

R, T, A_per_layer, profiles = RT(struct, Air, Air, options)
#b = [Interface(texture = surf), Layer(material = Si, width = si('100um')),Interface(texture = surf_back)]
#for i, element in enumerate(a):
#    print(type(element) == Interface, '\n')

R[17] + T[17]+sum(A_per_layer[17])

A = np.trapz(profiles, z_pos)

plt.figure()
plt.plot(wavelengths*1e9, R)
plt.plot(wavelengths*1e9, A)
plt.plot(wavelengths*1e9, T)
plt.plot(wavelengths*1e9, R+A+T)
plt.show()

plt.figure()
plt.plot(z_pos, profiles[19])

plt.show()

