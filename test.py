from structure import Structure, Layer, Interface, Group
from ray_tracing.rt import RTSurface, RT
import numpy as np
import math
from solcore import material
from solcore import si

Si = material('Si')()
Air = material('Air')()

char_angle = math.radians(55)
Lx = 1
Ly = 1
h = Lx*math.tan(char_angle)/2
x = np.array([0, Lx/2, Lx, 0, Lx])
y = np.array([0, Ly/2, 0, Ly, Ly])
z = np.array([0, h, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf = RTSurface(Points)

z = np.array([0, -h, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf_back = RTSurface(Points)

options =  {'wavelengths': si(np.linspace(300, 1200, 20), 'nm'), 'I_thresh': 1e-7, 'theta': 0, 'phi': 0, 'n_rays': 100}

theta = 0
phi = 0

a = [Interface(texture = surf), Layer(material = Si, width = si('100um'), n_depths = 100), \
     Interface(texture = surf_back)]

struct = Group(a, 'RT')

R, T, profiles = RT(struct, Air, Air, options)
#b = [Interface(texture = surf), Layer(material = Si, width = si('100um')),Interface(texture = surf_back)]
#for i, element in enumerate(a):
#    print(type(element) == Interface, '\n')