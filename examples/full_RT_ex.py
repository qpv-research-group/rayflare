from ray_tracing.rt_lookup import full_RT
from structure import RTgroup
from textures.standard_rt_textures import regular_pyramids, planar_surface
from solcore import material
from solcore import si
import numpy as np

Air = material('Air')()
Si = material('Si')()
GaAs = material('GaAs')()

def __init__(self, textures, materials=[], widths=[], depth_spacing=1):
    self.materials = materials
    self.textures = textures
    self.widths = widths
    self.depth_spacing = depth_spacing


flat_surf = planar_surface()

options = {'wavelengths': np.linspace(300, 900, 10)*1e-9, 'theta': 0, 'phi': 0, 'I_thresh': 1e-4, 'nx': 5, 'ny': 5,
           'parallel': False, 'pol': 'u'}
structure = RTgroup(textures=[flat_surf, flat_surf, flat_surf, flat_surf], materials = [Si, GaAs, Si],
                    widths=[si('100um'), si('20um'), si('50um')], depth_spacing=1)

full_RT(structure, Air, Air, options)
