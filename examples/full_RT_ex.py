from ray_tracing.rt_lookup import RT
from structure import RTgroup
from textures.standard_rt_textures import regular_pyramids, planar_surface
from solcore import material
from solcore import si

Air = material('Air')()
Si = material('Si')()
GaAs = material('GaAs')

def __init__(self, textures, materials=[], widths=[], depth_spacing=1):
    self.materials = materials
    self.textures = textures
    self.widths = widths
    self.depth_spacing = depth_spacing


flat_surf = planar_surface()


structure = RTgroup([flat_surf, flat_surf, flat_surf, flat_surf], materials = [Si, GaAs, Si], widths=[si('100um'), si('20um', si('50um'))], depth_spacing=1)

RT(group, incidence, transmission, surf_name, options, Fr_or_TMM = 0, front_or_rear = 'front',
       n_absorbing_layers=0, calc_profile=True, only_incidence_angle=True)
