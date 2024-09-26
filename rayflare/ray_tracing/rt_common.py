# Copyright (C) 2021-2024 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU Lesser General Public License (LGPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: p.pearce@unsw.edu.au

import numpy as np
from cmath import sqrt, acos, atan
from math import atan2
from random import random
from copy import deepcopy
from scipy.spatial import Delaunay
from rayflare.utilities import process_pol

from numba import jit

unit_cell_N = np.array(
    [[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]]
)  # surface normals: top, right, bottom, left

class Ray:
    """Class to store ray information for ray-tracing calculations."""
    def __init__(self,
                 intensity,
                 direction,
                 current_location,
                 s_vector,
                 p_vector,
                 pol,
                 ):
        """
        :param intensity: intensity of the ray (initial = 1)
        :param direction: tuple of the form (theta, phi) in radians
        :param current_location: 3D coordinate of the form (x, y, z)
        :param polarization: length 2 np.array of the form (s, p) where s and p are the fractions of
          the intensity in which are s and p polarized (always need to sum to 1)
        """

        self.d = direction
        self.r_a = current_location
        self.I = intensity

        self.s_vector = s_vector
        self.p_vector = p_vector

        self.pol = pol


class RTSurface:
    """Class which is used to store information about the surface which is used for ray-tracing."""

    def __init__(self, Points, interface_layers=None,
                 phong=False,
                 analytical=False,
                 **kwargs):
        """Initializes the surface.
        Parameters:

        :param Points: A numpy array of shape (n, 3) where n is the number of points on the surface. The columns are the
                        x, y and z coordinates of the points.
        :param interface_layers: a list of layers (typically, Solcore Layer objects) which are on the interface. Optional.
        :param coverage_height: The height at which the surface is expected to cover the whole unit cell. If this is not
                                provided (None), this function will try to guess the coverage height by finding the height
                                at which both the x and y coordinate are minimized.
        """

        if "height_distribution" in kwargs:
        #     return from a probability distribution instead of fixed values. This
        #     will change the height of the pyramids (assume only pyramids for now)
        #     but not the size of the unit cell. Simplices also do not change.
        #     Changing: Points, P_0s, P_1s, P_2s, crossP, N, z_min, z_max, z_cov if
        #     it's not 0 (but for pyramids it is).
            self.distribution = kwargs["height_distribution"]

        else:
            self.distribution = None

        tri = Delaunay(Points[:, [0, 1]])
        self.simplices = tri.simplices
        self.Points = Points
        self.original_Points = deepcopy(Points)
        self.height_ind = np.argmax(np.abs(self.Points[:,2]))
        self.P_0s = Points[tri.simplices[:, 0]]
        self.P_1s = Points[tri.simplices[:, 1]]
        self.P_2s = Points[tri.simplices[:, 2]]
        self.crossP = np.cross(self.P_1s - self.P_0s, self.P_2s - self.P_0s)
        self.N = self.crossP / np.linalg.norm(self.crossP, axis=1)[:, None]
        self.size = self.P_0s.shape[0]
        self.Lx = abs(min(Points[:, 0]) - max(Points[:, 0]))
        self.Ly = abs(min(Points[:, 1]) - max(Points[:, 1]))
        self.x_min = min(Points[:, 0])
        self.x_max = max(Points[:, 0])
        self.y_min = min(Points[:, 1])
        self.y_max = max(Points[:, 1])
        self.z_min = min(Points[:, 2])
        self.z_max = max(Points[:, 2])

        self.phong = phong

        if "phong_options" in kwargs:
            self.phong_options = kwargs["phong_options"]

        else:
            self.phong_options = [25, True]

        self.analytical = analytical

        if "n_analytical_interactions" in kwargs:
            self.n_analytical_interactions = kwargs["n_analytical_interactions"]

        else:
            self.n_analytical_interactions = 3

        # zcov is the height at which the surface covers the whole unit cell; i.e. it is safe to aim a ray at the unit
        # cell at this height and be sure that it will hit the surface. The method below works well for regular textures
        # like regular pyramids but doesn't work well for e.g. AFM scans, hyperhemisphere

        if "coverage_height" in kwargs:
            self.zcov = kwargs["coverage_height"]

        # catch exception here in case the surface is not regular
        else:
            self.zcov = Points[:, 2][
                np.all(
                    np.array(
                        [
                            Points[:, 0] == min(Points[:, 0]),
                            Points[:, 1] == min(Points[:, 1]),
                        ]
                    ),
                    axis=0,
                )
            ][0]

        if "name" in kwargs:
            self.name = kwargs["name"]

        else:
            self.name = ""

        if interface_layers is not None:
            self.interface_layers = interface_layers

            if "coherency_list" in kwargs:
                self.coherency_list = kwargs["coherency_list"]

            if "prof_layers" in kwargs:
                self.prof_layers = kwargs["prof_layers"]

    def __deepcopy__(self, memo):
        copy = type(self)(Points=self.Points, coverage_height=self.zcov)
        memo[id(self)] = copy

        keys = self.__dict__.keys()

        for key in keys:
            if key != "interface_layers":
                setattr(copy, key, deepcopy(getattr(self, key), memo))

        return copy

    def find_area(self):
        xyz = np.stack((self.P_0s, self.P_1s, self.P_2s))
        cos_theta = np.sum((xyz[0] - xyz[1]) * (xyz[2] - xyz[1]), 1)

        theta = np.arccos(cos_theta)
        self.area = np.sum(
            (
                0.5
                * np.linalg.norm(xyz[0] - xyz[1], axis=1)
                * np.linalg.norm(xyz[2] - xyz[1], axis=1)
                * np.sin(theta)
            )
        ) / (self.Lx * self.Ly)

    def shift(self, z_shift):
        self.Points[:, 2] = self.Points[:, 2] - z_shift
        self.P_0s = self.Points[self.simplices[:, 0]]
        self.P_1s = self.Points[self.simplices[:, 1]]
        self.P_2s = self.Points[self.simplices[:, 2]]
        self.crossP = np.cross(self.P_1s - self.P_0s, self.P_2s - self.P_0s)
        self.z_min = min(self.Points[:, 2])
        self.z_max = max(self.Points[:, 2])

        self.zcov = self.zcov - z_shift

    def refresh(self):

        if self.distribution is not None:

            new_height = np.random.choice(self.distribution["h"],
                                             p=self.distribution["p"])

            scaling = np.abs(new_height/(self.original_Points[:,2][self.height_ind] - self.zcov))

            self.Points[:,2] = scaling*(self.original_Points[:,2] - self.zcov) + self.zcov

            self.P_0s = self.Points[self.simplices[:, 0]]
            self.P_1s = self.Points[self.simplices[:, 1]]
            self.P_2s = self.Points[self.simplices[:, 2]]
            self.crossP = np.cross(self.P_1s - self.P_0s, self.P_2s - self.P_0s)
            self.N = self.crossP / np.linalg.norm(self.crossP, axis=1)[:, None]
            self.z_min = min(self.Points[:, 2])
            self.z_max = max(self.Points[:, 2])

def make_pol_vectors(pol_string, theta, phi):
    pol = process_pol(pol_string)
    pol = np.array(pol) / np.sum(pol)

    initial_p_dir = np.array([np.cos(theta) * np.cos(phi),
                              np.cos(theta) * np.sin(phi),
                              -np.sin(theta)])

    initial_s_dir = np.array([-np.sin(phi), np.cos(phi), np.zeros_like(theta)])

    initial_pol_vectors = np.array([initial_s_dir, initial_p_dir])

    return pol, initial_pol_vectors

@jit(nopython=True, error_model="numpy")
def calculate_tuv(d, tri_size, tri_crossP, r_a, tri_P_0s, tri_P_2s, tri_P_1s):

    D = (-d).repeat(tri_size).reshape((-1, tri_size)).T # this is significantly faster than np.tile
    pref = 1 / np.sum(D * tri_crossP, axis=1)
    corner = r_a - tri_P_0s

    t = pref * np.sum(tri_crossP * corner, axis=1)
    u = pref * np.sum(np.cross(tri_P_2s - tri_P_0s, D) * corner, axis=1)
    v = pref * np.sum(np.cross(D, tri_P_1s - tri_P_0s) * corner, axis=1)

    return t, u, v


@jit(nopython=True, error_model="numpy")
def normalize(x):
    return x / norm(x)

@jit(nopython=True)
def norm(x):
    s = x[0]**2 + x[1]**2 + x[2]**2
    return np.sqrt(s)

@jit(nopython=True, error_model="numpy")
def calc_intersection_properties(t, which_intersect, r_a, d, tri_N):
    t = t[which_intersect]
    ind = np.argmin(t)
    t = min(t)

    intersn = r_a + t * d

    N = tri_N[which_intersect][ind]

    Nxd = np.cross(N, -d)

    theta = atan(norm(Nxd) / np.dot(N, -d))

    return intersn, theta, N

def check_intersect(r_a, d, tri_size, tri_crossP, tri_P_0s, tri_P_2s, tri_P_1s, tri_N):

    t, u, v = calculate_tuv(d, tri_size, tri_crossP, r_a, tri_P_0s, tri_P_2s, tri_P_1s)

    # for some reason numba doesn't like this bit:
    which_intersect = (
        (u + v <= 1) & (np.all(np.vstack((u, v)) >= -1e-10, axis=0)) & (t > 0)
    )
    # get errors if set lower limit exactly to zero, hence 1e-10

    if sum(which_intersect) > 0:
        return calc_intersection_properties(t, which_intersect, r_a, d, tri_N)

    else:
        return False

def update_ray_d_pol(ray, rnd, R, T, Rs, Rp, Ts, Tp, A_per_layer, n0, n1, N, side,
                     ray_plane_s_direction, s_comp_sq):

    # TODO: is it necessary to normalize here or will it already be normalized?
    # comment out to make faster if not necessary
    if np.abs(norm(ray.d) - 1) > 1e-2:
        raise ValueError(f"Ray direction not normalized {norm(ray.d)}")

    if rnd <= R:  # REFLECTION
        ray.d = np.real(ray.d - 2 * np.dot(ray.d, N) * N)

        ray.s_vector = ray_plane_s_direction
        ray.p_vector = normalize(np.cross(ray.d, ray.s_vector))

        # the relative s/p weighting of the reflected ray needs to be weighted by
        # Rs and Rp:
        ray.pol = np.array([Rs / R, Rp / R])

        A = None

    elif (rnd > R) & (rnd <= (R + T)):  # TRANSMISSION
        # transmission, refraction
        # tr_par = (np.real(n0) / np.real(n1)) * (d - np.dot(d, N) * N)
        tr_par = (n0 / n1) * (ray.d - np.dot(ray.d, N) * N)
        tr_perp = -sqrt(1 - norm(tr_par) ** 2) * N

        side = -side
        ray.d = np.real(tr_par + tr_perp)
        ray.s_vector = ray_plane_s_direction

        ray.p_vector = normalize(np.cross(ray.d, ray.s_vector))

        # the relative s/p weighting of the transmitted ray needs to be weighted by
        # Rs and Rp:
        ray.pol = np.array([Ts / T, Tp / T])

        A = None

    else:
        # absorption
        ray.pol = np.array([s_comp_sq, 1 - s_comp_sq])
        A = A_per_layer

    ray.d = normalize(ray.d)

    return side, A

def decide_RT_Fresnel(ray, n0, n1, theta, N, side, rnd,
                        lookuptable=None, d_theta=None):

    s_component_sq, p_component_sq, ray_plane_s_direction = (
        get_pol_component_direction(theta, ray.d, N, ray.s_vector, ray.p_vector, ray.pol))

    ratio = np.clip(np.real(n1) / np.real(n0), -1, 1)

    if abs(theta) > np.arcsin(ratio):
        Rs, Rp = 1, 1

    else:
        Rs, Rp = calc_R(n0, n1, theta, np.array([s_component_sq, p_component_sq]))

    Ts = 1 - Rs
    Tp = 1 - Rp

    Rs = Rs * s_component_sq
    Rp = Rp * p_component_sq

    Ts = Ts * s_component_sq
    Tp = Tp * p_component_sq

    # R = ray.pol[0]*Rs + ray.pol[1]*Rp
    R = Rs + Rp
    T = Ts + Tp
    # R_plus_T = 1

    side, _ = update_ray_d_pol(ray, rnd, R, T, Rs, Rp, Ts, Tp, 0,
                              n0, n1, N, side, ray_plane_s_direction, s_component_sq)

    return side, None  # never absorbed, A = False


def decide_RT_TMM(ray, n0, n1, theta, N, side, rnd, lookuptable, d_theta):

    s_component_sq, p_component_sq, ray_plane_s_direction = (
        get_pol_component_direction(theta, ray.d, N, ray.s_vector, ray.p_vector, ray.pol))
    # print('s/p', s_component_sq, 1-s_component_sq)

    data_s = lookuptable.loc[dict(side=side)]

    Rs, Rp, Ts, Tp, A_per_layer = get_RT_data(theta, d_theta,
                                      data_s.R.data, data_s.T.data, data_s.Alayer.data,
                                           np.array([s_component_sq, p_component_sq]))

    R = Rs + Rp # overall probability this ray will reflect. Rs and Rp are already scaled (in
    # get_RT_data) by the incoming s and p component

    T = Ts + Tp # overall probability this ray will transmit

    side, A = update_ray_d_pol(ray, rnd, R, T, Rs, Rp, Ts, Tp,
                           A_per_layer,
                              n0, n1, N, side, ray_plane_s_direction,
                               s_component_sq)

    return side, A

@jit(nopython=True)
def get_RT_data(theta, d_theta, R_data, T_data, Alayer_data, pol):
    # theta HAS to be positive here!
    angle_ind = int(np.floor(theta/d_theta)) # floor to avoid issues when theta = np.pi/2

    [Rs, Rp] = R_data[:, angle_ind]*pol
    [Ts, Tp] = T_data[:, angle_ind]*pol
    A_per_layer = np.sum(Alayer_data[:, angle_ind].T * pol, 1)

    return Rs, Rp, Ts, Tp, A_per_layer

@jit(nopython=True)
def get_pol_component_direction(theta, d, N, current_s_vector, current_p_vector, ray_pol):
    if theta > 1e-4:  # non-normal incidence only, otherwise cannot take cross product between ray.d and N
        ray_plane_s_direction = normalize(np.cross(d, N))
        s_component = np.array([np.dot(current_s_vector, ray_plane_s_direction),
                                np.dot(current_p_vector, ray_plane_s_direction)])
        s_component_sq = (ray_pol[0] * s_component[0] ** 2 + ray_pol[1] * s_component[1] ** 2)

        # by definition:
        p_component_sq = 1 - s_component_sq

    else:
        [s_component_sq, p_component_sq] = ray_pol
        ray_plane_s_direction = current_s_vector

    return s_component_sq, p_component_sq, ray_plane_s_direction

@jit(nopython=True)
def calc_R(n1, n2, theta, pol_comps):
    theta_t = np.arcsin((n1 / n2) * np.sin(theta))
    if pol_comps[0] == 1: # 100% s-polarized. Only need to calculate Rs, do not need to update ray.pol
        Rs = (
                np.abs(
                    (n1 * np.cos(theta) - n2 * np.cos(theta_t))
                    / (n1 * np.cos(theta) + n2 * np.cos(theta_t))
                )
                ** 2
        )
        return Rs, 0

    elif pol_comps[1] == 1: # 100% p-polarized. Only need to calculate Rp, do not need to update ray.pol
        Rp = (
                np.abs(
                    (n1 * np.cos(theta_t) - n2 * np.cos(theta))
                    / (n1 * np.cos(theta_t) + n2 * np.cos(theta))
                )
                ** 2
        )
        return 0, Rp

    else:
        Rs = (
                np.abs(
                    (n1 * np.cos(theta) - n2 * np.cos(theta_t))
                    / (n1 * np.cos(theta) + n2 * np.cos(theta_t))
                )
                ** 2
        )
        Rp = (
                np.abs(
                    (n1 * np.cos(theta_t) - n2 * np.cos(theta))
                    / (n1 * np.cos(theta_t) + n2 * np.cos(theta))
                )
                ** 2
        )

        # need to update ray.pol: if Rs > Rp, ray becomes more s-polarized and vice-versa

        return Rs, Rp

def single_cell_check(
    ray,
    ni,
    nj,
    tri,
    Lx,
    Ly,
    side,
    z_cov,
    d_theta,
    n_interactions=0,
    wl=None,
    Fr_or_TMM=0,
    lookuptable=None,
):
    decide = {0: decide_RT_Fresnel, 1: decide_RT_TMM}

    theta = 0  # needs to be assigned so no issue with return in case of miss
    # print('side', side)
    d0 = ray.d
    intersect = True
    n_ints_loop = 0
    # [top, right, bottom, left]
    n_misses = 0
    i1 = 0

    while intersect:
        i1 = i1 + 1
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # there will be divide by 0/multiply by inf - this is fine but gives lots of warnings
            result = check_intersect(ray.r_a, ray.d, tri.size, tri.crossP, tri.P_0s,
                                         tri.P_2s, tri.P_1s, tri.N)

        if result is False:

            n_misses += 1

            o_t = np.real(acos(ray.d[2] / norm(ray.d)))
            o_p = np.real(atan2(ray.d[1], ray.d[0]))

            if np.sign(d0[2]) == np.sign(ray.d[2]):
                intersect = False
                final_res = 1

            else:
                intersect = False
                final_res = 0

            return (
                final_res,
                o_t,
                o_p,
                theta,
                n_interactions,
                side,
            )  # theta is LOCAL incidence angle (relative to texture)

        else:

            # there has been an intersection
            n_interactions += 1
            n_ints_loop += 1

            intersn = result[0]  # coordinate of the intersection (3D)

            theta = result[1]

            N = (
                result[2] * side
            )  # so angles get worked out correctly, relative to incident face normal

            if side == 1:
                n0 = ni
                n1 = nj

            else:
                n0 = nj
                n1 = ni

            rnd = random()

            side, A = decide[Fr_or_TMM](
                ray, n0, n1, abs(theta), N, side, rnd, lookuptable, d_theta
            )

            ray.r_a = np.real(
                intersn + ray.d / 1e9
            )  # this is to make sure the raytracer doesn't immediately just find the same intersection again

            if A is not None:
                # intersect = False
                # checked_translation = True
                final_res = 2
                o_t = A
                o_p = 0

                return (
                    final_res,
                    o_t,  # A array
                    o_p,
                    theta,  # LOCAL incidence angle
                    n_interactions,
                    side,
                )

@jit(nopython=True)
def exit_side(r_a, d, p_0):
    denom = np.sum(d * unit_cell_N, axis=1)
    denom[denom == 0] = 1e-12
    t = np.sum((p_0 - r_a) * unit_cell_N, axis=1) / denom  # r_intersect = r_a + t*d
    which_intersect = t > 0  # only want intersections of forward-travelling ray
    t[~which_intersect] = np.inf  # set others to inf to avoid finding when doing min
    which_side = np.argmin(t)  # find closest plane

    return which_side, t[which_side]

def single_interface_check(
    ray,
    ni,
    nj,
    tri,
    Lx,
    Ly,
    side,
    z_cov,
    d_theta,
    n_interactions=0,
    wl=None,
    Fr_or_TMM=0,
    lookuptable=None,
):
    decide = {0: decide_RT_Fresnel, 1: decide_RT_TMM}

    p_0 = np.array(
        [[0, Ly, 0], [Lx, 0, 0], [0, 0, 0], [0, 0, 0]]
    )  # points on each plane

    d0 = ray.d
    intersect = True
    checked_translation = False
    # [top, right, bottom, left]
    translation = np.array([[0, -Ly, 0], [-Lx, 0, 0], [0, Ly, 0], [Lx, 0, 0]])
    n_misses = 0
    i1 = 0

    tri.refresh()

    while intersect:
        i1 = i1 + 1
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # there will be divide by 0/multiply by inf - this is fine but gives lots of warnings
            # result = check_intersect(ray.r_a, ray.d, tri)
            result = check_intersect(ray.r_a, ray.d, tri.size, tri.crossP, tri.P_0s,
                                         tri.P_2s, tri.P_1s, tri.N)
        # print('result (intersn, theta, N)', result)

        if result is False and not checked_translation:
            if i1 > 1:
                which_side, _ = exit_side(ray.r_a, ray.d, p_0)
                ray.r_a = ray.r_a + translation[which_side]
                # if random pyramid, need to change surface at this point
                tri.refresh()
                checked_translation = True

            else:
                if n_misses < 100:
                    # misses surface. Try again
                    if ray.d[2] < 0:  # coming from above
                        ray.r_a = np.array(
                            [
                                np.random.rand() * Lx,
                                np.random.rand() * Ly,
                                tri.z_max + 0.01,
                            ]
                        )
                    else:
                        ray.r_a = np.array(
                            [
                                np.random.rand() * Lx,
                                np.random.rand() * Ly,
                                tri.z_min - 0.01,
                            ]
                        )
                    n_misses += 1
                    i1 = 0

                else:
                    # ray keeps missing, probably because it's travelling (almost) exactly perpendicular to surface.
                    # assume it is reflected back into layer it came from
                    ray.d[2] = -ray.d[2]

                    o_t = np.real(acos(ray.d[2] / norm(ray.d)))
                    o_p = np.real(atan2(ray.d[1], ray.d[0]))
                    return 0, o_t, o_p, 0, n_interactions, side

        elif result is False and checked_translation:

            if (side == 1 and ray.d[2] < 0 and ray.r_a[2] > tri.z_min) or (
                side == -1 and ray.d[2] > 0 and ray.r_a[2] < tri.z_max
            ):
                # going down but above surface

                if ray.r_a[0] > Lx or ray.r_a[0] < 0:
                    ray.r_a[0] = (
                        ray.r_a[0] % Lx
                    )  # translate back into until cell before doing any additional translation
                if ray.r_a[1] > Ly or ray.r_a[1] < 0:
                    ray.r_a[1] = (
                        ray.r_a[1] % Ly
                    )  # translate back into until cell before doing any additional translation
                ex, t = exit_side(ray.r_a, ray.d, p_0)

                ray.r_a = ray.r_a + t * ray.d + translation[ex]
                # also change surface here
                tri.refresh()

                checked_translation = True

            else:

                o_t = np.real(acos(ray.d[2] / norm(ray.d)))
                o_p = np.real(atan2(ray.d[1], ray.d[0]))

                if np.sign(d0[2]) == np.sign(ray.d[2]):
                    intersect = False
                    final_res = 1

                else:
                    intersect = False
                    final_res = 0

                # # TODO: is this necessary?
                # if ray.r_a[0] > Lx or ray.r_a[0] < 0:
                #     ray.r_a[0] = (
                #         ray.r_a[0] % Lx
                #     )  # translate back into until cell before next ray
                # if ray.r_a[1] > Ly or ray.r_a[1] < 0:
                #     ray.r_a[1] = (
                #         ray.r_a[1] % Ly
                #     )  # translate back into until cell before next ray

                return (
                    final_res,
                    o_t,  # theta with respect to horizontal
                    o_p,
                    theta,  # LOCAL incidence angle
                    n_interactions,
                    side,
                )  # theta is LOCAL incidence angle (relative to texture)

        else:

            # there has been an intersection
            n_interactions += 1

            intersn = result[0]  # coordinate of the intersection (3D)

            theta = result[1]

            N = (
                result[2] * side
            )  # so angles get worked out correctly, relative to incident face normal

            if side == 1:
                n0 = ni
                n1 = nj

            else:
                n0 = nj
                n1 = ni

            rnd = random()

            side, A = decide[Fr_or_TMM](
                ray, n0, n1, abs(theta), N, side, rnd, lookuptable, d_theta,
            )

            ray.r_a = np.real(
                intersn + ray.d / 1e9
            )  # this is to make sure the raytracer doesn't immediately just find the same intersection again

            checked_translation = False  # reset, need to be able to translate the ray back into the unit cell again if necessary

            if A is not None:
                # intersect = False
                # checked_translation = True
                final_res = 2
                o_t = A
                o_p = 0

                return (
                    final_res,
                    o_t, # A array, NOT theta (only in the case of absorption)
                    o_p,
                    theta,  # LOCAL incidence angle
                    n_interactions,
                    side,
                )