Treatment of polarization
==========================

Note: for ray-tracing, the below applies to RayFlare version 2.0.0 (September 2024) and later.

The polarization of light is a vector quantity that describes the orientation of the electric field of an electromagnetic
wave. In the context of ray-tracing, the polarization of light is important when considering the reflection and
transmission of light at interfaces. Across all methods (ray-tracing, RCWA and TMM) we decompose the polarization into
two orthogonal components, *s* and *p*, following
the standard convention: the s-polarization component is perpendicular to the plane of incidence (i.e. the plane which
contains the ray and the surface normal), while the p-polarization component is parallel to the plane of incidence.
This is defined with respect to the x-y plane, with the following convention for the direction of the incident ray and
the unit vectors for the direction of the E field in s and p polarization:

.. math::

   \begin{aligned}
    & \hat{d}=(-\sin \theta \cos \phi,-\sin \theta \sin \phi,-\cos \theta) \\
    & \hat{s}=(-\sin \phi, \cos \phi, 0) \\
    & \hat{p}=(\cos \theta \cos \phi, \cos \theta \sin \phi,-\sin \theta)
    \end{aligned}

These three vectors are mutually orthogonal. For normal incidence (:math:`\theta = \phi = 0`), the ray points in the
(0, 0, -1) direction, and the s and p polarization vectors are (0, 1, 0) and (1, 0, 0) respectively.

The treatment of polarization in TMM and RCWA are described in the relevant publications, and were implemented in
the original packages which are used by RayFlare for these methods (modified versions of the `tmm` Python package originally
developed by Steven Byrnes and :math:`\S^4`, originally developed by Victor Liu), and will not be discussed here.

For planar surfaces, the *s* and *p* planes directions stay the same throughout a simulations. However,
for textured surfaces, the *s and *p* components of a ray relative to the surface it is hitting change depending on the
orientation of the ray and the surface. RayFlare stores information on the *s* and *p* components of the ray and the
vector directions of these polarizations (relative to the incidence definition, or the previous surface the ray interacted
with), in the ray object. Each time a ray interacts with a surface, the following steps happen:

1. Project the ray's current polarization vector directions onto the new vectors which define
   the *s* and *p* directions for the ray-plane system for the ray's current interaction:

   .. math::

        \begin{aligned}
        & \text{pol} = (s, p) \\
        & \mathbf{\hat{s}}_{\text {new }}=\mathbf{\hat{d}} \times \mathbf{\hat{N}} \\
        & f_s=s\left|\mathbf{\hat{s}} \cdot \mathbf{\hat{s}}_{\text {new }}\right|^2+p\left|\mathbf{p} \cdot \mathbf{\hat{s}}_{\text {new }}\right|^2 \\
        & f_p=1-f_s
        \end{aligned}

   Here, the new *s* direction is given by the cross product of the rays incident direction and the normal
   to the surface. We then project the two old polarization vectors onto this new *s* directions,
   weighted by the old *s* and *p* components of the ray.

2. Calculate/look up :math:`R_s`, :math:`R_p`, :math:`T_s`, :math:`T_p`, which are the reflection and transmission coefficients
   for *s* and *p* polarization (in the reference frame of the ray-plane system for the current surface interaction).
   These can be calculated either with the Fresnel equations, or using the transfer-matrix method, but this does not
   affect the treatment of polarization. The total reflection probability :math:`R = f_s R_s + f_p R_p` and total transmission
   probability :math:`T = f_s T_s + f_p T_p` determine the probability the ray is reflected or transmitted.

3.  After deciding whether the ray is transmitted or reflected, the new direction of the *p* vector is then given by the
    cross product of the new *s* direction and the new ray direction, after reflection/refraction,
    and the new *s* and *p* components of the ray are calculated from :math:`R_s` and :math:`R_p` (reflection) or
    :math:`T_s` and :math:`T_p` (transmission):

    .. math::

          \begin{aligned}
          & \mathbf{\hat{p}}_{\text {new }}=\mathbf{\hat{s}}_{\text {new }} \times \mathbf{\hat{d}_\text{new}} \\
          & \text{pol}_{\text {new }}=(\frac{f_s R_s}{R}, \frac{f_p R_p}{R}) \qquad \text{or} \qquad (\frac{f_s T_s}{T}, \frac{f_p T_p}{T})\\
          \end{aligned}

    and the ray object is updated with the new polarization components and directions.





