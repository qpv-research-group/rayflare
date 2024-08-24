Treatment of polarization in ray-tracing
==========================================

Note: for ray-tracing, the below applies to RayFlare version 2.0.0 (August 2024) and later.

The polarization of light is a vector quantity that describes the orientation of the electric field of an electromagnetic
wave. In the context of ray-tracing, the polarization of light is important when considering the reflection and
transmission of light at interfaces. Across all methods (ray-tracing, RCWA and TMM) we decompose the polarization into two orthogonal components, *s* and *p*, following
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

For planar surfaces, the s and p planes directions stay the same throughout a simulations. However,
in ray-tracing, the s and p components of a ray relative to the surface it is hitting change depending on the
orientation of the ray and the surface, and so it is necessary to calculate the component of the ray which lies
in the s and p direction of the ray-surface plane. This is done by projecting the ray vector onto the s and p
directions

:math:`R_s`, :math:`R_s`, :math:`T_s`, :math:`T_p` are the reflection and transmission coefficients for s and p polarization.
These can be calculated either with the Fresnel equations, or using the transfer-matrix method, but this does
not affect the treatment of polarization
