Overview & background
====================================================
.. _overview:

A detailed look at the motivation, implementation, mathematics etc. behind
RayFlare can be found :download:`here <RayFlare_theory.pdf>`. This includes an
explanation of the ray-tracing algorithm, ray-tracing with integrated TMM,
the implementation of the angular redistribution matrix framework and how it is used to calculate useful
quantities such as absorption per layer and absorption profiles, and several
examples to validate the matrix framework with different combinations of methods
for the front and back surface. A brief overview is given below.

The idea behind RayFlare is to provide a comprehensive and flexible optical modelling framework. At the moment,
the following base methods are included:

- **Ray-tracing (RT)**: Geometric optics. Suitable for calculating reflection and refraction due to large-scale (compared to the wavelength)
  textures, which are described in terms of a triangulated surface.
- **Transfer-matrix method (TMM)**: Suitable for calculating interference effects in thin-films (transparent or absorbing).
- **Rigorous coupled-wave analysis (RCWA)**: Maxwell solver for calculating diffraction effects from periodic structures

All of these methods can be used to simulate a whole structure (using the rt_structure, tmm_structure and rcwa_structure
classes, respectively). In this case, the relevant materials and thicknesses of the layers, surface textures (RT) or grating shapes
(RCWA) are defined by the user and the reflection, transmission, absorption per layer and (if specified) absorption profile
in the structure are calculated by RayFlare.

However, they can also be used in combination with each other by using the angular redistribution matrix formalism (not to be confused
with TMM, the transfer matrix method!). In this
framework, the front and rear surface of the structure are decoupled from one another, and suitable methods are used for each surface
to calculate so-called redistribution matrices. These matrices described how light striking the surface from different directions
is scattered into other directions (which can be in transmission or reflection) or absorbed by any surface layers. The front and
back surface are coupled by some bulk medium, in which simple Beer-Lambert absorption is assumed. This reduces the problem down
to a matrix multiplication process to calculate total reflection, transmission, and absorption in the
surface layers and bulk medium. Note that for the Beer-Lambert absorption approximation to be valid, the bulk medium must be thick
enough that interference effects in it can be neglected, i.e. significantly thicker than the maximum incident wavelength of light
being considered.

While this matrix multiplication framework has been previously established by e.g. `OPTOS`_ and `GenPro4`_, the novelty of RayFlare is that
it is open-source and includes support for grating structures (unlike GenPro4) and that it will automatically calculate the necessary redistribution matrices
for the user-defined structure (using any of the methods listed below) at the required wavelengths (unlike OPTOS), including tracking absorption in
any surface layers on the
front and rear surface. It can also calculate absorption profiles in the surface and bulk layers, something which has not previously been
demonstrated to my knowledge.

The methods currently available to generate redistribution matrices are:

- **Ray-tracing with the Fresnel equations**. This is suitable for simple interfaces with some large-scale texture and no surface layers.
- **Integrated ray-tracing and transfer-matrix method**: The probability of reflection/transmission
  (or absorption in interface layers) are calculated using TMM rather than through the Fresnel equations. This is suitable for large-scale
  textures with one or more thin (compared to the size of the texture features) surface layers.
- **TMM**: Suitable for planar surfaces with multiple layers.
- **RCWA**: For gratings, which can be made of multiple layers including planar layers.
- **Ideal/theoretical reference cases**: currently, a perfect mirror or a perfect Lambertian scatterer.

Some examples of structures which can be treated in this way:

- A structure with a planar front surface with an anti-reflection coating, and a pyramidal back surface. The redistribution matrix
  for the front can be calculated with TMM (very fast) and only the back needs to be treated with ray-tracing (slower).
- A structure with pyramids on the front surface and a diffracting grating on the rear surface. The front surface can be treated with
  TMM (or ray-tracing + TMM) while the rear surface is treated with RCWA.

See the :ref:`Examples <examples>` section for more.


.. _OPTOS: https://doi.org/10.1364/OE.23.0A1720
.. _GenPro4: https://doi.org/10.1109/JPHOTOV.2017.2669640