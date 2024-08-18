Treatment of polarization in ray-tracing
==========================================

Note: the below applied to RayFlare version 2.0.0 (August 2024) and later. Prior to this,
the polarization could be set to 's', 'p' or 'u' (unpolarized), with unpolarized light being
an equal mixture of *s* and *p*. While results for pure *s* or *p*
polarization are unchanged between the current version and earlier version, the treatment
for unpolarized light in previous versions was not rigorous, as it was assumed that an initially
unpolarized would remain an equal mixture of *s* and *p* through the ray-tracing procedure. For non-normal incidence,
the reflectance of s and p polarized light is not the same, and thus the ratio *s*:*p*-polarized light will change
each time a ray interacts with a surface.

