Analytical ray-tracing
=======================

**Note that analytical ray-tracing currently only works for incidence from the incidence medium of the
front surface of the structure (i.e., options.initial_direction = 1, options.initial_material = 0,
which are the default options). Currently, it has been left up to the user to determine whether
the analytical method is suitable for their structure; the default behaviour is NOT to use it. Whether
or not the analytical method can be used is set for each texture or Interface.**

For the most common surface texture used in simulating silicon solar cells, regular pyramids
with an opening angle around 54 degrees, light at normal incidence will always have at
most two reflections from the front surface; the ray may enter the bulk after the first
interaction, or reflect and hit the opposite face of on adjacent pyramid. The ray may
then again enter the bulk, or reflect; if it reflects, it will leave the surface and cannot
hit another pyramids. This simplies the ray-tracing problem significantly, and we can
run an analytical calculation (also for off-normal incidence, as long as the maximum number
of interactions is known in advance on is the same for each ray, regardless of where on the
unit cell the ray first hits).

