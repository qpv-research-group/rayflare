Analytical ray-tracing
=======================

For the most common surface texture used in simulating silicon solar cells, regular pyramids
with an opening angle around 54 degrees, light at normal incidence will always have at
most two reflections from the front surface; the ray may enter the bulk after the first
interaction, or reflect and hit the opposite face of on adjacent pyramid. The ray may
then again enter the bulk, or reflect; if it reflects, it will leave the surface and cannot
hit another pyramids. This simplies the ray-tracing problem significantly, and we can
run an analytical calculation (also for off-normal incidence, as long as the maximum number
of interactions is known in advance on is the same for each ray, regardless of where on the
unit cell the ray first hits). For upright pyramids, the following table summarises regimes
where the maximum number of interactions is the same for all rays:

