Ray-tracing
============
.. _ray_tracing:

There are two distinct uses of the ray-tracing code: to create redistribution matrices for the matrix framework,
and to define and calculate structures which are treated in their entirety by ray-tracing (so the matrix framework is not used).
Both of these methods can be used in combination with TMM lookup tables to calculate realistic reflection, transmission
and absorption probabilities.
The function :literal:`RT` is used to create redistribution matrices, while the class :literal:`rt_structure` is used to define
structures for ray-tracing. Note that for both of these methods, the default behaviour when using TMM lookup tables is to
NOT recalculate the lookup tables if they already exist; currently, RayFlare does not detect whether you have e.g. changed
the wavelength range or spacing, the interface layers, thicknesses, materials, or their coherence. To force recalculation, you can set the :literal:`overwrite`
parameter to :literal:`True` in the :literal:`RT` function or the :literal:`rt_structure` class. Note that if you keep
the interface layers the same but change the surface texture (e.g. opening angle of pyramids), you do not
need to recalculate the lookup tables.


###################################
Individual function documentation:
###################################

.. automodule:: rayflare.ray_tracing.rt_structure
    :members:
    :undoc-members:


.. automodule:: rayflare.ray_tracing.rt_matrix
    :members:
    :undoc-members:


.. automodule:: rayflare.ray_tracing.rt_common
    :members:
    :undoc-members:

