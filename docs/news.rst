News & Updates
================
.. _news:

To update to the latest version of RayFlare, run the following command in your terminal:

.. code-block:: bash

   pip install rayflare --upgrade

Version 1.2.1 released (2023-11-19)
------------------------------------
**Highlights:**

- New RCWA method ([Inkstone](https://github.com/alexysong/inkstone) now intergrated with RayFlare.
  This is an all-Python program which is therefore easy to install (unlike S4). This means all core u
  functionality of RayFlare is now available without the need to compile anything on your computer.
  Users can toggle between S4 and Inkstone by setting the ``RCWA_method`` option.
- Calculations with unpolarized light using RCWA should be faster now.
- The ``wavelengths`` user options is now called ``wavelength``, for compatibility with Solcore.
  ``wavelengths`` still works, but will be deprecated in future.

Version 1.2.0 released (2023-03-21)
------------------------------------

**Highlights:**

- New functionality with ray-tracing, useful for e.g.
  `perovskite/Si <https://rayflare.readthedocs.io/en/latest/Examples/perovskite_Si_rt.html>`_ or III-V/Si tandems.  \
- New functions to generate textures: rough pyramids, rough planar surfaces, and (rough or smooth) hemispherical caps.
  See :ref:`here <textures>` for further details.


**Possible backwards compatibility issues:**

- The :literal:`make_absorption_function` function now requires a slightly different input format (the full result
  of the optical calculation, rather than just the profile part). This is to standardize its behaviour across all
  methods. See :ref:`the documentation <utilities>` for more details.


This is a major release with a lot of new features and fixes. The biggest change is the introduction
of functionality for ray-tracing simulations: previously, it was possible to use ray-tracing in conjunction with the
transfer-matrix model to calculate reflection/transmission/absorption probabilities at interfaces only when using the
angular redistribution method. It is now possible to do this in a pure ray-tracing simulation. Generally, if you do not
need to use another method for one of the interfaces, doing a pure ray-tracing simulation is faster than using the
angular redistribution matrix method with ray-tracing. An example of the new simulation, which is useful for modelling
e.g. perovskite on Si or III-V on Si solar cells with textured silicon surfaces, can be found
`here <https://rayflare.readthedocs.io/en/latest/Examples/perovskite_Si_rt.html>`_.

It is now also possible to specify that previous results for the angular redistribution matrices should be overwritten
rather than loaded. This is useful if you have changed the geometry of the system and want to recalculate without changing
the project name or manually deleting files every time. This can be done by setting the :literal:`overwrite` keyword
argument to :literal:`True` when calling :literal:`process_structure`. See :ref:`the docstring <matrix>` here.

There were also two bugs relating to the calculation of the absorption profile, both of which were apparent only in
very specific circumstances which are unlikely to occur in a real solar cell (highly transparent absorbers). The details
of what was changed and what was wrong previously are explained
`here <https://github.com/qpv-research-group/rayflare/issues/56>`_.

