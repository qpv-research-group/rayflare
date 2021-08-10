Examples
========
.. _examples:

Structures using a single simulation method per structure
----------------------------------------------------------

These examples do not use the angular redistribution matrix method; the whole structure to be simulated
is treated with the same optical method (ray-tracing, TMM or RCWA).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   rcwa_examples.ipynb
   rcwa_tmm_validation.ipynb
   rt_pyramids.ipynb
   create_rt_texture.ipynb


Using the angular matrix method
--------------------------------

These examples demonstrate the angular matrix method, where the front and back surfaces of the
structure are treated separately and can be simulated with different methods.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   grating_pyramids_OPTOS.ipynb
   compare_models_3Jsolarcell.ipynb
   compare_models_3Jsolarcell_profile.ipynb
   HIT_emissivity.ipynb