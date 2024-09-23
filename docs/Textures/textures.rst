Textures
==========================
.. _textures:

This module is used to generate textures for interfaces (currently only for ray-tracing textures, but
this could be expanded to surfaces for RCWA calculations). It contains both standard textures and
functions to define your own ray-tracing textures from input coordinates.

Whether you are defining your own surface textures or using standard textures, you can pass additional arguments to these
surfaces to specify surface layers, which will be treated optically using TMM. The relevant keyword arguments are:

- :literal:`interface_layers`: a list of Solcore Layer objects
- :literal:`coherency_list`: This is a list of the same length as :literal:`interface_layers`
  specifying which layers are coherent ("c") and which are incoherent ("i"). If not provided, all layers are
  assumed to be coherent.
- :literal:`name`: optional; name of the interface under which the TMM lookuptable will be saved.

See `here`_ for an example of how to use this in practice.

You can also specific whether you want to apply `Phong scattering`_ to the
rays which interact with the surface:

- :literal:`phong`: whether to apply Phong scattering to rays which interact with this surface. Default is False.
- :literal:`phong_options`: a list with two entries. The first entry is alpha, which describes how Lambertian the
  scattering is (alpha = 1 is Lambertian, alpha = infinity is perfectly specular; see paper linked above).
  The second entry is whether to randomize the polarization of the ray after applying the scattering. Default is
  [25, True].

And whether you want the surface to be treated analytically (if possible):

- :literal:`analytical`: whether to treat the surface analytically. Default is False.
- :literal:`n_analytical_interactions`: the maximum number if interactions to consider when doing analytical
  ray-tracing (default is 3). Note that this maximum does not apply when the calculation switches to full ray-tracing.

###################################
Individual function documentation:
###################################

.. automodule:: rayflare.textures.define_textures
    :members:
    :undoc-members:

.. automodule:: rayflare.textures.standard_rt_textures
    :members:
    :undoc-members:


.. _here: https://rayflare.readthedocs.io/en/latest/Examples/perovskite_Si_rt.html
.. _Phong scattering: https://ieeexplore.ieee.org/document/8638770