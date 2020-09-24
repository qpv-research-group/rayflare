Angular redistribution matrix method
====================================

There are three distinct uses of the transfer matrix method code: to create redistribution matrices for the matrix framework,
and to define and calculate structures which are treated in their entirety using TMM (so the matrix framework is not used).
In addition, the TMM is used to create lookup tables which are used by the ray-tracing methods.
The function :literal:`TMM` is used to create redistribution matrices, while the class :literal:`tmm_structure` is used to define
structures for ray-tracing. The function :literal:`make_TMM_lookuptable` is used to generate and save lookup tables for
the ray-tracer.


.. automodule:: rayflare.matrix_formalism.process_structure
    :members:
    :undoc-members:

.. automodule:: rayflare.matrix_formalism.multiply_matrices
    :members:
    :undoc-members:

.. automodule:: rayflare.matrix_formalism.ideal_cases
    :members:
    :undoc-members: