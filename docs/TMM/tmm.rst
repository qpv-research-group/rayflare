Transfer-matrix method
======================

There are three distinct uses of the ray-tracing code: to create redistribution matrices for the matrix framework, to
generate lookup tables which are used by the ray-tracer, and to define and calculate structures which are treated in their entirety
using TMM(so the matrix framework is not used).
The function :literal:`TMM` is used to create redistribution matrices, while the class :literal:`tmm_structure` is used to define
structures for ray-tracing. The function :literal:`make_TMM_lookuptable` generates lookup tables for the ray-tracer.

.. automodule:: rayflare.transfer_matrix_method.tmm
    :members:
    :undoc-members:

.. automodule:: rayflare.transfer_matrix_method.lookup_table
    :members:
    :undoc-members:

