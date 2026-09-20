FMMAX backend
=============

`FMMAX <https://github.com/facebookresearch/fmmax>`_ is an optional JAX-based
Fourier modal method backend. Install the reproducible version used by this
adapter with::

    pip install "rayflare[fmmax]"

The ``fmmax`` extra pins FMMAX 1.7.1. FMMAX and JAX are imported lazily, so a
normal RayFlare import and the default S4 backend do not require them. Select
FMMAX without changing the existing RCWA API::

    options.RCWA_method = "FMMAX"

The default remains ``"S4"``. FMMAX can be selected for both
:class:`~rayflare.rigorous_coupled_wave_analysis.rcwa.rcwa_structure` and the
:func:`~rayflare.rigorous_coupled_wave_analysis.rcwa.RCWA` redistribution
matrix function.

Conventions
-----------

* Wavelengths, finite-layer thicknesses, lattice vectors, shape coordinates,
  and the FMMAX raster spacing are passed to FMMAX in nm. Public RayFlare
  wavelengths and ``depth_spacing`` remain in metres as elsewhere in RayFlare.
* RayFlare's complex index is ``n + i k``. The adapter passes
  ``epsilon = (n + i k)^2`` to FMMAX. This is passive (positive loss) with the
  ``exp(-i omega t)`` time convention used by FMMAX.
* ``size=((ux, uy), (vx, vy))`` gives the two primitive lattice vectors in
  Cartesian coordinates. Shape centres and vertices are Cartesian coordinates
  in the same nm unit. Shapes are periodically wrapped into FMMAX's cell,
  whose vertices are ``0``, ``u``, ``v``, and ``u + v``.
* ``theta_in`` is measured from +z and ``phi_in`` from +x toward +y. Both are
  specified in radians in RayFlare options. Geometry rotation angles retain
  RayFlare's degree convention.
* ``pol="s"`` and ``pol="p"`` create unit transverse-electric-field
  excitations. ``pol="u"`` is an incoherent, equal-intensity average of
  independent s and p solves. A two-element ``(s, p)`` tuple remains a coherent
  amplitude superposition.
* Every reflected, transmitted, layer-absorption, and profile result is divided
  by the incident z-directed Poynting flux. Layer absorption is the difference
  between unit-cell-averaged flux at the two layer boundaries. Profiles are
  calculated from ``omega Im(epsilon) |E|^2 / 2`` and have units nm\ :sup:`-1`.
* ``options.orders`` is FMMAX's *approximate* number of Fourier terms. Symmetric
  circular or parallelogramic truncation can change the actual count; for
  example, requested counts 25 and 49 produce 21 and 45 terms for the square
  validation cell below.

Geometry and options
--------------------

Isotropic planar layers and rasterized ``circle``, ``ellipse``, ``rectangle``,
and ``polygon`` entries in ``Layer.geometry`` are supported. Multiple shapes
and patterned layers are allowed; later entries overwrite earlier entries in
overlap regions. A periodic feature must fit within one primitive-cell span.
An absorbing incidence medium, unknown shape type, degenerate lattice, invalid
grid, or unsupported option raises an explicit exception.

FMMAX-specific settings are supplied in ``options.FMMAX_options``:

===================  ====================  =====================================
Option               Default               Meaning
===================  ====================  =====================================
``formulation``      ``"fft"``             FMMAX formulation enum value
``truncation``       ``"circular"``        ``circular`` or ``parallelogramic``
``grid_spacing``     ``5.0``                Maximum requested raster spacing (nm)
``grid_shape``       ``None``               Explicit ``(nx, ny)`` override
``force_x64_solve``  ``False``              Use FMMAX's x64 matrix-solve path
``max_grid_points``  ``1000000``            Guard against accidental large grids
===================  ====================  =====================================

The raster is automatically enlarged if required by the Fourier expansion. An
explicit ``grid_shape`` that is too small is rejected. As with any RCWA solve,
both Fourier terms and geometry-grid resolution must be converged for the
specific structure.

Validation and tolerances
-------------------------

The following checks used FMMAX 1.7.1, JAX/JAXlib 0.11.2, single-precision JAX
arrays, and one CPU device. Errors are absolute because R, T, and absorption are
bounded power fractions and some reference values approach zero.

.. list-table::
    :header-rows: 1

    * - Check
      - Maximum absolute error
    * - Planar vs. coherent TMM, 27 cases
      - ``3.80e-7`` in R, T, or layer A
    * - Planar ``R + T + sum(A)``
      - ``3.65e-7``
    * - 0.5 nm profile integration vs. layer A
      - ``2.47e-4``
    * - Patterned vs. Inkstone, 25 requested terms
      - ``1.42e-2``
    * - Patterned vs. Inkstone, 81 terms/2 nm grid
      - ``4.27e-3``
    * - Patterned ``R + T + sum(A)``
      - ``5.4e-7``

The planar sweep used wavelengths 450, 650, and 850 nm; incidence angles 0,
35, and 60 degrees; and s, p, and unpolarized illumination. The patterned case
was a 400 nm square cell containing a 100 nm-radius circular air hole in a
120 nm layer with index ``2.2 + 0.04i``, at 600 nm, ``theta=0.35`` rad and
``phi=0.2`` rad. At 2 nm raster spacing, increasing requested terms from 49 to
81 changed the largest R/T/A component by ``5.42e-3``. At 81 terms, reducing
raster spacing from 4 to 2 nm changed it by ``2.38e-3``. Regression tolerances
are therefore ``2e-6`` for planar R/T/A, ``3e-4`` for profile integrals,
``2e-2`` for the inexpensive patterned cross-backend test, ``6e-3`` for the
Fourier-order check, and ``3e-3`` for the raster check. These are validation
tolerances, not universal convergence guarantees.

On the same patterned 25-term/4 nm case, an explicitly synchronized CPU run
took 11.32 s including first compilation and ``0.637 +/- 0.023`` s over five
repeated solves. Both visible A100 GPUs were occupied by another process
(approximately 37.2 of 40 GB on each), so no GPU package was installed and no
CPU/GPU agreement or GPU timing is reported.

Limitations
-----------

Only isotropic, nonmagnetic material data from RayFlare is adapted in this
milestone. ``A_per_order``, ``detailed_rcwa``, ``detail_layer`` redistribution,
and S4/Inkstone-specific field-plotting methods are rejected for FMMAX. Total
R/T, layer absorption, and depth-resolved unit-cell-averaged absorption are
supported.

FMMAX itself supports JAX batching, JIT compilation, accelerators, and automatic
differentiation. RayFlare currently evaluates material data in NumPy, loops over
wavelengths and matrix angles through joblib, rasterizes Python geometry, and
converts solver outputs to NumPy. Consequently this adapter does not preserve
FMMAX batching across the RayFlare boundary and is **not** an end-to-end
differentiable RayFlare interface. No differentiability claim is made. For JAX
workloads, ``options.parallel=False`` also avoids launching multiple joblib
processes that compile independently.
