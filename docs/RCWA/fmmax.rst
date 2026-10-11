FMMAX backend
=============

`FMMAX <https://github.com/matt8s/fmmax>`_ is an optional JAX-based
Fourier modal method backend.
It provides an alternative to S4 for the isotropic structures and outputs
listed below, not a replacement for every S4-specific API.
There are two installation paths: the pinned PyPI baseline and the maintained
source at ``matt8s/fmmax``.
Install the PyPI baseline with::

    pip install "rayflare[fmmax]"

The ``fmmax`` extra pins FMMAX 1.7.1.
That distribution identifies ``invrs-io/fmmax`` as its source; it is not a release
of ``matt8s/fmmax`` or the archived ``facebookresearch/fmmax`` repository.
FMMAX and JAX are imported lazily, so a normal RayFlare import and the default
S4 backend do not require them.
Select FMMAX without changing the existing RCWA API::

    options.RCWA_method = "FMMAX"

The default remains ``"S4"``. FMMAX can be selected for both
:class:`~rayflare.rigorous_coupled_wave_analysis.rcwa.rcwa_structure` and the
:func:`~rayflare.rigorous_coupled_wave_analysis.rcwa.RCWA` redistribution
matrix function.

RayFlare's existing saved-matrix filenames do not include the backend or its
settings.
When switching from S4 or Inkstone, changing the FMMAX source revision, or
changing solver settings, use a fresh matrix directory/surface name or request
``overwrite=True`` when calling ``RCWA``.
Otherwise a previously cached matrix can be returned without running FMMAX.

Maintained-source installation
------------------------------

The integration also has a separate compatibility lane for
`matt8s/fmmax <https://github.com/matt8s/fmmax>`_.
In a fresh environment, install RayFlare without the ``fmmax`` extra and then
install the reviewed source revision::

    python -m pip install rayflare
    python -m pip install "fmmax @ git+https://github.com/matt8s/fmmax.git@88f50196189b57a3de66c1da4a69030cda51bcc9"

For development from the RayFlare checkout, use::

    python -m pip install -e .
    python -m pip install -r requirements/fmmax-maintained.txt

Do not combine this source requirement with ``rayflare[fmmax]``: the current
source metadata reports version 0.11.0, which conflicts with the extra's 1.7.1
pin.
The full commit is the source identifier; the package version alone does not
identify the repository or establish compatibility.
``python -m pip freeze`` records the VCS revision, and installed VCS packages
also record it in their ``direct_url.json`` metadata.

The adapter selects public API capabilities rather than comparing version
strings.
It accepts the grid-size helper in either ``basis`` or ``fft``, uses the solver's
own grid coordinates for rasterization, passes an explicit unit-cell count to
field reconstruction, and normalizes physical loss density by
``0.5 Re(E x H*)`` of the incident fields.
These choices account for API, sampling-origin, and modal-flux prefactor
differences between the two sources without changing either solver.

The dedicated FMMAX workflow installs each source in a separate CPU job and
runs the same adapter tests, including planar TMM comparisons, energy balance,
profile integration, geometry-grid alignment, and patterned Inkstone comparisons.
Workflow results, not the existence of a job definition, establish compatibility
for a particular revision.

The historical `fork compatibility run
<https://github.com/matt8s/rayflare/actions/runs/37257793818>`_ at RayFlare
``88e56e89680afc6e5ecfae0ccb61901fb0eb6d0d`` passed all four CPU jobs.
Each source was tested with Python 3.11/JAX 0.4.38 and Python 3.12/JAX 0.11.2.
Each job passed 30 focused tests, dependency and provenance checks, and the
semiconductor/metal example.
This fork result does not establish an upstream CI pass or GPU agreement.
From each source's separate environment, reproduce the current focused checks with::

    JAX_PLATFORMS=cpu python -m pytest tests/test_fmmax_optional.py tests/test_fmmax.py -q

RayFlare's existing dependency metadata pins NumPy 2.4.0, which the package
index now marks as yanked.
That inherited pin was retained for these reproducibility checks; it is not a
recommendation to select a yanked release for new projects.

Distribution handoff
--------------------

GitHub repository access and PyPI publishing access are separate.
Until the maintained source has a verified release on the existing ``fmmax``
PyPI project, the extra remains on the baseline and the maintained-source lane
remains commit-pinned.
To switch the extra after publishing access is granted:

* Coordinate the release with the current PyPI owners and verify that the
  package is built from the intended maintained-source tag.
* Choose an unused, appropriately newer package version; the inherited 0.11.0
  version must not be reused as the successor to PyPI 1.7.1.
* Establish working release credentials or trusted publishing, update package
  and documentation links, and verify the wheel and sdist in a clean environment.
* Run the RayFlare numerical and optional-dependency tests against that released
  wheel, then update the extra pin, installation-error version, CI source matrix,
  and validation record together in a new commit.
* Retain the existing baseline evidence and compatibility lane until the intended
  support policy is agreed with RayFlare's maintainers.

This handoff does not require a new RayFlare backend name or API.

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
* Each solve uses JAX's ``highest`` matrix-multiply precision locally, while
  retaining the selected array dtype and restoring the caller's precision setting.
  Reduced-mantissa GPU defaults can otherwise violate the power/flux tolerances
  even for planar stacks.
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
Polygon vertices are relative to the centre and rotate with the shape, as in
the S4 interface; Inkstone's current RayFlare interface ignores polygon rotation.
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
``eig_backend``      ``"default"``          Optional maintained-source eigensolver
===================  ====================  =====================================

The raster is automatically enlarged if required by the Fourier expansion. An
explicit ``grid_shape`` that is too small is rejected. As with any RCWA solve,
both Fourier terms and geometry-grid resolution must be converged for the
specific structure.

``eig_backend="default"`` leaves the installed solver's default unchanged.
The maintained source additionally exposes ``"cusolver"`` for supported NVIDIA
GPU/JAX installations.
An installation without the selection API raises an explicit exception for a
nondefault request rather than silently ignoring it.
Selecting ``"cusolver"`` is not a performance guarantee.
The pinned maintained source's default eigensolver uses a host callback on GPU
when the optional ``jeig`` package is absent.
That revision initializes its CPU callback at import time, including when
``cusolver`` is selected, so enable both backends with ``JAX_PLATFORMS=cuda,cpu``.
JAX requires both listed backends to initialize and uses CUDA as the default;
the validation script additionally asserts that default and solver-array devices.
PyPI 1.7.1 instead delegates to JAX's default eigensolver; its execution location
depends on the JAX release and installed CUDA libraries.
In JAX 0.11.2, right-eigenvector-only solves can use native cuSOLVER when its
``geev`` implementation is available.
GPU-resident result arrays alone do not prove that eigensolving stayed on GPU.

Validation and tolerances
-------------------------

The following recorded baseline checks used FMMAX 1.7.1, JAX/JAXlib 0.11.2,
single-precision JAX arrays, and one CPU device, before the maintained-source
compatibility changes.
They do not establish validation of the maintained source.
Errors are absolute because R, T, and absorption are bounded power fractions
and some reference values approach zero.

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

In the historical baseline, on the same patterned 25-term/4 nm case, an explicitly synchronized CPU run
took 11.32 s including first compilation and ``0.637 +/- 0.023`` s over five
repeated solves. Both visible A100 GPUs were occupied by another process
(approximately 37.2 of 40 GB on each), so no GPU package was installed and no
CPU/GPU agreement or GPU timing was reported in that baseline.

Reproducing CPU/GPU comparisons
-------------------------------

``tests/fmmax_validation.py`` saves matching planar, circular-pattern,
off-centre lossy-pattern profile, and front/rear diffracting-matrix fixtures.
It asserts the selected default backend and checks eigensolve, scattering,
and reconstructed-field arrays on that backend.
Snapshots record the RayFlare SHA, source provenance, precision, package
versions, environment settings, and representative eigensolver lowering.
They also record ordered hashes and shapes of the material-permittivity rasters;
comparisons reject different geometry samples before checking numerical errors.
Run CPU and GPU snapshots in separate processes and keep their outputs outside
the checkout::

    JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python tests/fmmax_validation.py snapshot /path/to/cpu.npz --backend cpu
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu \
        MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python tests/fmmax_validation.py snapshot /path/to/gpu.npz --backend gpu
    python tests/fmmax_validation.py compare /path/to/cpu.npz /path/to/gpu.npz

Use the same source, runtime, precision, and committed checkout for both runs.
For maintained-source GPU eigensolving, add ``--eig-backend cusolver`` to a
separate GPU snapshot; the script requires the native cuSOLVER lowering and
compares its observables with the default snapshot.
PyPI baseline rejects this adapter option explicitly.
The comparison reports absolute errors in powers and matrices, pointwise
profile errors, and profile-integral errors, including individual planar layers.
It is a correctness check, not a timing benchmark.
Repeat snapshots with ``JAX_ENABLE_X64=true`` for a double-precision control.
Float32 power comparisons use the original ``2e-6`` absolute floor or the
modal accumulation scale ``gamma_(2N) = 2N epsilon / (1 - 2N epsilon)``,
whichever is larger, where ``N`` is the actual Fourier-term count.
For the 21- and 45-term patterned fixtures this is approximately ``5.01e-6``
and ``1.07e-5``.
This is a fixture-specific roundoff-scale criterion, not a general eigensolver
error bound; eigenpair residuals, eigenvector conditioning, double-precision
controls, physical bounds, and convergence are checked separately.
Planar TMM and energy-balance regressions retain their ``2e-6`` tolerances.
Profile comparisons retain ``2e-8`` nm\ :sup:`-1` pointwise and ``2e-6``
integrated absolute/L1 limits.

The expanded regressions use independent TMM powers and local loss density for
multilayer public ``calculate_profile()`` checks at 450, 650, and 850 nm,
including a lossless stack.
Patterned-profile checks use a rotated, off-centre ellipse with different loss
in feature and background and separately refine Fourier terms, raster spacing,
and midpoint depth quadrature.
Diffraction checks use a 900 nm period at 500 nm and nonzero azimuth; the
grating equation determines each propagating order's reflected/transmitted bin,
including accumulated power when several orders share a bin.
These references test normalization and angular placement, not only total energy.

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
