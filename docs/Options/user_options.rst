Options
=======

The function default_options is provided so not every
value has to be set manually by the user each time. All the user options, and their meanings, are listed below. As a general
rule, values with units of distance (wavelength, depth spacing) are in m while angles are in radians.

General options/options for matrix framework
---------------------------------------------

- **wavelengths**: wavelengths at which to carry out calculations (in m)
- **theta_in**: polar angle of incident light (in radians)
- **phi_in**: azimuthal angle of incident light (in radians)
- **phi_symmetry**: used by the matrix framework. Defines symmetry element [0, phi_symmetry] into which values of phi can be collapsed (in radians)
- **n_theta_bins**: used by the matrix framework. Number of polar angle bins used in the angular redistribution matrices (integer)
- **I_thresh**: used by the matrix framework and the ray-tracer. The fraction of incident power remaining below which is
  a ray is considered to have been absorbed
- **depth_spacing**: if absorption profiles for the surface layers are being calculated, the spacing of points at which
  the absorbed energy density should be calculated (float, in m)
- **bulk_profile**: Whether or not to calculate the absorption profile in the bulk material. Boolean.
- **depth_spacing_bulk**: if absorption profile is being calculated in the bulk, the spacing of points at which the absorbed
  energy density should be calculated (float, in m)
- **pol**: Polarization of incident light. Can be 's', 'p' or 'u' (unpolarized, an equal mix of 's' and 'p'). Default 'u'
- **parallel**: whether or not to execute computations in parallel, where relevant (e.g. for the RCWA and ray-tracing). Boolean.
- **n_jobs**: the n_jobs option passed to joblib for parallel computation. Default = -1, which leads to all cores being used.
  Check joblib documentation for further details.
- **c_azimuth**: azimuthal discretization. A higher number means more azimuthal bins.
- **only_incidence_angle**: used by the matrix framework.
  If True, the matrices describing light incident on the front surface from outside the cell are only calculated at the theta_in specified

Options used only by RCWA (S4)
---------------------------------------------

- **A_per_order**: whether or not to calculate absorption per diffraction order (Boolean)
- **S4_options**: options which are passed to S4. See S4 documentation for details
- **orders**: the number of Fourier orders to retain. The actual number of orders used may be different, depending on
  the lattice truncation rule used.


Options used only by the ray-tracer
---------------------------------------------

- **n_rays**: number of rays to trace per wavelength
- **nx**: number of surface points to scan across in the x direction
- **ny**: number of surface points to scan across in the y direction
- **random_ray_position**: if True, rather than scanning across the surface, nx*ny random surface points for ray incidence are generated
- **avoid_edges**: avoid edge of the unit cell (may be useful for surfaces based on AFM scans etc.)
- **randomize_surface**: used only by the ray-tracing algorithm for a whole structure (i.e. not if using ray-tracing in the matrix framework).
  If True, the ray position is randomized before interaction with every surface, mimicking the effect of e.g. random pyramids even though
  the unit cell corresponds to regular pyramids
- **random_ray_angles**: used in the matrix framework. Rather than scanning through incidence angles and populating the matrix that way,
  random incident ray directions are generated

**Note on the number of rays**:

- for ray-tracing structures, *n_rays* is the number of rays traced per wavelengths. This has to
  be considered in combination with the choice of *nx* and *ny*; if you specify *n_rays* as 100 and both *nx* and *ny* as 10,
  that is 100 surface points. So one ray incident on each surface point will be checked.
- for the matrix method, *n_rays* is also the number of rays traced per wavelength, but in this case this number of rays
  has to populate the whole angular redistribution matrix. So if *nx* and *ny* are 10 (100 surface points) and the number
  of angle bins in a half-plane (total bins, not just polar angle theta bins!) is 100, you will need at least 10,000 rays
  to make sure the whole matrix is populated. This is assuming you are not choosing *random_ray_angles* = True, in which case
  there is no guarantee the whole matrix will be populated.

The following options are not set by default, but can be supplied by the user for certain unusual situations:

- **x_limits** and **y_limits**: By default, the ray-tracer
  will scan over the whole surface of the unit cell, but if these options are provided in the form of a list/tuple as [minimum, maximum],
  the ray-tracer will generate x and y points between those limits instead.
- **periodic**: By default, the ray-tracer assumes that
  the unit cell that is provided is periodic, i.e. the surface is made up of an infinite number of these unit cells (square/rectangular tesselation).
  However, in some cases (e.g. ray-tracing for a single lens) you want to consider the unit cell in isolation. Setting this options to False
  achieves this. Note that this option currently only works with rt_structure and not to generate redistribution matrices.
- **initial_material**: The default starting position for the rays is outside the structure, i.e. incident from the incident medium above
  (usually, but not necessarily, this will be air) on the first material in the stack. However, if you want the rays to start inside a different
  material the initial_material options can be set. The incident medium has index 0, the first layer in the stack has index 1, etc. Changing
  this option currently only works for rt_structure and not to generate redistribution matrices.
- **initial_direction**: The default starting direction for the rays is downwards, from medium 0 to medium 1, corresponding to
  initial_direction = 1. If you want to start in the other direction, you can set initial_direction = -1.


Options used only by the TMM
---------------------------------------------

- **lookuptable_angles**: when using combined ray-tracing/TMM in the matrix framework, how many local indicence angles should be
  stored in the TMM lookup table.
- **coherent**: for a full TMM structure calculation (not using the matrix framework), True if the structure is fully coherent, otherwise False.
- **coherency_list**: for a full TMM structure calculation, a list of strings specifying whether each layer should be treated coherently ('c') or
  incoherently ('i'). The incidence and transmission medium are not included, the first element in the list corresponds to the
  first layer in the structure.


.. automodule:: rayflare.options
    :members:
    :undoc-members:
