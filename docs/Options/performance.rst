Performance
===========

Ray-tracing and RCWA are both computationally intensive. RayFlare currently has two approaches for
increasing the speed of calculations:

- **Parallel computation**: RayFlare uses the joblib library to parallelize computations (over CPUs) where possible.
  In ray-tracing, the parallelisation is done over rays (one wavelength calculated at a time, multiple rays - note that
  prior to v2.0.0 the parallelisation was over wavelengths instead). In RCWA, the parallelisation is over wavelengths.
- **Numba JIT (just-in-time) compilation**: RayFlare uses the Numba library to compile many ray-tracing functions to machine
  code, which can result in significant speed-ups.

By default, both of these options are enabled. However, depending on the specific use case and the hardware available,
they may in fact slow down the execution of your script, because in both cases, there is an overhead cost involved:

- **Parallel computation**: there is an overhead cost associated with initialising the separate threads.
- **Numba JIT compilation**: there is an overhead cost associated with compiling the code to machine code. This happens only
  the first time the function is called, but if the function is not called many times, this overhead cost can outweigh the
  time saved on function execution.

To experiment with what settings work best for your script, you can toggle these options on and off as follows:

- **Disable parallel computation**: set the `parallel` user option to `False`:

    .. code-block::

         from rayflare.options import default_options
         options = default_options
         options.parallel = False


  You can also control the number of workers which are spawned by setting the `n_jobs` user option. By default, it is
  set to -1, which means all available CPUs are used, but it can also be set to a specific number.

- **Disable Numba JIT compilation**: add the following at the start of your script - note that this must be before
  any RayFlare classes/functions are imported:

    .. code-block::

         from numba import config
         config.DISABLE_JIT = True