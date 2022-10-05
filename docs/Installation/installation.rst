======================================================
Installation instructions
======================================================

.. toctree::
    :maxdepth: 2

    python_install

Most of RayFlare will run on any platform (Windows, Linux, MacOS). However, if you want to use the rigorous coupled-wave analysis (RCWA)
functionality, this requires installing the Python package S4, which must be compiled locally (on your computer). Instructions for
setting up S4 are available for Ubuntu (Linux) and MacOS, but not for Windows. If you don't intend to use RayFlare's RCWA functionality,
you can skip the 'Setting up' steps for your platform below and go straight to the :ref:`Installing RayFlare <install>` section.

Setting up to install S4 on Ubuntu
-----------------------------------

This is assuming an Ubuntu computer or Virtual Machine, or MacOS. For an Ubuntu  Virtual Machine running on a Windows platform,
I have used both VirtualBox and Hyper-V manager to run the VM. You can download an Ubuntu disk image from the `official website`_.

There are lots of guides online for how to get an Ubuntu VM working in Windows, for example
here_. Alternatively, you can `dual boot`_ your computer with Windows and Ubuntu. Both methods have some advantages and disadvantages (running a Virtual Machine
is somewhat safer and you can access it without having to reboot your computer. However, the performance won't be
the same as running the operating system directly on the computer's hardware).

These instructions follow installation on Ubuntu version 18.04 but have also been tested on Ubuntu16 and Ubuntu20.

Make sure that 'pip' points to the right version of Python or the virtual environment you are using; you may
have to use pip3.

First, we need to install some compilers and some libraries which S4 uses.

.. code-block:: console

    sudo apt install make git gcc g++ gfortran
    sudo apt-get install libopenblas-dev libfftw3-dev libsuitesparse-dev libboost-all-dev

In the first line, we install the 'make' command which is needed to install S4, as well as git and compilers we
will need for both Fortran (in the PDD solver in Solcore) and C++ (in S4). Then we install some external libraries which
S4 needs to build the Python extension.

Setting up to install S4 on MacOS
-----------------------------------

There are multiple ways to install Python in MacOS - you can use homebrew_. If you install
it using :literal:`brew install python3` (after installing homebrew), you will automatically have venv available and can make
a virtual environment easily with:

.. code-block:: console

    python3 -m venv mainenv
    source mainenv/bin/activate


For the relevant compilers and the :literal:`make` and :literal:`git` commands, these are all available through Apple Developer Tools (which
you can install through the App store), or you can install them using Homebrew. To install the relevant libraries for S4,
you can again use homebrew:

.. code-block:: console

    brew install fftw suite-sparse openblas lapack boost


Installing S4 (Ubuntu and MacOS)
---------------------------------

Install some packages we will need:

.. code-block:: console

    pip install wheel setuptools numba

Downloading and installing S4 (start from the directory in which you want to download the S4 folder):

.. code-block:: console

    git clone https://github.com/phoebe-p/S4.git
    cd S4
    make S4_pyext

Here we download my fork of S4 (modified to be compatible with Python3), move to the downloaded directory, and then we make the Python extension.
If you activated the virtual environment before running this, S4 should automatically install into that virtual environment.

**Note for users of new Mac devices with Apple M1/ARM chips:** You can install packages using Homebrew as described above,
but instead of :literal:`make S4_pyext`, you should run :literal:`make S4_pyext --file="Makefile.m1"`.

.. _install:

Installing RayFlare
---------------------

To do a normal installation from PyPI (using :literal:`pip`):

.. code-block:: console

    pip install setuptools wheel numba
    pip install rayflare

This will install install the most recent version uploaded to PyPI (currently 1.1.0), which is not necessarily the most
up to date version.

To install RayFlare from source (GitHub):

.. code-block:: console

    pip install setuptools wheel numba
    git clone https://github.com/qpv-research-group/rayflare.git
    cd rayflare
    pip install .

If you want to install in development (so that changes you make to the directory where you downloaded RayFlare are
reflected in the installed version without having to reinstall), change the last line to:

.. code-block:: console

    pip install -e .'[dev]'


Setting things up to use/develop
-----------------------------------

If you want to run stuff from the command line, you just need to make sure you activate the
relevant Python environment. If you are using PyCharm, you can tell it which Python interpreter (version) to use
by going to File > Settings > Project > Project Interpreter (or it may be under PyCharm > Preferences > Project,
depending on the PyCharm version). Click on the little arrow next to the current Python interpreter, click 'Show all',
click on the +. Then you can select that you want to use an existing interpreter and navigate to the python
executable file for the environment you want to use (so for the 'mainenv' environment we made above that is
mainenv/bin/python). PyCharm will remember that you want to use this interpreter, and another nice thing is
that when you open a Terminal (command line) window in PyCharm it will also automatically load the venv, so running
things like 'pip install' will point to the right place.

As an alternative to the venv method above, PyCharm also has built-in support for making a new virtualenv environment
based on one of your 'system interpreters' (these are the versions of Python installed -- if you followed the
instructions above and are on Ubuntu 20, these are probably Python2.7, Python3.8 and possibly the newly-installed
Python3.10). You can create one of those environments by going to Project Interpreter in the same way, adding a new
environment and selecting 'Virtualenv environment' with the relevant base interpreter.

Installing S4 on Windows
-----------------------------------

To date, I have not yet found a way to make S4 work on Windows (although it is certainly possible, because
the developers state that it was developed on Windows).


Troubleshooting
-----------------------------------

- **The 'make boost' command gives loads of output and takes a long time.**

    This is normal. Have a coffee/tea/stretching break :)

- **I am trying to use Solcore/RayFlare but some examples crash without executing.**

    In PyCharm: 'Process finished with exit code Process finished with exit code -1073741571'

    In Spyder: 'Restarting Kernel...'

    This issue appears to be related to attempting to import solcore.poisson_drift_diffusion when using IPython
    as the kernel, even when the PDD was successfully installed as above or is not being used directly.
    Examples will run successfully on the command
    line but not when using an IDE, and even examples which don't directly try to import or use the PDD will fail because
    they use :literal:`solar_cell_solver` or another Solcore file which tries to import the PDD, despite the current
    try/except formulation. For now, the only way to avoid this is to not use
    IPython; in PyCharm, go to Preferences/Setting > Build, Execution and Deployment > Console and untick the 'Use IPython if available'
    options. Current version of Spyder only have access to IPython and no other Python consoles, so the only option
    seems to be to use a different IDE.


.. _official website: https://releases.ubuntu.com/18.04/
.. _here: https://itsfoss.com/install-linux-in-virtualbox/
.. _dual boot: https://linuxconfig.org/how-to-install-ubuntu-20-04-alongside-windows-10-dual-boot
.. _homebrew: https://brew.sh/