==========================================
Python versions and installing Python3.10
==========================================

Ubuntu
-------

On Ubuntu, Python is an integral part of the operating system, so there will already be a version
(likely more than one version, usually a version of Python2 and a version of Python3) installed
on the operating system out of the box. However, **because** Python forms an integral part of the operating
system, it is not a good idea to mess around with these built-in versions of Python, so I recommend
working in a virtual environment. This also avoids having to use 'sudo' to install or uninstall packages.
There are various packages for creating virtual environments, all with confusingly similar names -- I use
venv (not to be confused with virtualenv or pipenv...).

On Ubuntu20, Python3.8 is the default Python3. You can install the ability to make venvs with:

.. code-block:: console

    sudo apt install python3-dev python3-venv

And make and activate the venv:

.. code-block:: console

    python3 -m venv mainenv
    source mainenv/bin/activate

This makes a virtual environment called 'mainenv' (first line) and then activates it (second line). If you close the
terminal and open it again you would have to reactivate the virtual environment to make
sure you are using the virtual environment version of Python3.10.

Ubuntu20 comes with Python 3.8 by default. Python 3.8 is sufficient for running RayFlare and necessary supporting packages,
but if you want to update to a more recent Python version (3.10), this is easy:

.. code-block:: console

    sudo apt install software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.10 python3.10-dev python3.10-venv

This installs Python3.10, and also the ability to make virtual environments with python3.10-venv.
You can now use your new version of Python3.10 by just typing 'python3.10' on the command line.

Now make the venv:

.. code-block:: console

    python3.10 -m venv mainenv
    source mainenv/bin/activate


MacOS
-------


On MacOS, Homebrew Python (current default version Python3.10) also comes with the venv ability. This can be installed
with:

.. code-block:: console

    brew install python