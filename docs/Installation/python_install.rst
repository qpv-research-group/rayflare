========================================
Python versions and installing Python3.8
========================================

On Ubuntu, Python is an integral part of the operating system, so there will already be a version
(likely more than one version, usually a version of Python2 and a version of Python3) installed
on the operating system out of the box. However, **because** Python forms an integral part of the operating
system, it is not a good idea to mess around with these built-in versions of Python, so I recommend
working in a virtual environment. This also avoids having to use 'sudo' to install or uninstall packages.
There are various packages for creating virtual environments, all with confusingly similar names -- I use
venv (not to be confused with virtualenv or pipenv...).

Currently, Ubuntu18 does not come with Python3.8 (the latest stable Python version), by default, but you can
install it pretty easily. From a terminal window (command line):

.. code-block::

    sudo apt install software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.8 python3.8-dev python3.8-venv


This installs Python3.8, and also the ability to make virtual environments with python3.8-venv.
You can now use your new version of Python3.8 by just typing 'python3.8' on the command line.

Now make the venv:

.. code-block::

    python3.8 -m venv mainenv
    source mainenv/bin/activate


This makes a virtual environment called 'mainenv' (first line) and then activates it (second line). If you close the terminal and open it again you would have to reactivate the virtual environment to make
sure you are using the virtual environment version of Python3.8.

On Ubuntu20, Python3.8 is the default Python3. You can install the ability to make venvs with:

.. code-block::

    sudo apt install software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3-dev python3-venv


And make the venv:

.. code-block::

    python3 -m venv mainenv
    source mainenv/bin/activate


On MacOS, Homebrew Python (default version Python3.8) also comes with the venv ability.