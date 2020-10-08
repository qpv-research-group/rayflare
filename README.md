[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![codecov](https://codecov.io/gh/qpv-research-group/rayflare/branch/devel/graph/badge.svg)](https://codecov.io/gh/qpv-research-group/rayflare)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7ff9180e5f7a460192440895d823ff15)](https://www.codacy.com/gh/qpv-research-group/rayflare?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=qpv-research-group/rayflare&amp;utm_campaign=Badge_Grade)
[![Documentation Status](https://readthedocs.org/projects/rayflare/badge/?version=latest)](https://rayflare.readthedocs.io/en/latest/?badge=latest)


# rayflare
Open-source, integrated optical modelling of complex stacks

You can view RayFlare's documentation [here](https://rayflare.readthedocs.io/en/latest/).

*Package requirements:*
- setuptools
- numpy
- matplotlib
- solcore
- xarray
- sparse
- joblib
- seaborn (for plotting in examples, not required otherwise)
- S4 (to use RCWA functionality, not required otherwise)

The aim of RayFlare is to provide a flexible, user-friendly Python environment to model complex optical stacks, with a focus on solar cells. 

INSTALLATION INSTRUCTIONS (tested on 64-bit Ubuntu 16)
-----
Use Python 3.x where x >= 6
Install setuptools, numpy, matplotlib, xarray, sparse, joblib and seaborn in the normal way (e.g. pip install)

You need the latest version of Solcore (not yet on PyPi). To install this:

```
git clone https://github.com/qpv-research-group/solcore5.git

cd solcore5

python setup.py install
```

You may need to replace "python" with the appropriate python version, i.e. a virtual environment or python3.

To install S4 (to use RCWA capability):
You need to install a modified version of S4, which works in Python3. This can be obtained from github.com/phoebe-p/S4
To install:

```
git clone https://github.com/phoebe-p/S4.git

cd S4

make boost

make S4_pyext
```


![poster](poster.png "RayFlare poster")
