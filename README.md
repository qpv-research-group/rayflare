# rayflare
Open-source, integrated optical modelling of complex stacks

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

*Note to new visitors from PVSC:* I am hard at work adding functionality and documentation! In the mean time, I appreciate any comments and feedback. - Phoebe

INSTALLATION INSTRUCTIONS (tested on 64-but Ubuntu 18)
-----
Install setuptools, numpy, matplotlib, xarray, sparse, joblib and seaborn in the normal way (e.g. pip install)

You need the latest version of Solcore (not yet on PyPi). To install this:

git clone https://github.com/qpv-research-group/solcore5.git
cd solcore5
python setup.py install

You may need to replace "python" with the appropriate python version, i.e. a virtual environment or python3.

To install S4 (to use RCWA capability):
You need to install a modified version of S4, which works in Python3. This can be obtained from github.com/phoebe-p/S4
To install:

git clone https://github.com/phoebe-p/S4.git
cd S4
make boost
make S4_pyext


![poster](poster.png "RayFlare poster")
