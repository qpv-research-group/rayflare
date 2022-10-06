from setuptools import find_packages, setup

import os
import sys
from configparser import ConfigParser


def gen_data_files(*dirs):
    """ Creates the list of files (not necessarily python files) that need to be
    installed together with the rest of stuff """
    results = []
    exclude = [".DS_Store", "__pycache__", "egg", ".git"]
    for src_dir in dirs:
        for root, dirs, files in os.walk(src_dir):
            if not any(sub in root for sub in exclude):
                results.append(
                    (root, [os.path.join(root, f) for f in files if f not in exclude])
                )
    return results


here = os.path.abspath(os.path.dirname(__file__))
default_config = os.path.join(here, "rayflare", "rayflare_config.txt")
config = ConfigParser()
config.read([default_config])

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "matplotlib",
    "scipy",
    "numpy",
    "solcore",
    "xarray",
    "sparse",
    "joblib",
    "seaborn"
]


setup(
    name="rayflare",
    version="1.1.0",
    description="Python-based integrated optical modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qpv-research-group/rayflare",
    project_urls={
        "Documentation": "http://rayflare.readthedocs.io",
    },
    author="Phoebe Pearce",
    author_email="phoebe.pearce15@imperial.ac.uk",
    license="GNU GPLv3",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="photovoltaics modelling physics optics",
    packages=find_packages(exclude=[]),
    package_data={"": ["*.*"]},
    include_package_data=True,
    setup_requires="pytest-runner",
    install_requires=install_requires,
    #tests_require=tests_require,
    #extras_require=extras_require,
)
