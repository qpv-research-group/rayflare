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

# Option for updating the manifest
if "update_manifest" in sys.argv:
    # Update the MANIFEST.in file with all the data from within solcore
    include = "rayflare"
    exclude = ["__pycache__", "egg", "darwin", "cpython"]
    with open("MANIFEST.in", "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk("."):
            if not any(sub in root for sub in exclude) and root[2:9] == include:
                try:
                    files.remove(".DS_Store")
                except ValueError:
                    pass
                for file in files:
                    if any(sub in file for sub in exclude):
                        continue
                    include_line = "include " + os.path.join(root[2:], file) + "\n"
                    f.write(include_line)

    sys.exit()

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
    version="1.0.0",
    description="Python-based integrated optical modelling",
    long_description=long_description,
    url="https://github.com/qpv-research-group/rayflare",
    project_urls={
        "Documentation": "http://rayflare.readthedocs.io",
    },
    author="Phoebe Pearce",
    author_email="phoebe.pearce15@imperial.ac.uk",
    license="GNU LGPL",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
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
