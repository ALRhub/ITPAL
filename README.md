# ITPAL

ITPAL provides efficient implementations for KL-divergence based projections of Gaussians. This package focuses specifically on the projection operations, optimized through C++ and parallelized using OpenMP and provides low level bindings for them. For JAX bindings see [ITPAL_JAX](https://github.com/ALRhub/ITPAL_JAX).

# Installation

## From Pre-built Binaries (Recommended)

We now provide pre-built binaries that should work on all systems if:
 - Hardware is 64-bit Intel or AMD and from 2009+
 - OS is Linux with glibc 2.17+ (from 2014+) following FHS (includes Ubuntu/Debian/Fedora/CentOS/...)
 - Python 3.6-3.12
 - gfortran 10+ is installed (e.g. `sudo apt-get install gfortran`, already installed on BwUni2 & HoReKa)
 - OpenMP is installed (optional, but increases performance of batched projections, already installed on BwUni2 & HoReKa)

Download the appropriate wheel for your Python version from the [releases](https://github.com/ALRhub/ITPAL/releases) page and install with pip:

```bash
pip install cpp_projection-{version}-cp{python_version}-manylinux2014_x86_64.whl
```

## Manual Installation (Legacy Method)

If you prefer to build from source or the pre-built binaries don't work for your system, you can follow the manual installation instructions below.

### Code Structure:

##### python:
   
contains reference implementation for non-batched projection (projection), tests (mostly central difference test of the gradients) and utility for testing 

##### cpp
containes tuned implementation for non-batched projection (MoreProjection.cpp/h) and batched-projection
 (BatchedProjection.cpp/h). The latter uses the former and is parallelized using openmp.
 
 Interface to python is provided using pybind11 (PyProject.cpp), conversion for numpy arrays to aramdillo vec/mat/ cube is provided
 in PyArmaConverter.h

Note that the C++ part uses column-major layout while the python part uses row-major

### Setup python
Tested with python 3.6.8 

Dependencies: numpy, nlopt

### Setup c++
##### 1. Install required packages and libraries 
Install required packages into your conda environment

```conda install --file requirements.yml```

##### 2. Install package 
go to `ITPAL/cpp/` and run 

```python3 setup.py install --user```

## License

This project, including all code, documentation, and provided binaries is licensed under the MIT License. See the [LICENSE file](LICENSE) for details.

### 3rd Party Licenses
The following 3rd party libraries are statically linked into cpp_projection binaries:
- OpenBLAS
- LAPACK
- NLOPT (without LGPL parts to remain MIT compatible)
- Armadillo

You can find the licenses of these libraries in the [LICENSE file](LICENSE).
