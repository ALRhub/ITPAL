# ITPAL

ITPAL provides efficient implementations for KL-divergence based projections of Gaussians. This package focuses specifically on the projection operations, optimized through C++ and parallelized using OpenMP and provides low level bindings for them. For JAX bindings see [ITPAL_JAX](https://github.com/ALRhub/ITPAL_JAX).

## Installation

### From Pre-built Binaries (Recommended)

We now provide pre-built binaries that should work on all systems if:
 - Hardware is 64-bit Intel or AMD and from 2009+
 - OS is Linux with glibc 2.17+ (from 2014+) following FHS (includes Ubuntu/Debian/Fedora/CentOS/...)
 - Python 3.6-3.12
 - gfortran 10+ is installed (e.g. `sudo apt-get install gfortran`, already installed on BwUni2 & HoReKa)
 - OpenMP is installed (optional, but increases performance of batched projections, already installed on BwUni2 & HoReKa)

Download the appropriate wheel for your Python version from the [releases](https://github.com/ALRhub/ITPAL/releases) page and install with pip:

```bash
pip install cpp_projection-{version}-{python_version_name}-manylinux2014_x86_64.whl
```

### Building Wheels from Source

This repository contains the build system for creating cpp_projection wheels with statically linked dependencies for the previously described range of systems.
Ensure you have Docker (used for building in manylinux2014 environment) and run the build script:

```bash
./build.sh
```

This will:
- Gather and Build all dependencies (OpenBLAS, LAPACK, NLOPT, ...) statically
- Create wheels for all supported Python versions
- Output wheels to the `cpp/dist/` directory
- Test all created wheels if a fitting local python version is found (tested outside container)

Then just install the generated wheels via
```bash
pip install cpp/dist/cpp_projection-{version}-{python_version_name}-manylinux2014_x86_64.whl
```

### Manual Direct Installation (Legacy Method using Conda)

If your system is not supported by our pre-built wheels and wheel-building process, you can resort to a manual direct installation using conda:

#### Setup python
Tested with python 3.6.8 

Dependencies: numpy, nlopt

#### Setup c++
##### 1. Install required packages and libraries 
Install required packages into your conda environment

```conda install --file conda_requirements.yml```

##### 2. Install package 
go to `ITPAL/cpp/`and run 

```bash
mv CMakeLists_conda.txt CMakeLists.txt
python3 setup.py install --user
```

## Python Implementations

The `python/` directory contains development tools and reference implementations that are not part of the main package:

- Reference implementation in vanilla Python (for testing and validation)
- Test suite with central difference gradient tests
- Plotting utilities for development and debugging
- Additional development tools and utilities

These tools are primarily for development and testing. If you wish to use cpp_projections directly in your own project (without e.g. ITPAL_JAX or Fancy_RL), you could also have a look at the python module for usage examples.

## License

This project, including all code, documentation, and provided binaries is licensed under the MIT License. See the [LICENSE file](LICENSE) for details.

### 3rd Party Licenses
The following 3rd party libraries are statically linked into cpp_projection binaries:
- OpenBLAS
- LAPACK
- NLOPT (without LGPL parts to remain MIT compatible)
- Armadillo

You can find the licenses of these libraries in the [LICENSE file](LICENSE).
