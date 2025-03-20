## Code Structure

The implementation is in C++ with Python bindings via pybind11:

Core Projections:
- `src/projection/MoreProjection.{cpp,h}`: Full KL projection with entropy constraints
- `src/projection/CovOnlyMoreProjection.{cpp,h}`: KL projection for covariance only
- `src/projection/DiagCovOnlyMoreProjection.{cpp,h}`: KL projection for diagonal covariance
- `src/projection/SplitDiagMoreProjection.{cpp,h}`: Split KL projection for mean and diagonal covariance

Batched Implementations:
- `src/projection/BatchedProjection.{cpp,h}`: OpenMP-parallelized batched full projection
- `src/projection/BatchedCovOnlyProjection.{cpp,h}`: Batched covariance-only projection
- `src/projection/BatchedDiagCovOnlyProjection.{cpp,h}`: Batched diagonal covariance projection
- `src/projection/BatchedSplitDiagMoreProjection.{cpp,h}`: Batched split diagonal projection

Python Interface:
- `src/projection/PyProjection.cpp`: Python bindings using pybind11
- `include/PyArmaConverter.h`: Numpy array conversion utilities

Note: The C++ implementation uses column-major layout while Python/NumPy uses row-major. The conversion is handled automatically in the bindings.

## Build System Details

Making this package easily installable on all systems presents some challenges:

- Traditional approach required users to compile all dependencies locally
- We want to bundle dependencies into our binary for portability
- Python extensions must be shared objects (dynamic libraries)
- Requires special compilation to embed static libraries into shared objects
- Core system libraries must remain dynamic (GLIBC, libgfortran, libgomp)
- Need to support wide range of Linux systems

### Implementation

#### Build Environment
- Uses manylinux2014 Docker container for building
- Ensures consistent build environment with older GLIBC (2.17)
- Provides access to all supported Python versions (3.6-3.12)

#### Static Linking Strategy
- Download and build all dependencies (OpenBLAS, LAPACK, NLOPT) from source
- Compile with PIC (Position Independent Code) to allow static linking into shared library
- Create static .a libraries for all dependencies
- Link these statically into our final Python module (dynamic shared object)

#### Dynamic Linking Strategy
We only dynamically link against essential system libraries:
- Universally available (GLIBC)
- Required for runtime features (libgfortran)

### 3rd Party Licenses
The following 3rd party libraries are statically linked into cpp_projection:
- OpenBLAS
- LAPACK
- NLOPT (without LGPL parts)
- Armadillo
