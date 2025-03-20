#!/bin/bash
set -e

# Use manylinux2014 if not already in container
if [ -z "$IN_CONTAINER" ]; then
    echo "Building in manylinux2014 container..."
    
    # Check if user can run docker without sudo
    if docker info >/dev/null 2>&1; then
        DOCKER_CMD="docker"
    else
        echo "Docker requires sudo access to run"
        DOCKER_CMD="sudo docker"
    fi
    
    $DOCKER_CMD run --rm -v $PWD:/src:Z -w /src quay.io/pypa/manylinux2014_x86_64 \
        bash -c "yum install -y wget make cmake devtoolset-10-gcc* devtoolset-10-gcc-gfortran && \
                 IN_CONTAINER=1 ./build.sh && \
                 chown -R $(id -u):$(id -g) cpp/dist/"
    
    echo "[i] Testing wheels outside container"
    ./test_wheels.sh

    exit $?
fi

cd cpp

# List of Python versions to build for
PYTHON_VERSIONS=("3.6" "3.7" "3.8" "3.9" "3.10" "3.11" "3.12" "3.13")

# Dependency versions
OPENBLAS_VERSION="0.3.21"   # Stable version with good performance
LAPACK_VERSION="3.11.0"     # Latest stable release
NLOPT_VERSION="2.7.1"       # Version specified in original build
ARMADILLO_VERSION="11.2.1"  # Compatible version (original build references 9.8000)
PYBIND11_VERSION="2.11.1"   # Latest stable with Python 3.10+ support

# Create build directory
mkdir -p build
cd build

# Enable devtoolset and set compiler paths
source /opt/rh/devtoolset-10/enable
ln -sf /opt/rh/devtoolset-10/root/usr/bin/gcc /usr/bin/gcc
ln -sf /opt/rh/devtoolset-10/root/usr/bin/g++ /usr/bin/g++
ln -sf /opt/rh/devtoolset-10/root/usr/bin/gfortran /usr/bin/gfortran

# Get absolute path to build directory
BUILD_DIR=$PWD

# Download pybind11 (header-only, no build needed)
if [ ! -d "pybind11" ]; then
    wget https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz
    tar xzf v${PYBIND11_VERSION}.tar.gz
    mv pybind11-${PYBIND11_VERSION} pybind11
    rm v${PYBIND11_VERSION}.tar.gz
fi
PYBIND11_DIR=$PWD/pybind11

# Build OpenBLAS
if [ ! -d "OpenBLAS" ]; then
    wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v${OPENBLAS_VERSION}.tar.gz
    tar xzf v${OPENBLAS_VERSION}.tar.gz
    mv OpenBLAS-${OPENBLAS_VERSION} OpenBLAS
    rm v${OPENBLAS_VERSION}.tar.gz
fi
cd OpenBLAS
make DYNAMIC_ARCH=1 TARGET=NEHALEM USE_OPENMP=1 NO_SHARED=1 CFLAGS="$COMMON_C_FLAGS" \
    BINARY=64 NO_AFFINITY=1 USE_THREAD=1 NUM_THREADS=128
make PREFIX=$PWD/installed NO_SHARED=1 CFLAGS="$CFLAGS" LDFLAGS="$LDFLAGS" install
OPENBLAS_DIR=$PWD/installed
cd ..

# Build LAPACK
if [ ! -d "lapack" ]; then
    wget https://github.com/Reference-LAPACK/lapack/archive/v${LAPACK_VERSION}.tar.gz
    tar xzf v${LAPACK_VERSION}.tar.gz
    mv lapack-${LAPACK_VERSION} lapack
    rm v${LAPACK_VERSION}.tar.gz
fi
cd lapack
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$PWD/installed \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_C_FLAGS="$CFLAGS" \
      -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
      -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
      -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make
make install
LAPACK_DIR=$PWD/installed
cd ../..

# Download and build NLOPT
if [ ! -d "nlopt" ]; then
    wget https://github.com/stevengj/nlopt/archive/v${NLOPT_VERSION}.tar.gz
    tar xzf v${NLOPT_VERSION}.tar.gz
    mv nlopt-${NLOPT_VERSION} nlopt
    rm v${NLOPT_VERSION}.tar.gz
fi
cd nlopt
mkdir -p build && cd build
# DNLOPT_LUKSAN=OFF required to use MIT license instead of LGPL.
cmake -DCMAKE_INSTALL_PREFIX=$PWD/installed \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_C_FLAGS="$CFLAGS" \
      -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
      -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
      -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DNLOPT_PYTHON=OFF \
      -DNLOPT_OCTAVE=OFF \
      -DNLOPT_MATLAB=OFF \
      -DNLOPT_GUILE=OFF \
      -DNLOPT_LUKSAN=OFF \
      -DNLOPT_SWIG=OFF \
      ..
make
make install
NLOPT_DIR=$PWD/installed
cd ../..

# Download Armadillo (header-only, no build needed)
if [ ! -d "armadillo" ]; then
    wget https://sourceforge.net/projects/arma/files/armadillo-${ARMADILLO_VERSION}.tar.xz
    tar xf armadillo-${ARMADILLO_VERSION}.tar.xz
    mv armadillo-${ARMADILLO_VERSION} armadillo
    rm armadillo-${ARMADILLO_VERSION}.tar.xz
fi
ARMADILLO_DIR=$PWD/armadillo

echo "--------------------------------"
echo "ARMADILLO_DIR: $ARMADILLO_DIR"
echo "OPENBLAS_DIR: $OPENBLAS_DIR"
echo "LAPACK_DIR: $LAPACK_DIR"
echo "NLOPT_DIR: $NLOPT_DIR"
echo "PYBIND11_DIR: $PYBIND11_DIR"
echo "--------------------------------"

# Build for each Python version
for PYVER in "${PYTHON_VERSIONS[@]}"; do
    echo "Building for Python $PYVER"
    
    # Create Python-specific build directory
    PYBUILD="build-py$PYVER"
    mkdir -p $PYBUILD
    cd $PYBUILD
    
    # Convert 3.6 -> cp36-cp36m etc
    PYTAG="cp${PYVER/./}"
    # Compare versions: if PYVER is less than 3.8 (sorts before or with 3.7), add 'm' suffix
    if [ "$(echo -e "$PYVER\n3.7" | sort -V | head -n1)" = "$PYVER" ]; then
        PYTAG="${PYTAG}-${PYTAG}m"  # Older versions use 'm' suffix
    else
        PYTAG="${PYTAG}-${PYTAG}"
    fi
    PYTHON="/opt/python/${PYTAG}/bin/python"
    
    # Check if this Python version is available
    if [ -d "/opt/python/${PYTAG}" ]; then
        # Build the project
        cmake ../.. \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_SHARED_LIBS=OFF \
            -DOpenBLAS_HOME=$OPENBLAS_DIR \
            -DLAPACK_HOME=$LAPACK_DIR \
            -DNLOPT_HOME=$NLOPT_DIR \
            -DARMADILLO_INCLUDE_DIR=$ARMADILLO_DIR/include \
            -DPYBIND11_DIR=$PYBIND11_DIR \
            -DPYTHON_EXECUTABLE=$PYTHON \
            -DPYTHON_INCLUDE_DIR=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('include'))") \
            -DPYTHON_LIBRARY=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython${PYTAG%%-*}.so
        
        make -j4
        
        # Create wheel for this version
        cd ../..
        $PYTHON -m pip wheel . --no-deps -w dist/
        
        cd build
    fi
done

cd ../..
echo "All wheels built in cpp/dist/"
