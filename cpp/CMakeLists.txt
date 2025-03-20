cmake_minimum_required(VERSION 3.10...3.28)

project(cpp_projection LANGUAGES C CXX Fortran)

# Force static linking
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Get paths from build script
set(OpenBLAS_HOME "" CACHE PATH "Path to OpenBLAS installation")
set(LAPACK_HOME "" CACHE PATH "Path to LAPACK installation")
set(NLOPT_HOME "" CACHE PATH "Path to NLOPT installation")
set(ARMADILLO_INCLUDE_DIR "" CACHE PATH "Path to Armadillo headers")
set(PYBIND11_DIR "" CACHE PATH "Path to pybind11")

# Find static libraries
set(CMAKE_FIND_LIBRARY_SUFFIXES .a)  # Force static for OpenBLAS, LAPACK, NLOPT

find_library(OPENBLAS_LIB 
    NAMES libopenblas.a
    PATHS ${OpenBLAS_HOME}/lib
    REQUIRED
)

find_library(LAPACK_LIB 
    NAMES liblapack.a
    PATHS ${LAPACK_HOME}/lib
    REQUIRED
)

find_library(NLOPT_LIB 
    NAMES libnlopt.a
    PATHS ${NLOPT_HOME}/lib
    REQUIRED
)

# Set OpenMP flags manually for devtoolset
set(OpenMP_C_FLAGS "-fopenmp")
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_Fortran_FLAGS "-fopenmp")
set(OpenMP_C_LIB_NAMES "gomp;pthread")
set(OpenMP_CXX_LIB_NAMES "gomp;pthread")
set(OpenMP_Fortran_LIB_NAMES "gomp;pthread")
set(OpenMP_gomp_LIBRARY "/opt/rh/devtoolset-10/root/usr/lib/gcc/x86_64-redhat-linux/10/libgomp.so")
set(OpenMP_pthread_LIBRARY "/usr/lib64/libpthread.so")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building with OpenMP support")
    add_definitions(-DUSE_OPENMP)
else()
    message(FATAL_ERROR "OpenMP is required but not found")
endif()

# Include directories
include_directories(
    ${OpenBLAS_HOME}/include
    ${LAPACK_HOME}/include
    ${NLOPT_HOME}/include
    ${ARMADILLO_INCLUDE_DIR}
    include
)

# Add pybind11
add_subdirectory(${PYBIND11_DIR} build/pybind11)
include_directories(${PYBIND11_DIR}/include)

# Build projection library
add_library(projectionlib STATIC
    src/projection/MoreProjection.cpp
    src/projection/BatchedProjection.cpp
    src/projection/DiagCovOnlyMoreProjection.cpp
    src/projection/BatchedDiagCovOnlyProjection.cpp
    src/projection/SplitDiagMoreProjection.cpp
    src/projection/BatchedSplitDiagMoreProjection.cpp
    src/projection/CovOnlyMoreProjection.cpp
    src/projection/BatchedCovOnlyProjection.cpp
)

# Force using shared gfortran
execute_process(
    COMMAND gfortran -print-file-name=libgfortran.so
    OUTPUT_VARIABLE GFORTRAN_LIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT EXISTS "${GFORTRAN_LIB}")
    message(FATAL_ERROR "Could not find libgfortran.so - please install gfortran")
endif()

target_link_libraries(projectionlib PRIVATE
    ${LAPACK_LIB}
    ${NLOPT_LIB}
    ${OPENBLAS_LIB}
    ${OpenMP_gomp_LIBRARY}
    ${OpenMP_pthread_LIBRARY}
)

target_compile_options(projectionlib PRIVATE ${OpenMP_CXX_FLAGS})

# Python module
pybind11_add_module(cpp_projection src/projection/PyProjection.cpp)
target_compile_options(cpp_projection PRIVATE 
    -D__GLIBC_PREREQ=0x20011
    -D_FORTIFY_SOURCE=0
)
target_link_libraries(cpp_projection PRIVATE 
    projectionlib
    ${LAPACK_LIB}
    ${OPENBLAS_LIB}
    ${GFORTRAN_LIB}
    ${OpenMP_gomp_LIBRARY}
    ${OpenMP_pthread_LIBRARY}
)

target_compile_options(cpp_projection PRIVATE ${OpenMP_CXX_FLAGS})

# Force older GLIBC symbol versions during linking
target_link_options(cpp_projection PRIVATE
    "-Wl,--version-script=${CMAKE_SOURCE_DIR}/version.map"
)
