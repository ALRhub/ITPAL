import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

# Minimum supported Python version
MIN_PYTHON = (3, 6)

conda = False
if '--conda' in sys.argv:
    conda = True
    sys.argv.remove("--conda")

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # Check Python version
        if sys.version_info < MIN_PYTHON:
            raise RuntimeError(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or later required")

        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                             ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if conda:
            return self.build_extension_conda(ext)
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            '-DBUILD_SHARED_LIBS=OFF',
            f'-DOpenBLAS_HOME={os.path.join(ext.sourcedir, "build/OpenBLAS/installed")}',
            f'-DLAPACK_HOME={os.path.join(ext.sourcedir, "build/lapack/build/installed")}',
            f'-DNLOPT_HOME={os.path.join(ext.sourcedir, "build/nlopt/build/installed")}',
            f'-DARMADILLO_INCLUDE_DIR={os.path.join(ext.sourcedir, "build/armadillo/include")}',
            f'-DPYBIND11_DIR={os.path.join(ext.sourcedir, "build/pybind11")}',
        ]

        # Use older but widely compatible CPU features
        cmake_args.append('-DCMAKE_CXX_FLAGS=-march=x86-64')

        # Get the Python library path
        python_path = sys.executable
        cmake_args.append(f'-DPYTHON_EXECUTABLE={python_path}')

        build_args = ['--config', 'Release']

        if platform.system() == "Windows":
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

    def build_extension_conda(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if conda:
            cmake_args = [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                '-DPYTHON_EXECUTABLE=' + os.environ['PYTHON'],
                '-DCMAKE_PREFIX_PATH=' + os.environ['PREFIX'],
                '-DBUILD_SHARED_LIBS=ON'
            ]
        else:
            cmake_args = [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                '-DPYTHON_EXECUTABLE=' + sys.executable,
                '-DCMAKE_PREFIX_PATH=' + os.environ['CONDA_PREFIX']
                ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            if conda:
                cmake_args.append(os.environ['CMAKE_ARGS'])
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

if not conda:
    # Add platform tags
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
        class bdist_wheel(_bdist_wheel):
            def finalize_options(self):
                _bdist_wheel.finalize_options(self)
                # Mark us as not tied to Python ABI
                self.root_is_pure = False
                self.plat_name_supplied = True
                self.plat_name = "manylinux2014_x86_64"
    except ImportError:
        bdist_wheel = None

setup(
    name='cpp_projection',
    version='1.0.0',
    author='Philipp Becker',
    author_email='philipp.becker@kit.edu',
    description='ITPAL (cpp_projection) provides efficient implementations for KL-divergence based projections used in the ITPAL framework.',
    long_description='ITPAL (cpp_projection) provides efficient implementations for KL-divergence based projections used in the ITPAL (Information Theoretic Projection As Layer) framework, optimized through C++ and parallelized using OpenMP. Written by Philipp Becker, installation made painless by Dominik Roth.',
    ext_modules=[CMakeExtension('cpp_projection')],
    cmdclass={
        'build_ext': CMakeBuild,
        'bdist_wheel': bdist_wheel,
    },
    zip_safe=False,
    python_requires=f'>={MIN_PYTHON[0]}.{MIN_PYTHON[1]}',
)
