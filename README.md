## Code Structure:

##### python:
   
contains reference implementation for non-batched projection (projection), tests (mostly central difference test of the gradients) and utility for testing 

##### cpp
containes tuned implementation for non-batched projection (MoreProjection.cpp/h) and batched-projection
 (BatchedProjection.cpp/h). The latter uses the former and is parallelized using openmp.
 
 Interface to python is provided using pybind11 (PyProject.cpp), conversion for numpy arrays to aramdillo vec/mat/ cube is provided
 in PyArmaConverter.h
    
Note that the C++ part uses column-major layout while the python part uses row-major

## Setup python
Tested with python 3.6.8 

Dependencies: numpy, nlopt

## Setup c++

##### Install required packages and libraries 
The following libraries are required: gcc, openmp, gfortran, openblas, lapack cmake. Install them using your package manager

(On Ubuntu: run

```sudo apt-get install gcc gfortran libopenblas-dev liblapack-dev cmake ```

) 
     
##### Install NLOPT (tested with version 2.6)
Follow the download and installation instruction here: https://nlopt.readthedocs.io/en/latest/
You do not need to install using sudo but can put the library anywhere.

Change the 'include_directories' and 'link_directories' statements in cpp/CMakeLists.txt such that they point to the
NLOPT headers and libraries. 

##### Install Armadillo (tested with version 9.8000)

Download Armadillo (http://arma.sourceforge.net/download.html) unpack and run ./configure . You do not need to build 
Armadillo

Change the 'include_directories' and statement in cpp/CMakeLists.txt such that it points to the
armadillo headers. 


##### Download pybind11 (tested with version 2.4)

Download from here https://github.com/pybind/pybind11/releases, unpack, rename to pybind11, and place the pybind11 
folder under cpp (you can put it somewhere else but then need to adapt the CMakeLists.txt such that pybind11 is found)

##### Install CppEIM package 
go to CppEIM and run 

```sudo python3 setup.py install``` or ``python3 setup.py install --user``
