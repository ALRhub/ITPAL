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
##### 1. Install required packages and libraries 
Install required packages into your conda environment

```conda install --file requirements.yml```

##### 2. Install package 
go to `ITPAL/cpp/` and run 

```python3 setup.py install --user```
