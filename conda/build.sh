# export LD_LIBRARY_PATH=${PREFIX}/lib:$LD_LIBRARY_PATH
$PYTHON -m pip install . -vv

# cmake . -DPYTHON_LIBRARY_DIR="$SP_DIR" -DPYTHON_EXECUTABLE="$PYTHON" -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=ON ${CMAKE_ARGS}
# make
# make install
# cmake --build . --config Release
# cmake --install . --config Release
# cp libprojectionlib.so $PREFIX/lib
cp build/*/libprojectionlib.so $PREFIX/lib
# cp build/*/cpp_projection.cpython-310-x86_64-linux-gnu.so $PREFIX/lib
