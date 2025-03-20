#!/bin/bash
set -e
cd cpp

echo "Deleting build and dist directories..."
rm -rf build
rm -rf dist
echo "Done!"

cd ..
