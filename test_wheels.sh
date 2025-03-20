#!/bin/bash
set -e
cd cpp

# List of Python versions to test
PYTHON_VERSIONS=("3.6" "3.7" "3.8" "3.9" "3.10" "3.11" "3.12" "3.13")

# Create test directory
TEST_DIR="test_installations_native"
rm -rf $TEST_DIR
mkdir -p $TEST_DIR
cd $TEST_DIR

# Test each Python version
for PYVER in "${PYTHON_VERSIONS[@]}"; do
    PYTHON="python$PYVER"
    
    # Check if this Python version is available
    if command -v $PYTHON &> /dev/null; then
        # Check if wheel exists for this version
        WHEEL_PATTERN="../dist/cpp_projection-*-cp${PYVER/./}*-*.whl"
        if ! ls $WHEEL_PATTERN 1> /dev/null 2>&1; then
            echo "No wheel found for Python $PYVER, skipping test..."
            echo "----------------------------------------"
            continue
        fi
        
        echo "Testing for Python $PYVER"
        
        # Create and activate virtual environment
        VENV="venv-py$PYVER"
        $PYTHON -m venv $VENV
        source $VENV/bin/activate
        
        # Install the wheel
        pip install $WHEEL_PATTERN

        # Try importing
        python -c "
import cpp_projection
print('Successfully imported cpp_projection')
"
        
        # Deactivate venv
        deactivate
        
        echo "Test successful for Python $PYVER"
        echo "----------------------------------------"
    else
        echo "Python $PYVER not found, skipping test..."
    fi
done

cd ..
rm -rf $TEST_DIR
cd ..
