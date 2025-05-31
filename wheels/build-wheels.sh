#!/bin/bash

# Build all wheels using the main cog container for compatibility
set -e

echo "Building all wheels for Hunyuan3D-2..."

# Get the parent directory (source code)
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Source directory: $PARENT_DIR"
echo "Wheels output directory: $WHEELS_DIR"

# Build the main cog image first
echo "=== Building main cog image ==="
cd "$WHEELS_DIR"
cog build -t hunyuan3d-wheels

# Clean existing wheels
cd "$WHEELS_DIR"
rm -f *.whl

# Create a temporary container, copy source, build wheels, then copy wheels back
echo "=== Building all wheels in isolated container ==="
CONTAINER_ID=$(docker create --platform linux/amd64 \
  -e "TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0;9.0+PTX" \
  -e FORCE_CUDA=1 \
  hunyuan3d-wheels \
  bash -c "
    # Clean and setup source directory
    rm -rf /src/* /src/.*[!.]* 2>/dev/null || true
    cp -r /tmp/source/. /src/
    
    # Activate virtual environment
    source /root/.venv/bin/activate
    
    echo '=== Environment Info ==='
    python --version
    pip --version
    which python pip
    echo 'Virtual env: $VIRTUAL_ENV'
    
    echo 'Building mesh_processor wheel...'
    cd /src/hy3dgen/texgen/differentiable_renderer
    python -m pip wheel . --wheel-dir /tmp/wheels --no-deps --no-build-isolation --verbose
    echo '✅ mesh_processor wheel built'
    
    echo 'Building custom_rasterizer wheel...'
    cd /src/hy3dgen/texgen/custom_rasterizer
    python -m pip wheel . --wheel-dir /tmp/wheels --no-deps --no-build-isolation --verbose
    echo '✅ custom_rasterizer wheel built'
    
    echo 'Building diso wheel...'
    cd /src
    python -m pip wheel diso --wheel-dir /tmp/wheels --no-deps --no-build-isolation --verbose
    echo '✅ diso wheel built'
    echo 'All wheels built successfully!'
  ")

# Copy source code into the container
echo "Copying source code into container..."
docker cp "$PARENT_DIR/." "$CONTAINER_ID:/tmp/source/"

# Start the container and run the build
echo "Running build process..."
docker start -a "$CONTAINER_ID"

# Copy wheels back to host
echo "Copying wheels back to host..."
docker cp "$CONTAINER_ID:/tmp/wheels/" "$WHEELS_DIR/"
mv "$WHEELS_DIR/wheels/"*.whl "$WHEELS_DIR/" 2>/dev/null || true
rmdir "$WHEELS_DIR/wheels" 2>/dev/null || true

# Clean up container
docker rm "$CONTAINER_ID"

echo "=== All wheels built! ==="
echo "Wheels directory contents:"
ls -la *.whl

echo "Done! All wheels are in the $WHEELS_DIR directory" 