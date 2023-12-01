Building wheels for publishing on PyPI
======================================

Linux wheels published on PyPI must be for one of the manylinux variants.
This means they should be built inside a manylinux Docker container.

For Cherab versions 1.4 and earlier (which depend on raysect 0.7.1 or earlier),
the procedure varies slightly depending on whether there is a raysect wheel
available for the Python version or not.

When Raysect wheels are available
---------------------------------

1. Retrieve the appropriate manylinux container (we'll use manylinux1 here as
   an example, but any appropriate version works the same way).
   ```bash
   sudo docker run -ti -v <path-to-cherab>:/cherab_core quay.io/pypa/manylinux1_x86_64 /bin/bash
   ```
   Or, if using singularity instead of Docker (where sudo rights are not available):
   ```bash
   singularity pull docker://quay.io/pypa/manylinux1_x86_64
   singularity run -B <path-to-cherab>:/cherab_core -W /tmp -c ./manylinux1_x86_64_latest.sif
   ```
2. Inside the container, change to the top level directory of this repository.
   ```bash
   cd /cherab_core
   ```
3. Configure pip to prefer older binary wheels over newer sdists. This is because
   the manylinux images have very few libraries installed so fail to build many
   packages.
   ```bash
   export PIP_PREFER_BINARY=1
   ```
4. (Optional) Configure a cache directory for pip that doesn't have a small quota.
   This will prevent `No space left on device` errors when collecting dependencies,
   or when pip does have to build certain packages (Numpy for example does successfully
   build on certain manylinux versions for certain Python versions).
   ```bash
   export PIP_CACHE_DIR=/tmp/pipcache
   ```
5. Use PyPA's `build` package to build the sdists and wheels, for each Python version
   we're building the wheels for. For example, for Python 3.8:
   ```bash
   /opt/python/cp38-cp38/bin/python -m build .
   ```
   Ensure you've done step 3 above before running this command!
6. Once the build has finished, repair the wheel to give it the correct manylinux tag.
   Once again, using Python 3.8 as an example:
   ```bash
   auditwheel repair ./dist/cherab-1.4.0rc1-cp38-cp38-linux_x86_64.whl
   ```
   This will produce a wheel in the `./wheelhouse` directory which can be uploaded to PyPI.

When Raysect wheels aren't available
------------------------------------

For Raysect 0.8 and later, the same procedure applies as above.
For Raysect 0.7.1 and below, a couple of additional steps are required.

1. Follow steps 1-4 above to set up the container environment.
2. Create a virtual environment for building the wheel. For example, for Python 3.10:
   ```bash
   /opt/python/cp310-cp310/bin/python -m venv /tmp/cp310
   . /tmp/cp310/bin/activate
   ```
3. Install build, wheel Cython and the oldest supported Numpy into this environment.
   ```bash
   pip install build wheel cython oldest-supported-numpy
   ```
4. Install the required Raysect version.
   ```bash
   pip install raysect==0.7.1
   ```
5. Use PyPA's `build` to build the wheel, but tell it to use this virtual
   environment rather than creating a new isolated one.
   ```bash
   python -m build -n .
   ```
6. Run the auditwheel command as given in step 6 above to produce wheels with
   the correct tag for uploading to PyPI.
