from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import sys
import numpy
import os
import os.path as path
import multiprocessing

threads = multiprocessing.cpu_count()
force = False
profile = False

if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]

compilation_includes = [".", numpy.get_include()]
compilation_args = []
cython_directives = {
    'language_level': 3
}

setup_path = path.dirname(path.abspath(__file__))

# build .pyx extension list
extensions = []
for root, dirs, files in os.walk(setup_path):
    for file in files:
        if path.splitext(file)[1] == ".pyx":
            pyx_file = path.relpath(path.join(root, file), setup_path)
            module = path.splitext(pyx_file)[0].replace("/", ".")
            extensions.append(Extension(module, [pyx_file], include_dirs=compilation_includes, extra_compile_args=compilation_args),)

if profile:
    cython_directives["profile"] = True

# generate .c files from .pyx
extensions = cythonize(extensions, nthreads=multiprocessing.cpu_count(), force=force, compiler_directives=cython_directives)

# parse the package version number
with open(path.join(path.dirname(__file__), 'cherab/core/VERSION')) as version_file:
    version = version_file.read().strip()

setup(
    name="cherab",
    version=version,
    license="EUPL 1.1",
    namespace_packages=['cherab'],
    description='Cherab spectroscopy framework',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    install_requires=['numpy', 'scipy', 'matplotlib', 'raysect>=0.5.4', 'cython>=0.28'],
    packages=find_packages(),
    include_package_data=True,
    ext_modules=extensions
)
