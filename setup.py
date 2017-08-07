from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import sys
import numpy
import os
import os.path as path
import multiprocessing


def add_extensions(setup_path, folder, extensions):
    """
    Adds .pyx files in the specified folder to the list of extensions.
    """
    source_path = path.join(setup_path, folder)
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if path.splitext(file)[1] == ".pyx":
                pyx_file = path.relpath(path.join(root, file), setup_path)
                module = path.splitext(pyx_file)[0].replace("/", ".")
                extensions.append(Extension(module, [pyx_file], include_dirs=compilation_includes),)


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

setup_path = path.dirname(path.abspath(__file__))
source_folders = ['cherab', 'demos']

# build extension list
extensions = []
for folder in source_folders:
    add_extensions(setup_path, folder, extensions)

if profile:
    directives = {"profile": True}
else:
    directives = {}

setup(
    name="cherab",
    version="1.0.0",
    license="EUPL 1.1",
    namespace_packages=['cherab'],
    packages=find_packages(),
    include_package_data=True,
    ext_modules=cythonize(extensions, nthreads=threads, force=force, compiler_directives=directives)
)

