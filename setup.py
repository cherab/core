from setuptools import setup, find_packages, Extension
import sys
import numpy
import os
import os.path as path
import multiprocessing
from Cython.Build import cythonize

multiprocessing.set_start_method('fork')

force = False
profile = False
line_profile = False
install_rates = False

if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]

if "--line-profile" in sys.argv:
    line_profile = True
    del sys.argv[sys.argv.index("--line-profile")]

if "--install-rates" in sys.argv:
    install_rates = True
    del sys.argv[sys.argv.index("--install-rates")]

source_paths = ["cherab", "demos"]
compilation_includes = [".", numpy.get_include()]
compilation_args = ["-O3"]
cython_directives = {"language_level": 3}
setup_path = path.dirname(path.abspath(__file__))
num_processes = int(os.getenv("CHERAB_NCPU", "-1"))
if num_processes == -1:
    num_processes = multiprocessing.cpu_count()

if line_profile:
    compilation_args.append("-DCYTHON_TRACE=1")
    compilation_args.append("-DCYTHON_TRACE_NOGIL=1")
    cython_directives["linetrace"] = True
if profile:
    cython_directives["profile"] = True


extensions = []
for package in source_paths:
    for root, dirs, files in os.walk(path.join(setup_path, package)):
        for file in files:
            if path.splitext(file)[1] == ".pyx":
                pyx_file = path.relpath(path.join(root, file), setup_path)
                module = path.splitext(pyx_file)[0].replace("/", ".")
                extensions.append(
                    Extension(
                        module,
                        [pyx_file],
                        include_dirs=compilation_includes,
                        extra_compile_args=compilation_args,
                    ),
                )


# generate .c files from .pyx
extensions = cythonize(
    extensions,
    nthreads=multiprocessing.cpu_count(),
    force=force,
    compiler_directives=cython_directives,
)

# parse the package version number
with open(path.join(path.dirname(__file__), "cherab/core/VERSION")) as version_file:
    version = version_file.read().strip()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="cherab",
    version=version,
    license="EUPL 1.1",
    namespace_packages=["cherab"],
    description="Cherab spectroscopy framework",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    url="https://github.com/cherab",
    project_urls=dict(
        Tracker="https://github.com/cherab/core/issues",
        Documentation="https://cherab.github.io/documentation/",
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.14",
        "scipy",
        "matplotlib",
        "raysect==0.8.1",
    ],
    extras_require={
        # Running ./dev/build_docs.sh runs setup.py, which requires cython.
        "docs": ["cython", "sphinx", "sphinx-rtd-theme", "sphinx-tabs"],
    },
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    ext_modules=extensions,
    options=dict(
        build_ext={"parallel": num_processes},
    ),
)

# setup a rate repository with common rates
if install_rates:
    try:
        from cherab.atomic import repository

        repository.populate()
    except ImportError:
        pass
