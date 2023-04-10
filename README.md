[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1206141.svg)](https://doi.org/10.5281/zenodo.1206141)
[![Build Status](https://github.com/cherab/core/actions/workflows/ci.yml/badge.svg)](https://github.com/cherab/core/actions/workflows/ci.yml/badge.svg)

Cherab
======

Welcome to the Cherab project.

Please see our [documentation](https://cherab.github.io/documentation/index.html)
for guidance on using the code.

Installation
------------

Cherab is a large code framework consisting of a core package and feature
packages. Users will generally install the core package and the specific
feature packages they need for their work. For example, users working on the
JET tokamak will require the `cherab-core` package, and the `cherab-jet`
package.

Unless developing new code for a cherab package, most users should clone the
master branch. When developing new features for cherab, the development branch
should be used as the base.

All cherab packages are standard python packages and basic installation is
achieved with:

```
pip install cherab
```

This will compile the Cherab cython extensions and install the package. If you
don't have administrator access to install the package, add the `--user` flag
to the above line to install the package under your own user account.
Alternatively, consider creating a [virtual environment](https://docs.python.org/3/tutorial/venv.html)
and installing `cherab` in the environment.

When developing cherab it is usually preferred that the packages be installed
in "editable" mode. Clone this repository and change directory to the root of
the repository, then run:

```
pip install -e .
```

This will cause the original installation folder to be added to the site-package
path. Modifications to the code will therefore be visible to python next time
the code is imported. A virtual environment or the ``--user`` flag should be
used if you do not have administrative permission for your python installation.
If you make any changes to Cython files you will need to run `./dev/build.sh` to
rebuild the relevant files.

As all the Cherab packages are dependent on the ``cherab-core`` package, this
package must be installed first. Note that other packages may have their own
inter-dependencies, see the specific package documentation for more information.

Cherab is organised as a namespace package, where each of the submodules is
installed in the same location as the core package. Any submodules using Cython
with a build-time dependency on Cherab need to use a Cython version newer than
3.0a5, due to a [bug](https://github.com/cython/cython/issues/2918) in how
earlier versions of Cython handle namespaces.

By default, pip will install from wheel archives on PyPI. If a binary wheel is not
available for your version of Python, or if you are installing in editable mode
for development, the package will be compiled locally on your machine. Compilation
is done in parallel by default, using all available processors, but can be
overridden by setting the environment variable `CHERAB_NCPU` to the number of
processors to use.

Governance
----------

The management of the project is divided into Scientific and Technical Project
Management. The Scientific management happens through the normal community
routes such as JET and MST1 task force meetings, ITPA meetings, etc.

The Technical Management Committee (TMC) is a smaller subset of the community,
being responsible for ensuring the integrity and high code quality of Cherab is
maintained. These TMC members would have in-depth knowledge of the code base
through a demonstrated history of contributing to the project. The TMC would
primarily be responsible for accepting / rejecting merge requests on the basis
of code / physics algorithm quality standards.


TMC Members
-----------

- Alys Brett (chairwoman, master account holder, responsible for delegation, UKAEA, UK)
- Matt Carr (External consultant, diagnostic physics models)
- Jack Lovell (Oak Ridge, USA)
- Alex Meakins (External consultant, Architecture, software integrity)
- Vlad Neverov (NRC Kurchatov Institute, Moscow)
- Matej Tomes (Compass, IPP, Prague)


Citing The Code
---------------
* Dr Carine Giroud, Dr Alex Meakins, Dr Matthew Carr, Dr Alfonso Baciero, &
Mr Corentin Bertrand. (2018, March 23). Cherab Spectroscopy Modelling Framework
(Version v0.1.0). Zenodo. http://doi.org/10.5281/zenodo.1206142
