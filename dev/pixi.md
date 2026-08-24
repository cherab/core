# 🧰 Pixi developer guide

This document describes the development environments and tasks configured in
[`pixi.toml`](../pixi.toml). Pixi manages the development dependencies in
isolated environments and provides a common interface for running tests,
building the documentation, and checking the source tree.

Pixi installs or updates the selected environment automatically when a command
is run. The workspace currently supports Linux x86-64, macOS x86-64, and macOS
Arm64.

## 🔎 Discovering tasks

List every task in the workspace with:

```console
pixi task list
```

To show only the tasks available in a particular environment, pass its name:

```console
pixi task list -e test
```

Use `pixi run --help` for general command help. When a task is available in
more than one environment, use `-e <environment>` to select the environment
explicitly.

## 🧩 Environments

| Environment | Purpose |
| --- | --- |
| `default` | Basic development tools; Cherab is not installed |
| `test` | Run the complete test suite using the latest supported Python; currently equivalent to `test-pylatest` |
| `test-pylatest` | Run the complete test suite using the latest supported Python |
| `test-pyoldest` | Run the complete test suite using the oldest supported Python |
| `test-opencl` | Run the OpenCL SART tests with Cherab's `opencl` extra |
| `docs` | Build the documentation |
| `lint` | Run formatting and static-analysis tools without installing Cherab |

See the [`pyoldest` and `pylatest` features and environment definitions in
`pixi.toml`](../pixi.toml#L145-L159) for the Python versions used by each
environment.

## 🛠️ Basic development tasks

Start an IPython session in the default environment:

```console
pixi run ipython
```

Remove generated C/Cython libraries and HTML files from the `cherab/` source
tree:

```console
pixi run clean
```

The `clean` task deletes files matching `*.c`, `*.so`, `*.pyd`, `*.dll`, and
`*.html` below `cherab/`.

## 🧪 Testing

Run the complete test suite with the latest supported Python:

```console
pixi run -e test test
```

The `test` environment currently uses the same solve group as
`test-pylatest`. The explicit alias can also be used:

```console
pixi run -e test-pylatest test
```

Run the suite with the oldest supported Python:

```console
pixi run -e test-pyoldest test
```

Run the OpenCL SART tests:

```console
pixi run -e test-opencl test-opencl
```

The regular test task runs `python -m unittest discover cherab -v`. The OpenCL
task runs `cherab.tools.tests.test_sart_opencl` only.

## 📚 Documentation

Build the HTML documentation:

```console
pixi run -e docs doc-build
```

`html` is the default Sphinx builder. A different builder can be supplied as
the final argument; for example, check external and internal links with:

```console
pixi run -e docs doc-build linkcheck
```

Build output is written below `docs/build/<builder>`. Remove all documentation
build output with:

```console
pixi run doc-clean
```

After building the HTML documentation, serve it locally on port 8000 with:

```console
pixi run doc-serve
```

Then open <http://localhost:8000> in a browser. To use a different port, pass
it as the final argument:

```console
pixi run doc-serve 8080
```

## 🧹 Formatting and static analysis

The `lint` environment keeps code-quality tools separate from the environments
that build and install Cherab. Because the task names below are unique to this
environment, Pixi selects it automatically; `-e lint` is not required.

| Task | Action |
| --- | --- |
| `lefthook` | Run Lefthook |
| `hooks` | Install the Git hooks managed by Lefthook |
| `pre-commit` | Run the Lefthook `pre-commit` group |
| `ruff-check` | Run `ruff check` |
| `ruff-format` | Run `ruff format` |
| `toml-format` | Run `tombi format` |
| `dprint` | Run `dprint fmt` |
| `typos` | Find and fix spelling errors |
| `actionlint` | Run Actionlint |
| `blacken-docs` | Format Python examples in documentation |
| `validate-pyproject` | Validate `pyproject.toml` |
| `cython-lint` | Run Cython-Lint |
| `lint` | Run the Lefthook `pre-commit` group on all files |

For example:

```console
pixi run ruff-check
pixi run toml-format
pixi run cython-lint
pixi run validate-pyproject
```

The `hooks`, `lefthook`, `pre-commit`, and aggregate `lint` tasks invoke
Lefthook.

> [!WARNING]
> A Lefthook configuration file has not been added to the repository yet, so
> these tasks are not currently available.

Install the Git hooks and run all configured checks with:

```console
pixi run hooks
pixi run lint
```

Running `pixi run hooks` installs the Git hooks once. After installation, the
configured pre-commit checks are triggered automatically for every commit. To
remove the installed hooks and stop the automatic checks, run:

```console
pixi run lefthook uninstall
```
