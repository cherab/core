target_path = 'cherab/core/tests'

# source files
py_files = ['__init__.py', 'test_beam.py', 'test_beamcxline.py', 'test_bremsstrahlung.py', 'test_line_emission.py', 'test_lineshapes.py', 'test_maxwellian.py', 'test_species.py', 'test_total_radiated_power.py']
pyx_files = []
pxd_files = []
data_files = []

# compile cython
foreach pyx_file: pyx_files
    py.extension_module(
        fs.replace_suffix(pyx_file, ''),
        pyx_file,
        dependencies: cython_dependencies,
        install: true,
        subdir: target_path,
        cython_args: cython_args,
        c_args: c_args,
    )
endforeach

# add python, pxd and data files to the build
py.install_sources(
    py_files + pxd_files + data_files,
    subdir: target_path
)
