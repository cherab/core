target_path = 'cherab/tools/tests'

# source files
py_files = ['__init__.py', 'test_admt.py', 'test_ionization_balance.py', 'test_observer_groups.py', 'test_plasma_slabs.py', 'test_raytransfer.py', 'test_sart_opencl.py', 'test_spectroscopic_instruments.py', 'test_voxels.py']
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

subdir('data')
