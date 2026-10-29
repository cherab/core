target_path = 'cherab/core'

# source files
py_files = ['__init__.py']
pyx_files = ['distribution.pyx', 'species.pyx']
pxd_files = ['__init__.pxd', 'distribution.pxd', 'species.pxd']
data_files = ['VERSION']

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

subdir('atomic')
subdir('beam')
subdir('laser')
subdir('math')
subdir('model')
subdir('plasma')
subdir('rebeam')
subdir('tests')
subdir('utility')
