target_path = 'cherab/core/atomic/data/lineshape/zeeman/parametrised'

# source files
py_files = []
pyx_files = []
pxd_files = []
data_files = ['b.json', 'be.json', 'c.json', 'd.json', 'h.json', 'he.json', 'he3.json', 'n.json', 'ne.json', 'o.json']

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
