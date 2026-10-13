target_path = 'cherab/generomak/machine/data/first_wall'

# source files
py_files = []
pyx_files = []
pxd_files = []
data_files = ['BottomBaffle.obj', 'BottomDivertorFloor.obj', 'BottomInnerVerticalTarget.obj', 'BottomOuterVerticalTarget.obj', 'InnerDivertorBaffle.obj', 'InnerWallLimiter.obj', 'OuterWallLimiter.obj', 'TopBaffle.obj', 'TopDivertorFloor.obj', 'TopInnerVerticalTarget.obj', 'TopOuterVerticalTarget.obj']

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
