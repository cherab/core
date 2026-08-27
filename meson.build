project('cherab', 'cython',
    default_options: ['python.install_env=auto'],
    version: run_command('cat', files('cherab/core/VERSION'), check: true).stdout().strip()
)

py = import('python').find_installation(pure: false)
numpy = dependency('numpy', method: 'config-tool')
fs = import('fs')

# Match current setup.py compile settings
c_args = ['-O3', '-Wno-unreachable-code-fallthrough', '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION']
cython_args = ['--annotate', '-X legacy_implicit_noexcept=True']
cython_dependencies = [numpy]

subdir('cherab')
