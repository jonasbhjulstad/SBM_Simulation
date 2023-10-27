import setuptools
import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from glob import glob


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


# class CMakeBuild(build_ext):
#     def run(self):
#         try:
#             out = subprocess.check_output(['cmake', '--version'])
#         except OSError:
#             raise RuntimeError(
#                 "CMake must be installed to build the following extensions: " +
#                 ", ".join(e.name for e in self.extensions))

#         if platform.system() == "Windows":
#             cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
#                                          out.decode()).group(1))
#             if cmake_version < '3.1.0':
#                 raise RuntimeError("CMake >= 3.1.0 is required on Windows")

#         for ext in self.extensions:
#             self.build_extension(ext)

#     def build_extension(self, ext):

#         cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/home/deb/Documents/SBM_Graph/build/test/_deps/static_rng-build]

#         cfg = 'Debug' if self.debug else 'Release'
#         build_args = ['--config', cfg]

#         if platform.system() == "Windows":
#             if sys.maxsize > 2**32:
#                 cmake_args += ['-A', 'x64']
#             build_args += ['--', '/m']
#         else:
#             cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
#             build_args += ['--', '-j2']

#         env = os.environ.copy()
#         env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
#             env.get('CXXFLAGS', ''),
#             self.distribution.get_version())
#         if not os.path.exists(self.build_temp):
#             os.makedirs(self.build_temp)
#         subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
#                               cwd=self.build_temp, env=env)
#         subprocess.check_call(['cmake', '--build', '.'] + build_args,
#                               cwd=self.build_temp)

# with open("README.md", "r") as f:
#     long_description = f.read()
ext_modules = [Pybind11Extension("JonasHj.Static_RNG._Static_RNG",
sources=sorted([glob("/home/deb/.CPM/static_rng/74696fc3f2eef1c48ce8288f6baebe0135654f45/binders/*.cpp")]),
include_dirs=["/home/deb/.CPM/static_rng/74696fc3f2eef1c48ce8288f6baebe0135654f45/include"])
]


setup(
    name="Static_RNG",
    version="0.1.0",
    author="Jonas Hjulstad",
    author_email="jonas-apple@hotmail.com",
    description="Statically allocated RNG for C++/Sycl",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages("/home/deb/.CPM/static_rng/74696fc3f2eef1c48ce8288f6baebe0135654f45/binders"),
    # package_dir={"":"../binders"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux"
    ],
    package_data={'Static_RNG':['/home/deb/Documents/SBM_Graph/build/test/_deps/static_rng-build/Binder.so']},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    # python_requires='>=3.6',
    # cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False
    # install_requires=['']
)
