import setuptools
from distutils.core import setup
import platform

from Cython.Build import cythonize
# from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except:
    from distutils.command import build_ext

# try:
#     from setuptools import setup
#     # from setuptools import Extension
# except ImportError:
#     from distutils.core import setup
#     # from distutils.extension import Extension

import numpy as np


__version__ = '0.2.8dev'

mappy_name = 'MpGlue'
maintainer = 'Jordan Graesser'
maintainer_email = 'graesser@bu.edu'
description = 'The Glue of MapPy'
git_url = 'http://github.com/jgrss/mpglue.git'

with open('README.md') as f:
    long_description = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

with open('AUTHORS.txt') as f:
    author_file = f.read()

required_packages = []


def get_pyx_list():

    return ['mpglue/classification/*.pyx',
            'mpglue/stats/*.pyx']


def get_packages():
    return setuptools.find_packages()


def get_package_data():

    return {'': ['*.md', '*.txt'],
            'mpglue': ['classification/*.pyx',
                       'data/*.tif',
                       'data/*.vrt',
                       'stats/*.pyx']}


def setup_package():

    include_dirs = [np.get_include()]

    metadata = dict(name=mappy_name,
                    maintainer=maintainer,
                    maintainer_email=maintainer_email,
                    description=description,
                    license=license_file,
                    version=__version__,
                    long_description=long_description,
                    author=author_file,
                    packages=get_packages(),
                    package_data=get_package_data(),
                    ext_modules=cythonize(get_pyx_list()),
                    cmdclass=dict(build_ext=build_ext),
                    download_url=git_url,
                    install_requires=required_packages,
                    include_dirs=include_dirs)

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
