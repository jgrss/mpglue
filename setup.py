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


__version__ = '0.1.4'

mappy_name = 'MpGlue'
maintainer = 'Jordan Graesser'
maintainer_email = 'graesser@bu.edu'
description = 'The Glue of MapPy'
git_url = 'http://github.com/jgrss/glue.git'

with open('README.md') as f:
    long_description = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

with open('AUTHORS.txt') as f:
    author_file = f.read()

required_packages = ['matplotlib', 'joblib', 'BeautifulSoup4']

if platform.system() != 'Windows':

    for pkg in ['numpy>=1.12.0',
                'scipy>=0.17.0',
                'scikit-image>=0.10.0',
                'gdal>=2.1',
                'tables>=3.3',
                'statsmodels>=0.8.0rc1',
                'cython>=0.25.2',
                'scikit-learn>=0.18.1',
                'pandas>=0.19.2']:

        required_packages.append(pkg)


def get_pyx_list():

    return ['mpglue/classification/*.pyx',
            'mpglue/stats/*.pyx']


def get_packages():
    return setuptools.find_packages()


def get_package_data():

    if platform.system() == 'Windows':

        return {'': ['*.md', '*.txt'],
                'mpglue': ['classification/*.pyx',
                           'classification/*.pyd',
                           'stats/*.pyx',
                           'stats/*.pyd']}

    else:

        return {'': ['*.md', '*.txt'],
                'mpglue': ['classification/*.pyx',
                           'classification/*.so',
                           'stats/*.pyx',
                           'stats/*.so']}


def get_console_dict():

    return {'console_scripts': ['change=mpglue.classification.change:main',
                                'classify=mpglue.classification.classify:main',
                                'sample-raster=mpglue.classification.sample_raster:main',
                                'reclassify=mpglue.classification.reclassify:main',
                                'recode=mpglue.classification.recode:main',
                                'raster-calc=mpglue.raster_calc:main',
                                'veg-indices=mpglue.veg_indices:main',
                                'vrt-builder=mpglue.vrt_builder:main',
                                'raster-tools=mpglue.raster_tools:main',
                                'vector-tools=mpglue.vector_tools:main']}


def setup_package():

    if platform.system() != 'Windows':
        include_dirs = [np.get_include()]
    else:
        include_dirs = None

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
                    zip_safe=False,
                    download_url=git_url,
                    install_requires=required_packages,
                    entry_points=get_console_dict(),
                    include_dirs=include_dirs)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
