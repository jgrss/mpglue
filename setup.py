import setuptools
from distutils.core import setup

try:
     from Cython.distutils import build_ext
except ImportError:
     from distutils.command import build_ext


__version__ = '0.0.1'

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

required_packages = ['numpy>=1.11.0', 'scipy>=0.17.1', 'scikit-learn>=0.17.1', 'scikit-image>=0.12.3', 'gdal>=2.1',
                     'tables>=3.2.2', 'pandas>=0.18.1', 'matplotlib', 'statsmodels',
                     'joblib', 'BeautifulSoup4']


def get_packages():
    return setuptools.find_packages()


def get_package_data():

    return {'': ['*.md',
                 '*.txt'],
            'mpglue': ['stats/*.pyx',
                       'stats/*.c',
                       'stats/*.so',
                       'stats/*.pyd']}


def get_console_dict():

    return {'console_scripts': ['classify=mpglue.classify:main',
                                'sample_raster=mpglue.sample_raster:main',
                                'veg_indices=mpglue.veg_indices:main']}


def get_pyx_list():
    return ['mpglue/stats/*.pyx']


def setup_package():

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
                    zip_safe=False,
                    download_url=git_url,
                    install_requires=required_packages,
                    entry_points=get_console_dict())

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
