import setuptools
from distutils.core import setup
import platform


__version__ = '0.0.6'

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

required_packages = ['numpy>=1.11.0', 'scipy>=0.17.1', 'scikit-learn>=0.17.1',
                     'scikit-image>=0.12.3', 'tables>=3.2.2', 'pandas>=0.18.1',
                     'matplotlib', 'joblib', 'BeautifulSoup4']

if platform.system() == 'Darwin':

    for pkg in ['gdal>=2.1', 'statsmodels>=0.8.0']:
        required_packages.append(pkg)


def get_packages():
    return setuptools.find_packages()


def get_package_data():

    return {'mpglue': ['*.md',
                       '*.txt',
                       'stats/*.so',
                       'stats/*.pyd']}


def get_console_dict():

    return {'console_scripts': ['classify=mpglue.classification.classify:main',
                                'sample-raster=mpglue.classification.sample_raster:main',
                                'veg-indices=mpglue.veg_indices:main',
                                'reclassify=mpglue.classification.reclassify:main',
                                'recode=mpglue.classification.recode:main']}


# def get_pyx_list():
#     return ['mpglue/stats/*.pyx']


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
