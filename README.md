MpGlue
------

The **glue** of MapPy.

Usage examples
-----

Image handling:

>>> import mpglue as gl
>>>
>>> # Load an image and get information.
>>> i_info = gl.rinfo('/your/image.tif')
>>> print(i_info.bands)
>>> print(i_info.shape)
>>>
>>> # Open an image as an array.
>>> my_array = i_info.mparray()
>>>
>>> # Open specific bands, starting indexes, and row/column dimensions.
>>> my_array = i_info.mparray(bands2open=[2, 3, 4], i=1000, j=2000, rows=500, cols=500)
>>> my_array[0]     # 1st index = band 2
>>>
>>> # Open all bands and index by map coordinates.
>>> my_array = i_info.mparray(bands2open=-1, y=1200000, x=4230000, rows=500, cols=500)
>>>
>>> # Open image bands as arrays with dictionary mappings.
>>> my_band_dict = i_info.mparray(bands2open={'red': 2, 'green': 3, 'nir': 4})
>>> my_band_dict['red']
>>>
>>> # Compute the NDVI.
>>> ndvi = i_info.mparray(compute_index='ndvi', sensor='Landsat')
>>>
>>> # Writing to file
>>>
>>> # Copy an image info object and modify it.
>>> o_info = i_info.copy()
>>> o_info.update_info(bands=3, storage='float32')
>>>
>>> # Create the raster object
>>> out_raster = gl.create_raster('/output_image.tif', o_info)
>>>
>>> # Write an array block to band 1
>>> array2write = <some 2d array data>
>>> out_raster.write_array(array2write, i=0, j=0, band=1)
>>> out_raster.close()

Installation
------------
#### Dependencies
- Python third-party libraries (see /notebooks/01_installation.pynb)

**Install stable release with pip**

1) Update setuptools:

> pip install -U setuptools

2) [Acquire the latest MpGlue tarball](https://github.com/jgrss/mpglue/releases)

3) To install:

> pip install MpGlue-<version>.tar.gz
> e.g., pip install MpGlue-0.0.1.tar.gz

4) To update:

> pip install -U MpGlue-<new version>.tar.gz

5) To uninstall:

> pip uninstall mpglue

Development
-----------
For questions or bugs, contact Jordan Graesser (graesser@bu.edu).





