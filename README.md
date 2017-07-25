MpGlue
---

The **glue** of [MapPy](https://github.com/jgrss/mappy).

Usage examples
---

### Command line tools

```bash
change
classify
reclassify
recode
raster-calc
raster-tools
sample-raster
veg-indices
vrt-builder
vector-tools
```

### Python library

#### Load the library:
    
```python
>>> import mpglue as gl
```

#### Opening an image:

```python
>>> # Load an image and get information.
>>> i_info = gl.ropen('/your/image.tif')
>>>
>>> print(i_info.bands)
>>> print(i_info.shape)
>>>
>>> # Close the dataset.
>>> i_info.close()
>>>
>>> # or
>>>
>>> with gl.ropen('/your/image.tif') as i_info:
>>>     print(i_info.bands)
```
    
#### Getting an array:

```python
>>> # Open an image as an array.
>>> with gl.ropen('/your/image.tif') as i_info:
>>>     my_array = i_info.read()
>>>
>>> # The 1st index = the first band of the image
>>> my_array[0]
>>>
>>> # Open specific bands, starting indexes, and row/column dimensions.
>>> with gl.open('/your/image.tif') as i_info:
>>>     my_array = i_info.read(bands2open=[2, 3, 4], i=1000, j=2000, rows=500, cols=500)
>>>
>>> # The array shape = (3, 500, 500)
>>> print(my_array.shape)
>>>
>>> # The 1st index = band 2
>>> my_array[0]
>>>
>>> # Open all bands (i.e., `bands2open` = -1) and index by map coordinates.
>>> with gl.ropen('/your/image.tif') as i_info:
>>>     my_array = i_info.read(bands2open=-1, y=1200000, x=4230000, rows=500, cols=500)
>>>
>>> # Open image bands as arrays with dictionary mappings.
>>> with gl.ropen('/your/image.tif') as i_info:
>>>     my_band_dict = i_info.read(bands2open={'red': 2, 'green': 3, 'nir': 4})
>>>
>>> my_band_dict['red']
>>>
>>> # Compute the NDVI.
>>> with gl.ropen('/your/image.tif') as i_info:
>>>     ndvi = i_info.read(compute_index='ndvi', sensor='Landsat')
>>>
>>> # or
>>>
>>> with gl.ropen('/your/image.tif') as i_info:
>>>     i_info.read(compute_index='ndvi', sensor='Landsat')
>>>     ndvi = i_info.array
```
    
#### Writing to file:

```python
>>> # Copy an image info object and modify it.
>>> o_info = i_info.copy()
>>> o_info.update_info(bands=3, storage='float32')
>>>
>>> # Create the raster object
>>> with gl.create_raster('/output_image.tif', o_info) as out_raster:
>>>
>>>     # Write an array block to band 1.
>>>     array2write = <some 2d array data>
>>>     out_raster.write_array(array2write, i=0, j=0, band=1)
```

#### Vegetation indices:

```python
>>> # Compute the NDVI.
>>> gl.veg_indices('/image.tif', '/out_index.tif', 'ndvi', 'Landsat')
>>>
>>> # Compute multiple spectral indices.
>>> gl.veg_indices('/image.tif', '/out_index.tif', ['ndvi', 'evi2'], 'Landsat')
```

#### Land cover sampling:

```python
>>> # Sample land cover data.
>>> gl.sample_raster('/train_samples.shp', '/image_variables.tif', class_id='Id')
```

#### Image classification:

```python
>>> # Initiate the classification object.
>>> cl = gl.classification()
>>>
>>> # Load and split land cover samples.
>>> cl.split_samples('/land_cover_samples.shp', perc_samp=.7)
>>> 
>>> # Train a classification model.
>>> cl.construct_model(classifier_info={'classifier': 'RF', 'trees': 100})
>>>
>>> # Print the error matrix report.
>>> print(cl.emat.report)
>>>
>>> # Predict class labels.
>>> cl.predict('/input_image.tif', '/output_map.tif')
```

#### Post-classification:

```python
# Reclassify a map
#   recode class 1 to class 2
>>> gl.reclassify('/input_map.tif', '/output_map.tif', {1: 2})

# Recode values within polygon features
#   In polygon 1, reclassify 6 to 5; in polygon 2, reclassify 2 to 5 and 3 to 5
>>> gl.recode('/input_poly.shp', '/input_map.tif', '/output_map.tif', {1: {6:5}, 2: {2:5, 3:5}})

# Change analysis
>>> gl.change('/map1.tif', '/map2.tif', out_report='/change_report.csv')
```

#### Thematic accuracy:

```python
>>> # Get map accuracy.
>>> gl.sample_raster('/test_samples.shp', 
>>>                  '/thematic_map.tif',
>>>                  class_id='Id', 
>>>                  accuracy=True)
```

#### Raster calculator

```python
>>> # Multiply raster A by raster B
>>> gl.raster_calc('/output_image.tif', 
>>>                equation='A*B', 
>>>                out_type='float32', 
>>>                A='/raster1.tif', 
>>>                B='/raster2.tif')
```

#### Build mixed-type VRT files:

```python
>>> # Fill a dictionary with image names.
>>> comp_dict = {'1': ['/im1.tif'], '2': ['/im2.tif']}
>>>
>>> # one-liner with many images
>>> # comp_dict = dict(zip(map(str, range(1, 3)), [['/im1.tif'], ['/im2.tif']]))
>>>
>>> # Stack the images.
>>> vrt_builder(comp_dict, '/out_image.vrt', force_type='float32')
```

Installation
---

#### Install from source (requires Cython)
 
1) Clone MpGlue

```commandline
cd <location to clone MpGlue>
git clone https://github.com/jgrss/mpglue.git
```

2) Build and install

```commandline
python setup.py build
python setup.py install
```

#### Install stable release with pip

1) Update setuptools:

```commandline
pip install -U setuptools
```

2) [Acquire the latest MpGlue tarball](https://github.com/jgrss/mpglue/releases)

3) To install:

```commandline
pip install MpGlue-<version>.tar.gz
```

For example:

```commandline
pip install MpGlue-0.1.0.tar.gz
```

4) To update:

```commandline
pip install -U MpGlue-<new version>.tar.gz
```

5) To uninstall:

```commandline
pip uninstall mpglue
```

Development
---
For questions or bugs, please [**submit an issue**](https://github.com/jgrss/mpglue/issues).
