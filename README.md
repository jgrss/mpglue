MpGlue
---

Current version
---

`0.2.8dev`

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
>>>     my_array = i_info.read(bands2open=-1, y=1200000., x=4230000., rows=500, cols=500)
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

#### Images can also be opened as [xarrays](http://xarray.pydata.org/en/stable/)

```python
>>> # Open an image as a xarray.
>>> with gl.ropen('/your/image.tif') as i_info:
>>>     my_array = i_info.read(as_xarray=True)
>>>
>>> # By default, dimensions are named ('z', 'y', 'x') or ('y', 'x').
>>> # Provide dimension names with `xarray_dims` (3-d temporal image below).
>>> with gl.ropen('/your/image.tif') as i_info:
>>>
>>>     my_array = i_info.read(as_xarray=True,
>>>                            xarray_dims=['time', 'y', 'x'])
>>>
>>> print(my_array.dims)
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
>>>     array2write = <some 3d array data>
>>>     out_raster.write_array(array2write, i=0, j=0, band=1)
>>>
>>> # or
>>>
>>> from mpglue import raster_tools
>>>
>>> raster_tools.write2raster(array2write, '/output_image.tif', o_info=o_info)
```

#### Band-wise statistics

```python
>>> from mpglue.raster_tools import pixel_stats
>>>
>>> # Calculate the band-wise mean
>>> pixel_stats('/image.tif', '/output.tif', stat='mean')
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

#### Train and test preparation:

```python
>>> # Initiate the classification object.
>>> cl = gl.classification()
>>>
>>> # Load and split land cover samples into test and train.
>>> # train = 70%
>>> # test = 30%
>>> cl.split_samples('/land_cover_samples.txt', perc_samp=.7)
>>>
>>> # Sample 70% from each class.
>>> cl.split_samples('/land_cover_samples.txt', perc_samp_each=.7)
>>>
>>> # Specify sampling percentage for each class.
>>> cl.split_samples('/land_cover_samples.txt', class_subs={1: .5, 2: .5, 3: .3})
>>>
>>> # Merge class 3 into class 2.
>>> cl.split_samples('/land_cover_samples.txt', recode_dict={1:1, 2:2, 3:2})
>>>
>>> # Remove classes 2 and 3.
>>> cl.split_samples('/land_cover_samples.txt', classes2remove=[2, 3])
>>>
>>> # Ignore predictive features 1 and 10.
>>> cl.split_samples('/land_cover_samples.txt', ignore_feas=[1, 10])
```

#### Image classification:

```python
>>> # Initiate the classification object.
>>> cl = gl.classification()
>>>
>>> # Load and split land cover samples into test (30%) and train (70%).
>>> cl.split_samples('/land_cover_samples.txt', perc_samp=0.7)
>>>
>>> # Train a Random Forest classification model.
>>> cl.construct_model(classifier_info={'classifier': 'rf', 'trees': 100})
>>>
>>> # Print the error matrix report.
>>> cl.emat.write_stats('/text_report.txt')
>>>
>>> # Predict class labels on all bands and samples.
>>> cl.predict('/input_image.tif', '/output_map.tif')
```

A more detailed example

```python
>>> cl = gl.classification()
>>>
>>> # Load and split land cover samples into test and train.
>>> # Use x,y coordinates as predictors (prepended to existing predictors)
>>> cl.split_samples('/land_cover_samples.txt', 
>>>                  use_xy=True,
>>>                  perc_samp=0.7)
>>>
>>> # Train a voting classification model (average of posteriors) and save to file.
>>> cl.construct_model(classifier_info={'classifier': ['bayes', 'rf', 'dt'], 'trees': 100},
>>>                    output_model='classifier.model')
>>>
>>> # Load a model from file
>>> cl.construct_model(input_model='classifier.model')
>>>
>>> # Specify prediction parameters
>>> cl.predict('/input_image.vrt',
>>>            '/output_map.tif',
>>>             bands2open=[1, 4, 5, 6],    # define bands to open (they need to match the training bands)
>>>             scale_factor=10000.0,       # apply a scale factor to the input image
>>>             n_jobs=-1,                  # number of processors for the model
>>>             n_jobs_vars=-1,             # number of processors for band loading
>>>             row_block_size=500,         # the row block processing size (i.e., row tile size)
>>>             col_block_size=500,         # the column block processing size (i.e., column tile size)
>>>             overwrite=True,             # overwrite an existing model
>>>             relax_probabilities=True,   # apply post-classification probability relaxation
>>>             morphology=True,            # apply post-classification morphology
>>>             use_xy=True,                # use x,y location as a predictor (*must be trained)
>>>             i=500,                      # starting row index to predict
>>>             j=500,                      # starting column index to predict
>>>             rows=1000,                  # the number of rows to predict
>>>             cols=1000)                  # the number of columns to predict
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
>>> comp_dict = {'001': ['/im1.tif', '/im2.tif'],   # VRT layer 1
>>>              '002': ['/im1.tif', '/im2.tif']}   # VRT layer 2
>>>
>>> # Stack the images.
>>> vrt_builder(comp_dict, '/out_image.vrt', force_type='float32')
```

#### Moving window operations
```python
>>> from mpglue import moving_window
>>>
>>> # Basic usage
>>> image_mean = moving_window(<image_array>,
>>>                            statistic='mean',
>>>                            window_size=5)
>>>
>>> # Compute the edge direction
>>> edge_theta = moving_window(edge_gradient_array,
>>>                            statistic='edge-direction',
>>>                            window_size=9)
>>>
>>> # Compute non-maximum suppression
>>> non_max = moving_window(edge_gradient_array,
>>>                         statistic='suppression',
>>>                         window_size=3)
>>> 
>>> # Compute a moving mean over an image.
>>> with gl.ropen('/input_image.tif') as i_info:
>>>
>>>     image_array = i_info.read()
>>>
>>>     image_mean = moving_window(image_array,
>>>                                statistic='mean',
>>>                                window_size=5)
>>>
>>> # Compute the percentage of binary pixels.
>>> with gl.ropen('/input_image.tif') as i_info:
>>>
>>>     image_array = i_info.read()
>>>
>>>     image_percent = moving_window(image_array,
>>>                                   statistic='percent',
>>>                                   window_size=25)
```

Installation
---

#### Dependencies

* Cython
* NumPy

#### Install the most up-to-date version from source (requires Cython)
 
1) Clone MpGlue

```commandline
cd <location to clone MpGlue>

# Clone to /mpglue
git clone https://github.com/jgrss/mpglue.git
```

2) Build and install

```commandline
cd mpglue

# Build the package
python setup.py build

# Install into /site-packages
python setup.py install
```

## OR

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

Testing
---

```python
>>> import mpglue as gl
>>> gl.test()
```

Development
---
For questions or bugs, please [**submit an issue**](https://github.com/jgrss/mpglue/issues).
