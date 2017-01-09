MpGlue
---

The **glue** of MapPy.

Usage examples
---

Load the library:
    
```python
>>> import mpglue as gl
```

Opening an image:

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
    
Getting an array:

```python
>>> # Open an image as an array.
>>> with gl.ropen('/your/image.tif') as i_info:
>>>     my_array = i_info.read()
>>>
>>> # Open specific bands, starting indexes, and row/column dimensions.
>>> with gl.ropen('/your/image.tif') as i_info:
>>>     my_array = i_info.read()
>>>
>>> # Open specific bands, starting indexes, and row/column dimensions.
>>> with gl.open('/your/image.tif') as i_info:
>>>     my_array = i_info.read(bands2open=[2, 3, 4], i=1000, j=2000, rows=500, cols=500)
>>>
>>> my_array[0]     # 1st index = band 2
>>>
>>> # Open all bands and index by map coordinates.
>>> with gl.ropen('/your/image.tif') as i_info:
>>>     my_array = i_info.read(bands2open=-1, y=1200000, x=4230000, rows=500, cols=500)
>>>
>>> # Open image bands as arrays with dictionary mappings.
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
```
    
Writing to file:

```python
>>> # Copy an image info object and modify it.
>>> o_info = i_info.copy()
>>> o_info.update_info(bands=3, storage='float32')
>>>
>>> # Create the raster object
>>> out_raster = gl.create_raster('/output_image.tif', o_info)
>>>
>>> # Write an array block to band 1.
>>> array2write = <some 2d array data>
>>> out_raster.write_array(array2write, i=0, j=0, band=1)
>>> out_raster.close()
```

Vegetation indices:

```python
>>> # Compute the NDVI.
>>> gl.veg_indices('/image.tif', '/out_index.tif', 'ndvi', 'Landsat')
>>>
>>> # Compute multiple indices.
>>> gl.veg_indices('/image.tif', '/out_index.tif', ['ndvi', 'evi2'], 'Landsat')
```

Land cover sampling:

```python
>>> # Sample land cover data.
>>> gl.sample_raster('/train_samples.shp', '/image_variables.tif',
>>>                  class_id='Id')
```

Image classification:

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

Thematic accuracy:

```python
>>> # Get map accuracy.
>>> gl.sample_raster('/test_samples.shp', '/thematic_map.tif',
>>>                  class_id='Id', accuracy=True)
```

Installation
---
#### Dependencies
- Python third-party libraries see [**the notebooks installation guide**](https://github.com/jgrss/mpglue/tree/master/mpglue/notebooks/01_installation.pynb).

**Install stable release with pip**

1) Update setuptools:

```text
pip install -U setuptools
```

2) [Acquire the latest MpGlue tarball](https://github.com/jgrss/mpglue/releases)

3) To install:

```text
pip install MpGlue-<version>.tar.gz
e.g., pip install MpGlue-0.0.1.tar.gz
```

4) To update:

```text
pip install -U MpGlue-<new version>.tar.gz
```

5) To uninstall:

```text
pip uninstall mpglue
```

Development
---
For questions or bugs, contact Jordan Graesser (graesser@bu.edu).





