#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 9/24/2011
"""

from __future__ import division, print_function
from future.utils import iteritems, viewitems
from builtins import int, map, zip

import os
import sys
import copy
import fnmatch
import time
import argparse
import inspect
from joblib import Parallel, delayed
import shutil
import platform
import subprocess
from collections import OrderedDict

from . import vector_tools
from .helpers import random_float, overwrite_file, check_and_create_dir, _iteration_parameters
from .errors import EmptyImage, LenError, MissingRequirement, ropenError, ArrayShapeError, ArrayOffsetError, logger
from .veg_indices import veg_indices, VegIndicesEquations

try:
    import deprecation
except:

    logger.error('deprecation must be installed (pip install deprecation)')
    raise ImportError

# GDAL
try:
    from osgeo import gdal, osr
    from osgeo.gdalconst import GA_ReadOnly, GA_Update
except:

    logger.error('  GDAL Python must be installed')
    raise ImportError

# NumPy
try:
    import numpy as np
except:

    logger.error('  NumPy must be installed')
    raise ImportError

# Matplotlib
try:
    import matplotlib as mpl

    if (os.environ.get('DISPLAY', '') == '') or (platform.system() == 'Darwin'):

        mpl.use('Agg')

        try:
            mpl.pyplot.switch_backend('agg')
        except:
            pass

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import ticker, colors, colorbar
    import matplotlib.cm as cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except:
    logger.warning('  Matplotlib must be installed for plotting')

# Scikit-learn
try:
    from sklearn.preprocessing import StandardScaler
except:
    logger.warning('  Scikit-learn must be installed for pixel_stats z-scores.')

# Scikit-image
try:
    from skimage import exposure
except:
    logger.warning('  Scikit-image must be installed for image color balancing.')

# SciPy
try:
    from scipy.stats import mode as sci_mode
    from scipy.ndimage.measurements import label as lab_img
except:

    logger.error('  SciPy must be installed')
    raise ImportError

# Pandas
try:
    import pandas as pd
except:
    logger.warning('  Pandas must be installed to parse metadata')

# OpenCV
try:
    import cv2
except:
    logger.warning('  OpenCV must be installed to use stat functions.')

# BeautifulSoup4
try:
    from bs4 import BeautifulSoup
except:
    logger.warning('  BeautifulSoup4 must be installed to parse metadata')


gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.SetCacheMax(int(2.0**30.0))

DRIVER_DICT = {'.bin': 'ENVI',
               '.bsq': 'ENVI',
               '.dat': 'ENVI',
               '.ecw': 'ECW',
               '.img': 'HFA',
               '.hdf': 'HDF4',
               '.hdf4': 'HDF4',
               '.hdf5': 'HDF5',
               '.h5': 'HDF5',
               '.hdr': 'ENVI',
               '.jp2': 'JPEG2000',
               '.kea': 'KEA',
               '.mem': 'MEM',
               '.nc': 'netCDF',
               '.ntf': 'NITF',
               '.pix': 'PCRaster',
               '.hgt': 'SRTMHGT',
               '.sid': 'MrSID',
               '.tif': 'GTiff',
               '.tiff': 'GTiff',
               '.til': 'TIL',
               '.vrt': 'VRT'}


FORMAT_DICT = dict((v, k) for k, v in iteritems(DRIVER_DICT))

STORAGE_DICT = {'byte': 'uint8',
                'int16': 'int16',
                'uint16': 'uint16',
                'int32': 'int32',
                'uint32': 'uint32',
                'int64': 'int64',
                'uint64': 'uint64',
                'float32': 'float32',
                'float64': 'float64'}

STORAGE_DICT_r = dict((v, k) for k, v in iteritems(STORAGE_DICT))

STORAGE_DICT_GDAL = {'unknown': gdal.GDT_Unknown,
                     'byte': gdal.GDT_Byte,
                     'uint16': gdal.GDT_UInt16,
                     'int16': gdal.GDT_Int16,
                     'uint32': gdal.GDT_UInt32,
                     'int32': gdal.GDT_Int32,
                     'float32': gdal.GDT_Float32,
                     'float64': gdal.GDT_Float64,
                     'cint16': gdal.GDT_CInt16,
                     'cint32': gdal.GDT_CInt32,
                     'cfloat32': gdal.GDT_CFloat32,
                     'cfloat64': gdal.GDT_CFloat64}

STORAGE_DICT_NUMPY = {'byte': np.uint8,
                      'int16': np.int16,
                      'uint16': np.uint16,
                      'int32': np.int32,
                      'uint32': np.uint32,
                      'int64': np.int64,
                      'uint64': np.uint64,
                      'float32': np.float32,
                      'float64': np.float64}

RESAMPLE_DICT = dict(average=gdal.GRA_Average,
                     bilinear=gdal.GRA_Bilinear,
                     nearest=gdal.GRA_NearestNeighbour,
                     cubic=gdal.GRA_Cubic)

PANSHARPEN_WEIGHTS = dict(oli_tirs=dict(bw=0.2,
                                        gw=1.0,
                                        rw=1.0,
                                        iw=0.1),
                          etm=dict(bw=0.1,
                                   gw=1.0,
                                   rw=1.0,
                                   iw=1.0))

GOOGLE_CLOUD_SENSORS = dict(oli_tirs='LC08',
                            etm='LE07',
                            tm='LT05')

SENSOR_DICT = {'landsat_tm': 'tm',
               'lt4': 'tm',
               'lt5': 'tm',
               'tm': 'tm',
               'landsat_etm_slc_off': 'etm',
               'landsat_etm': 'etm',
               'landsat_et': 'etm',
               'landsat_etm_slc_on': 'etm',
               'etm': 'etm',
               'le7': 'etm',
               'lt7': 'etm',
               'landsat_oli_tirs': 'oli_tirs',
               'oli_tirs': 'oli_tirs',
               'oli': 'oli_tirs',
               'tirs': 'oli_tirs',
               'lc8': 'oli_tirs',
               'lt8': 'oli_tirs'}


def create_memory_raster(image_info,
                         rows,
                         cols,
                         left,
                         top):

    """
    Creates an in-memory raster object

    Args:
        image_info (object)
        rows (int)
        cols (int)
        left (float)
        top (float)

    Returns:
        Datasource object
    """

    # Create a raster to rasterize into.
    target_ds = gdal.GetDriverByName('MEM').Create('', cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform([left, image_info.cellY, 0.0, top, 0.0, -image_info.cellY])
    target_ds.SetProjection(image_info.projection)

    return target_ds


def nd_to_rgb(array2reshape):

    """
    Reshapes an array from nd layout to RGB
    """

    if len(array2reshape.shape) != 3:

        logger.error('  The array must be 3 dimensions.')
        raise LenError

    if array2reshape.shape[0] != 3:

        logger.error('  The array must be 3 bands.')
        raise ArrayShapeError

    return np.ascontiguousarray(array2reshape.transpose(1, 2, 0))


def rgb_to_nd(array2reshape):

    """
    Reshapes an array RGB layout to nd layout
    """

    if len(array2reshape.shape) != 3:

        logger.error('  The array must be 3 dimensions.')
        raise LenError

    if array2reshape.shape[2] != 3:

        logger.error('  The array must be 3 bands.')
        raise ArrayShapeError

    return np.ascontiguousarray(array2reshape.transpose(2, 0, 1))


def nd_to_columns(array2reshape, layers, rows, columns):

    """
    Reshapes an array from nd layout to [samples (rows*columns) x dimensions]
    """

    if layers == 1:
        return array2reshape.flatten()[:, np.newaxis]
    else:
        return array2reshape.reshape(layers, rows, columns).transpose(1, 2, 0).reshape(rows*columns, layers)


def columns_to_nd(array2reshape, layers, rows, columns):

    """
    Reshapes an array from columns layout to [n layers x rows x columns]
    """

    if layers == 1:
        return array2reshape.reshape(columns, rows).T
    else:
        return array2reshape.T.reshape(layers, rows, columns)


class ReadWrite(object):

    def read(self,
             bands2open=1,
             i=0,
             j=0,
             rows=-1,
             cols=-1,
             d_type=None,
             compute_index='none',
             sensor='Landsat',
             sort_bands2open=True,
             predictions=False,
             y=0.,
             x=0.,
             check_x=None,
             check_y=None,
             as_xarray=False,
             xarray_dims=None,
             xarray_coords=None,
             **viargs):

        """
        Reads a raster as an array

        Args:
            bands2open (Optional[int or int list or dict]: Band position to open, list of bands to open, or a
                dictionary of name-band pairs. Default is 1.

                Examples:
                    bands2open = 1        (open band 1)
                    bands2open = [1,2,3]  (open first three bands)
                    bands2open = [4,3,2]  (open bands in a specific order)
                        *When opening bands in a specific order, be sure to set ``sort_bands2open`` as ``False``.
                    bands2open = -1       (open all bands)
                    bands2open = {'blue': 1, 'green': 2, 'nir': 4}  (open bands 1, 2, and 4)

            i (Optional[int]): Starting row position. Default is 0, or first row.
            j (Optional[int]): Starting column position. Default is 0, or first column.
            rows (Optional[int]): Number of rows to extract. Default is -1, or all rows.
            cols (Optional[int]): Number of columns to extract. Default is -1, or all columns.
            d_type (Optional[str]): Type of array to return. Choices are ['byte', 'int16', 'uint16',
                'int32', 'uint32', 'int64', 'uint64', 'float32', 'float64']. Default is None, or gathered
                from <i_info>.
            compute_index (Optional[str]): A spectral index to compute. Default is 'none'.
            sensor (Optional[str]): The input sensor type (used with ``compute_index``). Default is 'Landsat'.
            sort_bands2open (Optional[bool]): Whether to sort ``bands2open``. Default is True.
            predictions (Optional[bool]): Whether to return reshaped array for Scikit-learn formatted
                predictions (i.e., samples x dimensions).
            y (Optional[float]): A y index coordinate (latitude, in map units). Default is 0.
                If greater than 0, overrides ``i``.
            x (Optional[float]): A x index coordinate (longitude, in map units). Default is 0.
                If greater than 0, overrides ``j``.
            check_x (Optional[float]): Check the x offset against ``check_x``. Default is None.
            check_y (Optional[float]): Check the y offset against ``check_y``. Default is None.
            as_xarray (Optional[bool]): Whether to open the array as a xarray, otherwise as a Numpy array.
                Default is False.
            xarray_dims (Optional[list]): Dimension names for xarray. Default is None.
            xarray_coords (Optional[list]): Coordinates for xarray. Default is None.
            viargs (Optional[dict]): Keyword arguments passed to `veg_indices`. Default is None.

        Attributes:
            array (ndarray)

        Returns:
            ``ndarray``, where shape is (rows x cols) if 1 band or (bands x rows x cols) if more than 1 band.

        Examples:
            >>> import mpglue as gl
            >>>
            >>> i_info = mp.ropen('image.tif')
            >>> i_info = mp.open('image.tif')
            >>>
            >>> # Open 1 band.
            >>> array = i_info.read(bands2open=1)
            >>>
            >>> # Open multiple bands.
            >>> array = i_info.read(bands2open=[1, 2, 3])
            >>> band_1 = array[0]
            >>>
            >>> # Open as a dictionary of arrays.
            >>> bands = i_info.read(bands2open={'blue': 1, 'red': 2, 'nir': 4})
            >>> red = bands['red']
            >>>
            >>> # Index an image by pixel positions.
            >>> array = i_info.read(i=1000, j=4000, rows=500, cols=500)
            >>>
            >>> # Index an image by map coordinates.
            >>> array = i_info.read(y=1200000., x=4230000., rows=500, cols=500)
        """

        self.i = i
        self.j = j

        # `self.rows` and `self.cols` are the
        #   image dimension info, so don't overwrite.
        self.rrows = rows
        self.ccols = cols

        self.sort_bands2open = sort_bands2open

        self.as_xarray = as_xarray
        self.xarray_dims = xarray_dims
        self.xarray_coords = xarray_coords

        if isinstance(bands2open, dict):

            if isinstance(d_type, str):
                self.d_type = STORAGE_DICT_NUMPY[d_type]
            else:
                self.d_type = STORAGE_DICT_NUMPY[self.storage.lower()]

        else:

            if isinstance(d_type, str):
                self.d_type = STORAGE_DICT[d_type]
            else:
                self.d_type = STORAGE_DICT[self.storage.lower()]

        if compute_index != 'none':

            bh = veg_indices.BandHandler(sensor)

            bh.get_band_order()

            # Overwrite the bands to open
            bands2open = bh.get_band_positions(bh.wavelength_lists[compute_index.upper()])

            self.d_type = 'float32'

        if self.rrows == -1:
            self.rrows = self.rows
        else:

            if self.rrows > self.rows:

                self.rrows = self.rows
                logger.warning('  The requested rows cannot be larger than the image rows.')

        if self.ccols == -1:
            self.ccols = self.cols
        else:

            if self.ccols > self.cols:

                self.ccols = self.cols
                logger.warning('  The requested columns cannot be larger than the image columns.')

        # Index the image by x, y coordinates (in map units).
        if (abs(y) > 0) and (abs(x) > 0):

            __, __, self.j, self.i = vector_tools.get_xy_offsets(self,
                                                                 x=x,
                                                                 y=y,
                                                                 check_position=False)
            
        if isinstance(check_x, float) and isinstance(check_y, float):

            __, __, x_offset, y_offset = vector_tools.get_xy_offsets(self,
                                                                     x=check_x,
                                                                     y=check_y,
                                                                     check_position=False)

            self.i += y_offset
            self.j += x_offset

        #################
        # Bounds checking
        #################

        # Row indices
        if self.i < 0:
            self.i = 0

        if self.i >= self.rows:
            self.i = self.rows - 1

        # Number of rows
        self.rrows = n_rows_cols(self.i, self.rrows, self.rows)

        # Column indices
        if self.j < 0:
            self.j = 0

        if self.j >= self.cols:
            self.j = self.cols - 1

        # Number of columns
        self.ccols = n_rows_cols(self.j, self.ccols, self.cols)

        #################

        # format_dict = {'byte': 'B', 'int16': 'i', 'uint16': 'I', 'float32': 'f', 'float64': 'd'}
        # values = struct.unpack('%d%s' % ((rows * cols * len(bands2open)), format_dict[i_info.storage.lower()]),
        #     i_info.datasource.ReadRaster(yoff=i, xoff=j, xsize=cols, ysize=rows, band_list=bands2open))

        if hasattr(self, 'band'):

            self.array = self.band.ReadAsArray(self.j,
                                               self.i,
                                               self.ccols,
                                               self.rrows).astype(self.d_type)

            self.array_shape = [1, self.rrows, self.ccols]

            if predictions:
                self._norm2predictions(1)

        else:

            # Check ``bands2open`` type.
            bands2open = self._check_band_list(bands2open)

            # Open the array.
            self._open_array(bands2open)

            if predictions:
                self._norm2predictions(len(bands2open))

        if compute_index != 'none':

            vie = VegIndicesEquations(self.array,
                                      chunk_size=-1)

            # exec('self.{} = vie.compute(compute_index.upper())'.format(compute_index.lower()))
            self.array = vie.compute(compute_index.upper(),
                                     **viargs)

        self.array[np.isnan(self.array) | np.isinf(self.array)] = 0

        if self.as_xarray:
            self._as_xarray()

        return self.array

    def _as_xarray(self):

        """
        Transforms the NumPy array to a xarray object
        """

        try:
            import xarray as xr
        except:
            logger.error('  Cannot import xarray')
            raise ImportError

        if len(self.array.shape) == 3:

            n_bands = self.array.shape[0]

            if not self.xarray_coords:
                self.xarray_coords = dict(z=('B' + ',B'.join(list(map(str, range(1, n_bands+1))))).split(','))

            if not self.xarray_dims:
                self.xarray_dims = ['z', 'y', 'x']

        else:

            if not self.xarray_dims:
                self.xarray_dims = ['y', 'x']

        self.array = xr.DataArray(self.array,
                                  coords=self.xarray_coords,
                                  dims=self.xarray_dims)

    def _open_array(self, bands2open):

        """
        Opens image bands into a ndarray.
        
        Args:
             bands2open (int or list)
        """

        # Open the image as a dictionary of arrays.
        if isinstance(bands2open, dict):

            self.array = dict()

            for band_name, band_position in viewitems(bands2open):

                if self.hdf_file:

                    self.array[band_name] = self.hdf_datasources[band_position-1].ReadAsArray(self.j,
                                                                                              self.i,
                                                                                              self.ccols,
                                                                                              self.rrows).astype(self.d_type)

                else:

                    self.array[band_name] = self.datasource.GetRasterBand(band_position).ReadAsArray(self.j,
                                                                                                     self.i,
                                                                                                     self.ccols,
                                                                                                     self.rrows).astype(self.d_type)

        # Open the image as an array.
        else:

            if self.hdf_file:

                self.array = np.asarray([self.hdf_datasources[band-1].ReadAsArray(self.j,
                                                                                  self.i,
                                                                                  self.ccols,
                                                                                  self.rrows) for band in bands2open],
                                        dtype=self.d_type)

            else:

                self.array = list()

                for band in bands2open:

                    arr = self.datasource.GetRasterBand(band).ReadAsArray(self.j,
                                                                          self.i,
                                                                          self.ccols,
                                                                          self.rrows)

                    if not isinstance(arr, np.ndarray):

                        logger.info(type(arr))
                        logger.error('  Band {:d} is not a NumPy array.'.format(band))

                        raise TypeError

                    self.array.append(arr)

                self.array = np.asarray(self.array, dtype=self.d_type)

            self.array = self._reshape(self.array, bands2open)

    def _predictions2norm(self, n_bands):

        """
        Reshapes an array from predictions to nd array

        Args:
            n_bands (int)
        """

        self.array = columns_to_nd(self.array, n_bands, self.rrows, self.ccols)

        self.array_shape = [n_bands, self.rrows, self.ccols]

    def _norm2predictions(self, n_bands):

        """
        Reshapes an array from normal layout to Scikit-learn compatible shape (i.e., samples x dimensions)
        
        Args:
            n_bands (int)
        """

        self.array = nd_to_columns(self.array, n_bands, self.rrows, self.ccols)

        self.array_shape = [1, self.rrows*self.ccols, n_bands]

    def _reshape(self, array2reshape, band_list):

        """
        Reshapes an array into [rows X columns] or [dimensions X rows X columns].
        
        Args:
            array2reshape (ndarray)
            band_list (list)
        """

        if len(band_list) == 1:
            array2reshape = array2reshape.reshape(self.rrows, self.ccols)
        else:
            array2reshape = array2reshape.reshape(len(band_list), self.rrows, self.ccols)

        self.array_shape = [len(band_list), self.rrows, self.ccols]

        return array2reshape

    def _check_band_list(self, bands2open):

        """
        Checks whether a band list is valid.
        
        Args:
            bands2open (dict, list, or int)
        """

        if isinstance(bands2open, dict):
            return bands2open
        elif isinstance(bands2open, list):

            if not bands2open:

                logger.error('  A band list must be declared.\n')
                raise LenError

            if 0 in bands2open:

                logger.error('  A band list cannot have any zeros. GDAL indexes starting at 1.\n')
                raise ValueError

            if not self.hdf_file:

                if max(bands2open) > self.bands:

                    logger.error('  The requested band position cannot be greater than the image bands.\n')
                    raise ValueError

        elif isinstance(bands2open, int):

            if not self.hdf_file:

                if bands2open > self.bands:

                    logger.error('  The requested band position cannot be greater than the image bands.\n')
                    raise ValueError

            if bands2open == -1:
                bands2open = list(range(1, self.bands+1))
            else:
                bands2open = [bands2open]

        else:

            logger.error('  The `bands2open` parameter must be a dict, list, or int.\n')
            raise TypeError

        if self.sort_bands2open and not isinstance(bands2open, dict):
            bands2open = sorted(bands2open)

        return bands2open

    def write2raster(self,
                     out_name,
                     write_which='array',
                     o_info=None,
                     x=0,
                     y=0,
                     out_rst=None,
                     write2bands=None,
                     compress='deflate',
                     tile=False,
                     close_band=True,
                     flush_final=False,
                     write_chunks=False,
                     **kwargs):

        """
        Writes an array to file.

        Args:
            out_name (str): The output image name.
            write_which (Optional[str]): The array type to write. Choices are ['array', '<spectral indices>'].
                Default is 'array'.
            o_info (Optional[object]): Output image information, instance of ``ropen``.
                Needed if <out_rst> not given. Default is None.
            x (Optional[int]): Column starting position. Default is 0.
            y (Optional[int]): Row starting position. Default is 0.
            out_rst (Optional[object]): GDAL object to right to, otherwise created. Default is None.
            write2bands (Optional[int or int list]): Band positions to write to, otherwise takes the order of the input
                array dimensions. Default is None.
            compress (Optional[str]): Needed if <out_rst> not given. Default is 'deflate'.
            tile (Optional[bool]): Needed if <out_rst> not given. Default is False.
            close_band (Optional[bool]): Whether to flush the band cache. Default is True.
            flush_final (Optional[bool]): Whether to flush the raster cache. Default is False.
            write_chunks (Optional[bool]): Whether to write to file in <write_chunks> chunks. Default is False.

        Returns:
            None, writes <out_name>.
        """

        if isinstance(write_which, str):

            if write_which == 'ndvi':
                out_arr = self.ndvi
            elif write_which == 'evi2':
                out_arr = self.evi2
            elif write_which == 'pca':
                out_arr = self.pca_components
            else:
                out_arr = self.array

        elif isinstance(write_which, np.ndarray):
            out_arr = write_which

        d_name, f_name = os.path.split(out_name)

        if not os.path.isdir(d_name):
            os.makedirs(d_name)

        array_shape = out_arr.shape

        if len(array_shape) == 2:

            out_rows, out_cols = out_arr.shape
            out_dims = 1

        else:

            out_dims, out_rows, out_cols = out_arr.shape

        new_file = False

        if not out_rst:

            new_file = True

            if kwargs:

                try:
                    o_info.storage = kwargs['storage']
                except:
                    pass

                try:
                    o_info.bands = kwargs['bands']
                except:
                    o_info.bands = out_dims

            o_info.rows = out_rows
            o_info.cols = out_cols

            out_rst = create_raster(out_name, o_info, compress=compress, tile=tile)

        # Specify a band to write to.
        if isinstance(write2bands, int) or isinstance(write2bands, list):

            if isinstance(write2bands, int):
                write2bands = [write2bands]

            for n_band in write2bands:

                out_rst.get_band(n_band)

                if write_chunks:

                    out_rst.get_chunk_size()

                    for i in range(0, out_rst.rows, out_rst.chunk_size):

                        n_rows = n_rows_cols(i, out_rst.chunk_size, out_rst.rows)

                        for j in range(0, out_rst.cols, out_rst.chunk_size):

                            n_cols = n_rows_cols(j, out_rst.chunk_size, out_rst.cols)

                            out_rst.write_array(out_arr[i:i+n_rows, j:j+n_cols], i=i, j=j)

                else:

                    out_rst.write_array(out_arr, i=y, j=x)

                if close_band:
                    out_rst.close_band()

        # Write in order of the 3rd array dimension.
        else:

            arr_shape = out_arr.shape

            if len(arr_shape) > 2:

                out_bands = arr_shape[0]

                for n_band in range(1, out_bands+1):

                    out_rst.write_array(out_arr[n_band-1], i=y, j=x, band=n_band)

                    if close_band:
                        out_rst.close_band()

            else:

                out_rst.write_array(out_arr, i=y, j=x, band=1)

                if close_band:
                    out_rst.close_band()

        # close the dataset if it was created or prompted by <flush_final>
        if flush_final or new_file:
            out_rst.close_file()


class DataChecks(object):

    """
    A class for spatial and cloud checks
    """

    def contains(self, iinfo):

        """
        Tests whether the open image contains another image.

        Args:
            iinfo (object): An image instance of ``ropen`` to test.
        """

        if (iinfo.left > self.left) and (iinfo.right < self.right) \
                and (iinfo.top < self.top) and (iinfo.bottom > self.bottom):

            return True

        else:
            return False

    def contains_value(self, value, array=None):

        """
        Tests whether a value is within the array

        Args:
            value (int): The value to check.
            array (Optional[ndarray]): An array to check. Otherwise, check self.array. Default is None.

        Returns:
            Whether the array contains `value` (bool)
        """

        if not isinstance(array, np.ndarray) and not hasattr(self, 'array'):
            logger.exception('  No array to check')

        if hasattr(self, 'array'):
            return np.in1d(np.array([value]), self.array)[0]
        else:
            return np.in1d(np.array([value]), array)[0]

    def intersects(self, iinfo):

        """
        Tests whether the open image intersects another image.

        Args:
            iinfo (object): An image instance of ``ropen`` to test.
        """

        image_intersects = False

        # At least within the longitude frame.
        if ((iinfo.left > self.left) and (iinfo.left < self.right)) or \
                ((iinfo.right < self.right) and (iinfo.right > self.left)):

            # Also within the latitude frame.
            if ((iinfo.bottom > self.bottom) and (iinfo.bottom < self.top)) or \
                    ((iinfo.top < self.top) and (iinfo.top > self.bottom)):

                image_intersects = True

        return image_intersects

    def within(self, iinfo):

        """
        Tests whether the open image falls within another image.

        Args:
            iinfo (object or dict): An image instance of ``ropen`` to test.
        """

        if isinstance(iinfo, ropen):
            iinfo = self._info2dict(iinfo)

        if (self.left > iinfo['left']) and (self.right < iinfo['right']) \
                and (self.top < iinfo['top']) and (self.bottom > iinfo['bottom']):

            return True

        else:
            return False

    def outside(self, iinfo):

        """
        Tests whether the open image falls outside coordinates

        Args:
            iinfo (object or dict): An image instance of ``ropen`` to test.
        """

        if isinstance(iinfo, dict):

            iif = ImageInfo()

            for k, v in viewitems(iinfo):
                setattr(iif, k, v)

            iinfo = iif.copy()

        has_extent = False

        if hasattr(self, 'left') and hasattr(self, 'right'):

            has_extent = True

            if self.left > iinfo.right:
                return True

            if self.right < iinfo.left:
                return True

        if hasattr(self, 'top') and hasattr(self, 'bottom'):

            has_extent = True

            if self.top < iinfo.bottom:
                return True

            if self.bottom > iinfo.top:
                return True

        if not has_extent:
            logger.error('The `iinfo` parameter did not contain extent information.')

        return False

    def check_clouds(self, cloud_band=7, clear_value=0, background_value=255):

        """
        Checks cloud information.

        Args:
            cloud_band (Optional[int]): The cloud band position. Default is 7.
            clear_value (Optional[int]): The clear pixel value. Default is 0.
            background_value (Optional[int]): The background pixel value. Default is 255.
        """

        cloud_array = self.read(bands2open=cloud_band)

        clear_pixels = (cloud_array == clear_value).sum()

        non_background_pixels = (cloud_array != background_value).sum()

        self.clear_percent = (float(clear_pixels) / float(non_background_pixels)) * 100.


class RegisterDriver(object):

    """
    Class handler for driver registration

    Args:
        out_name (str): The file to register.
        in_memory (bool): Whether to create the file in memory.

    Attributes:
        out_name (str)
        driver (object)
        file_format (str)
    """

    def __init__(self, out_name, in_memory):

        gdal.AllRegister()

        if not in_memory:

            self._get_file_format(out_name)

            self.driver = gdal.GetDriverByName(self.file_format)

        else:
            self.driver = gdal.GetDriverByName('MEM')

        self.driver.Register

    def _get_file_format(self, image_name):

        d_name, f_name = os.path.split(image_name)
        __, file_extension = os.path.splitext(f_name)

        self.hdr_file = False

        if os.path.isfile(os.path.join(d_name, '{}.hdr'.format(f_name))):

            file_extension = '.hdr'
            self.hdr_file = True

        self.file_format = self._get_driver_name(file_extension)

    @staticmethod
    def _get_driver_name(file_extension):

        if file_extension.lower() not in DRIVER_DICT:

            logger.error('{} is not an image, or is not a supported raster format.'.format(file_extension))
            raise TypeError

        else:
            return DRIVER_DICT[file_extension.lower()]


class CreateDriver(RegisterDriver):

    """
    Class handler for driver creation

    Args:
        out_name (str): The output file name.
        out_rows (int): The output number of rows.
        out_cols (int): The output number of columns.
        n_bands (int): The output number of bands.
        storage_type (str): The output storage type.
        in_memory (bool): Whether to create the file in memory.
        overwrite (bool): Whether to overwrite an existing file.
        parameters (str list): A list of GDAL creation parameters.

    Attributes:
        datasource (object)
    """

    def __init__(self, out_name, out_rows, out_cols, n_bands, storage_type, in_memory, overwrite, parameters):

        RegisterDriver.__init__(self, out_name, in_memory)

        if overwrite and not in_memory:

            if os.path.isfile(out_name):
                os.remove(out_name)

        # Create the output driver.
        if in_memory:
            self.datasource = self.driver.Create('', out_cols, out_rows, n_bands, storage_type)
        else:
            self.datasource = self.driver.Create(out_name, out_cols, out_rows, n_bands, storage_type, parameters)


class DatasourceInfo(object):

    def datasource_info(self):

        if self.datasource is None:

            if hasattr(self, 'file_name'):

                logger.error('  {} appears to be empty.'.format(self.file_name))
                raise EmptyImage

            else:

                logger.error('  The datasource appears to be empty.')
                raise EmptyImage

        try:
            self.meta_dict = self.datasource.GetMetadata_Dict()
        except:

            logger.error(gdal.GetLastErrorMsg())
            self.meta_dict = 'none'

        try:
            self.storage = gdal.GetDataTypeName(self.datasource.GetRasterBand(1).DataType)
        except:
            self.storage = 'none'

        if hasattr(self, 'file_name'):
            self.directory, self.filename = os.path.split(self.file_name)

        if self.hdf_file:
            self.bands = len(self.hdf_datasources)
        else:
            self.bands = self.datasource.RasterCount

        # Initiate the data checks object.
        # DataChecks.__init__(self)

        # Check if any of the bands are corrupted.
        if hasattr(self, 'check_corrupted'):

            if self.check_corrupted:
                self.check_corrupted_bands()

        self.projection = self.datasource.GetProjection()

        self.sp_ref = osr.SpatialReference()
        self.sp_ref.ImportFromWkt(self.projection)
        self.proj4 = self.sp_ref.ExportToProj4()

        self.color_interpretation = self.datasource.GetRasterBand(1).GetRasterColorInterpretation()

        if 'PROJ' in self.projection[:4]:

            if self.sp_ref.GetAttrValue('PROJCS|AUTHORITY', 1):
                self.epsg = int(self.sp_ref.GetAttrValue('PROJCS|AUTHORITY', 1))
            else:
                self.epsg = 'none'

        elif 'GEOG' in self.projection[:4]:

            try:
                self.epsg = int(self.sp_ref.GetAttrValue('GEOGCS|AUTHORITY', 1))
            except:

                logger.error(gdal.GetLastErrorMsg())

                if 'WGS' in self.sp_ref.GetAttrValue('GEOGCS') and '84' in self.sp_ref.GetAttrValue('GEOGCS'):
                    self.epsg = 4326  # WGS 1984
                else:
                    self.epsg = 'none'
        else:
            self.epsg = 'none'

        # Set georeference and projection.
        self.geo_transform = self.datasource.GetGeoTransform()

        # adfGeoTransform[0] :: top left x
        # adfGeoTransform[1] :: w-e pixel resolution
        # adfGeoTransform[2] :: rotation, 0 if image is north up
        # adfGeoTransform[3] :: top left y
        # adfGeoTransform[4] :: rotation, 0 if image is north up
        # adfGeoTransform[5] :: n-s pixel resolution

        self.left = self.geo_transform[0]  # get left extent
        self.top = self.geo_transform[3]  # get top extent
        self.cellY = self.geo_transform[1]  # pixel height
        self.cellX = self.geo_transform[5]  # pixel width

        self.rotation1 = self.geo_transform[2]
        self.rotation2 = self.geo_transform[4]

        self.rows = self.datasource.RasterYSize  # get number of rows
        self.cols = self.datasource.RasterXSize  # get number of columns

        self.center_x = self.left + ((self.cols / 2) * self.cellY)
        self.center_y = self.top - ((self.rows / 2) * self.cellY)

        if not self.projection:
            self._get_hdr_info()

        self.shape = dict(bands=self.bands,
                          rows='{:,d}'.format(self.rows),
                          columns='{:,d}'.format(self.cols),
                          pixels='{:,d}'.format(self.bands * self.rows * self.cols),
                          row_units='{:,.2f}'.format(self.rows * self.cellY),
                          col_units='{:,.2f}'.format(self.cols * self.cellY))

        self.right = self.left + (self.cols * abs(self.cellY))  # get right extent
        self.bottom = self.top - (self.rows * abs(self.cellX))  # get bottom extent

        self.image_envelope = [self.left, self.right, self.bottom, self.top]

        self.extent = dict(left=self.left,
                           right=self.right,
                           bottom=self.bottom,
                           top=self.top)

        self.name = self.datasource.GetDriver().ShortName

        try:
            self.block_x = self.datasource.GetRasterBand(1).GetBlockSize()[0]
            self.block_y = self.datasource.GetRasterBand(1).GetBlockSize()[1]
        except:

            logger.error(gdal.GetLastErrorMsg())

            self.block_x = 'none'
            self.block_y = 'none'


class FileManager(DataChecks, RegisterDriver, DatasourceInfo):

    """
    Class for file handling

    Args:
        open2read (bool)
        hdf_band (int)
        check_corrupted (bool)

    Attributes:
        band (GDAL object)
        datasource (GDAL object)
        chunk_size (int)

    Methods:
        build_overviews
        get_band
        write_array
        close_band
        close_file
        close_all
        get_chunk_size
        remove_overviews

    Returns:
        None
    """

    def get_image_info(self, open2read, hdf_band, check_corrupted):

        self.hdf_file = False
        self.check_corrupted = check_corrupted

        # HDF subdatasets given as files in `vrt_builder`.
        #   Find the file name in the subdataset name.
        if '.hdf' in self.file_name.lower() and not self.file_name.lower().endswith('.hdf'):

            stris = [stri for stri, strt in enumerate(self.file_name) if strt == ':']
            self.file_name = self.file_name[stris[1]+2:stris[2]-1]

        if not os.path.isfile(self.file_name):
            raise IOError('\n{} does not exist.\n'.format(self.file_name))

        self._get_file_format(self.file_name)

        # Open input raster.
        try:

            if open2read:

                self.datasource = gdal.Open(self.file_name, GA_ReadOnly)
                self.image_mode = 'read only'

            else:

                self.datasource = gdal.Open(self.file_name, GA_Update)
                self.image_mode = 'update'

            self.file_open = True

        except:

            logger.error(gdal.GetLastErrorMsg())
            logger.warning('\nCould not open {}.\n'.format(self.file_name))

            return

        if self.file_name.lower().endswith('.hdf'):

            self.hdf_file = True

            if self.datasource is None:

                logger.warning('\n1) {} appears to be empty.\n'.format(self.file_name))
                return

            # self.hdf_layers = self.datasource.GetSubDatasets()
            self.hdf_layers = self.datasource.GetMetadata('SUBDATASETS')

            self.hdf_key_list = [k for k in list(self.hdf_layers) if '_NAME' in k]

            self.hdf_name_dict = dict()

            for hdf_str in self.hdf_key_list:

                str_digit = hdf_str[hdf_str.find('_')+1:len(hdf_str)-hdf_str[::-1].find('_')-1]

                if len(str_digit) == 1:
                    self.hdf_name_dict[hdf_str.replace(str_digit, '0{}'.format(str_digit))] = self.hdf_layers[hdf_str]
                else:
                    self.hdf_name_dict[hdf_str] = self.hdf_layers[hdf_str]

            self.hdf_name_list = [self.hdf_name_dict[k] for k in sorted(self.hdf_name_dict)]

            # self.hdf_name_list = [self.hdf_layers[k] for k in list(self.hdf_layers) if '_NAME' in k]

            self.hdf_datasources = [self._open_dataset(hdf_name, True) for hdf_name in self.hdf_name_list]

            self.datasource = self.hdf_datasources[hdf_band-1]

            # self.datasource = gdal.Open(self.datasource.GetSubDatasets()[hdf_band - 1][0], GA_ReadOnly)

        self.datasource_info()

    @staticmethod
    def _open_dataset(image_name, open2read):

        """
        Opens the image dataset.
        
        Args:
            image_name (str): The full path, name, and extension of the image to open.
            open2read (bool): Whether to open the image in 'Read Only' mode.

        Returns:
            The datasource pointer.
        """

        if open2read:
            source_dataset = gdal.Open(image_name, GA_ReadOnly)
        else:
            source_dataset = gdal.Open(image_name, GA_Update)

        if source_dataset is None:
            logger.error('{} appears to be empty'.format(image_name))
            raise EmptyImage

        return source_dataset

    def _get_hdr_info(self):

        hdr_file = '{}.hdr'.format(self.file_name)

        if not os.path.isfile(hdr_file):
            return

        with open(hdr_file, 'rb') as hdr_open:

            for line in hdr_open:

                if line.startswith('samples'):

                    line_parsed = line.replace('samples = ', '')

                    self.rows = int(line_parsed)

                elif line.startswith('lines'):

                    line_parsed = line.replace('lines = ', '')

                    self.cols = int(line_parsed)

                elif line.startswith('map info'):

                    line_parsed = line.replace('map info = {', '')
                    line_parsed = line_parsed.replace('}', '').split(',')

                    self.left = float(line_parsed[3].strip())
                    self.top = float(line_parsed[4].strip())
                    self.cellY = float(line_parsed[5].strip())
                    self.cellX = -self.cellY

                elif line.startswith('coordinate'):

                    self.projection = line.replace('coordinate system string = {', '')
                    self.projection = self.projection.replace('}\n', '')

    def build_overviews(self, sampling_method='nearest', levels=None, be_quiet=False):

        """
        Builds image overviews.

        Args:
            sampling_method (Optional[str]): The sampling method to use. Default is 'nearest'.
            levels (Optional[int list]): The levels to build. Default is [2, 4, 8, 16].
            be_quiet (Optional[bool]): Whether to be quiet and do not print progress. Default is False.
        """

        if not levels:
            levels = [2, 4, 8, 16]
        else:
            levels = list(map(int, levels))

        try:

            if not be_quiet:
                logger.info('  Building pyramid overviews ...')

            self.datasource.BuildOverviews(sampling_method.upper(), overviewlist=levels)

        except:

            logger.error(gdal.GetLastErrorMsg())
            raise ValueError('Failed to build overviews.')

    def get_band(self, band_position):

        """
        Loads a raster band pointer.

        Args:
            band_position (int): The band position to load.
        """

        if not isinstance(band_position, int) or band_position < 1:

            logger.error('The band position must be an integer > 0.')
            raise ValueError

        try:

            self.band = self.datasource.GetRasterBand(band_position)
            self.band_open = True

        except:

            logger.error(gdal.GetLastErrorMsg())
            raise ValueError('Failed to load the band.')

    def get_stats(self, band_position):

        """
        Get band statistics.
        
        Args:
            band_position (int)
        
        Returns:
            Minimum, Maximum, Mean, Standard deviation
        """

        self.get_band(band_position)

        return self.band.GetStatistics(1, 1)

    def check_corrupted_bands(self):

        """Checks whether corrupted bands exist."""

        self.corrupted_bands = list()

        for band in range(1, self.bands+1):

            try:
                self.datasource.GetRasterBand(band).Checksum()

                if gdal.GetLastErrorType() != 0:

                    logger.info('\nBand {:d} of {} appears to be corrupted.\n'.format(band, self.file_name))
                    self.corrupted_bands.append(str(band))

            except:

                logger.error(gdal.GetLastErrorMsg())

                logger.info('\nBand {:d} of {} appears to be corrupted.\n'.format(band, self.file_name))
                self.corrupted_bands.append(str(band))

    def write_array(self, array2write, i=0, j=0, band=None):

        """
        Writes array to the loaded band object (``self.band`` of ``get_band``).

        Args:
            array2write (ndarray): The array to write.
            i (Optional[int]): The starting row position to write to. Default is 0.
            j (Optional[int]): The starting column position to write to. Default is 0.
            band (Optional[int]): The band position to write to. Default is None. If None, an object of
                ``get_band`` must be open.
        """

        if not isinstance(array2write, np.ndarray):

            logger.error('  The array must be an ndarray.')
            raise ValueError

        if not isinstance(i, int) or (i < 0):

            logger.error('  The row index must be a positive integer.')
            raise ValueError

        if not isinstance(j, int) or (j < 0):

            logger.error('  The column index must be a positive integer.')
            raise ValueError

        if isinstance(band, int):
            self.get_band(band_position=band)

        try:
            self.band.WriteArray(array2write, j, i)
        except:

            logger.error(gdal.GetLastErrorMsg())

            if (array2write.shape[0] > self.rows) or (array2write.shape[1] > self.cols):

                logger.error('\nThe array is larger than the file size.\n')
                raise ArrayShapeError

            elif (i + array2write.shape[0]) > self.rows:

                logger.error('\nThe starting row position + the array rows spills over.\n')
                raise ArrayOffsetError

            elif (j + array2write.shape[j]) > self.cols:

                logger.error('\nThe starting column position + the array columns spills over.\n')
                raise ArrayOffsetError

            else:

                if not hasattr(self, 'band'):

                    logger.error('\nThe band must be set either with `get_band` or `write_array`.\n')
                    raise AttributeError

                else:
                    logger.error('\nFailed to write the array to file (issue not apparent).')

    def close_band(self):

        """Closes a band object"""

        if hasattr(self, 'band') and self.band_open:

            # try:
            #     self.band.SetColorInterpretation(self.color_interpretation)
            #     self.band.SetRasterColorInterpretation(self.color_interpretation)
            # except:
            #     logger.warning('The band color could not be set.')
            #     logger.error(gdal.GetLastErrorMsg())
            #     pass

            try:
                self.band.GetStatistics(0, 1)
            except:

                logger.warning('The band statistics could not be calculated.')
                logger.warning(gdal.GetLastErrorMsg())

            try:
                self.band.FlushCache()
            except:

                logger.warning('The band statistics could not be flushed.')
                logger.warning(gdal.GetLastErrorMsg())

        self.band = None
        self.band_open = False

    def close_file(self):

        """Closes a file object"""

        if hasattr(self, 'datasource'):

            if hasattr(self, 'hdf_file'):

                if self.hdf_file:

                    if self.hdf_datasources:

                        for hdfd in range(0, len(self.hdf_datasources)):

                            if hasattr(self.hdf_datasources[hdfd], 'FlushCache'):

                                self.hdf_datasources[hdfd].FlushCache()
                                self.hdf_datasources[hdfd] = None

                            # try:
                            #     hdfd.FlushCache()
                            # except:
                            #
                            #     logger.warning('The HDF subdataset could not be flushed.')
                            #     logger.error(gdal.GetLastErrorMsg())
                            #
                            #     pass
                            # hdfd = None

            if hasattr(self.datasource, 'FlushCache'):

                try:
                    self.datasource.FlushCache()
                except:
                    logger.warning('The dataset could not be flushed.')

        if hasattr(self, 'output_image'):

            # Unlink memory images
            if self.output_image.lower().endswith('.mem'):

                gdal.Unlink(self.output_image)

                try:
                    os.remove(self.output_image)
                except:
                    pass

        self.datasource = None
        self.hdf_datasources = None
        self.file_open = False

    def close_all(self):

        """Closes a band object and a file object"""

        self.close_band()
        self.close_file()

    def fill(self, fill_value, band=None):

        """
        Fills a band with a specified value.

        Args:
            fill_value (int): The value to fill.
            band (Optional[int]): The band to fill. Default is None.
        """

        if isinstance(band, int):
            self.get_band(band_position=band)

        self.band.Fill(fill_value)

    def get_chunk_size(self):

        """Gets the band block size"""

        try:
            self.chunk_size = self.band.GetBlockSize()[0]
        except:
            raise IOError('\nFailed to get the block size.\n')

    def remove_overviews(self):

        """Removes image overviews"""

        if self.image_mode != 'update':
            raise NameError('\nOpen the image in update mode (open2read=False) to remove overviews.\n')
        else:
            self.build_overviews(levels=[])

    def calculate_stats(self, band=1):

        """
        Calculates image statistics and can be used to check for empty images.

        Args:
            band (Optional[int])
        """

        self.get_band(band_position=band)

        image_metadata = self.band.GetMetadata()

        use_exceptions = gdal.GetUseExceptions()
        gdal.UseExceptions()

        try:

            image_min, image_max, image_mu, image_std = self.band.GetStatistics(False, True)

            image_metadata['STATISTICS_MINIMUM'] = repr(image_min)
            image_metadata['STATISTICS_MAXIMUM'] = repr(image_max)
            image_metadata['STATISTICS_MEAN'] = repr(image_mu)
            image_metadata['STATISTICS_STDDEV'] = repr(image_std)

            image_metadata['STATISTICS_SKIPFACTORX'] = '1'
            image_metadata['STATISTICS_SKIPFACTORY'] = '1'

            if not use_exceptions:
                gdal.DontUseExceptions()

            self.band.SetMetadata(image_metadata)

            return True

        except:

            logger.error(gdal.GetLastErrorMsg())

            if not use_exceptions:
                gdal.DontUseExceptions()

            return False


class UpdateInfo(object):

    """A class for updating attributes"""

    def update_info(self, **kwargs):

        for k, v in viewitems(kwargs):
            setattr(self, k, v)


class ImageInfo(UpdateInfo, ReadWrite, FileManager, DatasourceInfo):

    """An empty class for passing image information"""

    def __init__(self):
        pass

    def copy(self):
        return copy.copy(self)


class LandsatParser(object):

    """
    A class to parse Landsat metadata

    Args:
        metadata (str)
        band_order (Optional[list])
    """

    def __init__(self, metadata, band_order=[]):

        self.bo = copy.copy(band_order)

        if metadata.endswith('MTL.txt'):
            self.parse_mtl(metadata)
        elif metadata.endswith('.xml'):
            self.parse_xml(metadata)
        else:
            raise NameError('Parser type not supported')

    def _cleanup(self):

        if os.path.isdir(self.temp_dir):

            for landsat_file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, landsat_file))

            shutil.rmtree(self.temp_dir)

    def parse_mtl(self, metadata):

        df = pd.read_table(metadata, header=None, sep='=')
        df.rename(columns={0: 'Variable', 1: 'Value'}, inplace=True)

        df['Variable'] = df['Variable'].str.strip()
        df['Value'] = df['Value'].str.strip()

        self.scene_id = df.loc[df['Variable'] == 'LANDSAT_SCENE_ID', 'Value'].values[0].replace('"', '').strip()

        if not df.loc[df['Variable'] == 'DATE_ACQUIRED', 'Value'].empty:
            self.date = df.loc[df['Variable'] == 'DATE_ACQUIRED', 'Value'].values[0].replace('"', '').strip()
        else:
            self.date = df.loc[df['Variable'] == 'ACQUISITION_DATE', 'Value'].values[0].replace('"', '').strip()

        self.date_ = self.date.split('-')
        self.year = self.date_[0]
        self.month = self.date_[1]
        self.day = self.date_[2]

        self.sensor = df.loc[df['Variable'] == 'SENSOR_ID', 'Value'].values[0].replace('"', '').strip()
        self.series = df.loc[df['Variable'] == 'SPACECRAFT_ID', 'Value'].values[0].replace('"', '').strip()

        self.path = df.loc[df['Variable'] == 'WRS_PATH', 'Value'].astype(int).astype(str).values[0].strip()

        if not df.loc[df['Variable'] == 'WRS_ROW', 'Value'].empty:
            self.row = df.loc[df['Variable'] == 'WRS_ROW', 'Value'].astype(int).astype(str).values[0].strip()
        else:
            self.row = df.loc[df['Variable'] == 'STARTING_ROW', 'Value'].astype(int).astype(str).values[0].strip()

        self.elev = df.loc[df['Variable'] == 'SUN_ELEVATION', 'Value'].astype(float).values[0]
        self.zenith = 90. - self.elev
        self.azimuth = df.loc[df['Variable'] == 'SUN_AZIMUTH', 'Value'].astype(float).values[0]
        self.cloudCover = df.loc[df['Variable'] == 'CLOUD_COVER', 'Value'].astype(float).astype(str).values[0].strip()

        try:
            self.imgQuality = df.loc[df['Variable'] == 'IMAGE_QUALITY', 'Value'].astype(int).astype(str).values[0].strip()
        except:

            self.img_quality_oli = df.loc[df['Variable'] ==
                                          'IMAGE_QUALITY_OLI', 'Value'].astype(int).astype(str).values[0].strip()

            self.img_quality_tirs = df.loc[df['Variable'] ==
                                           'IMAGE_QUALITY_TIRS', 'Value'].astype(int).astype(str).values[0].strip()

        self.LMAX_dict = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0.}
        self.LMIN_dict = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0.}
        self.no_coeff = 999

        # Landsat 8 radiance
        self.rad_mult_dict = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0., 10: 0., 11: 0.}
        self.rad_add_dict = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0., 10: 0., 11: 0.}

        # Landsat 8 reflectance
        self.refl_mult_dict = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0.}
        self.refl_add_dict = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0., 9: 0.}

        self.k1 = 0
        self.k2 = 0

        if self.sensor.lower() == 'oli_tirs':

            if not self.bo:
                self.bo = [2, 3, 4, 5, 6, 7]

        else:

            if not self.bo:
                self.bo = [1, 2, 3, 4, 5, 7]

        for bi in self.bo:

            if not df.loc[df['Variable'] == 'RADIANCE_MAXIMUM_BAND_{:d}'.format(bi), 'Value'].empty:

                self.LMAX_dict[bi] = df.loc[df['Variable'] == 'RADIANCE_MAXIMUM_BAND_{:d}'.format(bi),
                                            'Value'].astype(float).values[0]

                self.LMIN_dict[bi] = df.loc[df['Variable'] == 'RADIANCE_MINIMUM_BAND_{:d}'.format(bi),
                                            'Value'].astype(float).values[0]

                self.no_coeff = 1000

            else:

                self.LMAX_dict[bi] = df.loc[df['Variable'] == 'LMAX_BAND{:d}'.format(bi),
                                            'Value'].astype(float).values[0]

                self.LMIN_dict[bi] = df.loc[df['Variable'] == 'LMIN_BAND{:d}'.format(bi),
                                            'Value'].astype(float).values[0]

                self.no_coeff = 1000

            if self.sensor.lower() == 'oli_tirs':

                self.rad_mult_dict[bi] = df.loc[df['Variable'] == 'RADIANCE_MULT_BAND_{:d}'.format(bi),
                                                'Value'].astype(float).values[0]

                self.rad_add_dict[bi] = df.loc[df['Variable'] == 'RADIANCE_ADD_BAND_{:d}'.format(bi),
                                               'Value'].astype(float).values[0]

                self.refl_mult_dict[bi] = df.loc[df['Variable'] == 'REFLECTANCE_MULT_BAND_{:d}'.format(bi),
                                                'Value'].astype(float).values[0]

                self.refl_add_dict[bi] = df.loc[df['Variable'] == 'REFLECTANCE_ADD_BAND_{:d}'.format(bi),
                                               'Value'].astype(float).values[0]

                # TODO: add k1 and k2 values
                # self.k1 =
                # self.k2 =

    def parse_xml(self, metadata):

        with open(metadata) as mo:

            meta = mo.read()

            soup = BeautifulSoup(meta)

            wrs = soup.find('wrs')

            try:
                self.product_id = str(soup.find('product_id').text.strip())
            except:
                self.product_id = None

            self.scene_id = soup.find('lpgs_metadata_file').text
            si_index = self.scene_id.find('_')
            self.scene_id = self.scene_id[:si_index].strip()

            self.path = wrs['path'].strip()

            self.row = wrs['row'].strip()

            self.sensor = soup.find('instrument').text.strip()

            self.series = soup.find('satellite').text.strip()

            self.date = soup.find('acquisition_date').text.strip()

            self.date_ = self.date.split('-')
            self.year = self.date_[0].strip()
            self.month = self.date_[1].strip()
            self.day = self.date_[2].strip()

            solar_angles = soup.find('solar_angles')

            self.solar_zenith = float(solar_angles['zenith'].strip())
            self.solar_azimuth = float(solar_angles['azimuth'].strip())
            self.solar_elevation = 90. - self.solar_zenith


class SentinelParser(object):

    """A class to parse Sentinel 2 metadata"""

    def parse_xml(self, metadata):

        """
        Args:
            metadata (str)
            mgrs (Optional[str])
        """

        self.complete = False

        # xmltodict
        try:
            import xmltodict
        except ImportError:
            raise ImportError('xmltodict must be installed to parse Sentinel data')

        if not metadata.endswith('.xml'):

            logger.warning('  Parser type not supported')
            return

        if metadata.endswith('_report.xml'):

            logger.warning('  Cannot process <report> files.')
            return

        with open(metadata) as xml_tree:
            xml_object = xmltodict.parse(xml_tree.read())

        safe_dir = os.path.split(metadata)[0]

        self.level = '1C' if 'n1:Level-1C_User_Product' in list(xml_object) else '2A'

        base_xml = xml_object['n1:Level-{LEVEL}_User_Product'.format(LEVEL=self.level)]

        general_info = base_xml['n1:General_Info']

        quality_info = base_xml['n1:Quality_Indicators_Info']

        self.cloud_cover = float(quality_info['Cloud_Coverage_Assessment'])

        product_info = general_info['L{LEVEL}_Product_Info'.format(LEVEL=self.level)]

        self.year, self.month, self.day = product_info['GENERATION_TIME'][:10].split('-')

        # self.band_list = product_info['Query_Options']['Band_List']
        # self.band_list = [bn for bn in self.band_list['BAND_NAME']]

        granule_list = product_info['L{LEVEL}_Product_Organisation'.format(LEVEL=self.level)]['Granule_List']

        self.band_name_dict = dict()

        for granule_index in range(0, len(granule_list)):

            granule_key = 'Granule' if 'Granule' in granule_list[granule_index] else 'Granules'

            image_key = 'IMAGE_FILE_2A' if 'IMAGE_FILE_2A' in granule_list[granule_index][granule_key] else 'IMAGE_ID_2A'

            granule_identifier = granule_list[granule_index][granule_key]['@granuleIdentifier']

            img_data_dir = os.path.join(safe_dir, 'GRANULE', granule_identifier, 'IMG_DATA')
            qi_data_dir = os.path.join(safe_dir, 'GRANULE', granule_identifier, 'QI_DATA')

            # List of image names
            granule_image_list = granule_list[granule_index][granule_key][image_key]

            mgrs_code = granule_image_list[0][-13:-8]

            # Check if the file name has 20m.
            if '20m' in granule_image_list[0]:

                granule_image_list_full = list()

                for granule_image in granule_image_list:

                    if '_CLD_' in granule_image:
                        granule_image_list_full.append(os.path.join(qi_data_dir, granule_image))
                    else:
                        granule_image_list_full.append(os.path.join(img_data_dir, 'R20m', granule_image))

                self.band_name_dict['{MGRS}-20m'.format(MGRS=mgrs_code)] = granule_image_list_full

            elif '10m' in granule_image_list[0]:

                granule_image_list_full = list()

                for granule_image in granule_image_list:

                    if '_CLD_' in granule_image:
                        granule_image_list_full.append(os.path.join(qi_data_dir, granule_image))
                    else:
                        granule_image_list_full.append(os.path.join(img_data_dir, 'R10m', granule_image))

                self.band_name_dict['{MGRS}-10m'.format(MGRS=mgrs_code)] = granule_image_list_full

            image_format = granule_list[granule_index][granule_key]['@imageFormat']

        # self.granule_dict = dict()
        #
        # for granule in granule_list:
        #
        #     tile = granule['Granules']
        #     tile_id = tile['@granuleIdentifier']
        #     image_ids = tile['IMAGE_ID']
        #
        #     image_format = tile['@imageFormat']
        #
        #     self.granule_dict[tile_id] = image_ids

        self.image_ext = FORMAT_DICT[image_format]

        # print self.granule_dict

        # self.level = product_info['PROCESSING_LEVEL']
        self.product = product_info['PRODUCT_TYPE']

        self.series = product_info['Datatake']['SPACECRAFT_NAME']

        self.no_data = int(general_info['L{LEVEL}_Product_Image_Characteristics'.format(LEVEL=self.level)]['Special_Values'][0]['SPECIAL_VALUE_INDEX'])
        self.saturated = int(general_info['L{LEVEL}_Product_Image_Characteristics'.format(LEVEL=self.level)]['Special_Values'][1]['SPECIAL_VALUE_INDEX'])

        self.complete = True


class ropen(FileManager, LandsatParser, SentinelParser, UpdateInfo, ReadWrite):

    """
    Gets image information and returns a file pointer object.

    Args:
        file_name (Optional[str]): Image location, name, and extension. Default is 'none'.
        open2read (Optional[bool]): Whether to open image as 'read only' (True) or writeable (False).
            Default is True.
        metadata (Optional[str]): A metadata file. Default is None.
        sensor (Optional[str]): The satellite sensor to parse with ``metadata``. Default is 'Landsat'. Choices are
            ['Landsat', 'Sentinel2']. This is only used for inplace spectral transformations. It will not
            affect the image otherwise.
        hdf_band (Optional[int])

    Attributes:
        file_name (str)
        datasource (object)
        directory (str)
        filename (str)
        bands (int)
        projection (str)
        geo_transform (list)
        left (float)
        top (float)
        right (float)
        bottom (float)
        cellY (float)
        cellX (float)
        rows (int)
        cols (int)
        shape (str)
        name (str)
        block_x (int)
        block_y (int)

    Returns:
        None

    Examples:
        >>> # typical usage
        >>> import mpglue as gl
        >>>
        >>> i_info = mp.ropen('/some_raster.tif')
        >>> # <ropen> has its own array instance
        >>> i_info = mp.open('/some_raster.tif')
        >>> # <rinfo> has its own array instance
        >>> array = i_info.read()    # opens band 1, all rows and columns
        >>> print array
        >>>
        >>> # use the <read> function
        >>> # open specific rows and columns
        >>> array = mp.read(i_info,
        >>>                 bands2open=[-1],
        >>>                 i=100, j=100,
        >>>                 rows=500, cols=500)
        >>>
        >>> # compute the NDVI (for Landsat-like band channels only)
        >>> i_info.read(compute_index='ndvi')
        >>> print i_info.ndvi
        >>> print i_info.array.shape    # note that the image array is a 2xrowsxcolumns array
        >>> # display the NDVI
        >>> i_info.show('ndvi')
        >>> # display band 1 of the image (band 1 of <array> is the red band)
        >>> i_info.show(band=1)
        >>> # write the NDVI to file
        >>> i_info.write2raster('/ndvi.tif', write_which='ndvi', \
        >>>                     o_info=i_info.copy(), storage='float32')
        >>>
        >>> # write an array to file
        >>> array = np.random.randn(3, 1000, 1000)
        >>> i_info.write2raster('/array.tif', write_which=array, \
        >>>                     o_info=i_info.copy(), storage='float32')
        >>>
        >>> # create info from scratch
        >>> i_info = mp.ropen('create', left=, right=, top=, bottom=, \
        >>> i_info = mp.open('create', left=, right=, top=, bottom=, \
        >>>                   cellY=, cellX=, bands=, storage=, projection=, \
        >>>                   rows=, cols=)
        >>>
        >>> # build overviews
        >>> i_info = mp.ropen('/some_raster.tif')
        >>> i_info = mp.open('/some_raster.tif')
        >>> i_info.build_overviews()
        >>> i_info.close()
        >>>
        >>> # remove overviews
        >>> i_info = mp.ropen('/some_raster.tif', open2read=False)
        >>> i_info = mp.open('/some_raster.tif', open2read=False)
        >>> i_info.remove_overviews()
        >>> i_info.close()
    """

    def __init__(self,
                 file_name='none',
                 open2read=True,
                 metadata=None,
                 sensor='Landsat',
                 hdf_band=1,
                 check_corrupted=False,
                 **kwargs):

        self.file_name = os.path.normpath(file_name)

        passed = True

        if file_name == 'create':
            self.update_info(**kwargs)
        elif file_name != 'none':
            self.get_image_info(open2read, hdf_band, check_corrupted)
        else:
            passed = False

        if isinstance(metadata, str):
            self.get_metadata(metadata, sensor)
        else:

            if not passed:
                logger.warning('  No image or metadata file was given.')

        # Check open files before closing.
        # atexit.register(self.close)

    def get_metadata(self, metadata, sensor):

        """
        Args:
            metadata (str): The metadata file.
            sensor (str): The satellite sensor to search. Default is 'Landsat'. Choices are ['Landsat', 'Sentinel2'].
        """

        if sensor == 'Landsat':
            LandsatParser.__init__(self, metadata)
        elif sensor == 'Sentinel2':
            SentinelParser.__init__(self, metadata)
        else:

            logger.error('The {} sensor is not an option.'.format(sensor))
            raise NameError

    def copy(self):
        return copy.copy(self)

    def close(self):

        """Closes the dataset"""
        
        self.close_all()

    def warp(self, output_image, epsg, resample='nearest', cell_size=0., **kwargs):

        """
        Warp transforms a dataset

        Args:
            output_image (str): The output image.
            epsg (int): The output EPSG projection code.
            resample (Optional[str]): The resampling method. Default is 'nearest'.N
            cell_size (Optional[float]): The output cell size. Default is 0.
            kwargs:
                format='GTiff', outputBounds=None (minX, minY, maxX, maxY), 
                outputBoundsSRS=None, targetAlignedPixels=False,
                width=0, height=0, srcAlpha=False, dstAlpha=False, warpOptions=None,
                errorThreshold=None, warpMemoryLimit=None,
                creationOptions=None, outputType=0, workingType=0,
                resampleAlg=resample_dict[resample], srcNodata=None, dstNodata=None,
                multithread=False, tps=False, rpc=False, geoloc=False,
                polynomialOrder=None, transformerOptions=None, cutlineDSName=None,
                cutlineLayer=None, cutlineWhere=None, cutlineSQL=None,
                cutlineBlend=None, cropToCutline=False, copyMetadata=True,
                metadataConflictValue=None, setColorInterpretation=False,
                callback=None, callback_data=None

        Returns:
            None, writes to `output_image'.
        """

        warp_options = gdal.WarpOptions(srcSRS=None, dstSRS='EPSG:{:d}'.format(epsg),
                                        xRes=cell_size, yRes=cell_size,
                                        resampleAlg=RESAMPLE_DICT[resample],
                                        **kwargs)

        out_ds = gdal.Warp(output_image, self.file_name, options=warp_options)

        out_ds = None

    def translate(self, output_image, cell_size=0, **kwargs):

        """
        Args:
            output_image (str): The output image.
            cell_size (Optional[float]): The output cell size. Default is 0.
            kwargs:
                format='GTiff', outputType=0, bandList=None, maskBand=None, width=0, height=0,
                widthPct=0.0, heightPct=0.0, xRes=0.0, yRes=0.0, creationOptions=None, srcWin=None,
                projWin=None, projWinSRS=None, strict=False, unscale=False, scaleParams=None,
                exponents=None, outputBounds=None, metadataOptions=None, outputSRS=None, GCPs=None,
                noData=None, rgbExpand=None, stats=False, rat=True, resampleAlg=None,
                callback=None, callback_data=None
        """

        translate_options = gdal.TranslateOptions(xRes=cell_size, yRes=cell_size, **kwargs)

        out_ds = gdal.Translate(output_image, self.file_name, options=translate_options)

        out_ds = None

    def hist(self,
             input_array=None,
             band=1,
             i=0,
             j=0,
             rows=-1,
             cols=-1,
             d_type='byte',
             name_dict=None,
             bins=256,
             **kwargs):

        """
        Prints the image histogram

        Args:
            input_array (Optional[2d array]): An array to get the histogram from, otherwise, open the array.
            band (Optional[int]): The band to get the histogram from.
            i (Optional[int]): The starting row position.
            j (Optional[int]): The starting column position.
            rows (Optional[int]): The number of rows to take.
            cols (Optional[int]): The number of columns to take.
            d_type (Optional[str]): The image data type.
            name_dict (Optional[dict]): A dictionary of {value: 'name'} for discrete value arrays.
            bins (Optional[int]): The number of bins.
            kwargs:
                Other arguments passed to `numpy.histogram`.
                range (Optional[tuple]): The histogram range.
                normed (Optional[bool])
                weights
                density

        Example:
            >>> import mpglue as gl
            >>>
            >>> i_info = gl.ropen('image_name.tif')
            >>>
            >>> i_info.hist()
            >>>
            >>> # Print the histogram dictionary.
            >>> print(i_info.hist_dict)
        """

        if 'range' not in kwargs:
            kwargs['range'] = (0, bins-1)

        if isinstance(input_array, np.ndarray):

            the_hist, bin_edges = np.histogram(input_array,
                                               bins=bins,
                                               **kwargs)

        elif hasattr(self, 'array') and not isinstance(input_array, np.ndarray):

            the_hist, bin_edges = np.histogram(self.array,
                                               bins=bins,
                                               **kwargs)

        else:

            the_hist, bin_edges = np.histogram(self.read(bands2open=band,
                                                         i=i,
                                                         j=j,
                                                         rows=rows,
                                                         cols=cols,
                                                         d_type=d_type),
                                               bins=bins,
                                               **kwargs)

        if kwargs['range'][0] == 0:
            self.total_samples = float(the_hist[1:].sum())
        else:
            self.total_samples = float(the_hist.sum())

        the_hist_pct = (the_hist / self.total_samples) * 100.

        self.hist_dict = dict()

        for i in range(0, bins):

            if the_hist[i] > 0:

                if isinstance(name_dict, dict):

                    if i not in name_dict:
                        label = 'unknown'
                    else:
                        label = name_dict[i]

                    self.hist_dict[i] = dict(value=i,
                                             name=label,
                                             count=the_hist[i],
                                             perc=round(the_hist_pct[i], 4))

                else:

                    self.hist_dict[i] = dict(value=i,
                                             count=the_hist[i],
                                             perc=round(the_hist_pct[i], 4))

        # Sort the values, largest to smallest
        self.hist_dict = OrderedDict(sorted(list(iteritems(self.hist_dict)),
                                            key=lambda item: item[1]['count'],
                                            reverse=True))

    def pca(self, n_components=3):

        """
        Computes Principle Components Analysis

        Args:
            n_components (Optional[int]): The number of components to return. Default is 3.

        Attributes:
            pca_components (ndarray)

        Returns:
            None
        """

        # Scikit-learn
        try:
            from sklearn import decomposition
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError('Scikit-learn must be installed to run PCA')

        if n_components > self.bands:
            n_components = self.bands

        embedder = decomposition.PCA(n_components=n_components)

        dims, rs, cs = self.array.shape

        x = self.array.T.reshape(rs*cs, dims)

        scaler = StandardScaler().fit(x)
        x = scaler.transform(x.astype(np.float32)).astype(np.float32)

        x_fit = embedder.fit(x.astype(np.float32))
        x_reduced = x_fit.transform(x)

        self.pca_components = x_reduced.reshape(cs, rs, n_components).T

    def show(self,
             show_which='array',
             band=1,
             color_map='gist_stern',
             discrete=False,
             class_list=None,
             out_fig=None,
             dpi=300,
             clip_percentiles=(2, 98),
             equalize_hist=False,
             equalize_adapthist=False,
             gammas=None,
             sigmoid=None):

        """
        Displays an array

        Args:
            show_which (Optional[str]): Which array to display. Default is 'array'. Choices are ['array',
                'evi2', 'gndvi', 'ndbai', 'ndvi', 'ndwi', 'savi'].
            band (Optional[int]): The band to display. Default is 1.
            color_map (Optional[str]): The colormap to use. Default is 'gist_stern'. For more colormaps, visit
                http://matplotlib.org/examples/color/colormaps_reference.html.
            discrete (Optional[bool]): Whether the colormap is discrete. Otherwise, continuous. Default is False.
            class_list (Optional[int list]): A list of the classes to display. Default is [].
            out_fig (Optional[str]): An output image to save to. Default is None.
            dpi (Optional[int]): The DPI of the output figure. Default is 300.
            clip_percentiles (Optional[tuple]): The lower and upper clip percentiles to rescale RGB images.
                Default is (2, 98).
            equalize_hist (Optional[bool]): Whether to equalize the histogram. Default is False.
            equalize_adapthist (Optional[bool]): Whether to equalize the histogram using a localized approach.
                Default is False.
            gammas (Optional[float list]): A list of gamma corrections for each band. Default is [].
            sigmoid (Optional[float list]): A list of sigmoid contrast and gain values. Default is [].

        Examples:
            >>> import mpglue as gl
            >>> i_info = mp.ropen('image')
            >>> i_info = mp.open('image')
            >>>
            >>> # Plot a discrete map with specified colors
            >>> color_map = ['#000000', '#DF7401', '#AEB404', '#0B6121', '#610B0B', '#A9D0F5',
            >>>              '#8181F7', '#BDBDBD', '#3A2F0B', '#F2F5A9', '#5F04B4']
            >>> i_info.show(color_map=color_map, discrete=True,
            >>>             class_list=[0,1,2,3,4,5,6,7,8,9,10])
            >>>
            >>> # Plot the NDVI
            >>> i_info.read(compute_index='ndvi')
            >>> i_info.show(show_which='ndvi')
            >>>
            >>> # Plot a single band array as greyscale
            >>> i_info.read(bands2open=4)
            >>> i_info.show(color_map='Greys')
            >>>
            >>> # Plot a 3-band array as RGB true color
            >>> i_info.read(bands2open=[3, 2, 1], sort_bands2open=False)
            >>> i_info.show(band='rgb')

        Returns:
            None
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.axis('off')

        if show_which == 'ndvi':

            self.array[self.array != 0] += 1.1

            if equalize_hist:
                self.array = exposure.equalize_hist(self.array)

            ip = ax.imshow(self.array)
            im_min = np.percentile(self.array, clip_percentiles[0])
            im_max = np.percentile(self.array, clip_percentiles[1])

        elif show_which == 'evi2':

            if equalize_hist:
                self.array = exposure.equalize_hist(self.array)

            ip = ax.imshow(self.array)
            im_min = np.percentile(self.array, clip_percentiles[0])
            im_max = np.percentile(self.array, clip_percentiles[1])

        elif show_which == 'pca':

            if equalize_hist:
                self.pca_components[band-1] = exposure.equalize_hist(self.pca_components[band-1])

            ip = ax.imshow(self.pca_components[band-1])
            im_min = np.percentile(self.pca_components[band-1], clip_percentiles[0])
            im_max = np.percentile(self.pca_components[band-1], clip_percentiles[1])

        else:

            if self.array_shape[0] > 1:

                if band == 'rgb':

                    for ii, im in enumerate(self.array):

                        pl, pu = np.percentile(im, clip_percentiles)
                        self.array[ii] = exposure.rescale_intensity(im, in_range=(pl, pu), out_range=(0, 255))

                        if equalize_hist:
                            self.array[ii] = exposure.equalize_hist(im)

                        if equalize_adapthist:
                            self.array[ii] = exposure.equalize_adapthist(im, ntiles_x=4, ntiles_y=4, clip_limit=0.5)

                        if gammas:
                            self.array[ii] = exposure.adjust_gamma(im, gammas[ii])

                        if sigmoid:
                            self.array[ii] = exposure.adjust_sigmoid(im, cutoff=sigmoid[0], gain=sigmoid[1])

                    # ip = ax.imshow(cv2.merge([self.array[2], self.array[1], self.array[0]]))
                    ip = ax.imshow(np.ascontiguousarray(self.array.transpose(1, 2, 0)))
                    # ip = ax.imshow(np.dstack((self.array[0], self.array[1], self.array[2])), interpolation='nearest')

                else:

                    ip = ax.imshow(self.array[band-1])
                    im_min = np.percentile(self.array[band-1], clip_percentiles[0])
                    im_max = np.percentile(self.array[band-1], clip_percentiles[1])

            else:

                ip = ax.imshow(self.array)
                im_min = np.percentile(self.array, clip_percentiles[0])
                im_max = np.percentile(self.array, clip_percentiles[1])

        ip.axes.get_xaxis().set_visible(False)
        ip.axes.get_yaxis().set_visible(False)

        if discrete:

            if isinstance(color_map, list):
                color_map = colors.ListedColormap(color_map)
                # color_map = colorbar.ColorbarBase(ax, cmap=color_map_)
                ip.set_cmap(color_map)
            elif color_map.lower() == 'random':
                ip.set_cmap(colors.ListedColormap(np.random.rand(len(class_list), 3)))
            else:
                ip.set_cmap(_discrete_cmap(len(class_list), base_cmap=color_map))

            ip.set_clim(min(class_list), max(class_list))

        else:
            if band != 'rgb':
                ip.set_cmap(color_map)
                ip.set_clim(im_min, im_max)

        cbar = plt.colorbar(ip, fraction=0.046, pad=0.04, orientation='horizontal')

        cbar.solids.set_edgecolor('face')

        # Remove colorbar container frame
        cbar.outline.set_visible(False)

        # cbar.set_ticks([])
        # cbar.set_ticklabels(class_list)

        if band == 'rgb':
            colorbar_label = 'RGB'
        else:
            if show_which == 'array':
                colorbar_label = 'Band {:d} of {:d} bands'.format(band, self.array_shape[0])
            else:
                colorbar_label = show_which.upper()

        cbar.ax.set_xlabel(colorbar_label)

        # Remove color bar tick lines, while keeping the tick labels
        cbarytks = plt.getp(cbar.ax.axes, 'xticklines')
        plt.setp(cbarytks, visible=False)

        if isinstance(out_fig, str):
            plt.savefig(out_fig, dpi=dpi, bbox_inches='tight', pad_inches=.1, transparent=True)
        else:
            plt.show()

        if show_which == 'ndvi':
            self.array[self.array != 0] -= 1.1

        plt.clf()
        plt.close(fig)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def __del__(self):
        self.__exit__(None, None, None)


class PanSharpen(object):

    """
    A class to pan sharpen an image

    Args:
        multi_image (str)
        pan_array (str)
        out_dir (Optional[str])
        method (Optional[str])

    Equation:
        DNF = (P - IW * I) / (RW * R + GW * G + BW * B)
        Red_out = R * DNF
        Green_out = G * DNF
        Blue_out = B * DNF
        Infrared_out = I * DNF

        pan / ((rgb[0] + rgb[1] + rgb[2] * weight) / (2 + weight))

    Example:
        >>> ps = PanSharpen('/multi.tif', '/pan.tif', method='brovey')
        >>> ps.sharpen()
    """

    def __init__(self, multi_image, pan_image, out_dir=None, method='brovey'):

        self.multi_image = multi_image
        self.pan_image = pan_image
        self.method = method

        if isinstance(out_dir, str):
            self.out_dir = out_dir
        else:
            self.out_dir = os.path.split(self.multi_image)[0]

        f_name = os.path.split(self.multi_image)[1]
        self.f_base, self.f_ext = os.path.splitext(f_name)

        self.multi_image_ps = os.path.join(self.out_dir, '{}_pan.tif'.format(self.f_base))

    def sharpen(self, bw=.2, gw=1., rw=1., iw=.5):

        self.bw = bw
        self.gw = gw
        self.rw = rw
        self.iw = iw

        # self._sharpen_gdal()

        self._warp_multi()
        self._sharpen()

    def _sharpen_gdal(self):

        with ropen(self.multi_image) as m_info:
            m_bands = m_info.bands

        m_info = None

        logger.info('\nPan-sharpening ...\n')

        if m_bands == 4:

            com = 'gdal_pansharpen.py {} {} {} ' \
                  '-w {:f} -w {:f} -w {:f} -w {:f} -r cubic ' \
                  '-bitdepth 16 -threads ALL_CPUS -co TILED=YES -co COMPRESS=DEFLATE'.format(self.pan_image,
                                                                                             self.multi_image,
                                                                                             self.multi_image_ps,
                                                                                             self.w1,
                                                                                             self.w2,
                                                                                             self.w3,
                                                                                             self.w4)

        else:

            com = 'gdal_pansharpen.py {} {} {} ' \
                  '-w {:f} -w {:f} -w {:f} -r cubic ' \
                  '-bitdepth 16 -threads ALL_CPUS -co TILED=YES -co COMPRESS=DEFLATE'.format(self.pan_image,
                                                                                             self.multi_image,
                                                                                             self.multi_image_ps,
                                                                                             self.w1,
                                                                                             self.w2,
                                                                                             self.w3)

        subprocess.call(com, shell=True)

    def _do_sharpen(self, im):

        try:
            import numexpr as ne
        except:
            raise ImportError('Numexpr is needed for pan-sharpening.')

        blue = im[0][0]
        green = im[0][1]
        red = im[0][2]
        pan_array = im[1]

        bw = self.bw
        gw = self.gw
        rw = self.rw
        iw = self.iw

        if im[0].shape[0] == 4:

            nir = im[0][3]

            if self.method == 'esri':
                dnf = ne.evaluate('pan_array - ((red*.166 + green*.167 + blue*.167 + nir*.5) / (.166+.167+.167+.5))')
            elif self.method == 'brovey':
                # dnf = ne.evaluate('(pan_array - (iw * nir)) / ((rw * red) + (gw * green) + (bw * blue))')
                dnf = ne.evaluate('pan_array / (((blue * bw) + (green * gw) + (red * rw) + (nir * iw)) / (bw + gw + rw + iw))')

        # TODO
        # else:
        #     dnf = ne.evaluate('(pan_array - iw * nir) / (rw * red + gw * green + bw * blue)')

        im = im[0]

        # plt.subplot(121)
        # plt.imshow(im[0]+dnf)
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(dnf)
        # plt.axis('off')
        # plt.show()
        # sys.exit()

        for bi in range(0, im.shape[0]):

            if self.method == 'esri':
                im[bi] += dnf
            elif self.method == 'brovey':
                im[bi] *= dnf

        return im

    def _sharpen(self):

        with ropen(self.multi_warped) as m_info, ropen(self.pan_image) as p_info:

            o_info = m_info.copy()

            bp = BlockFunc(self._do_sharpen,
                           [m_info, p_info],
                           self.multi_image_ps,
                           o_info,
                           band_list=[range(1, m_info.bands+1), 1],
                           d_types=['float32', 'float32'],
                           block_rows=4000,
                           block_cols=4000,
                           print_statement='\nPan sharpening using {} ...\n'.format(self.method.title()),
                           method=self.method)

            bp.run()

        m_info = None
        p_info = None

    def _warp_multi(self):

        # Get warping info.
        with ropen(self.pan_image) as p_info:

            extent = p_info.extent
            cell_size = p_info.cellY

        p_info = None

        self.multi_warped = os.path.join(self.out_dir, '{}_warped.tif'.format(self.f_base))

        logger.info('Resampling to pan scale ...')

        # Resample the multi-spectral bands.
        warp(self.multi_image,
             self.multi_warped,
             cell_size=cell_size,
             resample='cubic',
             outputBounds=[extent['left'],
                           extent['bottom'],
                           extent['right'],
                           extent['top']],
             multithread=True,
             creationOptions=['GDAL_CACHEMAX=256',
                              'TILED=YES'])


def gdal_open(image2open, band):

    """
    A direct file open from GDAL.
    """

    driver_o = gdal.Open(image2open, GA_ReadOnly)

    return driver_o, driver_o.GetRasterBand(band)


def gdal_read(image2open, band, i, j, rows, cols):

    """
    A direct array read from GDAL.
    """

    driver_o = gdal.Open(image2open, GA_ReadOnly)

    if isinstance(band, list):

        band_array = []

        for bd in band:

            band_object_o = driver_o.GetRasterBand(bd)
            band_array.append(band_object_o.ReadAsArray(j, i, cols, rows))

            band_object_o = None

        driver_o = None

        return np.array(band_array, dtype='float32').reshape(len(band), rows, cols)

    else:

        band_object_o = driver_o.GetRasterBand(band)
        band_array = np.float32(band_object_o.ReadAsArray(j, i, cols, rows))

    band_object_o = None
    driver_o = None

    return band_array


def gdal_write(band_object_w, array2write, io=0, jo=0):
    band_object_w.WriteArray(array2write, jo, io)


def gdal_close_band(band_object_c):

    try:
        band_object_c.FlushCache()
    except:
        logger.error(gdal.GetLastErrorMsg())
        pass

    band_object_c = None

    return band_object_c


def gdal_close_datasource(datasource_d):

    try:
        datasource_d.FlushCache()
    except:
        logger.error(gdal.GetLastErrorMsg())
        pass

    datasource_d = None

    return datasource_d


def gdal_register(image_name, in_memory=False):

    __, f_name = os.path.split(image_name)
    __, file_extension = os.path.splitext(f_name)

    if file_extension.lower() not in DRIVER_DICT:
        raise TypeError('{} is not an image, or is not a supported raster format.'.format(file_extension))
    else:
        file_format = DRIVER_DICT[file_extension.lower()]

    gdal.AllRegister()

    if in_memory:
        driver_r = gdal.GetDriverByName('MEM')
    else:
        driver_r = gdal.GetDriverByName(file_format)

    driver_r.Register

    return driver_r


def gdal_create(image_name, driver_cr, out_rows, out_cols, n_bands, storage_type,
                left, top, cellY, cellX, projection,
                in_memory=False, overwrite=False, parameters=[]):

    if overwrite:

        if os.path.isfile(image_name):
            os.remove(image_name)

    # Create the output driver.
    if in_memory:
        return driver_cr.Create('', out_cols, out_rows, n_bands, storage_type)
    else:
        ds = driver_cr.Create(image_name, out_cols, out_rows, n_bands, storage_type, parameters)

        # Set the geo-transformation.
        ds.SetGeoTransform([left, cellY, 0., top, 0., cellX])

        # Set the projection.
        ds.SetProjection(projection)

        return ds


def gdal_get_band(datasource_b, band_position):
    return datasource_b.GetRasterBand(band_position)


def _parallel_blocks(out_image,
                     band_list,
                     ii,
                     jj,
                     y_offset,
                     x_offset,
                     nn_rows,
                     nn_cols,
                     left,
                     top,
                     cellY,
                     cellX,
                     projection,
                     **kwargs):

    """
    Args:
        out_image:
        band_list:
        ii:
        jj:
        y_offset:
        x_offset:
        n_rows:
        n_cols:
        **kwargs:

    Returns:

    """

    # out_info_tile = out_info.copy()
    # out_info_tile.update_info(rows=nn_rows, cols=nn_cols,
    #                           left=out_info.left+(jj*out_info.cellY),
    #                           top=out_info.top-(ii*out_info.cellY))

    d_name_, f_name_ = os.path.split(out_image)
    f_base_, f_ext_ = os.path.splitext(f_name_)

    d_name_ = os.path.join(d_name_, 'temp')

    rsn = '{:f}'.format(abs(np.random.randn(1)[0]))[-4:]

    out_image_tile = os.path.join(d_name_, '{}_{}{}'.format(f_base_, rsn, f_ext_))

    datasource = gdal_create(out_image_tile,
                             driver_pp,
                             nn_rows,
                             nn_cols,
                             1,
                             STORAGE_DICT_GDAL['float32'],
                             left,
                             top,
                             cellY,
                             cellX,
                             projection)

    band_object = gdal_get_band(datasource, 1)

    # out_raster = create_raster(out_image_tile, out_info_tile)
    # out_raster.get_band(1)

    image_arrays = [gdal_read(image_infos_list[imi],
                              band_list[imi],
                              ii+y_offset[imi],
                              jj+x_offset[imi],
                              nn_rows,
                              nn_cols) for imi in range(0, len(image_infos_list))]

    output = block_func(image_arrays, **kwargs)

    gdal_write(band_object, output)

    band_object = gdal_close_band(band_object)
    datasource = gdal_close_datasource(datasource)

    return out_image_tile


class BlockFunc(object):

    """
    A class for block by block processing

    Args:
        func
        image_infos (list): A list of ``ropen`` instances.
        out_image (str): The output image.
        out_info (object): An instance of ``ropen``.
        band_list (Optional[list]): A list of band positions. Default is [].
        proc_info (Optional[object]): An instance of ``ropen``. Overrides image_infos[0]. Default is None.
        y_offset (Optional[list]): The row offset. Default is [0].
        x_offset (Optional[list]): The column offset. Default is [0].
        y_pad (Optional[list]): The row padding. Default is [0].
        x_pad (Optional[list]): The column padding. Default is [0].
        block_rows (Optional[int]): The block row chunk size. Default is 2048.
        block_cols (Optional[int]): The block column chunk size. Default is 2048.
        d_types (Optional[str list]): A list of read data types. Default is None.
        be_quiet (Optional[bool]): Whether to be quiet and do not print progress. Default is False.
        print_statement (Optional[str]): A string to print. Default is None.
        out_attributes (Optional[list]): A list of output attribute names. Default is [].
        write_array (Optional[bool]): Whether to write the output array to file. Default is True.
        bigtiff (Optional[str]): GDAL option passed to `create_raster`. Default is 'no'. See `create_raster`
            for details.
        boundary_file (Optional[str]): A file to use for block intersection. Default is None.
            Skip blocks that do not intersect ``boundary_file``.
        mask_file (Optional[str]): A file to use for block masking. Default is None.
            Recode blocks to binary 1 and 0 that intersect ``mask_file``.
        n_jobs (Optional[int]): The number of blocks to process in parallel. Default is 1.
        no_data_values (Optional[list]): A list of no data values for each image. Default is None.
        kwargs (Optional[dict]): Function specific parameters.

    Returns:
        None, writes to ``out_image``.
    """

    def __init__(self,
                 func,
                 image_infos,
                 out_image,
                 out_info,
                 band_list=None,
                 proc_info=None,
                 y_offset=None,
                 x_offset=None,
                 y_pad=None,
                 x_pad=None,
                 block_rows=2000,
                 block_cols=2000,
                 be_quiet=False,
                 d_types=None,
                 print_statement=None,
                 out_attributes=None,
                 write_array=True,
                 bigtiff='no',
                 boundary_file=None,
                 mask_file=None,
                 n_jobs=1,
                 close_files=True,
                 no_data_values=None,
                 overwrite=False,
                 **kwargs):

        self.func = func
        self.image_infos = image_infos
        self.out_image = out_image
        self.out_info = out_info
        self.band_list = band_list
        self.proc_info = proc_info
        self.y_offset = y_offset
        self.x_offset = x_offset
        self.y_pad = y_pad
        self.x_pad = x_pad
        self.block_rows = block_rows
        self.block_cols = block_cols
        self.d_types = d_types
        self.be_quiet = be_quiet
        self.print_statement = print_statement
        self.out_attributes = out_attributes
        self.write_array = write_array
        self.bigtiff = bigtiff
        self.boundary_file = boundary_file
        self.mask_file = mask_file
        self.n_jobs = n_jobs
        self.close_files = close_files
        self.no_data_values = no_data_values
        self.kwargs = kwargs

        self.out_attributes_dict = dict()

        if not isinstance(self.d_types, list):
            self.d_types = ['byte'] * len(self.image_infos)

        if not self.y_offset:
            self.y_offset = [0] * len(self.image_infos)

        if not self.x_offset:
            self.x_offset = [0] * len(self.image_infos)

        if not isinstance(self.out_image, str) and write_array:

            logger.error('  The output image was not given.')
            raise NameError

        if overwrite:

            if os.path.isfile(self.out_image):
                os.remove(self.out_image)

        if self.n_jobs in [0, 1]:

            if not self.proc_info:
                self.proc_info = self.image_infos[0]

            for imi in range(0, len(self.image_infos)):

                if not isinstance(self.image_infos[imi], ropen):

                    if not isinstance(self.image_infos[imi], GetMinExtent):

                        if not isinstance(self.image_infos[imi], ImageInfo):

                            logger.error('  The image info list should be instances of `ropen`, `GetMinExtent`, or `ImageInfo`.')
                            raise ropenError

        if not isinstance(self.band_list, list) and isinstance(self.band_list, int):
            self.band_list = [self.band_list] * len(self.image_infos)
        else:

            if self.band_list:

                if len(self.band_list) != len(self.image_infos):

                    logger.error('  The band list and image info list much be the same length.')
                    raise LenError

            else:
                self.band_list = [1] * len(self.image_infos)

        if isinstance(out_image, str):

            if not isinstance(self.out_info, ropen):

                if not isinstance(self.out_info, GetMinExtent):

                    logger.error('  The output image object is not a `raster_tools` instance.')
                    raise ropenError

        if not isinstance(self.image_infos, list):

            logger.error('  The image infos must be given as a list.')
            raise TypeError

        if not len(self.y_offset) == len(self.x_offset) == len(self.image_infos):

            logger.error('  The offset lists and input image info lists must be the same length.')
            raise LenError

    def run(self):

        global block_func, image_infos_list, driver_pp

        if self.n_jobs in [0, 1]:

            for imi in range(0, len(self.image_infos)):
                if isinstance(self.band_list[imi], int):
                    self.image_infos[imi].get_band(self.band_list[imi])

            self._process_blocks()

        else:

            block_func = self.func
            image_infos_list = self.image_infos

            self._get_pairs()

            dn, fn = os.path.split(self.out_image)

            if not dn and not os.path.isabs(fn):
                dn = os.path.join(os.path.abspath('.'), 'temp')
            else:
                check_and_create_dir(os.path.join(dn, 'temp'))

            driver_pp = gdal_register(self.out_image)

            tile_list = Parallel(n_jobs=self.n_jobs,
                                 max_nbytes=None)(delayed(_parallel_blocks)(self.out_image,
                                                                            self.band_list,
                                                                            pair[0], pair[1],
                                                                            self.y_offset,
                                                                            self.x_offset,
                                                                            pair[2], pair[3],
                                                                            self.out_info.left+(pair[1]*self.out_info.cellY),
                                                                            self.out_info.top-(pair[0]*self.out_info.cellY),
                                                                            self.out_info.cellY,
                                                                            self.out_info.cellX,
                                                                            self.out_info.projection,
                                                                            **self.kwargs) for pair in self.pairs)

    def _get_pairs(self):

        self.pairs = []

        for i in range(0, self.proc_info.rows, self.block_rows):

            n_rows = n_rows_cols(i, self.block_rows, self.proc_info.rows)

            for j in range(0, self.proc_info.cols, self.block_cols):

                n_cols = n_rows_cols(j, self.block_cols, self.proc_info.cols)

                self.pairs.append((i, j, n_rows, n_cols))

    def _process_blocks(self):

        if self.write_array:

            out_raster = create_raster(self.out_image,
                                       self.out_info,
                                       bigtiff=self.bigtiff)

        # n_blocks = 0
        # for i in range(0, self.proc_info.rows, self.block_rows):
        #     for j in range(0, self.proc_info.cols, self.block_cols):
        #         n_blocks += 1
        #
        # n_block = 1

        if isinstance(self.print_statement, str):
            logger.info(self.print_statement)

        # set widget and pbar
        if not self.be_quiet:
            ctr, pbar = _iteration_parameters(self.proc_info.rows, self.proc_info.cols,
                                              self.block_rows, self.block_cols)

        # iterate over the images and get change pixels
        for i in range(0, self.proc_info.rows, self.block_rows):

            n_rows = n_rows_cols(i, self.block_rows, self.proc_info.rows)

            if isinstance(self.y_pad, int):
                y_pad_minus = 0 if i == 0 else self.y_pad
                y_pad_plus = 0 if i + n_rows + self.y_pad > self.proc_info.rows else self.proc_info.rows - (i + n_rows)
            else:
                y_pad_minus = 0
                y_pad_plus = 0

            for j in range(0, self.proc_info.cols, self.block_cols):

                n_cols = n_rows_cols(j, self.block_cols, self.proc_info.cols)

                if isinstance(self.x_pad, int):
                    x_pad_minus = 0 if j == 0 else self.x_pad
                    x_pad_plus = 0 if j + n_cols + self.x_pad > self.proc_info.cols else self.proc_info.cols - (j + n_cols)
                else:
                    x_pad_minus = 0
                    x_pad_plus = 0

                if isinstance(self.boundary_file, str):

                    # Get the extent of the current block.
                    self.get_block_extent(i, j, n_rows, n_cols)

                    # Check if the block intersects the boundary file.
                    if not vector_tools.intersects_boundary(self.extent_dict, self.boundary_file):
                        continue

                # if not self.be_quiet:
                #
                #     if n_block == 1:
                #         print 'Blocks 1--19 of {:,d} ...'.format(n_blocks)
                #     elif n_block % 20 == 0:
                #         n_block_ = n_block + 19 if n_blocks - n_block > 20 else n_blocks
                #         print 'Block {:,d}--{:,d} of {:,d} ...'.format(n_block, n_block_, n_blocks)
                #
                #     n_block += 1
                image_arrays = [self.image_infos[imi].read(bands2open=self.band_list[imi],
                                                           i=i+self.y_offset[imi]-y_pad_minus,
                                                           j=j+self.x_offset[imi]-x_pad_minus,
                                                           rows=n_rows+y_pad_plus,
                                                           cols=n_cols+x_pad_plus,
                                                           d_type=self.d_types[imi])
                                for imi in range(0, len(self.image_infos))]

                skip_block = False

                # Check for no data values.
                if isinstance(self.no_data_values, list):

                    for no_data, im_block in zip(self.no_data_values, image_arrays):

                        if isinstance(no_data, int) or isinstance(no_data, float):

                            if im_block.max() == no_data:

                                skip_block = True
                                break

                if skip_block:
                    continue

                if isinstance(self.mask_file, str):

                    self.get_block_extent(i, j, n_rows, n_cols)

                    orw = create_raster('none',
                                        None,
                                        in_memory=True,
                                        rows=n_rows,
                                        cols=n_cols,
                                        bands=1,
                                        projection=self.proc_info.projection,
                                        cellY=self.proc_info.cellY,
                                        cellX=self.proc_info.cellX,
                                        left=self.extent_dict['UL'][0],
                                        top=self.extent_dict['UL'][1],
                                        storage='byte')

                    # Rasterize the vector at the current block.
                    with vector_tools.vopen(self.mask_file) as v_info:

                        gdal.RasterizeLayer(orw.datasource, [1], v_info.lyr, burn_values=[1])
                        block_array = orw.datasource.GetRasterBand(1).ReadAsArray(0, 0, n_cols, n_rows)

                        for imib, image_array in enumerate(image_arrays):

                            image_array[block_array == 0] = 0
                            image_arrays[imib] = image_array

                    v_info = None

                    gdal.Unlink('none')

                output = self.func(image_arrays,
                                   **self.kwargs)

                if isinstance(output, tuple):

                    if self.write_array:

                        if output[0].shape[0] > 1:

                            for obi, obb in enumerate(output[0]):

                                out_raster.write_array(obb,
                                                       i=i,
                                                       j=j,
                                                       band=obi + 1)

                        else:

                            out_raster.write_array(output[0],
                                                   i=i,
                                                   j=j,
                                                   band=1)

                    # Get the other results.
                    for ri in range(1, len(output)):

                        # self.kwargs[self.out_attributes[ri-1]] = output[ri]

                        if self.out_attributes[ri-1] not in self.out_attributes_dict:
                            self.out_attributes_dict[self.out_attributes[ri-1]] = [output[ri]]
                        else:
                            self.out_attributes_dict[self.out_attributes[ri-1]].append(output[ri])

                else:

                    if self.write_array:

                        if len(output.shape) > 2:

                            for obi, obb in enumerate(output):

                                out_raster.write_array(obb,
                                                       i=i,
                                                       j=j,
                                                       band=obi+1)

                        else:

                            out_raster.write_array(output,
                                                   i=i,
                                                   j=j,
                                                   band=1)

                if not self.be_quiet:

                    pbar.update(ctr)
                    ctr += 1

        if self.out_attributes_dict:

            for ri in range(1, len(output)):
                setattr(self, self.out_attributes[ri-1], self.out_attributes_dict[self.out_attributes[ri-1]])

        if not self.be_quiet:
            pbar.finish()

        if isinstance(self.out_image, str):

            if self.close_files:

                for imi in range(0, len(self.image_infos)):
                    self.image_infos[imi].close()

                self.out_info.close()

            if self.write_array:
                out_raster.close_all()

    def get_block_extent(self, ii, jj, nn_rows, nn_cols):

        adj_left = self.proc_info.left + (jj * self.proc_info.cellY)
        adj_right = adj_left + (nn_cols * self.proc_info.cellY) + self.proc_info.cellY
        adj_top = self.proc_info.top - (ii * self.proc_info.cellY)
        adj_bottom = adj_top - (nn_rows * self.proc_info.cellY) - self.proc_info.cellY

        self.extent_dict = {'UL': [adj_left, adj_top],
                            'UR': [adj_right, adj_top],
                            'LL': [adj_left, adj_bottom],
                            'LR': [adj_right, adj_bottom]}


def _read_parallel(image, image_info, bands2open, y, x, rows2open, columns2open, n_jobs, d_type, predictions):

    """
    Opens image bands into arrays using multiple processes

    Args:
        image (str): The image to open.
        image_info (instance)
        bands2open (int or int list: Band position to open or list of bands to open.
        y (int): Starting row position.
        x (int): Starting column position.
        rows2open (int): Number of rows to extract.
        columns2open (int): Number of columns to extract.
        n_jobs (int): The number of jobs to run in parallel.
        d_type (str): Type of array to return.
        predictions (bool): Whether to return reshaped array for predictions.

    Returns:
        Ndarray where [rows, cols] if 1 band and [bands, rows, cols] if more than 1 band
    """

    if isinstance(bands2open, list):

        if max(bands2open) > image_info.bands:
            raise ValueError('\nCannot open more bands than exist in the image.\n')

    else:

        if bands2open == -1:
            bands2open = list(range(1, image_info.bands+1))

    if rows2open == -1:
        rows2open = image_info.rows

    if columns2open == -1:
        columns2open = image_info.cols

    image_info.close()

    band_arrays = Parallel(n_jobs=n_jobs)(delayed(gdal_read)(image,
                                                             band2open,
                                                             y,
                                                             x,
                                                             rows2open,
                                                             columns2open)
                                          for band2open in bands2open)

    if predictions:

        # Check for empty images.
        band_arrays = [b_ if b_.shape else np.zeros((rows2open, columns2open), dtype=d_type) for b_ in band_arrays]

        return np.array(band_arrays,
                        dtype=d_type).reshape(len(bands2open),
                                              rows2open,
                                              columns2open).transpose(1, 2, 0).reshape(rows2open*columns2open,
                                                                                       len(bands2open))

    else:
        return np.array(band_arrays, dtype=d_type).reshape(len(bands2open), rows2open, columns2open)


def read(image2open=None,
         i_info=None,
         bands2open=1,
         i=0,
         j=0,
         rows=-1,
         cols=-1,
         d_type=None,
         predictions=False,
         sort_bands2open=True,
         y=0.,
         x=0.,
         n_jobs=0):

    """
    Reads a raster as an array

    Args:
        image2open (Optional[str]): An image to open. Default is None.
        i_info (Optional[object]): An instance of `ropen`. Default is None
        bands2open (Optional[int list or int]: Band position to open or list of bands to open. Default is 1.
            Examples:
                bands2open = 1        (open band 1)
                bands2open = [1,2,3]  (open first three bands)
                bands2open = -1       (open all bands)
        i (Optional[int]): Starting row position. Default is 0, or first row.
        j (Optional[int]): Starting column position. Default is 0, or first column.
        rows (Optional[int]): Number of rows to extract. Default is all rows.
        cols (Optional[int]): Number of columns to extract. Default is all columns.
        d_type (Optional[str]): Type of array to return. Default is None, or gathered from <i_info>.
            Choices are ['uint8', 'int8', 'uint16', 'uint32', 'int16', 'float32', 'float64'].
        predictions (Optional[bool]): Whether to return reshaped array for predictions.
        sort_bands2open (Optional[bool]): Whether to sort ``bands2open``. Default is True.
        y (Optional[float]): A y index coordinate. Default is 0. If greater than 0, overrides `i`.
        x (Optional[float]): A x index coordinate. Default is 0. If greater than 0, overrides `j`.
        n_jobs (Optional[int]): The number of bands to open in parallel. Default is 0.

    Attributes:
        array (ndarray)

    Returns:
        Ndarray where [rows, cols] if 1 band and [bands, rows, cols] if more than 1 band

    Examples:
        >>> import mpglue as gl
        >>>
        >>> array = mp.read('image.tif')
        >>>
        >>> array = mp.read('image.tif', bands2open=[1, 2, 3])
        >>> print(a.shape)
        >>>
        >>> array = mp.read('image.tif', bands2open={'green': 3, 'nir': 4})
        >>> print(len(array))
        >>> print(array['nir'].shape)
    """

    if not isinstance(i_info, ropen) and not isinstance(image2open, str):

        logger.error('Either `i_info` or `image2open` must be declared.')
        raise MissingRequirement

    elif isinstance(i_info, ropen) and isinstance(image2open, str):

        logger.error('Choose either `i_info` or `image2open`, but not both.')
        raise OverflowError

    elif not isinstance(i_info, ropen) and isinstance(image2open, str):
        i_info = ropen(image2open)

    rrows = copy.copy(rows)
    ccols = copy.copy(cols)

    if rrows == -1:
        rrows = i_info.rows
    else:

        if rrows > i_info.rows:

            rrows = i_info.rows
            logger.warning('  The requested rows cannot be larger than the image rows.')

    if ccols == -1:
        ccols = i_info.cols
    else:

        if ccols > i_info.cols:

            ccols = i_info.cols
            logger.warning('  The requested columns cannot be larger than the image columns.')

    #################
    # Bounds checking
    #################

    # Row indices
    if i < 0:
        i = 0

    if i >= i_info.rows:
        i = i_info.rows - 1

    # Number of rows
    rrows = n_rows_cols(i, rrows, i_info.rows)

    # Column indices
    if j < 0:
        j = 0

    if j >= i_info.cols:
        j = i_info.cols - 1

    # Number of columns
    ccols = n_rows_cols(j, ccols, i_info.cols)

    if isinstance(bands2open, list):

        if len(bands2open) == 0:
            raise ValueError('\nA band list must be declared.\n')

        if max(bands2open) > i_info.bands:
            raise ValueError('\nThe requested band position cannot be greater than the image bands.\n')

    elif isinstance(bands2open, int):

        if bands2open > i_info.bands:
            raise ValueError('\nThe requested band position cannot be greater than the image bands.\n')

        if bands2open == -1:
            bands2open = list(range(1, i_info.bands+1))
        else:
            bands2open = [bands2open]

    if sort_bands2open:
        bands2open = sorted(bands2open)

    # Index the image by x, y coordinates (in map units).
    if abs(y) > 0:
        __, __, __, i = vector_tools.get_xy_offsets(i_info, x=x, y=y)

    if abs(x) > 0:
        __, __, j, __ = vector_tools.get_xy_offsets(i_info, x=x, y=y)

    if (n_jobs in [0, 1]) and not predictions:

        kwargs = dict(bands2open=bands2open,
                      i=i,
                      j=j,
                      rows=rrows,
                      cols=ccols,
                      d_type=d_type,
                      sort_bands2open=sort_bands2open,
                      y=y,
                      x=x)

        return i_info.read(**kwargs)

    else:

        # Convert to NumPy dtype.
        if isinstance(d_type, str):
            d_type = STORAGE_DICT[d_type]
        else:
            d_type = STORAGE_DICT[i_info.storage.lower()]

        # format_dict = {'byte': 'B', 'int16': 'i', 'uint16': 'I', 'float32': 'f', 'float64': 'd'}

        if n_jobs in [0, 1]:

            values = np.asarray([i_info.datasource.GetRasterBand(band).ReadAsArray(j, i, ccols, rrows)
                                 for band in bands2open], dtype=d_type)

            # values = struct.unpack('%d%s' % ((rows * cols * len(bands2open)), format_dict[i_info.storage.lower()]),
            #                        i_info.datasource.ReadRaster(yoff=i, xoff=j, xsize=cols, ysize=rows, band_list=bands2open))

            if predictions:

                return values.reshape(len(bands2open), rrows, ccols).transpose(1, 2, 0).reshape(rrows*ccols,
                                                                                                len(bands2open))

            else:

                if len(bands2open) == 1:
                    return values.reshape(rrows, ccols)
                else:
                    return values.reshape(len(bands2open), rrows, ccols)

            # only close the image if it was opened internally
            # if isinstance(image2open, str):
            #     i_info.close()

        else:
            return _read_parallel(image2open, i_info, bands2open, i, j, rrows, ccols, n_jobs, d_type, predictions)


def build_vrt(file_list,
              output_image,
              cell_size=0.0,
              return_datasource=False,
              overwrite=False,
              **kwargs):

    """
    Build a VRT file

    Args:
        file_list (str list): A list of files.
        output_image (str): The output image.
        cell_size (Optional[float]): The output cell size. Default is 0.
        return_datasource (Optional[bool]: Whether to return the raster datasource. Default is False.
        overwrite (Optional[bool]): Whether to overwrite an existing VRT file. Default is False.
        kwargs:
             resolution=None,
             outputBounds=None (minX, minY, maxX, maxY),
             targetAlignedPixels=None,
             separate=None,
             bandList=None,
             addAlpha=None,
             resampleAlg=None,
             outputSRS=None,
             allowProjectionDifference=None,
             srcNodata=None,
             VRTNodata=None,
             hideNodata=None,
             callback=None,
             callback_data=None
    """

    if overwrite:

        if os.path.isfile(output_image):
            os.remove(output_image)

    vrt_options = gdal.BuildVRTOptions(xRes=cell_size,
                                       yRes=cell_size,
                                       **kwargs)

    out_ds = gdal.BuildVRT(output_image,
                           file_list,
                           options=vrt_options)

    if return_datasource:
        return out_ds
    else:
        out_ds = None


def _merge_dicts(dict1, dict2):

    dict3 = dict1.copy()
    dict3.update(dict2)

    return dict3


def warp(input_image,
         output_image,
         out_epsg=None,
         out_proj=None,
         in_epsg=None,
         in_proj=None,
         resample='nearest',
         cell_size=0,
         d_type=None,
         return_datasource=False,
         overwrite=False,
         **kwargs):

    """
    Warp transforms a dataset

    Args:
        input_image (str): The image to warp.
        output_image (str): The output image.
        out_epsg (Optional[int]): The output EPSG projection code.
        out_proj (Optional[str]): The output proj4 projection code.
        in_epsg (Optional[int]): An input EPSG code. Default is None.
        in_proj (Optional[str]): An input projection string. Default is None.
        resample (Optional[str]): The resampling method. Default is 'nearest'.
        cell_size (Optional[float]): The output cell size. Default is 0.
        d_type (Optional[str]): Data type to overwrite `outputType`. Default is None.
        return_datasource (Optional[bool]): Whether to return the datasource object. Default is False.
        overwrite (Optional[bool]): Whether to overwrite `out_vrt`, if it exists. Default is False.
        kwargs:
            format=None, outputBounds=None (minX, minY, maxX, maxY), 
            outputBoundsSRS=None, targetAlignedPixels=False,
             width=0, height=0, srcAlpha=False, dstAlpha=False, warpOptions=None,
             errorThreshold=None, warpMemoryLimit=None,
             creationOptions=None, outputType=0, workingType=0,
             resampleAlg=resample_dict[resample], srcNodata=None, dstNodata=None,
             multithread=False, tps=False, rpc=False, geoloc=False,
             polynomialOrder=None, transformerOptions=None, cutlineDSName=None,
             cutlineLayer=None, cutlineWhere=None, cutlineSQL=None,
             cutlineBlend=None, cropToCutline=False, copyMetadata=True,
             metadataConflictValue=None, setColorInterpretation=False,
             callback=None, callback_data=None

             E.g.,
                creationOptions=['GDAL_CACHEMAX=256', 'TILED=YES']

    Returns:
        None, writes to `output_image'.

    Examples:
        >>> from mpglue import raster_tools
        >>>
        >>> # Resample a subset of an image in memory
        >>> warp_info = raster_tools.warp('/input_image.tif',
        >>>                               'memory_image.mem',
        >>>                               resample='nearest',
        >>>                               cell_size=10.0,
        >>>                               return_datasource=True,
        >>>                               outputBounds=[<left, bottom, right, top>])
        >>>
        >>> # Load the resampled array
        >>> resampled_array = warp_info.read()
    """

    if output_image.endswith('.mem'):

        while True:

            output_image = '{:f}'.format(abs(np.random.randn(1)[0]))[-5:] + '.mem'

            if not os.path.isfile(output_image):
                break

    else:

        d_name, f_name = os.path.split(output_image)

        if not d_name and not os.path.isabs(f_name):
            d_name = os.path.abspath('.')
        else:
            check_and_create_dir(d_name)

    if isinstance(out_epsg, int):
        out_proj = 'EPSG:{:d}'.format(out_epsg)

    if isinstance(in_epsg, int):
        in_proj = 'EPSG:{:d}'.format(in_epsg)

    if cell_size == 0:
        cell_size = (None, None)
    else:
        cell_size = (cell_size, -cell_size)

    if overwrite:

        if os.path.isfile(output_image):
            os.remove(output_image)

    if isinstance(d_type, str):

        awargs = _merge_dicts(dict(srcSRS=in_proj,
                                   dstSRS=out_proj,
                                   xRes=cell_size[0],
                                   yRes=cell_size[1],
                                   outputType=STORAGE_DICT_GDAL[d_type],
                                   resampleAlg=RESAMPLE_DICT[resample]),
                              kwargs)

    else:

        awargs = _merge_dicts(dict(srcSRS=in_proj,
                                   dstSRS=out_proj,
                                   xRes=cell_size[0],
                                   yRes=cell_size[1],
                                   resampleAlg=RESAMPLE_DICT[resample]),
                              kwargs)

    warp_options = gdal.WarpOptions(**awargs)

    try:

        out_ds = gdal.Warp(output_image,
                           input_image,
                           options=warp_options)

    except:

        if 'outputBounds' in awargs:

            logger.info('  Input image extent:')

            with ropen(input_image) as info:
                logger.info(info.extent)

            info = None

            logger.info('')

            logger.info('  Requested image extent (left, bottom, right, top):')
            logger.info(awargs['outputBounds'])

        logger.warning('  GDAL returned an exception--check the output file, {}.'.format(output_image))

        out_ds = None

    if return_datasource:

        if out_ds is None:
            return None

        i_info = ImageInfo()

        i_info.update_info(datasource=out_ds,
                           hdf_file=False,
                           output_image=output_image)

        i_info.datasource_info()

        out_ds = None

        return i_info

    else:

        out_ds = None

        if output_image.endswith('.mem'):
            gdal.Unlink(output_image)


def translate(input_image,
              output_image,
              cell_size=0,
              d_type=None,
              return_datasource=False,
              overwrite=False,
              **kwargs):

    """
    Args:
        input_image (str): The image to translate.
        output_image (str): The output image.
        cell_size (Optional[float]): The output cell size. Default is 0.
        d_type (Optional[str]): Data type to overwrite `outputType`. Default is None.
        return_datasource (Optional[bool]): Whether to return the datasource object. Default is False.
        overwrite (Optional[bool]): Whether to overwrite `out_vrt`, if it exists. Default is False.
        kwargs:
            format='GTiff', outputType=0, bandList=None, maskBand=None, width=0, height=0,
            widthPct=0.0, heightPct=0.0, xRes=0.0, yRes=0.0, creationOptions=None, srcWin=None,
            projWin=None [ulx, uly, lrx, lry], projWinSRS=None, strict=False, unscale=False,
            scaleParams=None [[srcmin, srcmax, dstmin, dstmax]],
            exponents=None, outputBounds=None, metadataOptions=None, outputSRS=None, GCPs=None,
            noData=None, rgbExpand=None, stats=False, rat=True, resampleAlg=None,
            callback=None, callback_data=None

        Examples:
            >>> from mpglue import raster_tools
            >>>
            >>> raster_tools.translate('input.tif', 'output.tif',
            >>>                        cell_size=30.,
            >>>                        format='GTiff', d_type='byte',
            >>>                        creationOptions=['GDAL_CACHEMAX=256', 'TILED=YES'])
    """

    d_name, f_name = os.path.split(output_image)

    if not d_name and not os.path.isabs(f_name):
        d_name = os.path.abspath('.')
    else:
        check_and_create_dir(d_name)

    if overwrite:

        if os.path.isfile(output_image):
            os.remove(output_image)

    if isinstance(d_type, str):

        translate_options = gdal.TranslateOptions(xRes=cell_size,
                                                  yRes=cell_size,
                                                  outputType=STORAGE_DICT_GDAL[d_type],
                                                  **kwargs)

    else:

        translate_options = gdal.TranslateOptions(xRes=cell_size,
                                                  yRes=cell_size,
                                                  **kwargs)

    try:
        out_ds = gdal.Translate(output_image, input_image, options=translate_options)
    except:
        logger.warning('  GDAL returned an exception--check the output file, {}.'.format(output_image))

    if return_datasource:

        i_info = ImageInfo()

        i_info.update_info(datasource=out_ds,
                           hdf_file=False)

        i_info.datasource_info()

        return i_info

    else:
        out_ds = None


def vis2rgb(image_array):

    """
    Converts a layer x rows x columns array to RGB
    """

    return image_array.transpose(1, 2, 0)


class create_raster(CreateDriver, FileManager, UpdateInfo):

    """
    Creates a raster driver to write to.

    Args:
        out_name (str): Output raster name.
        o_info (object): Instance of ``ropen``.
        compress (Optional[str]): The type of compression to use. Default is 'deflate'.
            Choices are ['none' 'lzw', 'packbits', 'deflate'].
        bigtiff (Optional[str]): How to manage large TIFF files. Default is 'no'.
            Choices are ['yes', 'no', 'if_needed', 'if_safer'].
        tile (Optional[bool]): Whether to tile the new image. Default is True.
        project_epsg (Optional[int]): Project the new raster to an EPSG code projection.
        create_tiles (Optional[str]): If positive, image is created in separate file tiles. Default is 0.
        overwrite (Optional[str]): Whether to overwrite an existing file. Default is False.
        in_memory (Optional[str]): Whether to create the raster dataset in memory. Default is False.

    Attributes:
        filename (str)
        rows (int)
        cols (int)
        bands (int)
        storage (str)

    Returns:
        Raster driver GDAL object or list of GDAL objects (if create_tiles > 0).
    """

    def __init__(self,
                 out_name,
                 o_info,
                 compress='deflate',
                 tile=True,
                 bigtiff='no',
                 project_epsg=None,
                 create_tiles=0,
                 overwrite=False,
                 in_memory=False,
                 **kwargs):

        if not in_memory:

            d_name, f_name = os.path.split(out_name)
            f_base, f_ext = os.path.splitext(f_name)

            if not d_name and not os.path.isabs(f_name):
                d_name = os.path.abspath('.')
            else:
                check_and_create_dir(d_name)

        storage_type = STORAGE_DICT_GDAL[o_info.storage.lower()] if 'storage' not in kwargs \
            else STORAGE_DICT_GDAL[kwargs['storage'].lower()]

        out_rows = o_info.rows if 'rows' not in kwargs else kwargs['rows']
        out_cols = o_info.cols if 'cols' not in kwargs else kwargs['cols']
        n_bands = o_info.bands if 'bands' not in kwargs else kwargs['bands']
        projection = o_info.projection if 'projection' not in kwargs else kwargs['projection']
        cellY = o_info.cellY if 'cellY' not in kwargs else kwargs['cellY']
        cellX = o_info.cellX if 'cellX' not in kwargs else kwargs['cellX']
        left = o_info.left if 'left' not in kwargs else kwargs['left']
        top = o_info.top if 'top' not in kwargs else kwargs['top']

        if tile:
            tile = 'YES'
        else:
            tile = 'NO'

        if abs(cellY) == 0:
            raise ValueError('The cell y size must be greater than 0.')

        if abs(cellX) == 0:
            raise ValueError('The cell x size must be greater than 0.')

        if cellX > 0:
            cellX *= -1.

        if cellY < 0:
            cellY *= -1.

        if out_name.lower().endswith('.img'):

            if compress.upper() == 'NONE':
                parameters = ['COMPRESS=NO']
            else:
                parameters = ['COMPRESS=YES']

        elif out_name.lower().endswith('.tif'):

            if compress.upper() == 'NONE':

                parameters = ['TILED={}'.format(tile),
                              'BIGTIFF={}'.format(bigtiff.upper())]

            else:

                parameters = ['TILED={}'.format(tile),
                              'COMPRESS={}'.format(compress.upper()),
                              'BIGTIFF={}'.format(bigtiff.upper())]

        elif (out_name.lower().endswith('.dat')) or (out_name.lower().endswith('.bin')):

            parameters = ['INTERLEAVE=BSQ']

        elif out_name.lower().endswith('.kea'):

            parameters = ['DEFLATE=1']

        else:
            parameters = list()

        if isinstance(project_epsg, int):

            osng = osr.SpatialReference()
            osng.ImportFromWkt(o_info.projection)

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(project_epsg)
            new_projection = srs.ExportToWkt()

            tx = osr.CoordinateTransformation(osng, srs)

            # Work out the boundaries of the new dataset in the target projection
            ulx, uly, ulz = tx.TransformPoint(o_info.left, o_info.top)
            lrx, lry, lrz = tx.TransformPoint(o_info.left + o_info.cellY*o_info.cols,
                                              o_info.top + o_info.cellX*o_info.rows)

            # project_rows = int((uly - lry) / o_info.cellY)
            # project_cols = int((lrx - ulx) / o_info.cellY)

            # Calculate the new geotransform
            new_geo = [ulx, o_info.cellY, o_info.rotation1, uly, o_info.rotation2, o_info.cellX]

            # out_rows = int((uly - lry) / o_info.cellY)
            # out_cols = int((lrx - ulx) / o_info.cellY)

        # Create driver for output image.
        if create_tiles > 0:

            d_name_tiles = os.path.join(d_name, '{}_tiles'.format(f_base))

            if not os.path.isdir(d_name_tiles):
                os.makedirs(d_name_tiles)

            out_rst = {}

            if out_rows >= create_tiles:
                blk_size_rows = create_tiles
            else:
                blk_size_rows = copy.copy(out_rows)

            if out_cols >= create_tiles:
                blk_size_cols = create_tiles
            else:
                blk_size_cols = copy.copy(out_cols)

            topo = copy.copy(top)

            image_counter = 1

            for i in range(0, out_rows, blk_size_rows):

                lefto = copy.copy(left)

                out_rows = n_rows_cols(i, blk_size_rows, out_rows)

                for j in range(0, out_cols, blk_size_cols):

                    out_cols = n_rows_cols(j, blk_size_cols, out_cols)

                    out_name = os.path.join(d_name_tiles, '{}_{:d}_{:d}{}'.format(f_base, i, j, f_ext))

                    out_rst[image_counter] = out_name

                    image_counter += 1

                    if overwrite:

                        if os.path.isfile(out_name):

                            try:
                                os.remove(out_name)
                            except OSError:
                                raise OSError('\nCould not delete {}.'.format(out_name))

                    else:

                        if os.path.isfile(out_name):

                            logger.warning('\n{} already exists.'.format(out_name))
                            continue

                    CreateDriver.__init__(self,
                                          out_name,
                                          out_rows,
                                          out_cols,
                                          n_bands,
                                          storage_type,
                                          in_memory,
                                          overwrite,
                                          parameters)

                    # FileManager.__init__(self)

                    # out_rst_ = self.driver.Create(out_name, out_cols, out_rows, bands, storage_type, parameters)

                    # set the geo-transformation
                    self.datasource.SetGeoTransform([lefto, cellY, 0.0, topo, 0.0, cellX])

                    # set the projection
                    self.datasource.SetProjection(projection)

                    self.close_file()

                    lefto += (out_cols * cellY)

                topo -= (out_rows * cellY)

        else:

            if not in_memory:

                if overwrite:

                    if os.path.isfile(out_name):

                        try:
                            os.remove(out_name)
                        except:
                            logger.warning('  Could not delete {}.\nWill attempt to write over the image'.format(out_name))

                else:

                    if os.path.isfile(out_name):

                        logger.warning(' {} already exists.\nWill not attempt to overwrite.'.format(out_name))
                        return

            CreateDriver.__init__(self,
                                  out_name,
                                  out_rows,
                                  out_cols,
                                  n_bands,
                                  storage_type,
                                  in_memory,
                                  overwrite,
                                  parameters)

            # FileManager.__init__(self)

            # self.datasource = self.driver.Create(out_name, out_cols, out_rows, bands, storage_type, parameters)

            if isinstance(project_epsg, int):

                # set the geo-transformation
                self.datasource.SetGeoTransform(new_geo)

                # set the projection
                self.datasource.SetProjection(new_projection)

                # gdal.ReprojectImage(o_info.datasource, out_rst, o_info.proj, new_projection, GRA_NearestNeighbour)

            else:

                # Set the geo-transformation.
                self.datasource.SetGeoTransform([left, cellY, 0., top, 0., cellX])

                # Set the projection.
                self.datasource.SetProjection(projection)

            self.filename = out_name
            self.rows = out_rows
            self.cols = out_cols
            self.bands = n_bands
            self.storage = storage_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close_all()


# @deprecation.deprecated(deprecated_in='0.1.3',
#                         removed_in='0.1.5',
#                         current_version=__version__,
#                         details='Variables `x` and `y` will be replaced with `j` and `i`, respectively.')
def write2raster(out_array,
                 out_name,
                 o_info=None,
                 x=0,
                 y=0,
                 out_rst=None,
                 write2bands=None,
                 close_band=True,
                 flush_final=False,
                 write_chunks=False,
                 **kwargs):

    """
    Writes an ndarray to file.

    Args:
        out_array (ndarray): The array to write to file.
        out_name (str): The output image name.
        o_info (Optional[object]): Output image information. Needed if ``out_rst`` not given. Default is None.
        x (Optional[int]): Column starting position. Default is 0.
        y (Optional[int]): Row starting position. Default is 0.
        out_rst (Optional[object]): GDAL object to write to, otherwise created. Default is None.
        write2bands (Optional[int or int list]): Band positions to write to, otherwise takes the order of the input
            array dimensions. Default is None.
        close_band (Optional[bool]): Whether to flush the band cache. Default is True.
        flush_final (Optional[bool]): Whether to flush the raster cache. Default is False.
        write_chunks (Optional[bool]): Whether to write to file in <write_chunks> chunks. Default is False.
        kwargs (Optional[dict]): Arguments passed to `create_raster`.

    Returns:
        None, writes <out_name>.

    Examples:
        >>> # Example
        >>> from mpglue import raster_tools
        >>> i_info = raster_tools.ropen('/in_raster.tif')
        >>>
        >>> out_array = np.random.randn(3, 100, 100).astype(np.float32)
        >>>
        >>> raster_tools.write2raster(out_array,
        >>>                           '/out_name.tif',
        >>>                           o_info=i_info.copy())
    """

    # Get the output information.
    d_name, f_name = os.path.split(out_name)

    if not d_name and not os.path.isabs(f_name):
        d_name = os.path.abspath('.')
    else:
        check_and_create_dir(d_name)

    array_shape = out_array.shape

    if len(array_shape) > 3:

        logger.error('The array shape should be 2d or 3d.')
        raise ArrayShapeError

    if len(array_shape) == 2:

        out_rows, out_cols = out_array.shape
        out_dims = 1

    else:
        out_dims, out_rows, out_cols = out_array.shape

    new_file = False

    # Does the file need to be created?
    if not out_rst:

        if not isinstance(o_info, ropen):

            if not isinstance(o_info, ImageInfo):

                logger.error('The output information must be set.')
                raise ropenError

        new_file = True

        o_info.update_info(bands=out_dims,
                           rows=out_rows,
                           cols=out_cols)

        if kwargs:
            out_rst = create_raster(out_name, o_info, **kwargs)
        else:
            out_rst = create_raster(out_name, o_info)

    ##########################
    # pack the data to binary
    ##########################
    # format_dict = {'byte': 'B', 'int16': 'i', 'uint16': 'I', 'float32': 'f', 'float64': 'd'}

    # specifiy a band to write to
    if isinstance(write2bands, int) or isinstance(write2bands, list):

        if isinstance(write2bands, int):
            write2bands = [write2bands]

        for n_band in write2bands:

            out_rst.get_band(n_band)

            if write_chunks:

                out_rst.get_chunk_size()

                for i in range(0, out_rst.rows, out_rst.chunk_size):

                    n_rows = n_rows_cols(i, out_rst.chunk_size, out_rst.rows)

                    for j in range(0, out_rst.cols, out_rst.chunk_size):

                        n_cols = n_rows_cols(j, out_rst.chunk_size, out_rst.cols)

                        out_rst.write_array(out_array[i:i+n_rows, j:j+n_cols], i=i, j=j)

            else:

                out_rst.write_array(out_array, i=y, j=x)

            if close_band:

                out_rst.close_band()

    else:

        if out_dims >= 2:

            for n_band in range(1, out_dims+1):

                out_rst.write_array(out_array[n_band-1], i=y, j=x, band=n_band)

                if close_band:
                    out_rst.close_band()

        else:

            out_rst.write_array(out_array, i=y, j=x, band=1)

            if close_band:
                out_rst.close_band()

    # close the dataset if it was created or prompted by <flush_final>
    if flush_final or new_file:
        out_rst.close_file()

    out_rst = None


class GetMinExtent(UpdateInfo):

    """
    Args:
        info1 (ropen or GetMinExtent object)
        info2 (ropen or GetMinExtent object)

    Attributes:
        Inherits from ``info1``.
    """

    def __init__(self, info1, info2):

        if not isinstance(info1, ropen):
            if not isinstance(info1, GetMinExtent):
                if not isinstance(info1, ImageInfo):
                    raise TypeError('The first info argument must be an instance of ropen, GetMinExtent, or ImageInfo.')

        if not isinstance(info2, ropen):
            if not isinstance(info2, GetMinExtent):
                if not isinstance(info2, ImageInfo):
                    if not isinstance(info2, vector_tools.vopen):
                        raise TypeError('The second info argument must be an instance of ropen, vopen, GetMinExtent, or ImageInfo.')

        # Pass the image info properties.
        attributes = inspect.getmembers(info1, lambda ia: not (inspect.isroutine(ia)))
        attributes = [ia for ia in attributes if not (ia[0].startswith('__') and ia[0].endswith('__'))]

        for attribute in attributes:
            setattr(self, attribute[0], attribute[1])

        self.get_overlap_info(info2)

    def copy(self):
        return copy.copy(self)

    def close(self):
        pass

    def get_overlap_info(self, info2):

        self.left = np.maximum(self.left, info2.left)
        self.right = np.minimum(self.right, info2.right)
        self.top = np.minimum(self.top, info2.top)
        self.bottom = np.maximum(self.bottom, info2.bottom)

        if (self.left < 0) and (self.right < 0) or (self.left >= 0) and (self.right >= 0):
            self.cols = int(abs(abs(self.right) - abs(self.left)) / self.cellY)
        elif (self.left < 0) and (self.right >= 0):
            self.cols = int(abs(abs(self.right) + abs(self.left)) / self.cellY)

        if (self.top < 0) and (self.bottom < 0) or (self.top >= 0) and (self.bottom >= 0):
            self.rows = int(abs(abs(self.top) - abs(self.bottom)) / self.cellY)
        elif (self.top >= 0) and (self.bottom < 0):
            self.rows = int(abs(abs(self.top) + abs(self.bottom)) / self.cellY)

        # Rounded dimensions for aligning pixels.
        left_max = np.minimum(self.left, info2.left)
        top_max = np.maximum(self.top, info2.top)

        if (left_max < 0) and (self.left < 0):
            n_col_pixels = int((abs(left_max) - abs(self.left)) / self.cellY)
            self.left_rounded = left_max + (n_col_pixels * self.cellY)
        elif (left_max >= 0) and (self.left >= 0):
            n_col_pixels = int((abs(left_max) - abs(self.left)) / self.cellY)
            self.left_rounded = left_max + (n_col_pixels * self.cellY)
        elif (left_max < 0) and (self.left >= 0):
            n_col_pixels1 = int(abs(left_max) / self.cellY)
            n_col_pixels2 = int(abs(self.left) / self.cellY)
            self.left_rounded = left_max + (n_col_pixels1 * self.cellY) + (n_col_pixels2 * self.cellY)

        if (top_max >= 0) and (self.top >= 0):
            n_row_pixels = int((abs(top_max) - abs(self.top)) / self.cellY)
            self.top_rounded = top_max - (n_row_pixels * self.cellY)
        elif (top_max < 0) and (self.top < 0):
            n_row_pixels = int((abs(top_max) - abs(self.top)) / self.cellY)
            self.top_rounded = top_max - (n_row_pixels * self.cellY)
        elif (top_max >= 0) and (self.top < 0):
            n_row_pixels1 = int(abs(top_max) / self.cellY)
            n_row_pixels2 = int(abs(self.top) / self.cellY)
            self.top_rounded = top_max - (n_row_pixels1 * self.cellY) - (n_row_pixels2 * self.cellY)

        if (self.left_rounded < 0) and (self.right < 0):
            n_col_pixels_r = int((abs(self.left_rounded) - abs(self.right)) / self.cellY)
            self.right_rounded = self.left_rounded + (n_col_pixels_r * self.cellY)
        elif (self.left_rounded >= 0) and (self.right >= 0):
            n_col_pixels_r = int((abs(self.left_rounded) - abs(self.right)) / self.cellY)
            self.right_rounded = self.left_rounded + (n_col_pixels_r * self.cellY)
        elif (self.left_rounded < 0) and (self.right >= 0):
            n_col_pixels_r1 = int(abs(self.left_rounded) / self.cellY)
            n_col_pixels_r2 = int(abs(self.right) / self.cellY)
            self.right_rounded = self.left_rounded + (n_col_pixels_r1 * self.cellY) + (n_col_pixels_r2 * self.cellY)

        if (self.top_rounded < 0) and (self.bottom < 0):
            n_row_pixels_r = int((abs(self.top_rounded) - abs(self.bottom)) / self.cellY)
            self.bottom_rounded = self.top_rounded - (n_row_pixels_r * self.cellY)
        elif (self.top_rounded >= 0) and (self.bottom >= 0):
            n_row_pixels_r = int((abs(self.top_rounded) - abs(self.bottom)) / self.cellY)
            self.bottom_rounded = self.top_rounded - (n_row_pixels_r * self.cellY)
        elif (self.top_rounded >= 0) and (self.bottom < 0):
            n_row_pixels_r1 = int(abs(self.top_rounded) / self.cellY)
            n_row_pixels_r2 = int(abs(self.bottom) / self.cellY)
            self.bottom_rounded = self.top_rounded - (n_row_pixels_r1 * self.cellY) + (n_row_pixels_r2 * self.cellY)


def get_min_extent(image1, image2):

    """
    Finds the minimum extent of two rasters

    Args:
        image1 (dict or object): The first image. If a ``dict``, {left: <left>, right: <right>,
            top: <top>, bottom: <bottom>}.
        image2 (dict or object): The second image. If a ``dict``, {left: <left>, right: <right>,
            top: <top>, bottom: <bottom>}.

    Returns:
        List as [left, right, top, bottom].
    """

    if isinstance(image1, ropen):
        left1 = image1.left
        top1 = image1.top
        right1 = image1.right
        bottom1 = image1.bottom
    else:
        left1 = image1['left']
        top1 = image1['top']
        right1 = image1['right']
        bottom1 = image1['bottom']

    if isinstance(image2, ropen):
        left2 = image2.left
        top2 = image2.top
        right2 = image2.right
        bottom2 = image2.bottom
    else:
        left2 = image2['left']
        top2 = image2['top']
        right2 = image2['right']
        bottom2 = image2['bottom']

    left = np.maximum(left1, left2)
    right = np.minimum(right1, right2)
    top = np.minimum(top1, top2)
    bottom = np.maximum(bottom1, bottom2)

    return left, right, top, bottom


def get_min_extent_list(image_list):

    lefto = image_list[0].left
    righto = image_list[0].right
    topo = image_list[0].top
    bottomo = image_list[0].bottom
    cell_size = image_list[0].cellY

    for img in image_list[1:]:

        lefto, righto, topo, bottomo = \
            get_min_extent(dict(left=lefto, right=righto, top=topo, bottom=bottomo),
                           dict(left=img.left, right=img.right, top=img.top, bottom=img.bottom))

    # Check for East/West, positive/negative dividing line.
    if (righto >= 0) and (lefto <= 0):
        cs = int((abs(lefto) + righto) / cell_size)
    else:
        cs = int(abs(abs(righto) - abs(lefto)) / cell_size)

    if (topo >= 0) and (bottomo <= 0):
        rs = int((abs(bottomo) + topo) / cell_size)
    else:
        rs = int(abs(abs(topo) - abs(bottomo)) / cell_size)

    return [lefto, topo, righto, bottomo, -cell_size, cell_size, rs, cs]


def get_new_dimensions(image_info, kernel_size):

    """
    Gets new [output] image dimensions based on kernel size used in processing.

    Args:
        image_info (object)
        kernel_size (int)

    Returns:
        ``new rows``, ``new columns``, ``new cell size y``, ``new cell size x``
    """

    image_info.rows = int(np.ceil(float(image_info.rows) / float(kernel_size)))
    image_info.cols = int(np.ceil(float(image_info.cols) / float(kernel_size)))

    image_info.cellY = float(kernel_size) * float(image_info.cellY)
    image_info.cellX = float(kernel_size) * float(image_info.cellX)

    return image_info


def n_rows_cols(pixel_index, block_size, rows_cols):

    """
    Adjusts block size for the end of image rows and columns.

    Args:
        pixel_index (int): The current pixel row or column index.
        block_size (int): The image block size.
        rows_cols (int): The total number of rows or columns in the image.

    Example:
        >>> n_rows = 5000
        >>> block_size = 1024
        >>> i = 4050
        >>> adjusted_block_size = n_rows_cols(i, block_size, n_rows)

    Returns:
        Adjusted block size as int.
    """

    return block_size if (pixel_index + block_size) < rows_cols else rows_cols - pixel_index


def n_i_j(pixel_index, offset):

    """
    Args:
        pixel_index (int): Current pixel index.
        block_size (int): Block size to use.

    Returns:
        int
    """

    if pixel_index - offset < 0:
        samp_out = 0
    else:
        samp_out = pixel_index - offset

    return samp_out


def block_dimensions(image_rows, image_cols, row_block_size=1024, col_block_size=1024):

    """
    Args:
        image_rows (int): The number of image rows.
        image_cols (int): The number of image columns.
        row_block_size (Optional[int]): Default is 1024.
        col_block_size (Optional[int]): Default is 1024.

    Returns:
        Row dimensions, Column dimensions
    """

    # set the block dimensions
    if image_rows >= row_block_size:
        row_blocks = row_block_size
    else:
        row_blocks = copy.copy(image_rows)

    if image_cols >= col_block_size:
        col_blocks = col_block_size
    else:
        col_blocks = copy.copy(image_cols)

    return row_blocks, col_blocks


def stats_func(im,
               ignore_value=None,
               stat=None,
               stats_functions=None,
               set_below=None,
               set_above=None,
               set_common=None,
               no_data_value=None):

    im = im[0][:]

    if isinstance(ignore_value, int):

        stat = 'nan{}'.format(stat)

        im[im == ignore_value] = np.nan

    if stat in stats_functions:
        out_array = stats_functions[stat](im, axis=0)
    elif stat == 'nancv':

        out_array = stats_functions['nanstd'](im, axis=0)
        out_array /= stats_functions['nanmean'](im, axis=0)

    elif stat == 'nanmode':
        out_array = sci_mode(im, axis=0, nan_policy='omit')
    elif stat == 'cv':

        out_array = im.std(axis=0)
        out_array /= im.mean(axis=0)

    elif stat == 'min':
        out_array = im.min(axis=0)
    elif stat == 'max':
        out_array = im.max(axis=0)
    elif stat == 'mean':
        out_array = im.mean(axis=0)
    elif stat == 'var':
        out_array = im.var(axis=0)
    elif stat == 'std':
        out_array = im.std(axis=0)
    elif stat == 'sum':
        out_array = im.sum(axis=0)
    elif stat == 'zscore':

        dims, rows, cols = im.shape

        scaler = StandardScaler(with_mean=True, with_std=True)
        out_array = columns_to_nd(scaler.fit_transform(nd_to_columns(im, dims, rows, cols)), dims, rows, cols)

    # Filter values.
    if isinstance(set_below, int):
        out_array[out_array < set_below] = no_data_value

    if isinstance(set_above, int):

        if set_common:

            # Mask unwanted to 1 above threshold.
            out_array[out_array > set_above] = set_above + 1

            # Invert the array values.
            __, out_array = cv2.threshold(np.uint8(out_array), 0, 1, cv2.THRESH_BINARY_INV)

            # Add the common value among all bands.
            out_array *= np.uint8(im[0])

        else:
            out_array[out_array > set_above] = no_data_value

    # Reset no data pixels
    out_array[np.isnan(out_array) | np.isinf(out_array)] = no_data_value

    return out_array


def pixel_stats(input_image,
                output_image,
                stat='mean',
                bands=-1,
                ignore_value=None,
                no_data_value=0,
                set_below=None,
                set_above=None,
                set_common=False,
                be_quiet=False,
                block_rows=1000,
                block_cols=1000,
                out_storage='float32',
                overwrite=False,
                bigtiff='no',
                n_jobs=1):

    """
    Computes statistics on n-dimensions

    Args:
        input_image (str): The (bands x rows x columns) input image to process.
        output_image (str): The output image.
        stat (Optional[str]): The statistic to calculate. Default is 'mean'.
            Choices are ['min', 'max', 'mean', 'median', 'mode', 'var', 'std', 'cv', 'sum', 'zscore'].
        bands (Optional[int or int list]): The bands to include in the statistics. Default is -1, or
            include all bands.
        ignore_value (Optional[int]): A value to ignore in the calculations. Default is None.
        no_data_value (Optional[int]): A no data value to set in ``output_image``. Default is 0.
        set_below (Optional[int]): Set values below ``set_below`` to ``no_data_values``. Default is None.
        set_above (Optional[int]): Set values above ``set_above`` to ``no_data_values``. Default is None.
        set_common (Optional[bool]): Whether to set threshold values to the common pixel among all bands.
            Default is False.
        be_quiet (Optional[bool]): Whether to be quiet and do not report progress status. Default is False.
        block_rows (Optional[int]): The number of rows in the block. Default is 1000.
        block_cols (Optional[int]): The number of columns in the block. Default is 1000.
        out_storage (Optional[str]): The output raster storage. Default is 'float32'.
        overwrite (Optional[bool]): Whether to overwrite the output image. Default is False.
        bigtiff (Optional[str]): See `create_raster` for details. Default is 'no'.
        n_jobs (Optional[int]): The number of blocks to process in parallel. Default is 1.

    Examples:
        >>> from mpglue.raster_tools import pixel_stats
        >>>
        >>> # Coefficient of variation on all dimensions.
        >>> pixel_stats('/image.tif', '/output.tif', stat='cv')
        >>>
        >>> # Calculate the mean of the first 3 bands, ignoring zeros, and
        >>> #   set the output no data pixels as -999.
        >>> pixel_stats('/image.tif',
        >>>             '/output.tif',
        >>>             stat='mean',
        >>>             bands=[1, 2, 3],
        >>>             ignore_value=0,
        >>>             no_data_value=-999)

    Returns:
        None, writes to ``output_image``.
    """

    if stat not in ['min', 'max', 'mean', 'median', 'mode', 'var', 'std', 'cv', 'sum', 'zscore']:

        logger.error('{} is not an option.'.format(stat))
        raise NameError

    stats_functions = dict(nanmean=np.nanmean,
                           nanmedian=np.nanmedian,
                           nanvar=np.nanvar,
                           nanstd=np.nanstd,
                           nanmin=np.nanmin,
                           nanmax=np.nanmax,
                           nansum=np.nansum,
                           median=np.median,
                           mode=sci_mode)

    params = dict(ignore_value=ignore_value,
                  stat=stat,
                  stats_functions=stats_functions,
                  set_below=set_below,
                  set_above=set_above,
                  set_common=set_common,
                  no_data_value=no_data_value)

    with ropen(input_image) as i_info:

        info_list = [i_info]

        if isinstance(bands, list):
            bands = [bands]
        elif isinstance(bands, int):

            if bands == -1:
                bands = [list(range(1, i_info.bands+1))]
            else:
                bands = [bands]

        if i_info.bands <= 1:

            logger.error('The input image only has {:d} band. It should have at least 2.'.format(i_info.bands))
            raise ValueError

        # Copy the input information.
        o_info = i_info.copy()

        o_info.update_info(bands=1,
                           storage=out_storage)

        bp = BlockFunc(stats_func,
                       info_list,
                       output_image,
                       o_info,
                       proc_info=i_info,
                       print_statement='\nGetting pixel stats for {} ...\n'.format(input_image),
                       d_types=['float32'],
                       be_quiet=be_quiet,
                       band_list=bands,
                       n_jobs=n_jobs,
                       block_rows=block_rows,
                       block_cols=block_cols,
                       overwrite=overwrite,
                       bigtiff=bigtiff,
                       **params)

        bp.run()

    i_info = None


# def hist_equalization(img, n_bins=256):
#
#     """
#     Computes histogram equalization on an image array
#
#     Args:
#         img (ndarray)
#         n_bins (Optional[int])
#
#     Returns:
#         Histogram equalized image & normalized cumulative distribution function
#     """
#
#     rows, cols = img.shape
#
#     imhist, bins = np.histogram(img.flat, n_bins, normed=True)  # get image histogram
#     cdf = imhist.cumsum()                               # cumulative distribution function
#     cdf = 255 * cdf / cdf[-1]                           # normalize
#
#     img = np.interp(img.flat, bins[:-1], cdf)           # use linear interpolation of cdf to find new pixel values
#
#     img = img.reshape(rows, cols)                        # reshape
#
#     return img, cdf


def match_histograms(source_array, target_hist, n_bins):

    image_rows, image_cols = source_array.shape

    source_array_flat = source_array.flatten()

    hist1, bins = np.histogram(source_array_flat, n_bins, range=[1, 255])

    # Cumulative distribution function.
    cdf1 = hist1.cumsum()
    cdf2 = target_hist.cumsum()

    # Normalize
    cdf1 = (255. * cdf1 / cdf1[-1]).astype(np.uint8)
    cdf2 = (255. * cdf2 / cdf2[-1]).astype(np.uint8)

    # cdf_m = np.ma.masked_equal(cdf,0)
    # 2 cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # 3 cdf = np.ma.filled(cdf_m,0).astype('uint8')

    matched_image = np.interp(source_array_flat, bins[:-1], cdf1)

    matched_image = np.interp(matched_image, cdf2, bins[:-1]).reshape(image_rows, image_cols)

    matched_image[source_array == 0] = 0

    return matched_image, hist1


def fill_ref_histogram(image_array, n_bins):

    source_array_flat = image_array.flatten()

    hist, __ = np.histogram(source_array_flat, n_bins, range=[1, 255])

    return hist


def histogram_matching(image2adjust, reference_list, output_image, band2match=-1, n_bins=254,
                       overwrite=False, vis_hist=False):

    """
    Adjust one reference image to another using image using histogram matching

    Args:
        image2adjust (str): The image to adjust.
        reference_list (str list): A list of reference images.
        output_image (str): The output adjusted image.
        band2match (Optional[int]): The band or bands to adjust. Default is -1, or all bands.
        n_bins (Optional[int]): The number of bins. Default is 254 (ignores 0).
        overwrite (Optional[bool]): Whether to overwrite an existing ``output_image``. Default is False.
        vis_hist (Optional[bool]): Whether to plot the band histograms. Default is False.

    Returns:
        None, writes to ``output_image``.
    """

    # Open the images
    with ropen(image2adjust) as match_info:

        if band2match == -1:
            bands = list(range(1, match_info.bands+1))
        else:
            bands = [band2match]

        # Copy the input information.
        o_info = match_info.copy()
        o_info.bands = len(bands)

        if overwrite:
            overwrite_file(output_image)

        # Create the output.
        with create_raster(output_image, o_info) as out_rst:

            color_list = ['r', 'g', 'b', 'o', 'c', 'k', 'y']

            # Match each band.
            for bi, band in enumerate(bands):

                match_array = match_info.read(bands2open=band)

                for ri, reference_image in enumerate(reference_list):

                    with ropen(reference_image) as ref_info:
                        ref_array = ref_info.read(bands2open=band)

                    ref_info = None

                    if ri == 0:
                        h2 = fill_ref_histogram(ref_array, n_bins)
                    else:
                        h2 += fill_ref_histogram(ref_array, n_bins)

                adjusted_array, h1 = match_histograms(match_array, h2, n_bins)

                out_rst.write_array(adjusted_array, band=band)

                out_rst.close_band()

                if vis_hist:
                    plt.plot(range(len(h1+1)), [0]+h1, color=color_list[bi], linestyle='-')
                    plt.plot(range(len(h2+1)), [0]+h2, color=color_list[bi], linestyle='--')

    match_info = None
    out_rst = None

    if vis_hist:
        plt.show()

    plt.close()


def _add_unique_values(segmented_objects):

    object_image_array = np.copy(segmented_objects)

    # binarize
    segmented_objects[segmented_objects > 0] = 1

    # Label the objects, in sequential order.
    objects, n_objects = lab_img(segmented_objects)

    index = np.unique(objects)

    # Get random values.
    random_values = np.random.uniform(2, 255, size=len(index))

    # Here we give each object a random value between 2-255.
    for noi, n_object in enumerate(index):

        if n_object == 0:
            continue

        # Check if any object has been labeled.
        # object_image_array[object_image_array > 1] = object_image_array

        # Give any value <= 1 a random value.
        object_image_array[(object_image_array <= 1) & (objects == n_object)] = int(random_values[noi])

    return object_image_array


def quick_plot(image_arrays, titles=['Field estimates'], colorbar_labels=['ha'], color_maps=['gist_stern'],
               out_fig=None, unique_values=False, dpi=300, font_size=12, colorbar_font_size=7,
               font_face='Calibri', fig_size=(5, 5), image_mins=[None], image_maxes=[None], discrete_list=[],
               class_list=[], layout='by', tile_size=256, clip_limit=1.):

    """
    Args:
        image_array (ndarray list): A list of image arrays to plot.
        titles (Optional[str list]): A list of subplot title labels. Default is ['Field estimates'].
        colorbar_labels (Optional[str list]): A list of colorbar labels. Default is ['ha'].
        color_maps (Optional[str list]): A list of colormaps to plot. Color maps can be found at
            http://matplotlib.org/examples/color/colormaps_reference.html. e.g., 'ocean', 'gist_earth',
            'terrain', 'gist_stern', 'brg', 'cubehelix', 'gnuplot', 'CMRmap'. Default is 'gist_stern'.
            Default is ['gist_stern'].
        out_fig (Optional[str]): An output figure to write to. Default is None.
        unique_values (Optional[bool]): Whether to create unique values for each object. Default is False.
        dpi (Optional[int]): The plot DPI. Default is 300.
        font_size (Optional[int]): The plot font size. Default is 12.
        font_face (Optional[str]): The plot font face type. Default is 'Calibri'.
        fig_size (Optional[int tuple]): The plot figure size (width, height). Default is (5, 5).
        discrete_list (Optional[bool]): Whether the colormap is discrete. Otherwise, continuous. Default is False.
        tile_size (Optional[int]): The tile size (in pixels) for CLAHE. Default is 256.
        clip_limit (Optional[float]): The clip percentage limit for CLAHE. Default is 1.

    Examples:
        >>> import mpglue as gl
        >>> from mappy import raster_tools
        >>>
        >>> i_info = mp.ropen('/image.tif')
        >>> arr = mp.read(i_info)
        >>> raster_tools.quick_plot([arr], colorbar_labels=['Hectares'], color_maps=['gist_earth'])
    """

    # set the parameters
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['font.family'] = font_face
    mpl.rcParams['axes.labelsize'] = font_size  # controls colorbar label size
    mpl.rcParams['xtick.labelsize'] = colorbar_font_size        # controls colorbar tick label size
    mpl.rcParams['ytick.labelsize'] = 9.
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['figure.edgecolor'] = 'white'

    mpl.rcParams['savefig.dpi'] = dpi
    mpl.rcParams['savefig.facecolor'] = 'white'
    mpl.rcParams['savefig.edgecolor'] = 'white'
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = .05

    if fig_size:
        mpl.rcParams['figure.figsize'] = fig_size[0], fig_size[1]      # width, height

    fig = plt.figure(frameon=False)

    if layout == 'over':
        gs = gridspec.GridSpec(len(image_arrays), 1)
    elif layout == 'by':
        gs = gridspec.GridSpec(1, len(image_arrays))
    else:
        raise NameError('The layout should by "by" or "over".')

    gs.update(hspace=.01)
    gs.update(wspace=.01)

    if not discrete_list or len(discrete_list) != len(image_arrays):
        discrete_list = [False] * len(image_arrays)

    zip_list = [range(0, len(image_arrays)), image_arrays, titles, colorbar_labels, color_maps,
                discrete_list, image_mins, image_maxes]

    for ic, image_array, title, colorbar_label, color_map, discrete, image_min, image_max in zip(*zip_list):

        # ax = fig.add_subplot(1, len(image_arrays), ic)
        if layout == 'over':
            ax = fig.add_subplot(gs[ic, 0])
        else:
            ax = fig.add_subplot(gs[0, ic])

        ax.set_title(title)

        image_shape = image_array.shape

        if unique_values:
            image_array = _add_unique_values(image_array)

        if len(image_shape) > 2:

            for ii, im in enumerate(image_array):

                # im_min = np.percentile(im, 2)
                # im_max = np.percentile(im, 98)

                # image_array[ii] = exposure.rescale_intensity(im,
                #                                     in_range=(mins[ii], maxs[ii]),
                #                                     out_range=(0, 255)).astype(np.uint8)

                # image_array[ii] = exposure.equalize_hist(im)
                image_array[ii] = exposure.equalize_adapthist(np.uint8(exposure.rescale_intensity(im,
                                                                                                  out_range=(0, 255))),
                                                              kernel_size=tile_size,
                                                              clip_limit=clip_limit)

            image_array = np.ascontiguousarray(image_array.transpose(1, 2, 0))

        else:

            if isinstance(image_min, int) or isinstance(image_min, float):
                im_min = image_min
            else:
                im_min = np.percentile(image_array, 2)

            if isinstance(image_max, int) or isinstance(image_max, float):
                im_max = image_max
            else:
                im_max = np.percentile(image_array, 98)

            # image_array[image_array < im_min] = 0
            image_array[image_array > im_max] = im_max

        plt.axis('off')

        if len(image_shape) == 2:

            if discrete:

                ip = ax.imshow(image_array)

                if isinstance(color_map, list):
                    color_map = colors.ListedColormap(color_map)
                    # color_map = colorbar.ColorbarBase(ax, cmap=color_map_)
                    ip.set_cmap(color_map)
                elif color_map.lower() == 'random':
                    ip.set_cmap(colors.ListedColormap(np.random.rand(len(class_list), 3)))
                else:
                    ip.set_cmap(_discrete_cmap(len(class_list), base_cmap=color_map))

                ip.set_clim(min(class_list), max(class_list))

            else:

                if color_map.lower() == 'random':

                    image_array = np.ma.masked_where(image_array == 0, image_array)
                    color_map = colors.ListedColormap(np.random.rand(256, 3))
                    color_map.set_bad('none')

                # my_cmap = cm.gist_stern
                # my_cmap.set_under('#E6E6E6', alpha=1)
                ip = ax.imshow(image_array, vmin=im_min, vmax=im_max, clim=[im_min, im_max])
                # modest_image.imshow(ax, image_array, vmin=im_min, vmax=im_max, clim=[im_min, im_max])

                ip.set_cmap(color_map)
                ip.set_clim(im_min, im_max)

        else:
            ip = ax.imshow(image_array)

        ip.axes.get_xaxis().set_visible(False)
        ip.axes.get_yaxis().set_visible(False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='3%', pad=.05)

        cbar = plt.colorbar(ip, orientation='horizontal', cax=cax)

        # cbar = plt.colorbar(ip, fraction=0.046, pad=0.04, orientation='horizontal')
        # cbar = plt.colorbar(ip, orientation='horizontal')#ticks=[-1, 0, 1],
        # cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])
        # cbar.set_label(colorbar_label)
        cbar.outline.set_linewidth(0)
        ticklines = cbar.ax.get_xticklines()
        for line in ticklines:
            line.set_visible(False)
        cbar.update_ticks()
        cbar.ax.set_xlabel(colorbar_label)

        # cbar.set_clim(im_min, im_max)

    gs.tight_layout(fig)

    if isinstance(out_fig, str):

        plt.savefig(out_fig)

        plt.clf()

    else:
        plt.show()

    plt.close(fig)


def cumulative_plot_table(table, label_field, data_field='DATA_LYR1', lw_field='MEAN_LYR1',
                          threshold_field='MAX_LYR1', threshold='none', small2large=True, out_fig=None,
                          plot_hist=False, labels2exclude=[], log_data=False, standardize=False,
                          color_by_adm1=False, line_weight_weighting=.00001, **kwargs):

    """
    Plots histograms from table data

    Args:
        table (str): The table with the data.
        label_field (str): The column in ``table`` with the label.
        data_field (Optional[str]): The column in ``table`` with the data. Default is 'DATA_LYR1'.
        lw_field (Optional[str]): The column in ``table`` with the line weights. Default is 'MEAN_LYR1'.
        threshold_field (Optional[str]): The column in ``table`` with the threshold cutoff data.
            Default is 'MAX_LYR1'.
        threshold (Optional[str]): The threshold with ``threshold_field``. Default is 'none'.
        small2large (Optional[bool]): Whether to sort the fields small to large. Default is True.
        out_fig (Optional[str]): The output figure (otherwise pyplot.show). Default is None.
        plot_hist (Optional[bool]): Whether to plot the regular histogram (otherwise cumulative histogram).
            Default is False.
        labels2exclude (Optional[str list]): A list of labels to exclude from the plot. Default is [].
        log_data (Optional[bool]): Whether to log the data. Default is False.
        standardize (Optional[bool]): Whether to standardize the data. Default is False.
        color_by_adm1 (Optional[bool]): Whether to color by ADM (otherwise random colors). Default is False.

    Examples:
        >>> from mappy import raster_tools
        >>> raster_tools.cumulative_plot_table('/PRY_all_bands_fields_centroids_stats_join.csv',
        >>>                                    'NAME_1', labels2exclude=['asuncin', 'central', 'cordillera'])
        >>> # or
        >>> raster_tools.cumulative_plot_table('/PRY_all_bands_fields_centroids_stats_join.csv', 'NAME_1',
        >>>                                    labels2exclude=['asuncin', 'central', 'cordillera'],
        >>>                                    log_data=True)
        >>> # or
        >>> # threshold defines what is shown
        >>> # here, only zones with less than 200 max are shown
        >>> raster_tools.cumulative_plot_table('/zonal_stats.csv', 'UNQ', lw_field='MED_LYR1',
        >>>                                    threshold_field='MAX_LYR1', threshold='<200')
    """

    # Pandas
    try:
        import pandas
    except ImportError:
        raise ImportError('Pandas must be installed')

    from itertools import cycle

    col_gen = cycle('bgrcmk')

    if isinstance(table, str):

        try:
            df = pandas.read_csv(table)
        except:
            df = pandas.read_excel(table)

    else:
        df = table

    if isinstance(out_fig, str):

        dpi = 300
        fig = plt.figure(figsize=(10, 7), dpi=dpi, facecolor='white')
        ax = fig.add_subplot(111, axisbg='white')

    else:

        ax = plt.subplot(111)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    text_positions = []

    if log_data:
        range_value = 1
        xr = .5
        yr1 = .25
        yr2 = .5
    else:
        range_value = 50
        xr = 10
        yr1 = 2.5
        yr2 = 5

    adm1_colors = {'BUENOS AIRES': '#8181F7', 'LA PAMPA': '#2ECCFA', 'CORDOBA': '#7401DF', 'ENTRE RIOS': '#00FF00',
                   'SANTIAGO DEL ESTERO': '#B40404', 'SAN LUIS': '#AEB404', 'MENDOZA': '#FF8000',
                   'RIO NEGRO': '#088A29', 'SANTA FE': '#0B615E'}

    # plot the data
    for di, df_row in df.iterrows():

        x = df_row[data_field].split(',')

        if log_data:
            x = [np.log(float(d)) for d in x]
        else:
            x = [float(d) for d in x]

        if threshold != 'none':

            # get the sign
            the_sign = threshold[0]

            # the threshold
            the_threshold = int(threshold[1:])

            if the_sign == '<':

                if df_row[threshold_field] >= the_threshold:
                    continue

            else:

                if df_row[threshold_field] < the_threshold:
                    continue

        try:
            line_weight = float(df_row[lw_field]) * line_weight_weighting
        except:
            line_weight = 1

        try:
            label = u''.join(df_row[label_field])
        except:
            label = df_row[label_field].decode('utf-8')

        if label.encode('ascii', 'ignore').lower() in labels2exclude:
            continue

        n_area = sum(x)

        y = [(float(n) / n_area) * 100. for n in x]

        y = np.sort(y).cumsum()

        # color = np.random.rand(3,)
        # color = col_gen.next()
        if color_by_adm1:
            try:
                color = adm1_colors[df_row.provincia]
            except:
                color = 'black'
        else:
            color = cm.nipy_spectral(random_float(0, 1))

        if plot_hist:

            h, b = np.histogram(x, bins=1000, range=(1, max(x)), density=True)
            h /= float(max(h))
            ax.plot(b[1:], h)

        else:

            if standardize:
                ax.plot(np.sort(x) / float(max(x)), y, c=color, lw=line_weight, alpha=.5)
            else:
                ax.plot(np.sort(x), y, c=color, lw=line_weight, alpha=.5)

            plot_x_position = np.sort(x)[-1]
            plot_y_position = y[-1]

            if text_positions:

                for text_position in text_positions:

                    if ((plot_x_position - text_position) > -range_value) and \
                            ((plot_x_position - text_position) < range_value):

                        plot_x_position -= xr
                        plot_y_position += yr1

                    elif ((plot_x_position - text_position) >= range_value) and \
                            ((plot_x_position - text_position) < range_value):

                        plot_x_position += xr
                        plot_y_position += yr2

            text_positions.append(plot_x_position)

            ax.text(plot_x_position, plot_y_position, label.title(), color=color, fontsize=20, alpha=.5)

    if not plot_hist:
        ax.set_ylim(0, 100)

    # ax.set_xlim(0, 500)

    plt.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='on',
                    left='off',
                    right='off',
                    labelleft='on')

    plt.ylabel('Percent of cropland total', fontsize=16)
    plt.xlabel('Field size (ha)', fontsize=16)

    if isinstance(out_fig, str):

        plt.savefig(out_fig, dpi=dpi, bbox_inches='tight', pad_inches=.1)
        plt.clf()

    else:
        plt.show()

    plt.close()


def cumulative_plot_array(image_array, small2large=True, out_fig=None):

    """
    Args:
        image_array (ndarray): Segments are area.
        small2large (Optional[bool]): Whether to sort the x-axis from small to large fields as range,
            otherwise sort by size. Default is True.
        out_fig (Optional[str])
    """

    # SciPy
    try:
        from scipy.ndimage.measurements import label as nd_label
    except ImportError:
        raise ImportError('SciPy must be installed')

    # Scikit-learn
    try:
        from skimage.measure import regionprops
    except ImportError:
        raise ImportError('Scikit-learn must be installed')

    image_shape = image_array.shape

    plot_multiple = False
    if len(image_shape) > 2:
        plot_multiple = True

    if isinstance(out_fig, str):

        dpi = 300
        fig = plt.figure(figsize=(10, 7), dpi=dpi)
        ax = fig.add_subplot(111)

    else:

        ax = plt.subplot(111)

    ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if plot_multiple:

        cs = []
        xs = []

        for image_band in image_array:

            # create binary
            arr_b = np.where(image_band > 0, 1, 0).astype(np.uint8)
            o, n_o = nd_label(arr_b)
            p = regionprops(o, intensity_image=image_band)

            # convert ha to square kilometers
            l = np.asarray([(pp.max_intensity / 100.) for pp in p]).astype(np.float32).sort()

            n_features = len(list(l))
            n_area = l.sum()

            x = []
            for i in range(1, n_features+1):
                x.append((i / n_features) * 100.)

            y = [(n / n_area) * 100. for n in l]

            c = np.sort(l).cumsum()

            if small2large:
                x = np.arange(np.sort(l).size)
            else:
                x = np.sort(l)

            cs.append([c[0], c[-1]])
            xs.append([x[0], x[-1]])

            ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

            ax.plot(x, c)

            ax.fill_between(x, c, facecolor=np.random.rand(3,), alpha=.5)

            cc = np.multiply(c, 100.)

            i5 = np.percentile(cc, 50)
            for i5_index in range(0, len(cc)):
                if i5+5 > cc[i5_index] > i5-5:
                    break

            i75 = np.percentile(cc, 75)
            for i75_index in range(0, len(cc)):
                if i75+5 > cc[i75_index] > i75-5:
                    break

            i90 = np.percentile(cc, 90)
            for i90_index in range(0, len(cc)):
                if i90+50 > cc[i90_index] > i90-50:
                    break

            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="white", lw=.5)

            # ax.scatter(x[i5_index], c[i5_index], c='black', s=40, marker='o', edgecolor='black')
            # ax.text(x[i5_index], c[i5_index], '50%', size=10, bbox=bbox_props)
            #
            # ax.scatter(x[i75_index], c[i75_index], c='black', s=40, marker='o', edgecolor='black')
            # ax.text(x[i75_index], c[i75_index], '75%', size=10, bbox=bbox_props)

            ax.stem([x[i90_index]], [c[i90_index]], linefmt='b-.', markerfmt='bo')

        plt.tick_params(axis="both",
                        which="both",
                        bottom="off",
                        top="off",
                        labelbottom="on",
                        left="off",
                        right="off",
                        labelleft="on")

        min_c, max_c, min_x, max_x = 0, 0, 0, 0

        for c, x in zip(cs, xs):

            min_c = min(c[0], min_c)
            max_c = max(c[1], max_c)

            min_x = min(x[0], min_x)
            max_x = max(x[1], max_x)

        plt.ylim(min_c, max_c)
        plt.xlim(min_x, max_x)

        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

    else:

        # create binary
        arr_b = np.where(image_array > 0, 1, 0).astype(np.uint8)
        o, n_o = nd_label(arr_b)
        p = regionprops(o, intensity_image=image_array)

        # convert ha to square kilometers
        l = np.asarray([(pp.max_intensity / 100.) for pp in p])

        c = np.sort(l).cumsum()

        if small2large:
            x = np.arange(np.sort(l).size)
        else:
            x = np.sort(l)

        plt.ylim(c[0], c[-1])
        plt.xlim(x[0], x[-1])

        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.tick_params(axis="both",
                        which="both",
                        bottom="off",
                        top="off",
                        labelbottom="on",
                        left="off",
                        right="off",
                        labelleft="on")

        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        ax.plot(x, c)

        ax.fill_between(x, c, facecolor='#3104B4', alpha=.7)#np.random.rand(3,))

        # h5 = np.percentile(c, 50)
        # i5 = np.where(c == h5)
        i5 = len(c) / 2
        i25 = int(len(c) * .25)
        i75 = int(len(c) * .75)

        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="white", lw=.5)

        ax.scatter(x[i5], c[i5], c='black', s=40, marker='o', edgecolor='black')
        ax.text(x[i5]-100, c[i5]-100, '50%', size=10, bbox=bbox_props)

        ax.scatter(x[i25], c[i25], c='black', s=40, marker='o', edgecolor='black')
        ax.text(x[i25]-10, c[i25]-10, '25%', size=10, bbox=bbox_props)

        ax.scatter(x[i75], c[i75], c='black', s=40, marker='o', edgecolor='black')
        ax.text(x[i75]-10, c[i75]-10, '75%', size=10, bbox=bbox_props)

        # ax2 = ax.twinx()
        # ax2.hist(c, 100, color="#3F5D7D", alpha=.7)

    if small2large:
        plt.xlabel('Sorted fields, small to large (order)', fontsize=16)
    else:
        plt.xlabel('Sorted fields, small to large (Square km)', fontsize=16)

    plt.ylabel('Square km', fontsize=16)

    if isinstance(out_fig, str):

        plt.savefig(out_fig, dpi=dpi, bbox_inches='tight', pad_inches=.1)
        plt.clf()

    else:
        plt.show()

    plt.close()


def rasterize_vector(in_vector,
                     out_raster,
                     burn_id='Id',
                     cell_size=None,
                     storage='float32',
                     match_raster=None,
                     bigtiff='no',
                     in_memory=False,
                     initial_value=0,
                     where_clause=None,
                     return_array=False,
                     all_touched=True,
                     **kwargs):

    """
    Rasterizes a vector dataset

    Args:
        in_vector (str): The vector to rasterize.
        out_raster (str): The output image.
        burn_id (Optional[str]): The attribute id of ``in_vector`` to burn into ``out_raster``. Default is 'Id'.
        cell_size (Optional[float]): The output raster cell size. Default is None. *Needs to be given if
            ``match_raster``=None.
        storage (Optional[str])
        match_raster (Optional[str]): A raster to match cell size. Default is None.
        bigtiff (Optional[str]): How to handle big TIFF creation option. Default is 'no'.
        in_memory (Optional[bool]): Whether to build ``out_raster`` in memory. Default is False.
        initial_value (Optional[int])
        where_clause (Optional[str])
        return_array (Optional[bool])
        all_touched (Optional[bool]): Whether to rasterize all pixels touched by the vector. Otherwise,
            only include pixels that have their centroids inside of the polygon. Default is True.
        kwargs (Optional[dict]): Creation options.

    Examples:
        >>> # rasterize to the extent of the matching raster
        >>> from mappy.raster_tools import rasterize_vector
        >>>
        >>> rasterize_vector('/in_vector.shp', 'out_image.tif',
        >>>                  match_raster='/some_image.tif')
        >>>
        >>> # rasterize to the extent of the input vector
        >>> rasterize_vector('/in_vector.shp', 'out_image.tif',
        >>>                  burn_id='UNQ', cell_size=30.)
        >>>
        >>> # rasterize to a given extent
        >>> rasterize_vector('/in_vector.shp', 'out_image.tif', burn_id='UNQ',
        >>>                  cell_size=30., top=10000, bottom=5000, left=-8000,
        >>>                  right=-2000, projection='')

    Returns:
        None, writes to ``out_raster``.
    """

    with vector_tools.vopen(in_vector) as v_info:

        if kwargs:

            if not isinstance(cell_size, float):
                raise ValueError('The cell size must be given.')

            if 'projection' not in kwargs:
                raise ValueError('The projection must be given.')

            if 'right' not in kwargs:

                if 'cols' not in kwargs:
                    raise ValueError('Either right or cols must be given.')

                kwargs['right'] = kwargs['left'] + (kwargs['cols'] * cell_size) + cell_size

            if 'bottom' not in kwargs:

                if 'rows' not in kwargs:
                    raise ValueError('Either bottom or rows must be given.')

                kwargs['bottom'] = kwargs['top'] - (kwargs['rows'] * cell_size) - cell_size

            # get rows and columns
            if 'rows' not in kwargs:

                if (kwargs['top'] > 0) and (kwargs['bottom'] >= 0):
                    kwargs['rows'] = int((kwargs['top'] - kwargs['bottom']) / cell_size)
                elif (kwargs['top'] > 0) and (kwargs['bottom'] < 0):
                    kwargs['rows'] = int((kwargs['top'] + abs(kwargs['bottom'])) / cell_size)
                elif (kwargs['top'] < 0) and (kwargs['bottom'] < 0):
                    kwargs['rows'] = int((abs(kwargs['bottom']) - abs(kwargs['top'])) / cell_size)

            if 'cols' not in kwargs:

                if (kwargs['right'] > 0) and (kwargs['left'] >= 0):
                    kwargs['cols'] = int((kwargs['right'] - kwargs['left']) / cell_size)
                elif (kwargs['right'] > 0) and (kwargs['left'] < 0):
                    kwargs['cols'] = int((kwargs['right'] + abs(kwargs['left'])) / cell_size)
                elif (kwargs['right'] < 0) and (kwargs['left'] < 0):
                    kwargs['cols'] = int((abs(kwargs['left']) - abs(kwargs['right'])) / cell_size)

            create_dict = dict(left=kwargs['left'], right=kwargs['right'], top=kwargs['top'],
                               bottom=kwargs['bottom'], projection=kwargs['projection'], storage=storage, bands=1,
                               cellY=cell_size, cellX=-cell_size, rows=kwargs['rows']+1, cols=kwargs['cols']+1)

        else:

            if not isinstance(cell_size, float):
                raise ValueError('The cell size must be given.')

            # get rows and columns
            rows = abs(int((abs(v_info.top) - abs(v_info.bottom)) / cell_size))
            cols = abs(int((abs(v_info.left) - abs(v_info.right)) / cell_size))

            create_dict = dict(left=v_info.left, right=v_info.right, top=v_info.top, bottom=v_info.bottom,
                               projection=v_info.projection, storage=storage, bands=1,
                               cellY=cell_size, cellX=-cell_size, rows=rows, cols=cols)

        if match_raster and not kwargs:

            with ropen(match_raster) as o_info:

                orw = create_raster(out_raster, o_info, bigtiff=bigtiff, in_memory=in_memory)

                orw.get_band(1)
                orw.fill(initial_value)

        else:

            with ropen('create', **create_dict) as o_info:

                orw = create_raster(out_raster, o_info, bigtiff=bigtiff, in_memory=in_memory)

                orw.get_band(1)
                orw.fill(initial_value)

        o_info = None

        # raster dataset, band(s) to rasterize, vector layer to rasterize,
        # burn a specific value, or values, matching the bands :: burn_values=[100]

        if isinstance(where_clause, str):
            v_info.lyr.SetAttributeFilter(where_clause)

        # rasterize_options = gdal.RasterizeOptions(attribute=burn_id,
        #                                           allTouched=all_touched)
        #
        # gdal.RasterizeLayer(orw.datasource,
        #                     [1],
        #                     v_info.lyr,
        #                     options=rasterize_options)

        gdal.RasterizeLayer(orw.datasource,
                            [1],
                            v_info.lyr,
                            options=['ATTRIBUTE={}'.format(burn_id),
                                     'ALL_TOUCHED={}'.format(str(all_touched).upper())])

    if in_memory:

        if isinstance(orw, create_raster) and hasattr(orw, 'datasource'):

            i_info = ImageInfo()

            i_info.update_info(datasource=orw.datasource,
                               hdf_file=False,
                               **create_dict)

    orw.close_file()
    orw = None
    v_info = None

    if in_memory:

        mem_array = i_info.read()

        gdal.Unlink(out_raster)

        if return_array:

            i_info.close()
            i_info = None

            return mem_array

        else:
            return i_info

    else:
        return None


def batch_manage_overviews(image_directory, build=True, image_extensions=['tif'], wildcard=None):

    """
    Creates images overviews for each image in a directory

    Args:
        image_directory (str): The directory to search in.
        build (Optional[bool]): Whether to build overviews (otherwise, remove overviews). Default is True.
        image_extensions (Optional[str list]): A list of image extensions to limit the search to. Default is ['tif'].
        wildcard (Optional[str]): A wildcard search parameter to limit the search to. Default is None.

    Examples:
        >>> import mpglue as gl
        >>>
        >>> # build overviews
        >>> mp.batch_manage_overviews('/image_directory', wildcard='p224*')
        >>>
        >>> # remove overviews
        >>> mp.batch_manage_overviews('/image_directory', build=False, wildcard='p224*')

    Returns:
        None, builds overviews in place for each image in ``image_directory``.
    """

    image_list = os.listdir(image_directory)

    image_extensions = ['*.{}'.format(se) for se in image_extensions]

    images_filtered = []
    for se in image_extensions:
        [images_filtered.append(fn) for fn in fnmatch.filter(image_list, se)]

    if isinstance(wildcard, str):
        images_filtered = fnmatch.filter(images_filtered, wildcard)

    for image in images_filtered:

        if build:
            info = ropen('{}/{}'.format(image_directory, image))
            info.build_overviews()
        else:
            info = ropen('{}/{}'.format(image_directory, image), open2read=False)
            info.remove_overviews()

        info.close()


def _discrete_cmap(n_classes, base_cmap='cubehelix'):

    """
    @original author: Jake VanderPlas
    License: BSD-style

    Creates an N-bin discrete colormap from the specified input map

    Args:
        n_classes (int): The number of classes in the colormap.
        base_cmap (Optional[str]): The colormap to use. Default is 'cubehelix'.
    """

    if not isinstance(n_classes, int):
        raise ValueError('\nThe number of classes must be given as an integer.\n')

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n_classes))
    cmap_name = base.name + str(n_classes)

    return base.from_list(cmap_name, color_list, n_classes)


class QAMasker(object):

    """A class for masking bit-packed quality flags"""

    def __init__(self, qa, sensor, mask_items=None, modis_qa_band=1, modis_quality=2, confidence_level='yes'):

        """
        Args:
            qa (2d array): The quality array.
            sensor (str): The sensor name. Choices are ['HLS', 'L8-pre', 'L8-C1', 'L-C1', 'MODIS']. 
                'L-C1' refers to Collection 1 L4-5 and L7. 'L8-C1' refers to Collection 1 L8.
            mask_items (str list): A list of items to mask.
            modis_qa_position (Optional[int]): The MODIS QA band position. Default is 1.
            modis_quality (Optional[int]): The MODIS quality level. Default is 2.
            confidence_level (Optional[str]): The confidence level. Choices are ['notdet', 'no', 'maybe', 'yes']. 

        References:
            Landsat Collection 1:
                https://landsat.usgs.gov/collectionqualityband

        Examples:
            >>> # Get the MODIS cloud mask.
            >>> qa = QAMasker(<array>, 'MODIS')
            >>>
            >>> # HLS
            >>> qa = QAMasker(<array>, 'HLS', ['cloud'])
            >>> qa.mask

        Returns:
            2d array with values:
                0: clear                
                1: water
                2: shadow
                3: snow or ice
                4: cloud
                5: cirrus cloud
                6: adjacent cloud
                7: saturated                
                8: dropped
                9: terrain occluded
                255: fill
        """

        self.qa = qa
        self.sensor = sensor
        self.modis_qa_band = modis_qa_band
        self.modis_quality = modis_quality

        self._set_dicts()

        if self.sensor == 'MODIS':
            self.mask = self.get_modis_qa_mask()
        else:

            self.mask = np.zeros(self.qa.shape, dtype='uint8')

            for mask_item in mask_items:

                if mask_item in self.qa_flags[self.sensor]:

                    if 'conf' in mask_item:

                        # Has high confidence that
                        #   this condition was met.
                        mask_value = self.conf_dict[confidence_level]

                    else:
                        mask_value = 1

                    self.mask[self.get_qa_mask(mask_item) >= mask_value] = self.fmask_dict[mask_item]

    def _set_dicts(self):

        self.fmask_dict = dict(clear=0,
                               water=1,
                               shadow=2,
                               shadowconf=2,
                               snow=3,
                               snowice=3,
                               snowiceconf=3,
                               cloud=4,
                               cloudconf=4,
                               cirrus=5,
                               cirrusconf=5,
                               adjacent=6,
                               saturated=7,
                               dropped=8,
                               terrain=0,
                               fill=255)

        self.conf_dict = dict(notdet=0,
                              no=1,
                              maybe=2,
                              yes=3)

        self.qa_flags = {'HLS': {'cirrus': (0, 0),
                                 'cloud': (1, 1),
                                 'adjacent': (2, 2),
                                 'shadow': (3, 3),
                                 'snowice': (4, 4),
                                 'water': (5, 5)},
                         'L8-pre': {'cirrus': (13, 12),
                                    'snowice': (11, 10),
                                    'water': (5, 4),
                                    'fill': (0, 0),
                                    'dropped': (1, 1),
                                    'terrain': (2, 2),
                                    'shadow': (7, 6),
                                    'vegconf': (9, 8),
                                    'snowiceconf': (11, 10),
                                    'cirrusconf': (13, 12),
                                    'cloudconf': (15, 14)},
                         'L8-C1': {'cirrus': (12, 11),
                                   'snowice': (10, 9),
                                   'shadowconf': (8, 7),
                                   'cloudconf': (6, 5),
                                   'cloud': (4, 4),
                                   'saturated': (3, 2),
                                   'terrain': (1, 1),
                                   'fill': (0, 0)},
                         'L-C1': {'fill': (0, 0),
                                  'dropped': (1, 1),
                                  'saturated': (3, 2),
                                  'cloud': (4, 4),
                                  'cloudconf': (6, 5),
                                  'shadowconf': (8, 7),
                                  'snowice': (10, 9)},
                         'ARD': {'fill': (0, 0),
                                 'clear': (1, 1),
                                 'water': (2, 2),
                                 'shadow': (3, 3),
                                 'snow': (4, 4),
                                 'cloud': (5, 5)},
                         'MODIS': {'cloud': (0, 0),
                                   'daynight': (3, 3),
                                   'sunglint': (4, 4),
                                   'snowice': (5, 5),
                                   'landwater': (7, 6)}}

        self.modis_bit_shifts = {1: 0,
                                 2: 4,
                                 3: 8,
                                 4: 12,
                                 5: 16,
                                 6: 20,
                                 7: 24}

    def qa_bits(self, what2mask):

        # For confidence bits
        # 0 = not determined
        # 1 = no
        # 2 = maybe
        # 3 = yes

        bit_location = self.qa_flags[self.sensor][what2mask]

        self.b1 = bit_location[0]
        self.b2 = bit_location[1]

    def get_modis_qa_mask(self):

        """
        Reference:
            https://github.com/haoliangyu/pymasker/blob/master/pymasker.py
        """

        # bit_pos = 0
        # bit_len = 2
        #
        # # 0 = high
        # # 1 = medium
        # # 2 = low
        # # 3 = low cloud
        # data_quality = 0
        #
        # bit_len = int('1' * bit_len, 2)
        # value = int(str(data_quality), 2)
        #
        # pos_value = bit_len << bit_pos
        # con_value = value << bit_pos
        #
        # return (self.qa & pos_value) == con_value

        # `modis_mask`
        #   0: best quality
        #   1: good quality
        #   4: fill value
        #
        # `output`
        #   0: good data = clear
        #   255: bad data = fill
        return np.where(np.uint8(self.qa >> self.modis_bit_shifts[self.modis_qa_band] & 4) <= self.modis_quality,
                        self.fmask_dict['clear'],
                        self.fmask_dict['fill'])

    def get_qa_mask(self, what2mask):

        """
        Reference:
            https://github.com/mapbox/landsat8-qa/blob/master/landsat8_qa/qa.py        
        """

        self.qa_bits(what2mask)

        width_int = int((self.b1 - self.b2 + 1) * '1', 2)

        return np.uint8(((self.qa >> self.b2) & width_int))


def _examples():

    sys.exit("""\

    # Get basic image information
    raster_tools.py -i /image.tif --method info

    # Compute the variance over all bands
    raster_tools.py -i /image.tif -o /output.tif --stat var

    # Compute the average over three bands
    raster_tools.py -i /image.tif -o /output.tif --stat mean --bands 1 2 3

    # Set pixels with variance to 0 and keep the common value among all layers.
    raster_tools.py -i /image.tif -o /output.tif --stat var --set-above 0 --set-common --out-storage int16 --no-data -1

    # Compute the majority value among all bands
    raster_tools.py -i /image.tif -o /output.tif --stat mode

    """)


def main():

    parser = argparse.ArgumentParser(description='Raster tools',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--input', dest='input', help='The input image', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output image', default=None)
    parser.add_argument('-m', '--method', dest='method', help='The method to run', default='pixel-stats',
                        choices=['info', 'pixel-stats'])
    parser.add_argument('-b', '--bands', dest='bands', help='A list of bands to process', default=[-1], nargs='+',
                        type=int)
    parser.add_argument('--stat', dest='stat', help='The statistic to compute', default='mean',
                        choices=['min', 'max', 'mean', 'median', 'mode', 'var', 'std', 'cv', 'sum'])
    parser.add_argument('--ignore', dest='ignore', help='A value to ignore', default=None, type=int)
    parser.add_argument('--no-data', dest='no_data', help='The output no data value', default=0, type=int)
    parser.add_argument('--set-above', dest='set_above', help='Set values above threshold to no-data', default=None,
                        type=int)
    parser.add_argument('--set-below', dest='set_below', help='Set values below threshold to no-data', default=None,
                        type=int)
    parser.add_argument('--set-common', dest='set_common',
                        help='Set values above or below thresholds to common values among all bands',
                        action='store_true')
    parser.add_argument('--out-storage', dest='out_storage', help='The output raster storage', default='float32')

    args = parser.parse_args()

    if args.examples:
        _examples()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    if args.method == 'info':

        i_info = ropen(args.input)

        logger.info('\nThe projection:\n')
        logger.info(i_info.projection)

        logger.info('\n======================================\n')

        logger.info('The extent (left, right, top, bottom):\n')
        logger.info('{:f}, {:f}, {:f}, {:f}'.format(i_info.left, i_info.right, i_info.top, i_info.bottom))

        storage_string = 'The data type: {}\n'.format(i_info.storage)

        logger.info('\n{}\n'.format(''.join(['=']*(len(storage_string)-1))))

        logger.info(storage_string)

        logger.info('=========\n')

        logger.info('The size:\n')
        logger.info('{:,d} rows'.format(i_info.rows))
        logger.info('{:,d} columns'.format(i_info.cols))

        if i_info.bands == 1:
            logger.info('{:,d} band'.format(i_info.bands))
        else:
            logger.info('{:,d} bands'.format(i_info.bands))

        logger.info('{:.2f} meter cell size'.format(i_info.cellY))

        i_info.close()

    elif args.method == 'pixel-stats':

        pixel_stats(args.input, args.output, stat=args.stat, bands2process=args.bands,
                    ignore_value=args.ignore, no_data_value=args.no_data,
                    set_below=args.set_below, set_above=args.set_above,
                    set_common=args.set_common, out_storage=args.out_storage)

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                (time.asctime(time.localtime(time.time())), (time.time()-start_time)))


if __name__ == '__main__':
    main()
