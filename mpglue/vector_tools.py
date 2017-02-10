#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 9/24/2011
""" 

import os
import sys
import shutil
import time
import argparse
import subprocess
import ast
import copy
import fnmatch
import atexit
import tarfile

from .paths import get_main_path
import raster_tools
from .errors import TransformError, logger

MAIN_PATH = get_main_path()

# GDAL
try:
    from osgeo import gdal, ogr, osr
    from osgeo.gdalconst import GA_ReadOnly, GA_Update
except ImportError:
    raise ImportError('GDAL must be installed')

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

# Pandas
try:
    import pandas as pd
except ImportError:
    raise ImportError('Pandas must be installed')

# SciPy
try:
    from scipy import stats
except ImportError:
    raise ImportError('SciPy must be installed')

# PySAL
try:
    import pysal
except:
    print('PySAL is not installed')

# Rtree
try:
    import rtree
    rtree_installed = True
except:
    rtree_installed = False

# Pickle
try:
    import cPickle as pickle
except:
    from six.moves import cPickle as pickle
else:
   import pickle


gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')


class RegisterDriver(object):

    """
    Registers a vector driver.

    Args:
        vector_file (str): The vector to register.

    Attributes:
        driver (object)
        f_base (str)
        file_format (str)
    """

    def __init__(self, vector_file):

        self.out_vector = vector_file

        self._get_file_format()

        self.driver = ogr.GetDriverByName(self.file_format)
        self.driver.Register

    def _get_file_format(self):

        __, f_name = os.path.split(self.out_vector)
        self.f_base, file_extension = os.path.splitext(f_name)

        # if 'shp' not in file_extension.lower():
        #     raise NameError('\nOnly shapefiles are currently supported.\n')

        formats = {'.shp': 'ESRI Shapefile',
                   '.mem': 'MEMORY'}

        self.file_format = formats[file_extension]


class vopen(RegisterDriver):

    """
    Gets vector information and file pointer object.

    Args:
        file_name (str): Vector location, name, and extension.
        open2read (Optional[bool]): Whether to open vector as 'read only' (True) or writeable (False).
            Default is True.
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.

    Attributes:
        shp (object)
        lyr (object)
        lyr_def (object)
        feature (object)
        shp_geom (list)
        shp_geom_name (str)
        n_feas (int)
        layer_count (int)
        projection (str list)
        extent (float list)
        left (float)
        top (float)
        right (float)
        bottom (float)
    """

    def __init__(self, file_name, open2read=True, epsg=None):

        self.file_name = file_name
        self.open2read = open2read
        self.epsg = epsg

        self.d_name, self.f_name = os.path.split(self.file_name)
        self.f_base, self.f_ext = os.path.splitext(self.f_name)

        RegisterDriver.__init__(self, self.file_name)

        self.read()
    
        self.get_info()

        # Check open files before closing.
        atexit.register(self.close)

    def read(self):

        self.file_open = False

        if self.open2read:
            self.shp = ogr.Open(self.file_name, GA_ReadOnly)
        else:
            self.shp = ogr.Open(self.file_name, GA_Update)

        if self.shp is None:
            raise NameError('Unable to open {}.'.format(self.file_name))

        self.file_open = True

    def get_info(self):

        # get the layer
        self.lyr = self.shp.GetLayer()
        self.lyr_def = self.lyr.GetLayerDefn()
        
        self.feature = self.lyr.GetFeature(0)
        
        self.shp_geom = self.feature.GetGeometryRef()

        try:
            self.shp_geom_name = self.shp_geom.GetGeometryName()
        except:
            self.shp_geom_name = None

        # get the number of features in the layer
        self.n_feas = self.lyr.GetFeatureCount()

        # get the number of layers in the shapefile
        self.layer_count = self.shp.GetLayerCount()

        # get the projection
        if isinstance(self.epsg, int):

            try:

                self.spatial_reference = osr.SpatialReference()
                self.spatial_reference.ImportFromEPSG(self.epsg)

            except:
                logger.warning('Could not get the spatial reference')

        else:

            try:
                self.spatial_reference = self.lyr.GetSpatialRef()
            except:
                logger.warning('Could not get the spatial reference')

        self.projection = self.spatial_reference.ExportToWkt()

        # get the extent
        self.extent = self.lyr.GetExtent()
        
        self.left = self.extent[0]
        self.top = self.extent[3]
        self.right = self.extent[1]
        self.bottom = self.extent[2]

        self.field_names = [self.lyr_def.GetFieldDefn(i).GetName() for i in xrange(0, self.lyr_def.GetFieldCount())]

    def copy(self):

        """
        Copies the object instance
        """

        return copy.copy(self)

    def copy2(self, output_file):

        """
        Copies the input vector to another vector

        Args:
            output_file (str): The output vector.

        Returns:
            None, writes to ``output_file``.
        """

        __ = self.driver.CopyDataSource(self.shp, output_file)

    def close(self):

        if hasattr(self.shp, 'feature'):
            self.shp.feature.Destroy()
            self.shp.feature = None

        if self.file_open:
            self.shp.Destroy()

        self.shp = None

        self.file_open = False

    def delete(self):

        """
        Deletes an open file
        """

        if not self.open2read:
            raise NameError('The file must be opened in read-only mode.')

        try:
            self.driver.DeleteDataSource(self.file_name)
        except IOError:
            logger.error(gdal.GetLastErrorMsg())
            raise IOError('{} could not be deleted. Check for a file lock.'.format(self.file_name))

        self._cleanup()

    def _cleanup(self):

        """
        Cleans undeleted files
        """

        file_list = fnmatch.filter(os.listdir(self.d_name), '{}*'.format(self.f_name))

        if file_list:

            for rf in file_list:
                os.remove('{}/{}'.format(self.d_name, rf))

    def exit(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


def copy_vector(file_name, output_file):

    """
    Copies a vector file

    Args:
        file_name (str): The file to copy.
        output_file (str): The file to copy to.

    Returns:
        None
    """

    if os.path.isfile(file_name):

        with vopen(file_name) as v_info:
            v_info.copy2(output_file)

        v_info = None


def delete_vector(file_name):

    """
    Deletes a vector file

    Args:
        file_name (str): The file to delete.

    Returns:
        None
    """

    if os.path.isfile(file_name):

        with vopen(file_name) as v_info:
            v_info.delete()

        v_info = None

    # Delete QGIS files.
    d_name, f_name = os.path.split(file_name)
    f_base, f_ext = os.path.splitext(f_name)

    for f in fnmatch.filter(os.listdir(d_name), '{}*.qpj'.format(f_base)):
        os.remove('{}/{}'.format(d_name, f))


class CreateDriver(RegisterDriver):

    """
    Creates a vector driver.

    Args:
        out_vector (str): The vector to create.
        overwrite (bool): Whether to overwrite an existing file.

    Attributes:
        datasource (object)
    """

    def __init__(self, out_vector, overwrite):

        RegisterDriver.__init__(self, out_vector)

        if overwrite:

            if os.path.isfile(out_vector):
                self.driver.DeleteDataSource(out_vector)

        # create the output driver
        self.datasource = self.driver.CreateDataSource(out_vector)

    def close(self):
        self.datasource.Destroy()


class create_vector(CreateDriver):

    """
    Creates a vector file.

    Args:
        out_vector (str): The output file name.
        field_names (Optional[str list]): The field names to create. Default is ['Id'].
        epsg (Optional[int]): The projection of the output vector, given by EPSG projection code. Default is 0.
        projection_from_file (Optional[str]): An file to grab the projection from. Default is None.
        projection (Optional[int]): The projection of the output vector, given as a string. Default is None.
        field_type (Optional[str]): The output field type. Default is 'int'.
        geom_type (Optional[str]): The output geometry type. Default is 'point'. Choices are ['point', 'polygon'].
        overwrite (Optional[bool]): Whether to overwrite an existing file. Default is True.

    Attributes:
        time_stamp (str)
        lyr (object)
        lyr_def (object)
        field_defs (object)

    Returns:
        None
    """

    def __init__(self, out_vector, field_names=['Id'], epsg=0, projection_from_file=None,
                 projection=None, field_type='int', geom_type='point', overwrite=True):

        self.time_stamp = time.asctime(time.localtime(time.time()))

        CreateDriver.__init__(self, out_vector, overwrite)

        if geom_type == 'point':
            geom_type = ogr.wkbPoint
        elif geom_type == 'polygon':
            geom_type = ogr.wkbPolygon

        if epsg > 0:

            sp_ref = osr.SpatialReference()
            sp_ref.ImportFromEPSG(epsg)

            # create the point layer
            self.lyr = self.datasource.CreateLayer(self.f_base, geom_type=geom_type, srs=sp_ref)

        elif isinstance(projection_from_file, str):

            with vopen(projection_from_file) as p_info:

                sp_ref = osr.SpatialReference()
                sp_ref.ImportFromWkt(p_info.projection)

            p_info = None

            # create the point layer
            self.lyr = self.datasource.CreateLayer(self.f_base, geom_type=geom_type, srs=sp_ref)

        elif isinstance(projection, str):

            sp_ref = osr.SpatialReference()
            sp_ref.ImportFromWkt(projection)

            self.lyr = self.datasource.CreateLayer(self.f_base, geom_type=geom_type, srs=sp_ref)

        else:

            self.lyr = self.datasource.CreateLayer(self.f_base, geom_type=geom_type)

        self.lyr_def = self.lyr.GetLayerDefn()

        # create the field
        if field_type == 'int':

            self.field_defs = [ogr.FieldDefn(field, ogr.OFTInteger) for field in field_names]

        elif field_type == 'float':

            self.field_defs = [ogr.FieldDefn(field, ogr.OFTReal) for field in field_names]

        elif field_type == 'string':

            self.field_defs = []

            for field in field_names:

                field_def = ogr.FieldDefn(field, ogr.OFTString)
                field_def.SetWidth(20)
                self.field_defs.append(field_def)

        # create the fields
        [self.lyr.CreateField(field_def) for field_def in self.field_defs]


def rename_vector(input_file, output_file):

    """
    Renames a shapefile and all of its associated files

    Args:
        input_file (str): The file to rename.
        output_file (str): The renamed file.

    Examples:
        >>> from mappy import vector_tools
        >>> vector_tools.rename_vector('/in_vector.shp', '/out_vector.shp')

    Returns:
        None
    """

    d_name, f_name = os.path.split(input_file)
    f_base, f_ext = os.path.splitext(f_name)

    od_name, of_name = os.path.split(output_file)
    of_base, of_ext = os.path.splitext(of_name)

    associated_files = os.listdir(d_name)

    for associated_file in associated_files:

        a_base, a_ext = os.path.splitext(associated_file)

        if a_base == f_base:

            try:
                os.rename('{}/{}'.format(d_name, associated_file), '{}/{}{}'.format(od_name, of_base, a_ext))
            except OSError:
                logger.error(gdal.GetLastErrorMsg())
                raise OSError('Could not write {} to file.'.format(of_base))


def merge_vectors(shps2merge, merged_shapefile):

    """
    Merges a list of shapefiles into one shapefile

    Args:
        shps2merge (str list): A list of shapefiles to merge.
        merged_shapefile (str): The output merged shapefile.

    Examples:
        >>> from mappy import vector_tools
        >>> vector_tools.merge_vectors(['/in_shp_01.shp', '/in_shp_02.shp'],
        >>>                            '/merged_file.shp')

    Returns:
        None, writes to ``merged_shapefile``.
    """

    from ._gdal import ogr2ogr

    # First copy any of the shapefiles in
    # the list so that we have something
    # to merge to.
    d_name, f_name = os.path.split(shps2merge[0])
    f_base, f_ext = os.path.splitext(f_name)

    od_name, of_name = os.path.split(merged_shapefile)
    of_base, of_ext = os.path.splitext(of_name)

    associated_files = os.listdir(d_name)

    for associated_file in associated_files:

        a_base, a_ext = os.path.splitext(associated_file)

        if a_base == f_base:

            out_file = '{}/{}{}'.format(od_name, of_base, a_ext)

            if not os.path.isfile(out_file):
                shutil.copy2('{}/{}'.format(d_name, associated_file), out_file)

    # Then merge each shapefile into the
    # output file.
    for shp2merge in shps2merge[1:]:

        print('Merging {} ...'.format(shp2merge))

        ogr2ogr.main(['', '-f', 'ESRI Shapefile', '-update', '-append', merged_shapefile, shp2merge, '-nln', of_base])


def add_point(x, y, layer_object, field, value2write):

    """
    Adds a point to an existing vector.

    Args:
        x (float)
        y (float)
        layer_object (object)
        field (str)
        value2write (str)

    Returns:
        None
    """

    pt_geom = ogr.Geometry(ogr.wkbPoint)

    # add the point
    pt_geom.AddPoint(x, y)

    # create a new feature
    feat = ogr.Feature(layer_object.lyr_def)

    feat.SetGeometry(pt_geom)

    # set the field value
    feat.SetField(field, value2write)

    # create the point
    layer_object.lyr.CreateFeature(feat)

    feat.Destroy()


def add_polygon(vector_object, xy_pairs=None, field_vals=None, geometry=None):

    """
    Args:
        vector_object (object): Class instance of ``create_vector``.
        xy_pairs (Optional[tuple]): List of x, y coordinates that make the feature. Default is None.
        field_vals (Optional[dict]): A dictionary of field values to write. They should match the order
            of ``field_defs``. Default is [].
        geometry (Optional[object]): A polygon geometry object to write (in place of ``xy_pairs``). Default is None.

    Returns:
        None
    """

    poly_geom = ogr.Geometry(ogr.wkbLinearRing)

    # Add the points
    if isinstance(xy_pairs, tuple) or isinstance(xy_pairs, list):

        for pair in xy_pairs:
            poly_geom.AddPoint(float(pair[0]), float(pair[1]))

        poly = ogr.Geometry(ogr.wkbPolygon)

        poly.AddGeometry(poly_geom)

    else:
        poly = geometry

    feature = ogr.Feature(vector_object.lyr_def)
    feature.SetGeometry(poly)

    # set the fields
    if field_vals:

        for field, val in field_vals.iteritems():
            feature.SetField(field, val)

    vector_object.lyr.CreateFeature(feature)

    vector_object.lyr.SetFeature(feature)

    feature.Destroy()


def dataframe2dbf(df, dbf_file, my_specs=None):

    """
    Converts a pandas.DataFrame into a dbf.

    Author:  Dani Arribas-Bel <darribas@asu.edu>
        https://github.com/GeoDaSandbox/sandbox/blob/master/pyGDsandbox/dataIO.py#L56

    Args:
        df (object): Pandas dataframe.
        dbf_file (str): The output .dbf file.
        my_specs (Optional[list]): List with the field_specs to use for each column. Defaults to None and
            applies the following scheme:
                int: ('N', 14, 0)
                float: ('N', 14, 14)
                str: ('C', 14, 0)

    Returns:
        None, writes to ``dbf_file``.
    """

    if my_specs:
        specs = my_specs
    else:
        type2spec = {int: ('N', 20, 0),
                     np.int64: ('N', 20, 0),
                     float: ('N', 36, 15),
                     np.float64: ('N', 36, 15),
                     str: ('C', 14, 0)}

        types = [type(df[i].iloc[0]) for i in df.columns]
        specs = [type2spec[t] for t in types]

    db = pysal.open(dbf_file, 'w')
    db.header = list(df.columns)

    db.field_spec = specs

    for i, row in df.T.iteritems():
        db.write(row)

    db.close()


def shp2dataframe(input_shp):

    """
    Uses PySAL to convert shapefile .dbf to a Pandas dataframe

    Author: Dani Arribas-Bel <darribas@asu.edu>
        https://github.com/GeoDaSandbox/sandbox/blob/master/pyGDsandbox/dataIO.py#L56

    Args:
        input_shp (str): The input shapefile

    Returns:
        Pandas dataframe
    """

    df = pysal.open(input_shp.replace('.shp', '.dbf'), 'r')

    df = dict([(col, np.array(df.by_col(col))) for col in df.header])

    return pd.DataFrame(df)


def is_within(x, y, image_info):

    """
    Checks whether x, y coordinates are within an image extent.

    Args:
        x (float): The x coordinate.
        y (float): The y coordinate.
        image_info (object): Object of ``mappy.ropen``.

    Returns:
        ``True`` if ``x`` and ``y`` are within ``image_info``, otherwise ``False``.
    """

    if not isinstance(image_info, raster_tools.ropen):
        raise TypeError('`image_info` must be an instance of `ropen`.')

    if not hasattr(image_info, 'left') or not hasattr(image_info, 'right') \
            or not hasattr(image_info, 'bottom') or not hasattr(image_info, 'top'):

        raise AttributeError('The `image_info` object must have left, right, bottom, top attributes.')

    if (x > image_info.left) and (x < image_info.right) and (y > image_info.bottom) and (y < image_info.top):
        return True
    else:
        return False


def create_point(coordinate_pair, projection_file):

    """
    Creates a point in memory

    Args:
        coordinate_pair (list or tuple): The x, y coordinate pair.
        projection_file (str): The file with the EPSG projection info.
    """

    rsn = '{:f}'.format(abs(np.random.randn(1)[0]))[-4:]

    # Create a temporary point file. The
    #   value field is created and called 'Value'.
    cv = create_vector('temp_points_{}.mem'.format(rsn),
                       field_names=['Value'],
                       projection_from_file=projection_file,
                       geom_type='point')

    # Add a point.
    add_point(coordinate_pair[0], coordinate_pair[1], cv, 'Value', 1)

    return cv


class Transform(object):

    """
    Transforms a x, y coordinate pair

    Args:
        x (float): The source x coordinate.
        y (float): The source y coordinate.
        source_epsg (int): The source EPSG code.
        target_epsg (int): The target EPSG code.

    Examples:
        >>> from mappy.vector_tools import Transfrom
        >>>
        >>> ptr = Transform(740000., 2260000., 102033, 4326)
        >>> print ptr.x, ptr.y
        >>> print ptr.x_transform, ptr.y_transform
    """

    def __init__(self, x, y, source_epsg, target_epsg):

        self.x = x
        self.y = y

        source_sr = osr.SpatialReference()
        target_sr = osr.SpatialReference()

        try:

            if isinstance(source_epsg, int):
                source_sr.ImportFromEPSG(source_epsg)
            elif isinstance(source_epsg, str):
                # source_sr.ImportFromProj4(source_epsg)
                source_sr.ImportFromWkt(source_epsg)

        except:
            logger.error(gdal.GetLastErrorMsg())
            print('EPSG:{:d}'.format(source_epsg))
            raise ValueError('The source EPSG code could not be read.')

        try:

            if isinstance(target_epsg, int):
                target_sr.ImportFromEPSG(target_epsg)
            elif isinstance(target_epsg, str):
                # target_sr.ImportFromProj4(source_epsg)
                target_sr.ImportFromWkt(target_epsg)

        except:
            logger.error(gdal.GetLastErrorMsg())
            print('EPSG:{:d}'.format(target_epsg))
            raise ValueError('The target EPSG code could not be read.')

        try:
            coord_trans = osr.CoordinateTransformation(source_sr, target_sr)
        except:
            logger.error(gdal.GetLastErrorMsg())
            raise TransformError('The coordinates could not be transformed.')

        self.point = ogr.Geometry(ogr.wkbPoint)
        
        self.point.AddPoint(self.x, self.y)
        self.point.Transform(coord_trans)

        self.x_transform = self.point.GetX()
        self.y_transform = self.point.GetY()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.point.Destroy()

    def __del__(self):
        self.__exit__(None, None, None)


class TransformUTM(object):

    """
    Converts an extent envelope to UTM without requiring the UTM EPSG code

    Args:
        grid_envelope (list)
        utm_zone (str or int)
        from_epsg (Optional[int])
    """

    def __init__(self, grid_envelope, utm_zone, to_epsg=None, from_epsg=None, hemisphere=None):

        # Envelope
        # left, right, bottom, top
        self.to_epsg = to_epsg
        self.from_epsg = from_epsg

        if isinstance(self.to_epsg, int):

            if isinstance(hemisphere, str):

                # Northern hemisphere
                if hemisphere == 'N':
                    self.from_epsg = int('326{}'.format(str(utm_zone)))
                else:
                    self.from_epsg = int('327{}'.format(str(utm_zone)))

            else:

                # Northern hemisphere
                if grid_envelope['top'] > 0:
                    self.from_epsg = int('326{}'.format(str(utm_zone)))
                else:
                    self.from_epsg = int('327{}'.format(str(utm_zone)))

        elif isinstance(self.from_epsg, int):

            if isinstance(hemisphere, str):

                # Northern hemisphere
                if hemisphere == 'N':
                    self.to_epsg = int('326{}'.format(str(utm_zone)))
                else:
                    self.to_epsg = int('327{}'.format(str(utm_zone)))

            else:

                # Northern hemisphere
                if grid_envelope['top'] > 0:
                    self.to_epsg = int('326{}'.format(str(utm_zone)))
                else:
                    self.to_epsg = int('327{}'.format(str(utm_zone)))

        else:
            raise NameError('The to or from EPSG code must be given.')

        ptr = Transform(grid_envelope['left'], grid_envelope['top'], self.from_epsg, self.to_epsg)

        self.left = copy.copy(ptr.x_transform)
        self.top = copy.copy(ptr.y_transform)

        ptr = Transform(grid_envelope['right'], grid_envelope['bottom'], self.from_epsg, self.to_epsg)

        self.right = copy.copy(ptr.x_transform)
        self.bottom = copy.copy(ptr.y_transform)


class TransformExtent(object):

    """
    Converts an extent envelope

    Args:
        grid_envelope (list)
        from_epsg (int)
        to_epsg (Optional[int])
    """

    def __init__(self, grid_envelope, from_epsg, to_epsg=4326):

        # Envelope
        # left, right, bottom, top

        if not grid_envelope:
            logger.error('The grid envelope list must be set.')
            raise TypeError('The grid envelope list must be set.')

        self.from_epsg = from_epsg
        self.to_epsg = to_epsg

        ptr = Transform(grid_envelope['left'], grid_envelope['bottom'], self.from_epsg, self.to_epsg)

        self.left = copy.copy(ptr.x_transform)
        self.bottom = copy.copy(ptr.y_transform)

        ptr = Transform(grid_envelope['right'], grid_envelope['top'], self.from_epsg, self.to_epsg)

        self.right = copy.copy(ptr.x_transform)
        self.top = copy.copy(ptr.y_transform)


class RTreeManager(object):

    def __init__(self, base_shapefile=None):

        self.utm_shp_path = '{}/utilities/sentinel'.format(MAIN_PATH.replace('mpglue', 'mappy'))

        # Setup the UTM MGRS shapefile
        if isinstance(base_shapefile, str):
            self.base_shapefile_ = base_shapefile
        else:

            self.base_shapefile_ = '{}/sentinel2_grid.shp'.format(self.utm_shp_path)

            if not os.path.isfile(self.base_shapefile_):

                with tarfile.open('{}/utm_shp.tar.bz2'.format(self.utm_shp_path), mode='r:bz2') as tar:
                    tar.extractall(path=self.utm_shp_path)

        if rtree_installed:
            self.rtree_index = rtree.index.Index(interleaved=False)
        else:
            self.rtree_index = dict()

        # Setup the RTree index
        with vopen(self.base_shapefile_) as bdy_info:

            for f in xrange(0, bdy_info.n_feas):

                bdy_feature = bdy_info.lyr.GetFeature(f)
                bdy_geometry = bdy_feature.GetGeometryRef()
                en = bdy_geometry.GetEnvelope()

                if rtree_installed:
                    self.rtree_index.insert(f, (en[0], en[1], en[2], en[3]))
                else:
                    self.rtree_index[f] = (en[0], en[1], en[2], en[3])

                # bdy_feature.Destroy()
                # bdy_feature = None
                # bdy_geometry = None

        bdy_info = None

        # Load the RTree info
        self.rtree_info = '{}/utilities/sentinel/utm_grid_info.txt'.format(MAIN_PATH.replace('mpglue', 'mappy'))
        self.field_dict = pickle.load(file(self.rtree_info, 'rb'))

    def get_intersecting_features(self, shapefile2intersect=None, envelope=None,
                                  epsg=None, proj4=None, proj=None, lat_lon=None):

        """
        Intersects the RTree index with a shapefile or extent envelope.
        """

        if isinstance(shapefile2intersect, str):

            # Open the base shapefile
            with vopen(shapefile2intersect) as bdy_info:

                bdy_feature = bdy_info.lyr.GetFeature(0)
                bdy_geometry = bdy_feature.GetGeometryRef()
                bdy_envelope = bdy_geometry.GetEnvelope()

            bdy_info = None

            # left, right, bottom, top
            envelope = [bdy_envelope[0], bdy_envelope[1], bdy_envelope[2], bdy_envelope[3]]

        image_envelope = dict(left=envelope[0],
                              right=envelope[1],
                              bottom=envelope[2],
                              top=envelope[3])

        # Transform the points from UTM to WGS84
        if isinstance(epsg, int):
            e2w = TransformExtent(image_envelope, epsg)
            envelope = [e2w.left, e2w.right, e2w.bottom, e2w.top]
        elif isinstance(proj4, str):
            e2w = TransformExtent(image_envelope, proj4)
            envelope = [e2w.left, e2w.right, e2w.bottom, e2w.top]
        elif isinstance(proj, str):
            e2w = TransformExtent(image_envelope, proj, to_epsg=lat_lon)
            envelope = [e2w.left, e2w.right, e2w.bottom, e2w.top]

        if rtree_installed:
            index_iter = self.rtree_index.intersection(envelope)
        else:
            index_iter = xrange(0, len(self.field_dict))

        self.grid_infos = []

        # Intersect the base shapefile bounding box
        #   with the UTM grids.
        for n in index_iter:

            grid_info = self.field_dict[n]

            # Create a polygon object from the coordinates.
            # 0:left, 1:right, 2:bottom, 3:top
            coord_wkt = 'POLYGON (({:f} {:f}, {:f} {:f}, {:f} {:f}, {:f} {:f}, {:f} {:f}))'.format(
                grid_info['extent']['left'],
                grid_info['extent']['top'],
                grid_info['extent']['right'],
                grid_info['extent']['top'],
                grid_info['extent']['right'],
                grid_info['extent']['bottom'],
                grid_info['extent']['left'],
                grid_info['extent']['bottom'],
                grid_info['extent']['left'],
                grid_info['extent']['top'])

            coord_poly = ogr.CreateGeometryFromWkt(coord_wkt)

            if isinstance(shapefile2intersect, str):

                # Check if the feature intersects
                #   the base shapefile.
                if not bdy_geometry.Intersection(coord_poly).IsEmpty():
                    self.grid_infos.append(grid_info)

            else:
                self.grid_infos.append(grid_info)

        # bdy_feature.Destroy()
        # bdy_feature = None
        # bdy_geometry = None

        self._cleanup()

    def _cleanup(self):

        if isinstance(self.base_shapefile_, str):
            delete_vector(self.base_shapefile_)


def intersects_shapefile(shapefile2intersect, base_shapefile=None):

    srt = RTreeManager(base_shapefile=base_shapefile)

    srt.get_intersecting_features(shapefile2intersect=shapefile2intersect)

    return srt.grid_infos


def difference(shapefile2cut, overlap_shapefile, output_shp):

    """
    Computes the difference between two shapefiles

    Args:
        shapefile2cut (str): The shapefile to 'punch' a hole into.
        overlap_shapefile (str): The shapefile that defines the hole.
        output_shp (str): The output shapefile.
    """

    with vopen(shapefile2cut) as cut_info, vopen(overlap_shapefile) as over_info:

        cut_feature = cut_info.lyr.GetFeature(0)
        cut_geometry = cut_feature.GetGeometryRef()

        over_feature = over_info.lyr.GetFeature(0)
        over_geometry = over_feature.GetGeometryRef()

        diff_geometry = cut_geometry.Difference(over_geometry)

        cv = create_vector(output_shp,
                           geom_type='polygon',
                           projection=cut_info.projection)

        diff_feature = ogr.Feature(cv.lyr_def)

        diff_feature.SetGeometry(diff_geometry)

        cv.lyr.CreateFeature(diff_feature)

        cv.close()


def intersects_boundary(meta_dict, boundary_file):

    """
    Checks if an image extent intersects a polygon boundary.

    Args:
        meta_dict (dict): A dictionary of extent information.
            E.g., dict(UL=[x, y], UR=[x, y], LL=[x, y], LR=[x, y]).
        boundary_file (str): A boundary shapefile to check.

    Returns:
        True if ``meta_dict`` coordinates intersect ``boundary_shp``, otherwise False.
    """

    with vopen(boundary_file) as bdy_info:

        bdy_feature = bdy_info.lyr.GetFeature(0)

        bdy_geometry = bdy_feature.GetGeometryRef()

        # Create a polygon object from the coordinates.
        coord_wkt = 'POLYGON (({:f} {:f}, {:f} {:f}, {:f} {:f}, {:f} {:f}, {:f} {:f}))'.format(meta_dict['UL'][0],
                                                                                               meta_dict['UL'][1],
                                                                                               meta_dict['UR'][0],
                                                                                               meta_dict['UR'][1],
                                                                                               meta_dict['LR'][0],
                                                                                               meta_dict['LR'][1],
                                                                                               meta_dict['LL'][0],
                                                                                               meta_dict['LL'][1],
                                                                                               meta_dict['UL'][0],
                                                                                               meta_dict['UL'][1])

        coord_poly = ogr.CreateGeometryFromWkt(coord_wkt)

        # If the polygon object is empty,
        #   then the two do not intersect.
        is_empty = bdy_geometry.Intersection(coord_poly).IsEmpty()

    if is_empty:
        return False
    else:
        return True

    # # for key, coordinate_pair in meta_dict.iteritems():
    #
    # # Create a point in memory.
    # # cv = create_point(coordinate_pair, boundary_shp)
    #
    # # Set a spatial filter to check if the
    # #   point is within the current feature
    # #   (i.e., envelope).
    # cv.lyr.SetSpatialFilterRect(poly_geometry)
    #
    # # Store the point in a list if it
    # #   is within the envelope.
    # n_points = cv.lyr.GetFeatureCount()
    #
    # # Clear the spatial filter.
    # cv.lyr.SetSpatialFilter(None)
    #
    # # Remove the temporary vector.
    # cv = None
    #
    # if n_points > 0:
    #
    #     poly_feature.Destroy()
    #
    #     return True
    #
    # poly_feature.Destroy()
    #
    # return False


def _get_xy_offsets(x, left, right, y, top, bottom, cell_size_x, cell_size_y, round_offset, check):

    # Xs (longitudes)
    if check:
        if (x < left) or (x > right):
            raise ValueError('The x is out of the image extent.')

    if ((x > 0) and (left < 0)) or ((left > 0) and (x < 0)):

        if round_offset:
            x_offset = int(round((abs(x) + abs(left)) / abs(cell_size_x)))
        else:
            x_offset = int((abs(x) + abs(left)) / abs(cell_size_x))

    else:

        if round_offset:
            x_offset = int(round(abs(abs(x) - abs(left)) / abs(cell_size_x)))
        else:
            x_offset = int(abs(abs(x) - abs(left)) / abs(cell_size_x))

    # Ys (latitudes)
    if check:
        if (y > top) or (y < bottom):
            raise ValueError('The y is out of the image extent.')

    if ((y > 0) and (top < 0)) or ((top > 0) and (y < 0)):

        if round_offset:
            y_offset = int(round((abs(y) + abs(top)) / cell_size_y))
        else:
            y_offset = int((abs(y) + abs(top)) / cell_size_y)

    else:

        if round_offset:
            y_offset = int(round(abs(abs(y) - abs(top)) / cell_size_y))
        else:
            y_offset = int(abs(abs(y) - abs(top)) / cell_size_y)

    return x_offset, y_offset


def get_xy_offsets(image_info=None, image_list=None, x=None, y=None, feature=None, xy_info=None,
                   round_offset=False, check_position=True):

    """
    Get coordinate offsets

    Args:
        image_info (object): Object of ``mappy.ropen``.
        image_list (Optional[list]): [left, top, right, bottom, cellx, celly]. Default is [].
        x (Optional[float]): An x coordinate. Default is None.
        y (Optional[float]): A y coordinate. Default is None.
        feature (Optional[object]): Object of ``ogr.Feature``. Default is None.
        xy_info (Optional[object]): Object of ``mappy.vopen`` or ``mappy.ropen``. Default is None.
        round_offset (Optional[bool]): Whether to round offsets. Default is False.
        check_position (Optional[bool])

    Examples:
        >>> from mappy import vector_tools
        >>>
        >>> # With an image and x, y coordinates.
        >>> x, y, x_offset, y_offset = vector_tools.get_xy_offsets(image_info=i_info, x=x, y=y)
        >>>
        >>> # With an image and a feature object.
        >>> x, y, x_offset, y_offset = vector_tools.get_xy_offsets(image_info=i_info, feature=feature)
        >>>
        >>> # With an image and a ``ropen`` or ``vopen` instance.
        >>> x, y, x_offset, y_offset = vector_tools.get_xy_offsets(image_info=i_info, xy_info=v_info)

    Returns:
        X coordinate, Y coordinate, X coordinate offset, Y coordinate offset
    """

    # The offset is given as a single x, y coordinate.
    if isinstance(feature, ogr.Feature):

        # Get point geometry.
        geometry = feature.GetGeometryRef()

        # Get X,Y coordinates.
        x = geometry.GetX()
        y = geometry.GetY()

    # The offset is from a vector or
    #   raster information object.
    elif isinstance(xy_info, vopen) or isinstance(xy_info, raster_tools.ropen):

        x = xy_info.left
        y = xy_info.top

    else:
        if not isinstance(x, float):
            raise ValueError('A coordinate or feature object must be given.')

    # Check if a list or an
    #   object/instance is given.
    if image_list:

        left = image_list[0]
        top = image_list[1]
        right = image_list[2]
        bottom = image_list[3]
        cell_x = image_list[4]
        cell_y = image_list[5]

    else:

        left = image_info.left
        top = image_info.top
        right = image_info.right
        bottom = image_info.bottom
        cell_x = image_info.cellX
        cell_y = image_info.cellY

    # Compute pixel offsets.
    x_offset, y_offset = _get_xy_offsets(x, left, right, y, top, bottom, cell_x, cell_y, round_offset, check_position)

    return x, y, x_offset, y_offset


class get_xy_coordinates(object):

    """
    Converts i, j indices to map coordinates.

    Args:
        i (int): The row index position.
        j (int): The column index position.
        rows (int): The number of rows in the array.
        cols (int): The number of columns in the array.
        image_info (object): An instance of ``raster_tools.ropen``.
    """

    def __init__(self, i, j, rows, cols, image_info):

        self.get_extent(i, j, rows, cols, image_info)

    def get_extent(self, i, j, rows, cols, image_info):

        if (image_info.top > 0) and (image_info.bottom < 0):

            # Get the number of pixels top of center.
            n_pixels_top = int(np.ceil(image_info.top / image_info.cellY))

            if i > n_pixels_top:
                self.top = -(i - n_pixels_top) * image_info.cellY
            else:
                self.top = image_info.top - (i * image_info.cellY)

        else:
            self.top = image_info.top - (i * image_info.cellY)

        if (image_info.right > 0) and (image_info.left < 0):

            # Get the number of pixels left of center.
            n_pixels_left = int(np.ceil(abs(image_info.left) / image_info.cellY))

            if j > n_pixels_left:
                self.left = (j - n_pixels_left) * image_info.cellY
            else:
                self.left = image_info.left + (j * image_info.cellY)

        else:
            self.left = image_info.left + (j * image_info.cellY)

        self.bottom = self.top - (rows * image_info.cellY)
        self.right = self.left + (cols * image_info.cellY)


def spatial_intersection(select_shp, intersect_shp, output_shp, epsg=None):

    """
    Creates a new shapefile from a spatial intersection of two shapefiles

    Args:
        select_shp (str): The shapefile to select from.
        intersect_shp (str): The shapefile to test for intersection.
        output_shp (str): The output shapefile.
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.
    """

    # Open the files.
    with vopen(select_shp, epsg=epsg) as select_info, vopen(intersect_shp, epsg=epsg) as intersect_info:

        tracker_list = []

        # Create the output shapefile
        field_names = list_field_names(select_shp, be_quiet=True)

        if epsg > 0:

            o_shp = create_vector(output_shp, field_names=field_names,
                                  geom_type=select_info.shp_geom_name.lower(), epsg=epsg)

        else:

            o_shp = create_vector(output_shp, field_names=field_names, projection_from_file=select_shp,
                                  geom_type=select_info.shp_geom_name.lower())

        # Iterate over each select feature in the polygon.
        for m in xrange(0, select_info.n_feas):

            if m % 500 == 0:

                if (m + 499) > select_info.n_feas:
                    end_feature = select_info.n_feas
                else:
                    end_feature = m + 499

                print('Select features {:d}--{:d} of {:d} ...'.format(m, end_feature, select_info.n_feas))

            # Get the current polygon feature.
            select_feature = select_info.lyr.GetFeature(m)

            # Set the polygon geometry.
            select_geometry = select_feature.GetGeometryRef()

            # Iterate over each intersecting feature in the polygon.
            for n in xrange(0, intersect_info.n_feas):

                # Get the current polygon feature.
                intersect_feature = intersect_info.lyr.GetFeature(n)

                # Set the polygon geometry.
                intersect_geometry = intersect_feature.GetGeometryRef()

                left, right, bottom, top = intersect_geometry.GetEnvelope()

                # No need to check intersecting features
                # if outside bounds.
                if (left > select_info.right) or (right < select_info.left) or (top < select_info.bottom) \
                        or (bottom > select_info.top):
                    continue

                # Test the intersection.
                if select_geometry.Intersect(intersect_geometry):

                    # Get the id name of the select feature.
                    # select_id = select_feature.GetField(select_field)

                    # Don't add a feature on top of existing one.
                    if m not in tracker_list:

                        field_values = {}

                        # Get the field names and values.
                        for field in field_names:
                            field_values[field] = select_feature.GetField(field)

                        # Add the feature.
                        add_polygon(o_shp, field_vals=field_values, geometry=select_geometry)

                        tracker_list.append(m)

        o_shp.close()


def select_and_save(file_name, out_vector, select_field=None, select_value=None,
                    expression=None, overwrite=True, epsg=None):

    """
    Selects a vector feature by an attribute and save to new file.

    Args:
        file_name (str): The file name to select from.
        out_vector (str): The output vector file.
        select_field (str): The field to select from.
        select_value (str): The field value to select.
        expression (Optional[str]): A conditional expression. E.g., "FIELD = 'VALUE' OR FIELD = 'VALUE2'".
        overwrite (Optional[bool]): Whether to overwrite an existing file. Default is True.
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.

    Returns:
        None

    Examples:
        >>> import mpglue as gl
        >>>
        >>> # Save features where 'Id' is equal to 1.
        >>> mp.select_and_save('/in_shapefile.shp', '/out_shapefile.shp', 'Id', '1')
    """

    if not os.path.isfile(file_name):
        raise NameError('{} does not exist'.format(file_name))

    d_name, f_name = os.path.split(out_vector)
    f_base, f_ext = os.path.splitext(f_name)

    if not os.path.isdir(d_name):
        os.makedirs(d_name)

    # Open the input shapefile.
    with vopen(file_name, epsg=epsg) as v_info:

        # Select the attribute by an expression.
        if not isinstance(expression, str):
            v_info.lyr.SetAttributeFilter("{} = '{}'".format(select_field, select_value))
        else:
            v_info.lyr.SetAttributeFilter(expression)

        # Create the output shapefile.
        out_driver_source = CreateDriver(out_vector, overwrite)

        out_lyr = out_driver_source.datasource.CopyLayer(v_info.lyr, f_base)

        out_lyr = None

    v_info = None


def list_field_names(in_shapefile, be_quiet=False, epsg=None):

    """
    Lists all field names in a shapefile

    Args:
        in_shapefile (str)
        be_quiet (Optional[bool])
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.

    Returns:
        List of field names
    """

    d_name, f_name = os.path.split(in_shapefile)

    # Open the input shapefile.
    with vopen(in_shapefile, epsg=epsg) as v_info:

        df_fields = pd.DataFrame(columns=['Name', 'Type', 'Length'])

        for i in xrange(0, v_info.lyr_def.GetFieldCount()):

            df_fields.loc[i, 'Name'] = v_info.lyr_def.GetFieldDefn(i).GetName()
            df_fields.loc[i, 'Type'] = v_info.lyr_def.GetFieldDefn(i).GetTypeName()
            df_fields.loc[i, 'Length'] = v_info.lyr_def.GetFieldDefn(i).GetWidth()

    if not be_quiet:

        print('\n{} has the following fields:\n'.format(f_name))
        print(df_fields)

    return df_fields


def buffer_vector(file_name, out_vector, distance=None, epsg=None, field_name=None):

    """
    Buffers a vector file.

    Args:
        file_name (str): The vector file to buffer.hex_shp
        out_vector (str): The output, buffered vector file.
        distance (Optional[float]): The buffer distance, in projection units. Default is None.
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.
        field_name (Optional[str]): A field to use as the buffer value. Default is None.

    Returns:
        None

    Examples:
        >>> import mpglue as gl
        >>>
        >>> # 10 km buffer
        >>> mp.buffer_vector('/in_shapefile.shp', '/out_buffer.shp', distance=10000.)
        >>>
        >>> # Buffer by field name
        >>> mp.buffer_vector('/in_shapefile.shp', '/out_buffer.shp', field_name='Buffer')
    """

    if not os.path.isfile(file_name):
        raise NameError('{} does not exist'.format(file_name))

    if not isinstance(distance, float) and not isinstance(field_name, str):
        raise ValueError('Either the distance or field name must be given.')

    d_name, f_name = os.path.split(out_vector)

    if not os.path.isdir(d_name):
        os.makedirs(d_name)

    # open the input shapefile
    with vopen(file_name, epsg=epsg) as v_info:

        if isinstance(distance, float):
            print('\nBuffering {} by {:f} distance ...'.format(f_name, distance))
        else:
            print('\nBuffering {} by field {} ...'.format(f_name, field_name))

        # create the output shapefile
        if isinstance(epsg, int):
            cv = create_vector(out_vector, epsg=epsg, geom_type='polygon')
        else:
            cv = create_vector(out_vector, projection_from_file=v_info.projection, geom_type='polygon')

        df_fields = list_field_names(file_name, be_quiet=True)

        field_names = df_fields['Name'].values.tolist()

        cv = create_fields(cv, field_names,
                           df_fields['Type'].values.tolist(),
                           df_fields['Length'].values.tolist())

        for feature in v_info.lyr:

            in_geom = feature.GetGeometryRef()

            if isinstance(field_name, str):

                try:
                    distance = float(feature.GetField(field_name))
                except:
                    continue

                if distance is None or distance == 'None':
                    continue

            geom_buffer = in_geom.Buffer(distance)

            out_feature = ogr.Feature(cv.lyr_def)
            out_feature.SetGeometry(geom_buffer)

            for fn in field_names:
                out_feature.SetField(fn, feature.GetField(fn))

            cv.lyr.CreateFeature(out_feature)

        cv.close()


def convex_hull(in_shp, out_shp):

    """
    Creates a convex hull of a polygon shapefile

    Reference:
        This code was slightly modified to fit into MapPy.

        Project:        Geothon (https://github.com/MBoustani/Geothon)
        File:           Conversion_Tools/shp_convex_hull.py
        Description:    This code generates convex hull shapefile for point, line and polygon shapefile
        Author:         Maziyar Boustani (github.com/MBoustani)

    Args:
        in_shp (str): The input vector file.
        out_shp (str): The output convex hull polygon vector file.

    Returns:
        None
    """

    v_info = vopen(in_shp)

    # output convex hull polygon
    cv = create_vector(out_shp, projection=v_info.projection, geom_type='polygon')

    # define convex hull feature
    convex_hull_feature = ogr.Feature(cv.lyr_def)

    # define multipoint geometry to store all points
    multipoint = ogr.Geometry(ogr.wkbMultiPoint)

    # iterate over each feature
    for each_feature in xrange(0, v_info.n_feas):

        shp_feature = v_info.lyr.GetFeature(each_feature)

        feature_geom = shp_feature.GetGeometryRef()

        # if geometry is MULTIPOLYGON then need to get POLYGON then LINEARRING to be able to get points
        if feature_geom.GetGeometryName() == 'MULTIPOLYGON':

            for polygon in feature_geom:
                for linearring in polygon:
                    points = linearring.GetPoints()
                    for point in points:
                        point_geom = ogr.Geometry(ogr.wkbPoint)
                        point_geom.AddPoint(point[0], point[1])
                        multipoint.AddGeometry(point_geom)

        # if geometry is POLYGON then need to get LINEARRING to be able to get points
        elif feature_geom.GetGeometryName() == 'POLYGON':

            for linearring in feature_geom:
                points = linearring.GetPoints()
                for point in points:
                    point_geom = ogr.Geometry(ogr.wkbPoint)
                    point_geom.AddPoint(point[0], point[1])
                    multipoint.AddGeometry(point_geom)

        # if geometry is MULTILINESTRING then need to get LINESTRING to be able to get points
        elif feature_geom.GetGeometryName() == 'MULTILINESTRING':

            for multilinestring in feature_geom:
                for linestring in multilinestring:
                    points = linestring.GetPoints()
                    for point in points:
                        point_geom = ogr.Geometry(ogr.wkbPoint)
                        point_geom.AddPoint(point[0], point[1])
                        multipoint.AddGeometry(point_geom)

        # if geometry is MULTIPOINT then need to get POINT to be able to get points
        elif feature_geom.GetGeometryName() == 'MULTIPOINT':

            for multipoint in feature_geom:
                for each_point in multipoint:
                    points = each_point.GetPoints()
                    for point in points:
                        point_geom = ogr.Geometry(ogr.wkbPoint)
                        point_geom.AddPoint(point[0], point[1])
                        multipoint.AddGeometry(point_geom)

        #  if the geometry is POINT or LINESTRING then get points
        else:
            points = feature_geom.GetPoints()
        for point in points:
            point_geom = ogr.Geometry(ogr.wkbPoint)
            point_geom.AddPoint(point[0], point[1])
            multipoint.AddGeometry(point_geom)

    # convert multipoint to convex hull geometry
    convex_hull = multipoint.ConvexHull()

    # set the geomerty of convex hull shapefile feature
    convex_hull_feature.SetGeometry(convex_hull)

    # add the feature to convex hull layer
    cv.lyr.CreateFeature(convex_hull_feature)

    # close the datasets
    v_info.close()
    cv.close()


def create_fields(v_info, field_names, field_types, field_widths):

    """
    Creates fields in an existing vector

    Args:
        v_info (object): A ``vopen`` object.
        field_names (str list): A list of field names to create.
        field_types (str list): A list of field types to create. Choices are ['real', 'float', 'int', str'].
        field_widths (int list): A list of field widths.

    Examples:
        >>> from mappy import vector_tools
        >>>
        >>> vector_tools.create_fields(v_info, ['Id'], ['int'], [5])

    Returns:
        The ``vopen`` object.
    """

    type_dict = {'float': ogr.OFTReal, 'Real': ogr.OFTReal,
                 'int': ogr.OFTInteger, 'Integer': ogr.OFTInteger,
                 'int64': ogr.OFTInteger64, 'Integer64': ogr.OFTInteger64,
                 'str': ogr.OFTString, 'String': ogr.OFTString}

    # Create the fields.
    field_defs = []

    for field_name, field_type, field_width in zip(field_names, field_types, field_widths):

        field_def = ogr.FieldDefn(field_name, type_dict[field_type])

        if field_type in ['str', 'String']:
            field_def.SetWidth(field_width)
        elif field_type in ['float', 'Real']:
            field_def.SetPrecision(4)

        field_defs.append(field_def)

        v_info.lyr.CreateField(field_def)

    return v_info


def add_fields(input_vector, output_vector=None, field_names=['x', 'y'], method='field-xy',
               area_units='km', constant=1, epsg=None, field_breaks=None, default_value=None,
               field_type=None):

    """
    Adds x, y coordinate fields to an existing vector.

    Args:
        input_vector (str): The input vector.
        output_vector (Optional[str]): An output vector iwth ``method``='dissolve'. Default is None.
        field_names (Optional[str list]): The field names. Default is ['x', 'y'].
        method (Optional[str]): The method to use. Default is 'field-xy'. Choices are
            ['field-xy', 'field-id', 'field-area', 'field-constant', 'field-dissolve'].
        area_units (Optional[str]): The units to use for calculating area. Default is 'km', or square km.
            *Assumes the input units are meters if you use 'km'. Choices area ['ha', 'km'].
        constant (Optional[int]): A constant value when ``method`` is equal to field-constant. Default is 1.
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.
        field_breaks (Optional[dict]): The field breaks. Default is None.
        default_value (Optional[int, float, or str]): The default break value. Default is None.

    Returns:
        None, writes to ``input_vector`` in place.
    """

    if method in ['field-xy', 'field-area']:
        field_type = 'float'
    elif method == 'field-id':
        field_type = 'int'
    elif method == 'field-merge':
        field_type = 'str'
    else:

        if not field_type:

            if isinstance(default_value, float):
                field_type = 'float'
            elif isinstance(default_value, int):
                field_type = 'int'
            elif isinstance(default_value, str):
                field_type = 'str'
            else:
                field_type = 'int'

    __, f_name = os.path.split(input_vector)
    f_base, __ = os.path.splitext(f_name)

    # First open the vector file.
    v_info = vopen(input_vector, open2read=False, epsg=epsg)

    # Create the new id field.
    field_names_ = [v_info.lyr_def.GetFieldDefn(i).GetName() for i in xrange(0, v_info.lyr_def.GetFieldCount())]

    if method == 'field-xy':

        if len(field_names) != 2:
            raise ValueError('There should be two {} field names'.format(method))

        for field_name in field_names:
            if field_name not in field_names_:
                v_info = create_fields(v_info, [field_name], [field_type], [None])

        # Add the centroids to each feature.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print('Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas))

            geometry = feature.GetGeometryRef()

            centroid = geometry.Centroid()

            feature.SetField(field_names[0], centroid.GetX())
            feature.SetField(field_names[1], centroid.GetY())

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-breaks':

        if len(field_names) != 2:
            raise ValueError('There should be two {} field names'.format(method))

        for field_name in field_names:
            if field_name not in field_names_:
                v_info = create_fields(v_info, [field_name], [field_type], [50])

        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print('Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas))

            try:
                field_value = float(feature.GetField(field_names[0]))
            except:
                continue

            if field_value is None or field_value == 'None':
                continue

            value_found = False

            for key, break_values in field_breaks.iteritems():

                if isinstance(break_values, int) or isinstance(break_values, float):

                    if field_value >= break_values:

                        feature.SetField(field_names[1], key)

                        value_found = True

                        break

                else:

                    if break_values[0] <= field_value < break_values[1]:

                        feature.SetField(field_names[1], key)

                        value_found = True

                        break

            if not value_found:
                feature.SetField(field_names[1], default_value)

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-id':

        if len(field_names) != 1:
            raise ValueError('There should be one {} field name'.format(method))

        if field_names[0] not in field_names_:
            v_info = create_fields(v_info, field_names, [field_type], [None])

        # Add the id to each feature.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print('Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas))

            feature.SetField(field_names[0], fi+1)

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-constant':

        if len(field_names) != 1:
            raise ValueError('There should be one {} field name'.format(method))

        if field_names[0] not in field_names_:
            v_info = create_fields(v_info, field_names, [field_type], [None])

        # Add the id to each feature.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print('Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas))

            feature.SetField(field_names[0], constant)

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-area':

        if len(field_names) != 1:
            raise ValueError('There should be one {} field name'.format(method))

        if field_names[0] not in field_names_:
            v_info = create_fields(v_info, field_names, [field_type], [None])

        # Add the id to each feature.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print('Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas))

            geometry = feature.GetGeometryRef()

            area = geometry.GetArea()

            # Convert square meters to square kilometers or to hectares
            if area_units == 'km':
                area *= .000001
            elif area_units == 'ha':
                area *= .0001

            # float('%.4f' % area)

            feature.SetField(field_names[0], area)

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-dissolve':

        if len(field_names) != 1:
            raise ValueError('There should be one {} field name'.format(method))

        if not isinstance(output_vector, str):
            raise ValueError('The output vector must be given.')

        # Dissolve the field.
        com = 'ogr2ogr {} {} -dialect sqlite -sql "SELECT ST_Union(geometry), \
        {} FROM {} GROUP BY {}"'.format(output_vector, input_vector, field_names[0], f_base, field_names[0])

        print('Dissolving {} by {} ...'.format(input_vector, field_names[0]))

        subprocess.call(com, shell=True)

    elif method == 'field-label':

        if len(field_names) != 2:
            raise ValueError('There should be two {} field names'.format(method))

        for field_name in field_names:
            if field_name not in field_names_:
                v_info = create_fields(v_info, [field_name], [field_type], [5])

        # Get the class names.
        all_class_names = []
        for fi, feature in enumerate(v_info.lyr):
            all_class_names.append(str(feature.GetField(field_names[0])))

        class_names = list(set(all_class_names))

        class_dictionary = dict(zip(class_names, range(1, len(class_names)+1)))

        # Reopen the shapefile.
        v_info.close()
        v_info = vopen(input_vector, open2read=False, epsg=epsg)

        # Add the class values.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print('Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas))

            feature.SetField(field_names[1], class_dictionary[str(feature.GetField(field_names[0]))])

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-merge':

        if len(field_names) != 3:
            raise ValueError('There should be three {} field names'.format(method))

        if field_names[2] not in field_names_:
            v_info = create_fields(v_info, [field_names[2]], [field_type], [20])

        # Merge the two fields.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + 99) < v_info.n_feas:
                remaining = fi + 99
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % 100 == 0:
                print('Features {:d}--{:d} of {:d} ...'.format(fi, remaining, v_info.n_feas))

            feature.SetField(field_names[2],
                             ''.join([str(feature.GetField(field_names[0])), str(feature.GetField(field_names[1]))]))

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    else:
        raise NameError('{} is not a method'.format(method))

    v_info.close()


def _examples():

    sys.exit("""\

    #############
    # INFORMATION
    #############

    # Get vector information
    vector_tools.py -i /in_shape.shp --method info

    # List the field names
    vector_tools.py -i /in_shape.shp -m fields

    ###########
    # PROCESSES
    ###########

    # Create a 10 km buffer
    vector_tools.py -i /in_shape.shp -o /out_buffer.shp -d 10000 -m buffer

    # Select the Id field where it is equal to 1, then save to new file
    vector_tools.py -i /in_shape.shp -o /out_selection.shp -f Id -v 1 -m select
    # OR
    vector_tools.py -i /in_shape.shp -o /out_selection.shp --expression "Id = '1'" --method select

    # Select features of A.shp that intersect B.shp
    vector_tools.py -iss A.shp -isi B.shp -o output.shp -m spatial --epsg 102033

    # Rename a shapefile in place
    vector_tools.py -i /in_vector.shp -o /out_vector.shp --method rename

    # Copy a shapefile
    vector_tools.py -i /in_vector.shp -o /out_vector.shp --method copy2

    # Delete a shapefile
    vector_tools.py -i /in_vector.shp --method delete

    # Merge multiple shapefiles
    vector_tools.py -sm /in_vector_01.shp /in_vector_02.shp -o /merged.shp --method merge

    # Dissolve a shapefile by a field.
    vector_tools.py -i /in_vector.shp -o /dissolved.shp --method field-dissolve --field-names DissolveField

    ########
    # FIELDS
    ########

    # Add x, y coordinate fields
    vector_tools.py -i /in_vector.shp --method field-xy --field-names X Y

    # Add an ordered id field
    vector_tools.py -i /in_vector.shp --method field-id --field-names id
'
    # Add an area field
    vector_tools.py -i /in_vector.shp --method field-area --field-names Area

    # Add unique class labels based on a named field. In this example, the field
    #   that contains the class names is 'Name' and the class id field to be
    #   created is 'Id'.
    vector_tools.py -i /in_vector.shp --method field-label --field-names Name Id

    # Merge two field names (f1 and f2) into a new field (merged).
    vector_tools.py -i /in_vector.shp --method field-merge --field-names f1 f2 merged

    # Set field values based on range parameters. In this example, the test is:
    #   If the value of 'f1' is --> (1 <= value < 10), then set field 'f2' as 1.
    #   If the value of 'f1' is --> (10 <= value < 20), then set field 'f2' as 2.
    vector_tools.py -i /in_vector.shp --method field-breaks --field-names f1 f2 --field-breaks "{1: [1, 10], 2: [10, 20]}"

    """)


def main():

    parser = argparse.ArgumentParser(description='Vector tools',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--input', dest='input', help='The input shapefile', default=None)
    parser.add_argument('-iss', '--input_select', dest='input_select', help='The select shapefile with -m spatial',
                        default=None)
    parser.add_argument('-isi', '--input_intersect', dest='input_intersect',
                        help='The intersect shapefile with -m spatial', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output shapefile', default=None)
    parser.add_argument('-m', '--method', dest='method', help='The method to run', default=None,
                        choices=['buffer', 'copy2', 'delete', 'fields', 'info', 'merge', 'rename', 'select', 'spatial',
                                 'field-xy', 'field-id', 'field-area', 'field-constant',
                                 'field-dissolve', 'field-merge', 'field-label', 'field-breaks'])
    parser.add_argument('-d', '--distance', dest='distance', help='The buffer distance', default=None, type=float)
    parser.add_argument('-f', '--field', dest='field', help='The field to select', default=None)
    parser.add_argument('-v', '--value', dest='value', help='The field selection value', default=None)
    parser.add_argument('-sm', '--shps2merge', dest='shps2merge', help='A list of shapefiles to merge', default=None,
                        nargs='+')
    parser.add_argument('-fn', '--field-name', dest='field_name', help='The field name', default=None)
    parser.add_argument('-fns', '--field-names', dest='field_names',
                        help='The field name(s) to add', default=['x', 'y'], nargs='+')
    parser.add_argument('-b', '--field-breaks', dest='field_breaks', help='The field breaks', default="{}")
    parser.add_argument('-dv', '--default-value', dest='default_value', help='The default break value', default=None)
    parser.add_argument('--area_units', dest='area_units', help='The units to use for area calcuation', default='km')
    parser.add_argument('--constant', dest='constant', help='A constant value for -m field-constant', default='1')
    parser.add_argument('--expression', dest='expression', help='A query expression', default=None)
    parser.add_argument('--epsg', dest='epsg', help='An EPSG projection code', default=0, type=int)

    args = parser.parse_args()

    if args.examples:
        _examples()

    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    if args.method == 'info':

        v_info = vopen(args.input, epsg=args.epsg)

        print('\nThe projection:\n')
        print(v_info.projection)

        print('\nThe extent (left, right, top, bottom):\n')
        print('{:f}, {:f}, {:f}, {:f}'.format(v_info.left, v_info.right, v_info.top, v_info.bottom))

        print('\nThe geometry:\n')
        print(v_info.shp_geom_name)

        v_info.close()

    elif args.method == 'buffer':
        buffer_vector(args.input, args.output, distance=args.distance,
                         epsg=args.epsg, field_name=args.field_name)
    elif args.method == 'fields':
        list_field_names(args.input, epsg=args.epsg)
    elif args.method == 'select':
        select_and_save(args.input, args.output, args.field, args.value,
                        expression=args.expression, epsg=args.epsg)
    elif args.method == 'spatial':
        spatial_intersection(args.input_select, args.input_intersect, args.output, epsg=args.epsg)
    elif args.method == 'rename':
        rename_vector(args.input, args.output)
    elif args.method == 'copy2':
        copy_vector(args.input, args.output)
    elif args.method == 'delete':
        delete_vector(args.input)
    elif args.method == 'merge':
        merge_vectors(args.shps2merge, args.output)
    elif args.method in ['field-xy', 'field-id', 'field-area', 'field-constant',
                         'field-dissolve', 'field-merge', 'field-label', 'field-breaks']:

        add_fields(args.input, output_vector=args.output, method=args.method,
                   field_names=args.field_names, area_units=args.area_units,
                   constant=args.constant, epsg=args.epsg,
                   field_breaks=ast.literal_eval(args.field_breaks),
                   default_value=args.default_value)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
          (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()
