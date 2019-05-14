#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 9/24/2011
""" 

from __future__ import division
from future.utils import viewitems
from builtins import int

import os
import sys
import shutil
import time
import argparse
import subprocess
import ast
import copy
import fnmatch
# import atexit
import tarfile

from .paths import get_main_path
from .errors import TransformError, logger
from .helpers import PickleIt

raster_tools = sys.modules['mpglue.raster_tools']

MAIN_PATH = get_main_path()

# GDAL
try:
    from osgeo import gdal, ogr, osr
    from osgeo.gdalconst import GA_ReadOnly, GA_Update
except ImportError:
    logger.error('GDAL Python must be installed')
    raise ImportError

# NumPy
try:
    import numpy as np
except ImportError:
    logger.error('NumPy must be installed')
    raise ImportError

# Pandas
try:
    import pandas as pd
    pandas_installed = True
except:
    pandas_installed = False

# PySal
try:
    import pysal
    pysal_installed = True
except:
    pysal_installed = False

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


def geometry2array(geometry, image_info, image=None):

    """
    Converts a polygon geometry to an array

    Args:
        geometry (object): The polygon geometry.
        image_info (object): The image information object. The object should have (cellY, projection).
        image (Optional[str]): An image to subset. Default is None.

    Returns:
        2d array
    """

    # Unpack the polygon bounds.
    left, bottom, right, top = geometry.bounds

    datasource, lyr = create_memory_layer(image_info.projection)

    field_def = ogr.FieldDefn('Value', ogr.OFTInteger)
    lyr.CreateField(field_def)

    feature = ogr.Feature(lyr.GetLayerDefn())
    feature.SetGeometryDirectly(ogr.Geometry(wkt=str(geometry)))
    feature.SetField('Value', 1)
    lyr.CreateFeature(feature)

    xcount = int((right - left) / image_info.cellY) + 1
    ycount = int((top - bottom) / image_info.cellY) + 1

    # Create a raster to rasterize into.
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)

    target_ds.SetGeoTransform([left, image_info.cellY, 0.0, top, 0.0, -image_info.cellY])
    target_ds.SetProjection(image_info.projection)

    # Rasterize
    gdal.RasterizeLayer(target_ds,
                        [1],
                        lyr,
                        options=['ATTRIBUTE=Value'])

    poly_array = np.uint8(target_ds.GetRasterBand(1).ReadAsArray())

    datasource = None
    target_ds = None

    if isinstance(image, str):

        with raster_tools.ropen(image) as i_info:

            image_array = i_info.read(bands2open=-1,
                                      x=left,
                                      y=top,
                                      rows=ycount,
                                      cols=xcount,
                                      d_type='float32')

        i_info = None

        return poly_array, image_array

    else:
        return poly_array


def feature_from_geometry(layer, geometry):

    """
    Creates a feature object from a Shapely geometry object

    Args:
        layer (object)
        geometry (Shapely geometry)

    Returns:
        OGR feature object
    """

    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometryDirectly(ogr.Geometry(wkt=str(geometry)))
    feature.SetField('Value', 1)

    return feature


def set_spatial_reference(projection):

    """
    Sets the spatial reference object

    Args:
        projection (str or int)

    Returns:
        Spatial reference object
    """

    sp_ref = osr.SpatialReference()

    if isinstance(projection, int):
        sp_ref.ImportFromEPSG(projection)
    else:

        if projection.startswith('+proj='):
            sp_ref.ImportFromProj4(projection)
        else:
            sp_ref.ImportFromWkt(projection)

    return sp_ref


def create_memory_layer(projection):

    """
    Creates an in-memory vector layer object

    Args:
        projection (str or int)

    Returns:
        Datasource object, Layer object
    """

    datasource = ogr.GetDriverByName('Memory').CreateDataSource('wrk')

    sp_ref = set_spatial_reference(projection)

    lyr = datasource.CreateLayer('',
                                 geom_type=ogr.wkbPolygon,
                                 srs=sp_ref)

    return datasource, lyr


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
                   '.gpkg': 'GPKG',
                   '.mem': 'MEMORY'}

        if file_extension not in formats:

            logger.error('  The file extension should be .shp or .mem.')
            raise NameError

        self.file_format = formats[file_extension]


class vopen(RegisterDriver):

    """
    Gets vector information and file pointer object.

    Args:
        file_name (str): Vector location, name, and extension.
        open2read (Optional[bool]): Whether to open vector as 'read only' (True) or writeable (False).
            Default is True.
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.
        delete (Optional[bool]): Whether to delete the datasource objects. Default is False.

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

    def __init__(self,
                 file_name,
                 open2read=True,
                 epsg=None,
                 delete=False):

        self.file_name = file_name
        self.open2read = open2read
        self.epsg = epsg

        self.d_name, self.f_name = os.path.split(self.file_name)
        self.f_base, self.f_ext = os.path.splitext(self.f_name)

        RegisterDriver.__init__(self, self.file_name)

        if not delete:

            self.read()

            self.get_info()

        # Check open files before closing.
        # atexit.register(self.close)

    def read(self):

        self.file_open = False

        if self.open2read:
            self.shp = ogr.Open(self.file_name, GA_ReadOnly)
        else:
            self.shp = ogr.Open(self.file_name, GA_Update)

        if self.shp is None:
            logger.error('Unable to open {}.'.format(self.file_name))
            raise NameError

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

        self.spatial_reference = osr.SpatialReference()

        # get the projection
        if isinstance(self.epsg, int):

            try:

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

        self.field_names = [self.lyr_def.GetFieldDefn(i).GetName() for i in range(0, self.lyr_def.GetFieldCount())]

    def copy(self):

        """Copies the object instance"""

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

        if hasattr(self, 'shp'):

            if hasattr(self.shp, 'feature'):

                self.shp.feature.Destroy()
                self.shp.feature = None

            if self.file_open:
                self.shp.Destroy()

        self.shp = None

        self.file_open = False

    def delete(self):

        """Deletes an open file"""

        if not self.open2read:

            logger.error('The file must be opened in read-only mode.')
            raise IOError

        try:
            self.driver.DeleteDataSource(self.file_name)
        except:

            logger.error(gdal.GetLastErrorMsg())
            logger.error('{} could not be deleted. Check for a file lock.'.format(self.file_name))
            raise IOError

        self._cleanup()

    def _cleanup(self):

        """Cleans undeleted files"""

        if self.d_name:
            file_list = fnmatch.filter(os.listdir(self.d_name), '{}*'.format(self.f_name))
        else:
            file_list = fnmatch.filter(os.listdir('.'), '{}*'.format(self.f_name))

        if file_list:

            for rf in file_list:

                file2remove = os.path.join(self.d_name, rf)

                if os.path.isfile(file2remove):
                    os.remove(file2remove)

    def exit(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def __del__(self):
        self.__exit__(None, None, None)


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

        with vopen(file_name, delete=True) as v_info:
            v_info.delete()

        v_info = None

    # Delete QGIS files.
    d_name, f_name = os.path.split(file_name)
    f_base, f_ext = os.path.splitext(f_name)

    if d_name:
        file_list = os.listdir(d_name)
    else:
        file_list = os.listdir('.')

    for f in fnmatch.filter(file_list, '{}*.qpj'.format(f_base)):

        if d_name:
            qpj_file = os.path.join(d_name, f)
        else:
            qpj_file = f

        if os.path.isfile(qpj_file):
            os.remove(qpj_file)


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
        self.datasource = None


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

    def __init__(self,
                 out_vector,
                 field_names=['Id'],
                 epsg=0,
                 projection_from_file=None,
                 projection=None,
                 field_type='int',
                 geom_type='point',
                 overwrite=True):

        self.time_stamp = time.asctime(time.localtime(time.time()))

        CreateDriver.__init__(self, out_vector, overwrite)

        if geom_type == 'point':
            geom_type = ogr.wkbPoint
        elif geom_type == 'polygon':
            geom_type = ogr.wkbPolygon

        sp_ref = None

        if isinstance(projection_from_file, str):

            with vopen(projection_from_file) as p_info:
                sp_ref = set_spatial_reference(p_info.projection)

            p_info = None

        else:

            if isinstance(epsg, int):
                sp_ref = set_spatial_reference(epsg)
            else:

                if isinstance(projection, str):
                    sp_ref = set_spatial_reference(projection)

        # Create the layer
        self.lyr = self.datasource.CreateLayer(self.f_base,
                                               geom_type=geom_type,
                                               srs=sp_ref)

        self.lyr_def = self.lyr.GetLayerDefn()

        # create the field
        if field_type == 'int':

            self.field_defs = [ogr.FieldDefn(field, ogr.OFTInteger) for field in field_names]

        elif field_type == 'float':

            self.field_defs = [ogr.FieldDefn(field, ogr.OFTReal) for field in field_names]

        elif field_type == 'string':

            self.field_defs = list()

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
        >>> from mpglue import vector_tools
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
                os.rename(os.path.join(d_name, associated_file), os.path.join(od_name, '{}{}'.format(of_base, a_ext)))
            except:

                logger.error(gdal.GetLastErrorMsg())
                logger.error('Could not write {} to file.'.format(of_base))
                raise IOError


def merge_vectors(shps2merge, merged_shapefile):

    """
    Merges a list of shapefiles into one shapefile

    Args:
        shps2merge (str list): A list of shapefiles to merge.
        merged_shapefile (str): The output merged shapefile.

    Examples:
        >>> from mpglue import vector_tools
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

    if not os.path.isdir(od_name):
        os.makedirs(od_name)

    associated_files = os.listdir(d_name)

    for associated_file in associated_files:

        a_base, a_ext = os.path.splitext(associated_file)

        if a_base == f_base:

            out_file = os.path.join(od_name, '{}{}'.format(of_base, a_ext))

            if not os.path.isfile(out_file):

                try:
                    shutil.copy2(os.path.join(d_name, associated_file), out_file)
                except:

                    logger.error('Could not copy {} to {}'.format(os.path.join(d_name, associated_file),
                                                                  out_file))
                    raise IOError

    # Then merge each shapefile into
    #   the output file.
    for shp2merge in shps2merge[1:]:

        logger.info('  Merging {} into {} ...'.format(shp2merge,
                                                      shps2merge[0]))

        try:

            ogr2ogr.main(['',
                          '-f',
                          'ESRI Shapefile',
                          '-update',
                          '-append',
                          merged_shapefile,
                          shp2merge,
                          '-nln',
                          of_base])

        except:

            logger.error('Could not merge the vector files.')
            raise IOError


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


def add_polygon(vector_object,
                xy_pairs=None,
                field_values=None,
                geometry=None):

    """
    Args:
        vector_object (object): Class instance of `create_vector`.
        xy_pairs (Optional[tuple]): List of x, y coordinates that make the feature. Default is None.
        field_values (Optional[dict]): A dictionary of field values to write. They should match the order
            of ``field_defs``. Default is None.
        geometry (Optional[object]): A polygon geometry object to write (in place of ``xy_pairs``).
            Default is None.

    Returns:
        None
    """

    # Add the points
    if isinstance(xy_pairs, tuple) or isinstance(xy_pairs, list):

        poly_geom = ogr.Geometry(ogr.wkbLinearRing)

        for pair in xy_pairs:
            poly_geom.AddPoint(float(pair[0]), float(pair[1]))

        geometry = ogr.Geometry(ogr.wkbPolygon)

        geometry.AddGeometry(poly_geom)

    feature = ogr.Feature(vector_object.lyr_def)
    feature.SetGeometry(geometry)

    vector_object.lyr.CreateFeature(feature)

    # set the fields
    if field_values:

        for field, value in viewitems(field_values):
            feature.SetField(field, value)

    vector_object.lyr.SetFeature(feature)

    feature.Destroy()


def dataframe2geo(pddf, x_field='X', y_field='Y', epsg=4326):

    """
    Converts a Pandas DataFrame to a GeoPandas DataFrame

    Args:
        pddf (Pandas DataFrame)
        x_field (str)
        y_field (str)
        epsg (int)
    """

    try:
        import geopandas as gpd
    except:

        logger.error('  GeoPandas is required')
        raise ImportError

    try:
        import shapely
        from shapely.geometry import Point
    except:

        logger.error('  Shapely is required')
        raise ImportError

    shapely.speedups.enable()

    geometry = [Point(xy) for xy in zip(pddf[x_field], pddf[y_field])]
    crs = dict(init='epsg:{:d}'.format(epsg))

    return gpd.GeoDataFrame(pddf.drop([x_field,
                                       y_field],
                                      axis=1),
                            crs=crs,
                            geometry=geometry)


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

    if not pandas_installed:

        logger.warning('Pandas must be installed to convert dataframes to shapefiles.')
        return

    if not pysal_installed:

        logger.warning('PySAL must be installed to convert dataframes to shapefiles.')
        return

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

    for i, row in viewitems(df.T):
        db.write(row)

    db.close()


def reproject(input_vector, output_vector, in_epsg=None, out_epsg=None, overwrite=False):

    """
    Re-projects a vector file

    Args:
        input_vector (str): The vector file to reproject.
        output_vector (str): The output, reprojected file.
        in_epsg (int)
        out_epsg (int)
        overwrite Optional[bool]: Whether to overwrite an existing output file. Default is False.
    """

    if not isinstance(in_epsg, int):

        logger.error('  The input EPSG code must be set.')
        raise TypeError

    if not isinstance(out_epsg, int):

        logger.error('  The output EPSG code must be set.')
        raise TypeError

    if os.path.isfile(output_vector) and overwrite:
        delete_vector(output_vector)

    from ._gdal import ogr2ogr

    ogr2ogr.main(['', '-f', 'ESRI Shapefile',
                  '-s_srs', 'EPSG:{:d}'.format(in_epsg),
                  '-t_srs', 'EPSG:{:d}'.format(out_epsg),
                  output_vector, input_vector])

    # input spatial reference
    # source_sr = osr.SpatialReference()
    # source_sr.ImportFromEPSG(in_epsg)
    #
    # # output spatial reference
    # target_sr = osr.SpatialReference()
    # target_sr.ImportFromEPSG(out_epsg)
    #
    # # Create the transformation.
    # coord_trans = osr.CoordinateTransformation(source_sr, target_sr)
    #
    # # Open the input layer.
    # with vopen(input_vector) as v_info:
    #
    #     vct_fields = list()
    #
    #     # Get the input fields to transfer over.
    #     for i in range(0, v_info.lyr_def.GetFieldCount()):
    #         vct_fields.append(v_info.lyr_def.GetFieldDefn(i).GetName())
    #
    #     # create the output layer
    #     cv = create_vector(output_vector,
    #                        field_names=vct_fields,
    #                        geom_type=v_info.shp_geom_name.lower(),
    #                        epsg=out_epsg)
    #
    #     # Iterate over the input features.
    #     in_feature = v_info.lyr.GetNextFeature()
    #
    #     while in_feature:
    #
    #         # get the input geometry
    #         geom = in_feature.GetGeometryRef()
    #
    #         # reproject the geometry
    #         geom.Transform(coord_trans)
    #
    #         # create a new feature
    #         out_feature = ogr.Feature(cv.lyr_def)
    #
    #         # set the geometry and attribute
    #         out_feature.SetGeometry(geom)
    #
    #         for i in range(0, cv.lyr_def.GetFieldCount()):
    #
    #             out_feature.SetField(cv.lyr_def.GetFieldDefn(i).GetNameRef(),
    #                                  in_feature.GetField(i))
    #
    #         # add the feature to the shapefile
    #         cv.lyr.CreateFeature(out_feature)
    #
    #         # dereference the features and get the next input feature
    #         out_feature = None
    #         in_feature = v_info.lyr.GetNextFeature()
    #
    #     cv.close()
    #
    # v_info = None


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

    if not pandas_installed:

        logger.warning('Pandas must be installed to convert shapefiles to dataframes.')
        return

    if not pysal_installed:

        logger.warning('PySAL must be installed to convert shapefiles to dataframes.')
        return

    df = pysal.open(input_shp.replace('.shp', '.dbf'), 'r')

    df = dict([(col, np.array(df.by_col(col))) for col in df.header])

    return pd.DataFrame(df)


def extent_within_boundary(image_info, geometry):

    """
    Checks whether an image extent falls entirely within a polygon boundary

    Args:
        image_info (object)
        geometry (OGR object)
    """

    # UL
    ul_ = ogr.CreateGeometryFromWkt('POINT ({:f} {:f})'.format(image_info.left,
                                                               image_info.top))

    # UR
    ur_ = ogr.CreateGeometryFromWkt('POINT ({:f} {:f})'.format(image_info.right,
                                                               image_info.top))

    # LL
    ll_ = ogr.CreateGeometryFromWkt('POINT ({:f} {:f})'.format(image_info.left,
                                                               image_info.bottom))

    # LR
    lr_ = ogr.CreateGeometryFromWkt('POINT ({:f} {:f})'.format(image_info.right,
                                                               image_info.bottom))

    if ul_.Within(geometry) and ur_.Within(geometry) and \
            ll_.Within(geometry) and lr_.Within(geometry):

        return True

    else:
        return False


def xy_within_image(x, y, image_info):

    """
    Checks whether x, y coordinates are within an image extent.

    Args:
        x (float): The x coordinate.
        y (float): The y coordinate.
        image_info (object): Object of ``mpglue.ropen``.

    Returns:
        ``True`` if ``x`` and ``y`` are within ``image_info``, otherwise ``False``.
    """

    if not isinstance(image_info, raster_tools.ropen):
        logger.error('`image_info` must be an instance of `ropen`.')
        raise TypeError

    if not hasattr(image_info, 'left') or not hasattr(image_info, 'right') \
            or not hasattr(image_info, 'bottom') or not hasattr(image_info, 'top'):

        logger.error('The `image_info` object must have left, right, bottom, top attributes.')
        raise AttributeError

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


class TransformGeometry(object):

    """
    Transforms a geometry

    Args:
        geometry (OGR geometry object)
        in_epsg (int)
        out_epsg (int)
    """

    def __init__(self, geometry, in_epsg, out_epsg):

        self.geometry = geometry

        source_sr = osr.SpatialReference()
        source_sr.ImportFromEPSG(in_epsg)

        target_sr = osr.SpatialReference()
        target_sr.ImportFromEPSG(out_epsg)

        # Create the transformation.
        self.coord_transform = osr.CoordinateTransformation(source_sr, target_sr)

        self.geometry.Transform(self.coord_transform)


class Transform(object):

    """
    Transforms a x, y coordinate pair

    Args:
        x (float): The source x coordinate.
        y (float): The source y coordinate.
        source_projection (int or str): The source projection code. Format can be EPSG, CS, or proj4.
        target_projection (int or str): The target projection code. Format can be EPSG, CS, or proj4.

    Examples:
        >>> from mpglue.vector_tools import Transform
        >>>
        >>> ptr = Transform(740000.0, 2260000.0, 102033, 4326)
        >>> print(ptr.x, ptr.y)
        >>> print(ptr.x_transform, ptr.y_transform)
    """

    def __init__(self, x, y, source_projection, target_projection):

        self.x = x
        self.y = y

        source_srs = osr.SpatialReference()
        target_srs = osr.SpatialReference()

        try:

            if isinstance(source_projection, int):
                source_srs.ImportFromEPSG(source_projection)
            elif isinstance(source_projection, str):

                if source_projection.startswith('PROJCS') or source_projection.startswith('GEOGCS'):
                    source_srs.ImportFromWkt(source_projection)
                elif source_projection.startswith('+proj'):
                    source_srs.ImportFromProj4(source_projection)
                else:

                    logger.error('  The source code could not be read.')
                    raise ValueError

        except:

            logger.error(gdal.GetLastErrorMsg())
            logger.error('  The source code could not be read.')
            raise ValueError

        try:

            if isinstance(target_projection, int):
                target_srs.ImportFromEPSG(target_projection)
            elif isinstance(target_projection, str):

                if target_projection.startswith('PROJCS') or target_projection.startswith('GEOGCS'):
                    target_srs.ImportFromWkt(target_projection)
                elif target_projection.startswith('+proj'):
                    target_srs.ImportFromProj4(target_projection)
                else:

                    logger.error('  The target code could not be read.')
                    raise ValueError

        except:

            logger.error(gdal.GetLastErrorMsg())
            logger.error('  The target code could not be read.')
            raise ValueError

        try:
            coord_trans = osr.CoordinateTransformation(source_srs, target_srs)
        except:

            logger.error(gdal.GetLastErrorMsg())
            logger.error('  The coordinates could not be transformed.')
            raise TransformError

        self.point = ogr.Geometry(ogr.wkbPoint)
        
        self.point.AddPoint(self.x, self.y)
        self.point.Transform(coord_trans)

        self.x_transform = self.point.GetX()
        self.y_transform = self.point.GetY()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self):

        self.point.Destroy()
        self.point = None

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
            logger.error('The to or from EPSG code must be given.')
            raise NameError

        ptr = Transform(grid_envelope['left'], grid_envelope['top'], self.from_epsg, self.to_epsg)

        self.left = copy.copy(ptr.x_transform)
        self.top = copy.copy(ptr.y_transform)

        ptr = Transform(grid_envelope['right'], grid_envelope['bottom'], self.from_epsg, self.to_epsg)

        self.right = copy.copy(ptr.x_transform)
        self.bottom = copy.copy(ptr.y_transform)


class TransfromEmpty(object):

    def __init__(self):
        pass

    def update_info(self, **kwargs):

        for k, v in viewitems(kwargs):
            setattr(self, k, v)


class TransformExtent(object):

    """
    Converts an extent envelope

    Args:
        grid_envelope (dict): A dictionary with 'left', 'right', 'top', 'bottom' mappings.
        from_epsg (int): A EPSG projection code.
        to_epsg (Optional[int]): A EPSG projection code.

    Attributes:
        left
        right
        top
        bottom
    """

    def __init__(self,
                 grid_envelope,
                 from_epsg,
                 to_epsg=4326):

        # Envelope
        # left, right, bottom, top

        if not grid_envelope:

            logger.error('The grid envelope list must be set.')
            raise TypeError

        self.from_epsg = from_epsg
        self.to_epsg = to_epsg

        ptr = Transform(grid_envelope['left'],
                        grid_envelope['bottom'],
                        self.from_epsg,
                        self.to_epsg)

        self.left = copy.copy(ptr.x_transform)
        self.bottom = copy.copy(ptr.y_transform)

        ptr = Transform(grid_envelope['right'],
                        grid_envelope['top'],
                        self.from_epsg,
                        self.to_epsg)

        self.right = copy.copy(ptr.x_transform)
        self.top = copy.copy(ptr.y_transform)


class RTreeManager(object):

    """A class to handle nearest neighbor lookups and spatial intersections with RTree"""

    def __init__(self, base_shapefile=None, file2pickle=None, do_not_pickle=False):

        """
        Args:
            base_shapefile (Optional[str]): The base shapefile that will be checked for intersecting features.
                Default is None, which uses the global MGRS grid.
            file2pickle (Optional[str])): A file to pickle.
            do_not_pickle (Optional[bool])
        """

        if not rtree_installed:

            logger.warning('Rtree and libspatialindex must be installed for spatial indexing')
            return

        # Setup the UTM MGRS shapefile
        if isinstance(base_shapefile, str):
            self.base_shapefile_ = base_shapefile
        else:

            import mappy

            # self.utm_shp_path = os.path.join(MAIN_PATH.replace('mpglue', 'mappy'), 'utilities', 'sentinel')
            self.utm_shp_path = os.path.join(os.path.dirname(os.path.realpath(mappy.__file__)), 'utilities', 'sentinel')

            self.base_shapefile_ = os.path.join(self.utm_shp_path, 'sentinel2_grid.shp')

            if not os.path.isfile(self.base_shapefile_):

                with tarfile.open(os.path.join(self.utm_shp_path, 'utm_shp.tar.gz'), mode='r') as tar:
                    tar.extractall(path=self.utm_shp_path)

        if rtree_installed:
            self.rtree_index = rtree.index.Index(interleaved=False)
        else:
            self.rtree_index = dict()

        # Load the information from the shapefile.
        # if isinstance(base_shapefile, str):
        #     self.field_dict = dict()

        # Setup the RTree index database.
        with vopen(self.base_shapefile_) as bdy_info:

            # Iterate over each feature.
            for f in range(0, bdy_info.n_feas):

                bdy_feature = bdy_info.lyr.GetFeature(f)
                bdy_geometry = bdy_feature.GetGeometryRef()
                en = bdy_geometry.GetEnvelope()

                if rtree_installed:
                    self.rtree_index.insert(f, (en[0], en[1], en[2], en[3]))
                else:
                    self.rtree_index[f] = (en[0], en[1], en[2], en[3])

        bdy_info = None
        bdy_feature = None
        bdy_geometry = None

        if do_not_pickle:

            self.field_dict = dict()

            # Setup the RTree index database.
            with vopen(self.base_shapefile_) as bdy_info:

                # Iterate over each feature.
                for f in range(0, bdy_info.n_feas):

                    bdy_feature = bdy_info.lyr.GetFeature(f)
                    latitude = bdy_feature.GetField('Latitude')
                    grid = bdy_feature.GetField('Grid')
                    name = bdy_feature.GetField('Name')

                    bdy_geometry = bdy_feature.GetGeometryRef()
                    en = bdy_geometry.GetEnvelope()

                    self.field_dict[f] = dict(latitude=latitude,
                                              grid=grid,
                                              name=name,
                                              extent=dict(left=en[0],
                                                          top=en[3],
                                                          right=en[1],
                                                          bottom=en[2]),
                                              utm='1')

            bdy_info = None
            bdy_feature = None
            bdy_geometry = None

        else:

            if isinstance(file2pickle, str):
                self.field_dict = PickleIt().load(file2pickle)
            else:

                import mappy

                # Load the RTree info for the MGRS global grid.
                # mgrs_info = os.path.join(MAIN_PATH.replace('mpglue', 'mappy'),
                #                          'utilities',
                #                          'sentinel',
                #                          'utm_grid_info.pickle')

                mgrs_info = os.path.join(os.path.dirname(os.path.realpath(mappy.__file__)),
                                         'utilities',
                                         'sentinel',
                                         'utm_grid_info.pickle')

                if not os.path.isfile(mgrs_info):

                    logger.exception('The MGRS global grid information file does not exist.')
                    raise NameError

                self.field_dict = PickleIt().load(mgrs_info)

    def get_intersecting_features(self,
                                  shapefile2intersect=None,
                                  get_centroid_feature=False,
                                  name_field=None,
                                  envelope=None,
                                  epsg=None,
                                  proj4=None,
                                  proj=None,
                                  lat_lon=None):

        """
        Intersects the RTree index with a shapefile or extent envelope.

        Args:
            shapefile2intersect (Optional[str]): The shapfile to intersect. The projection of
                `shapefile2intersect` should match the projection of `base_shapefile`.
            get_centroid_feature (Optional[bool]): Whether to check if a feature's centroid is
                within the intersecting grid. Default is False, i.e., get all intersecting features.
            name_field (Optional[str]): The value field for `shapefile2intersect`.
            envelope (Optional[list]): [left, right, bottom, top]. Default is to extract from `shapefile2intersect`.
            epsg (Optional[int])
            proj4 (Optional[str])
            proj (Optional[str])
            lat_lon (Optional[int])
        """

        self.grid_infos = list()

        if isinstance(shapefile2intersect, str):

            # Open the base shapefile
            with vopen(shapefile2intersect) as bdy_info:

                # Get the extent of 1 feature.
                if bdy_info.n_feas == 1:

                    bdy_feature = bdy_info.lyr.GetFeature(0)
                    bdy_geometry = bdy_feature.GetGeometryRef()
                    bdy_envelope = bdy_geometry.GetEnvelope()

                    # left, right, bottom, top
                    envelope = [bdy_envelope[0], bdy_envelope[1], bdy_envelope[2], bdy_envelope[3]]

                # Get the maximum extent of
                #   all the features.
                else:

                    envelope = [10000000., -10000000., 10000000., -10000000.]

                    for fea in range(0, bdy_info.n_feas):

                        bdy_feature = bdy_info.lyr.GetFeature(fea)
                        bdy_geometry = bdy_feature.GetGeometryRef()
                        bdy_envelope = bdy_geometry.GetEnvelope()

                        envelope[0] = min(envelope[0], bdy_envelope[0])
                        envelope[1] = max(envelope[1], bdy_envelope[1])
                        envelope[2] = min(envelope[2], bdy_envelope[2])
                        envelope[3] = max(envelope[3], bdy_envelope[3])

            bdy_info = None

        if not isinstance(envelope, list):

            logger.error('The study area envelope was not loaded.')
            raise NameError

        image_envelope = dict(left=envelope[0],
                              right=envelope[1],
                              bottom=envelope[2],
                              top=envelope[3])

        # Transform the points from ___ to WGS84.
        if isinstance(epsg, int):

            e2w = TransformExtent(image_envelope, epsg)
            envelope = [e2w.left, e2w.right, e2w.bottom, e2w.top]

        elif isinstance(proj4, str):

            e2w = TransformExtent(image_envelope, proj4)
            envelope = [e2w.left, e2w.right, e2w.bottom, e2w.top]

        elif isinstance(proj, str):

            e2w = TransformExtent(image_envelope, proj, to_epsg=lat_lon)
            envelope = [e2w.left, e2w.right, e2w.bottom, e2w.top]

        else:

            e2w = TransfromEmpty()
            e2w.update_info(left=image_envelope['left'],
                            top=image_envelope['top'],
                            right=image_envelope['right'],
                            bottom=image_envelope['bottom'])

        if rtree_installed:
            index_iter = self.rtree_index.intersection(envelope)
        else:
            index_iter = list(range(0, len(self.field_dict)))

        # Intersect the base shapefile bounding box
        #   with the UTM grids.
        for n in index_iter:

            grid_info = self.field_dict[n]

            if get_centroid_feature:

                # Get the polygon coordinates.
                coord_poly = self._get_coord_poly(e2w)

                with vopen(self.base_shapefile_) as svi_info:

                    for svi_fea_iter in range(0, svi_info.n_feas):

                        svi_feature = svi_info.lyr.GetFeature(svi_fea_iter)
                        svi_field = svi_feature.GetField(name_field)

                        if not isinstance(svi_field, str):
                            continue

                        if svi_field.strip() == grid_info['name']:

                            svi_geometry = svi_feature.GetGeometryRef()

                            # Check if the feature's centroid
                            #   is within the base shapefile.
                            if coord_poly.Centroid().Within(svi_geometry):

                                if grid_info not in self.grid_infos:

                                    grid_info['intersecting_centroid'] = str(coord_poly.Centroid()).strip()

                                    self.grid_infos.append(grid_info)

                        svi_feature.Destroy()

            #     # Open the shapefile and check each feature.
            #     with vopen(shapefile2intersect) as svi_info:
            #
            #         if svi_info.n_feas == 1:
            #
            #             svi_feature = svi_info.lyr.GetFeature(0)
            #             svi_geometry = svi_feature.GetGeometryRef()
            #
            #             # Check if the feature intersects
            #             #   the base shapefile.
            #             if coord_poly.Intersects(svi_geometry):
            #                 self.grid_infos.append(grid_info)
            #
            #         else:
            #
            #             for svi_fea_iter in range(0, svi_info.n_feas):
            #
            #                 svi_feature = svi_info.lyr.GetFeature(svi_fea_iter)
            #                 # svi_field = svi_feature.GetField(name_field)
            #
            #                 # if svi_field.strip() == grid_info['name']:
            #
            #                 svi_geometry = svi_feature.GetGeometryRef()
            #
            #                 # Check if the feature's centroid
            #                 #   is within the base shapefile.
            #                 if coord_poly.Centroid().Within(svi_geometry):
            #
            #                     if grid_info not in self.grid_infos:
            #                         self.grid_infos.append(grid_info)

            else:
                self.grid_infos.append(grid_info)

    def _get_coord_poly(self, the_extent_info):

        if isinstance(the_extent_info, dict):

            # Create a polygon object from the coordinates.
            # 0:left, 1:right, 2:bottom, 3:top
            coord_wkt = 'POLYGON (({:f} {:f}, {:f} {:f}, {:f} {:f}, {:f} {:f}, {:f} {:f}))'.format(
                the_extent_info['extent']['left'],
                the_extent_info['extent']['top'],
                the_extent_info['extent']['right'],
                the_extent_info['extent']['top'],
                the_extent_info['extent']['right'],
                the_extent_info['extent']['bottom'],
                the_extent_info['extent']['left'],
                the_extent_info['extent']['bottom'],
                the_extent_info['extent']['left'],
                the_extent_info['extent']['top'])

        else:

            # Create a polygon object from the coordinates.
            # 0:left, 1:right, 2:bottom, 3:top
            coord_wkt = 'POLYGON (({:f} {:f}, {:f} {:f}, {:f} {:f}, {:f} {:f}, {:f} {:f}))'.format(
                the_extent_info.left,
                the_extent_info.top,
                the_extent_info.right,
                the_extent_info.top,
                the_extent_info.right,
                the_extent_info.bottom,
                the_extent_info.left,
                the_extent_info.bottom,
                the_extent_info.left,
                the_extent_info.top)

        return ogr.CreateGeometryFromWkt(coord_wkt)

    def copy(self):
        return copy.copy(self)

    def _cleanup(self):

        if isinstance(self.base_shapefile_, str):
            delete_vector(self.base_shapefile_)


def create_poly_geom_from_image(the_extent_info):

    """
    Args:
        the_extent_info (ropen object): An instance of `ropen`.
    """

    # Create a polygon object from the coordinates.
    # 0:left, 1:right, 2:bottom, 3:top
    coord_wkt = 'POLYGON (({:f} {:f}, {:f} {:f}, {:f} {:f}, {:f} {:f}, {:f} {:f}))'.format(
        the_extent_info.left,
        the_extent_info.top,
        the_extent_info.right,
        the_extent_info.top,
        the_extent_info.right,
        the_extent_info.bottom,
        the_extent_info.left,
        the_extent_info.bottom,
        the_extent_info.left,
        the_extent_info.top)

    return ogr.CreateGeometryFromWkt(coord_wkt)


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

    if isinstance(boundary_file, ogr.Geometry):
        bdy_geometry = boundary_file
    else:

        with vopen(boundary_file) as bdy_info:

            bdy_feature = bdy_info.lyr.GetFeature(0)

            bdy_geometry = bdy_feature.GetGeometryRef()

        bdy_info = None

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
    if not bdy_geometry.Intersection(coord_poly):
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


def _get_xy_offsets(x,
                    left,
                    right,
                    y,
                    top,
                    bottom,
                    cell_size_x,
                    cell_size_y,
                    round_offset,
                    check):

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


def get_xy_offsets(image_info=None,
                   image_list=None,
                   x=None,
                   y=None,
                   feature=None,
                   xy_info=None,
                   round_offset=False,
                   check_position=True):

    """
    Get coordinate offsets

    Args:
        image_info (object): Object of ``mpglue.ropen``.
        image_list (Optional[list]): [left, top, right, bottom, cellx, celly]. Default is [].
        x (Optional[float]): The x coordinate. Default is None.
        y (Optional[float]): The y coordinate. Default is None.
        feature (Optional[object]): Object of ``ogr.Feature``. Default is None.
        xy_info (Optional[object]): Object of ``mpglue.vopen`` or ``mpglue.ropen``. Default is None.
        round_offset (Optional[bool]): Whether to round offsets. Default is False.
        check_position (Optional[bool]): Whether to check if `x` and `y` are within the extent bounds. Default is False.

    Examples:
        >>> from mpglue import vector_tools
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
            
            logger.error('A coordinate or feature object must be given.')
            raise ValueError

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
    x_offset, y_offset = _get_xy_offsets(x,
                                         left,
                                         right,
                                         y,
                                         top,
                                         bottom,
                                         cell_x,
                                         cell_y,
                                         round_offset,
                                         check_position)

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

        tracker_list = list()

        # Create the output shapefile
        field_names = list_field_names(select_shp, be_quiet=True)

        if epsg > 0:

            o_shp = create_vector(output_shp,
                                  field_names=field_names,
                                  geom_type=select_info.shp_geom_name.lower(),
                                  epsg=epsg)

        else:

            o_shp = create_vector(output_shp,
                                  field_names=field_names,
                                  projection_from_file=select_shp,
                                  geom_type=select_info.shp_geom_name.lower())

        # Iterate over each select feature in the polygon.
        for m in range(0, select_info.n_feas):

            if m % 500 == 0:

                if (m + 499) > select_info.n_feas:
                    end_feature = select_info.n_feas
                else:
                    end_feature = m + 499

                logger.info('  Intersecting features {:d}--{:d} of {:d} ...'.format(m, end_feature, select_info.n_feas))

            # Get the current polygon feature.
            select_feature = select_info.lyr.GetFeature(m)

            # Set the polygon geometry.
            select_geometry = select_feature.GetGeometryRef()

            # Iterate over each intersecting feature in the polygon.
            for n in range(0, intersect_info.n_feas):

                # Get the current polygon feature.
                intersect_feature = intersect_info.lyr.GetFeature(n)

                # Set the polygon geometry.
                intersect_geometry = intersect_feature.GetGeometryRef()

                left, right, bottom, top = intersect_geometry.GetEnvelope()

                # No need to check intersecting features
                # if outside bounds.
                if (left > select_info.right) or (right < select_info.left) or \
                        (top < select_info.bottom) or (bottom > select_info.top):

                    continue

                # Test the intersection.
                if select_geometry.Intersect(intersect_geometry):

                    # Get the id name of the select feature.
                    # select_id = select_feature.GetField(select_field)

                    # Don't add a feature on top of existing one.
                    if m not in tracker_list:

                        field_values = dict()

                        # Get the field names and values.
                        for field in field_names:
                            field_values[field] = select_feature.GetField(field)

                        # Add the feature.
                        add_polygon(o_shp, field_values=field_values, geometry=select_geometry)

                        tracker_list.append(m)

        o_shp.close()


def select_and_save(file_name,
                    out_vector,
                    select_field=None,
                    select_value=None,
                    expression=None,
                    overwrite=True,
                    epsg=None):

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
        >>> gl.vector_tools.select_and_save('/in_shapefile.shp', '/out_shapefile.shp', 'Id', '1')
    """

    if not os.path.isfile(file_name):
        
        logger.error('{} does not exist'.format(file_name))
        raise NameError

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

    if not pandas_installed:

        logger.warning('Pandas must be installed to load field names.')
        return

    d_name, f_name = os.path.split(in_shapefile)

    # Open the input shapefile.
    with vopen(in_shapefile, epsg=epsg) as v_info:

        df_fields = pd.DataFrame(columns=['Name', 'Type', 'Length'])

        for i in range(0, v_info.lyr_def.GetFieldCount()):

            df_fields.loc[i, 'Name'] = v_info.lyr_def.GetFieldDefn(i).GetName()
            df_fields.loc[i, 'Type'] = v_info.lyr_def.GetFieldDefn(i).GetTypeName()
            df_fields.loc[i, 'Length'] = v_info.lyr_def.GetFieldDefn(i).GetWidth()

    if not be_quiet:

        logger.info('\n{} has the following fields:\n'.format(f_name))
        logger.info(df_fields)

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
        logger.error('{} does not exist'.format(file_name))
        raise NameError

    if not isinstance(distance, float) and not isinstance(field_name, str):
        logger.error('Either the distance or field name must be given.')
        raise ValueError

    d_name, f_name = os.path.split(out_vector)

    if not os.path.isdir(d_name):
        os.makedirs(d_name)

    # open the input shapefile
    with vopen(file_name, epsg=epsg) as v_info:

        if isinstance(distance, float):
            logger.info('  Buffering {} by {:f} distance ...'.format(f_name, distance))
        else:
            logger.info('  Buffering {} by field {} ...'.format(f_name, field_name))

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
        This code was slightly modified to fit into MpGlue.

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
    for each_feature in range(0, v_info.n_feas):

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


def get_field_ids(in_file, field_name):

    """
    Gets a list of field ids

    Args:
        in_file (str): The input vector file.
        field_name (str): The field name to get ids from.
    """

    field_names = list()

    with vopen(in_file) as v_info:

        for feature in v_info.lyr:

            field_names.append(feature.GetField(field_name))
            feature.Destroy()

    v_info = None

    return field_names


def create_fields(v_info, field_names, field_types, field_widths):

    """
    Creates fields in an existing vector

    Args:
        v_info (object): A ``vopen`` object.
        field_names (str list): A list of field names to create.
        field_types (str list): A list of field types to create. Choices are ['real', 'float', 'int', str'].
        field_widths (int list): A list of field widths.

    Examples:
        >>> from mpglue import vector_tools
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
    field_defs = list()

    for field_name, field_type, field_width in zip(field_names, field_types, field_widths):

        field_def = ogr.FieldDefn(field_name, type_dict[field_type])

        if field_type in ['str', 'String']:
            field_def.SetWidth(field_width)
        elif field_type in ['float', 'Real']:
            field_def.SetPrecision(4)

        field_defs.append(field_def)

        v_info.lyr.CreateField(field_def)

    return v_info


def add_fields(input_vector,
               output_vector=None,
               field_names=None,
               method='field-xy',
               area_units='ha',
               buffer_out=0.0,
               buffer_in=0.0,
               simplify_geometry=False,
               simplify_tolerance=1.0,
               boundary_mask=None,
               constant=1,
               epsg=None,
               field_breaks=None,
               default_value=None,
               field_type=None,
               random_range=None,
               print_skip=100):

    """
    Adds fields to an existing vector

    Args:
        input_vector (str): The input vector.
        output_vector (Optional[str]): An output vector with ``method``='dissolve'. Default is None.
        field_names (Optional[str list]): The field names. Default is ['x', 'y'].
        method (Optional[str]): The method to use. Default is 'field-xy'. Choices are
            ['field-xy', 'field-id', 'field-area', 'field-constant', 'field-dissolve'].
        area_units (Optional[str]): The units to use for calculating area. Default is 'ha', or hectares.
            *Assumes the input units are meters. Choices area ['ha', 'km2', 'm2'].
        buffer_out (Optional[float]): A buffer distance to apply to each feature. Default is 0.
        buffer_in (Optional[float]): A buffer distance to apply to each feature. Default is 0.
        simplify_geometry (Optional[bool]): Whether to simplify geometry and write to `output_vector`. Default is False.
        simplify_tolerance (Optional[float]): The tolerance for geometry `ogr.Simplify`. Default is 1.0.
        boundary_mask (Optional[OGR geometry]): A boundary to use as a mask. Default is None.
        constant (Optional[int]): A constant value when ``method`` is equal to field-constant. Default is 1.
        epsg (Optional[int]): An EPSG code to declare when the .prj file is missing. Default is None.
        field_breaks (Optional[dict]): The field breaks. Default is None.
        default_value (Optional[int, float, or str]): The default break value. Default is None.
        field_type (Optional[str])
        random_range (Optional[list or tuple])
        print_skip (Optional[int])

    Returns:
        None, writes to ``input_vector`` in place.
    """

    if not field_names:
        field_names = ['x', 'y']

    if method in ['field-xy', 'field-area']:
        field_type = 'float'
    elif method in ['field-id', 'field-random']:
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

    f_name = os.path.split(input_vector)[1]
    f_base = os.path.splitext(f_name)[0]

    # First open the vector file.
    v_info = vopen(input_vector,
                   open2read=False,
                   epsg=epsg)

    # Create the new id field.
    field_names_ = [v_info.lyr_def.GetFieldDefn(i).GetName() for i in range(0, v_info.lyr_def.GetFieldCount())]

    if method == 'field-xy':

        if len(field_names) != 2:
            raise ValueError('There should be two {} field names'.format(method))

        for field_name in field_names:
            if field_name not in field_names_:
                v_info = create_fields(v_info, [field_name], [field_type], [None])

        # Add the centroids to each feature.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + print_skip-1) < v_info.n_feas:
                remaining = fi + print_skip-1
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % print_skip == 0:
                logger.info('  Features {:,d}--{:,d} of {:,d} ...'.format(fi, remaining, v_info.n_feas))

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

            if (fi + print_skip-1) < v_info.n_feas:
                remaining = fi + print_skip-1
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % print_skip == 0:
                logger.info('  Features {:,d}--{:,d} of {:,d} ...'.format(fi, remaining, v_info.n_feas))

            try:
                field_value = float(feature.GetField(field_names[0]))
            except:
                feature.Destroy()
                continue

            if field_value is None or field_value == 'None':
                feature.Destroy()
                continue

            value_found = False

            for key, break_values in viewitems(field_breaks):

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

            if (fi + print_skip-1) < v_info.n_feas:
                remaining = fi + print_skip-1
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % print_skip == 0:
                logger.info('  Features {:,d}--{:,d} of {:,d} ...'.format(fi, remaining, v_info.n_feas))

            feature.SetField(field_names[0], fi+1)

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-random':

        if len(field_names) != 1:
            raise ValueError('There should be one {} field name'.format(method))

        if not random_range:
            raise ValueError('The random range should be given.')

        if field_names[0] not in field_names_:
            v_info = create_fields(v_info, field_names, [field_type], [None])

        # Add the id to each feature.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + print_skip-1) < v_info.n_feas:
                remaining = fi + print_skip-1
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % print_skip == 0:
                logger.info('  Features {:,d}--{:,d} of {:,d} ...'.format(fi, remaining, v_info.n_feas))

            feature.SetField(field_names[0],
                             int(np.random.randint(random_range[0], random_range[1], 1)))

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-constant':

        if len(field_names) != 1:
            raise ValueError('There should be one {} field name'.format(method))

        if field_names[0] not in field_names_:
            v_info = create_fields(v_info, field_names, [field_type], [None])

        # Add the id to each feature.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + print_skip-1) < v_info.n_feas:
                remaining = fi + print_skip-1
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % print_skip == 0:
                logger.info('  Features {:,d}--{:,d} of {:,d} ...'.format(fi, remaining, v_info.n_feas))

            feature.SetField(field_names[0], constant)

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-area':

        if len(field_names) != 1:
            raise ValueError('There should be one {} field name'.format(method))

        if field_names[0] not in field_names_:
            v_info = create_fields(v_info, field_names, [field_type], [None])

        if area_units not in ['ha', 'km2', 'm2']:

            logger.error('  The area units were not recognized.')
            raise ValueError

        if simplify_geometry:

            if not isinstance(output_vector, str):

                logger.error('  The output vector must be given to simplify geometry.')
                raise NameError

            # # Get the field names + the Area field.
            # df_fields = list_field_names(input_vector,
            #                              be_quiet=True)
            #
            # field_names_in = df_fields['Name'].values.tolist()
            #
            # if field_names[0] not in field_names_in:
            #     field_names = field_names_in + field_names
            # else:
            #     field_names = field_names_in

            # Create the new shapefile.
            o_info = create_vector(output_vector,
                                   field_names=field_names,
                                   projection=v_info.projection,
                                   field_type='float',
                                   geom_type='polygon')

        # Iterate over each feature.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + print_skip-1) < v_info.n_feas:
                remaining = fi + print_skip-1
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % print_skip == 0:
                logger.info('  Features {:,d}--{:,d} of {:,d} ...'.format(fi, remaining, v_info.n_feas))

            # Get the polygon feature geometry.
            geometry = feature.GetGeometryRef()

            if boundary_mask:

                if not geometry.Intersects(boundary_mask):
                    feature.Destroy()
                    continue

            if simplify_geometry:
                geometry = geometry.Simplify(simplify_tolerance)

            if buffer_out > 0:
                geometry = geometry.Buffer(buffer_out)

            # Get the polygon feature area, in m^2
            area = geometry.GetArea()

            # Convert m^2 to km^2 or ha
            if area_units == 'km2':
                area *= 1e-06
            elif area_units == 'ha':
                area *= 1e-04

            if buffer_in > 0:
                geometry = geometry.Buffer(-buffer_in)

            if simplify_geometry:

                field_values = {field_names[0]: area}

                # Add the feature.
                add_polygon(o_info,
                            field_values=field_values,
                            geometry=geometry)

            else:

                # Add the feature area to the vector file.
                feature.SetField(field_names[0], area)

                v_info.lyr.SetFeature(feature)

            feature.Destroy()

        if simplify_geometry:
            o_info.close()

    elif method == 'field-dissolve':

        if len(field_names) != 1:
            logger.error('There should be one {} field name'.format(method))
            raise ValueError

        if not isinstance(output_vector, str):
            logger.error('The output vector must be given.')
            raise ValueError

        # Dissolve the field.
        com = 'ogr2ogr {} {} -dialect sqlite -sql "SELECT ST_Union(geometry), \
        {} FROM {} GROUP BY {}"'.format(output_vector, input_vector, field_names[0], f_base, field_names[0])

        logger.info('  Dissolving {} by {} ...'.format(input_vector, field_names[0]))

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

            if (fi + print_skip-1) < v_info.n_feas:
                remaining = fi + print_skip-1
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % print_skip == 0:
                logger.info('  Features {:,d}--{:,d} of {:,d} ...'.format(fi, remaining, v_info.n_feas))

            feature.SetField(field_names[1], class_dictionary[str(feature.GetField(field_names[0]))])

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    elif method == 'field-merge':

        if len(field_names) != 3:
            logger.error('There should be three {} field names'.format(method))
            raise ValueError

        if field_names[2] not in field_names_:
            v_info = create_fields(v_info, [field_names[2]], [field_type], [20])

        # Merge the two fields.
        for fi, feature in enumerate(v_info.lyr):

            if (fi + print_skip-1) < v_info.n_feas:
                remaining = fi + print_skip-1
            else:
                remaining = (v_info.n_feas - fi) + fi

            if fi % print_skip == 0:
                logger.info('  Features {:,d}--{:,d} of {:,d} ...'.format(fi, remaining, v_info.n_feas))

            feature.SetField(field_names[2],
                             ''.join([str(feature.GetField(field_names[0])), str(feature.GetField(field_names[1]))]))

            v_info.lyr.SetFeature(feature)

            feature.Destroy()

    else:

        logger.error('{} is not a method'.format(method))
        raise NameError

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
                                 'field-dissolve', 'field-merge', 'field-label', 'field-breaks', 'field-random'])
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
    parser.add_argument('--area-units', dest='area_units', help='The units to use for area calculations',
                        default='ha', choices=['ha', 'km2', 'm2'])
    parser.add_argument('--buffer-out', dest='buffer_out', help='A buffer to apply to each polygon feature',
                        default=0.0, type=float)
    parser.add_argument('--buffer-in', dest='buffer_in', help='A buffer to apply to each polygon feature',
                        default=0.0, type=float)
    parser.add_argument('--simplify-geometry', dest='simplify_geometry', help='Whether to simplify geometry',
                        action='store_true')
    parser.add_argument('--simplify-tolerance', dest='simplify_tolerance', help='A tolerance level to simplify by',
                        default=1.0, type=float)
    parser.add_argument('--random-range', dest='random_range', help='A min/max range for random numbers',
                        default=None, nargs='+', type=int)
    parser.add_argument('--constant', dest='constant', help='A constant value for -m field-constant', default='1')
    parser.add_argument('--expression', dest='expression', help='A query expression', default=None)
    parser.add_argument('--epsg', dest='epsg', help='An EPSG projection code', default=0, type=int)
    parser.add_argument('--print-skip', dest='print_skip', help='A skip factor for feature progress',
                        default=100, type=int)

    args = parser.parse_args()

    if args.examples:
        _examples()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    if args.method == 'info':

        v_info = vopen(args.input, epsg=args.epsg)

        logger.info('\nThe projection:\n')
        logger.info(v_info.projection)

        logger.info('\nThe extent (left, right, top, bottom):\n')
        logger.info('{:f}, {:f}, {:f}, {:f}'.format(v_info.left, v_info.right, v_info.top, v_info.bottom))

        logger.info('\nThe geometry:\n')
        logger.info(v_info.shp_geom_name)

        v_info.close()

    elif args.method == 'buffer':

        buffer_vector(args.input,
                      args.output,
                      distance=args.distance,
                      epsg=args.epsg,
                      field_name=args.field_name)

    elif args.method == 'fields':

        list_field_names(args.input, epsg=args.epsg)

    elif args.method == 'select':

        select_and_save(args.input,
                        args.output,
                        args.field,
                        args.value,
                        expression=args.expression,
                        epsg=args.epsg)

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

    elif args.method in ['field-xy',
                         'field-id',
                         'field-area',
                         'field-constant',
                         'field-dissolve',
                         'field-merge',
                         'field-label',
                         'field-breaks',
                         'field-random']:

        add_fields(args.input,
                   output_vector=args.output,
                   method=args.method,
                   field_names=args.field_names,
                   area_units=args.area_units,
                   buffer_out=args.buffer_out,
                   buffer_in=args.buffer_in,
                   simplify_geometry=args.simplify_geometry,
                   simplify_tolerance=args.simplify_tolerance,
                   constant=args.constant,
                   epsg=args.epsg,
                   field_breaks=ast.literal_eval(args.field_breaks),
                   default_value=args.default_value,
                   random_range=args.random_range,
                   print_skip=args.print_skip)

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                (time.asctime(time.localtime(time.time())), (time.time()-start_time)))


if __name__ == '__main__':
    main()
