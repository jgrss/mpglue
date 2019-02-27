#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 9/24/2011
"""

from future.utils import iteritems, itervalues
from builtins import int, dict

import os
import sys
import time
import subprocess
from copy import copy
import argparse
import fnmatch
from joblib import Parallel, delayed

from .. import raster_tools
from .. import vector_tools
from ..helpers import _iteration_parameters_values
from ..errors import ArrayOffsetError, logger
from .poly_to_points import poly_to_points
from .error_matrix import error_matrix

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy is not installed')

# Pandas
try:
    import pandas as pd
except ImportError:
    raise ImportError('Pandas is not installed')

# GDAL
try:
    from osgeo import gdal
    from osgeo.gdalconst import GA_ReadOnly
except ImportError:
    raise ImportError('GDAL is not installed')


def _sample_parallel(band_position,
                     image_name,
                     c_list,
                     accuracy,
                     feature_length):

    datasource = gdal.Open(image_name,
                           GA_ReadOnly)

    band_object = datasource.GetRasterBand(band_position)

    logger.info('Band {:d} of {:d} ...'.format(band_position, datasource.RasterCount))

    # 1d array to store image values.
    value_list = np.zeros(feature_length, dtype='float32')

    # Sort by feature index position.
    for vi, values in enumerate(sorted(c_list)):

        # values[1] = [x, y, x offset, y offset, label]
        # values[1][2] = x offset position
        # values[1][3] = y offset position
        # pixel_value = image_info.read(i=values[1][3],
        #                                  j=values[1][2],
        #                                  rows=1,
        #                                  cols=1,
        #                                  d_type='float32')[0, 0]
        try:
            pixel_value = band_object.ReadAsArray(values[1][2], values[1][3], 1, 1)[0, 0]
        except:
            pixel_value = -999.0

        if not accuracy:
            pixel_value = round(float(pixel_value), 4)
        else:
            pixel_value = int(pixel_value)

        # Update the list with raster values.
        value_list[vi] = pixel_value

    band_object = None
    datasource = None

    return value_list


class SampleImage(object):

    """
    A class for image sampling

    Args:
        points_file (str): The shapefile.
        image_file (str): The raster file to sample.
        out_dir (str)
        class_id (str)
        accuracy (Optional[bool])
        n_jobs (Optional[int])
        neighbors (Optional[bool])
        field_type (Optional[str])
        transform_xy_proj (Optional[int or proj4 str]): A transformation for the x, y coordinate output.
            Default is None.
        use_extent (Optional[bool])
        append_name (Optional[str]): A base name to append to the samples file name.
        check_corrupted_bands (Optional[bool]): Whether to perform a corrupted band check. Default is True.
        verbose (Optional[int]): The level of verbosity for print statements. Default is 1.
    """

    def __init__(self,
                 points_file,
                 image_file,
                 out_dir,
                 class_id,
                 accuracy=False,
                 n_jobs=0,
                 neighbors=False,
                 field_type='int',
                 transform_xy_proj=None,
                 use_extent=True,
                 append_name=None,
                 sql_expression_attr=None,
                 sql_expression_field='Id',
                 check_corrupted_bands=True,
                 verbose=1):

        self.points_file = points_file
        self.image_file = image_file
        self.out_dir = out_dir
        self.class_id = class_id
        self.accuracy = accuracy
        self.n_jobs = n_jobs
        self.neighbors = neighbors
        self.field_type = field_type
        self.transform_xy_proj = transform_xy_proj
        self.use_extent = use_extent
        self.append_name = append_name
        self.sql_expression_attr = sql_expression_attr
        self.sql_expression_field = sql_expression_field
        self.check_corrupted_bands = check_corrupted_bands
        self.verbose = verbose

        self.count_dict = None
        self.class_list = None
        self.n_classes = None

        self.d_type = 'uint8' if self.field_type == 'int' else 'float32'

        if not os.path.isfile(self.points_file):
            raise IOError('\n{} does not exist. It should be a point shapefile.'.format(self.points_file))

        if not os.path.isfile(self.image_file):
            raise IOError('\n{} does not exist. It should be a raster image.'.format(self.image_file))

        if self.neighbors and (self.n_jobs != 0):

            logger.info('Cannot sample neighbors in parallel, so setting ``n_jobs`` to 0.')
            self.n_jobs = 0

        self.d_name_points, f_name_points = os.path.split(self.points_file)
        self.f_base_points = os.path.splitext(f_name_points)[0]

        # Filter by SQL expression.
        if self.sql_expression_attr:
            self.points_file = self.sql()

        self.f_name_rst = os.path.split(self.image_file)[1]
        self.f_base_rst = os.path.splitext(self.f_name_rst)[0]

        if not self.out_dir:

            self.out_dir = copy(self.d_name_points)
            logger.info('\nNo output directory was given. Results will be saved to {}'.format(self.out_dir))

        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        self.setup_names()

    def sql(self):

        out_points = '{}/{}_sql.shp'.format(self.d_name_points, self.f_base_points)

        ogr_com = 'ogr2ogr -overwrite {} {}'.format(out_points, self.points_file)

        ogr_com_sql = '"SELECT * FROM {} WHERE {}'.format(self.f_base_points, self.sql_expression_field)

        for attr in range(0, len(self.sql_expression_attr)):

            if attr == 0:
                ogr_com_sql = '{} = \'{}\''.format(ogr_com_sql, self.sql_expression_attr[attr])
            else:
                ogr_com_sql = '{} OR {} = \'{}\''.format(ogr_com_sql, self.sql_expression_field,
                                                         self.sql_expression_attr[attr])

        ogr_com = '{} -sql {}"'.format(ogr_com, ogr_com_sql)

        logger.info('\nSubsetting {} by classes {} ...\n'.format(self.points_file, ','.join(self.sql_expression_attr)))

        subprocess.call(ogr_com, shell=True)

        self.d_name_points, f_name_points = os.path.split(out_points)
        self.f_base_points, __ = os.path.splitext(f_name_points)

        return out_points

    def setup_names(self):

        """
        Sets the file and directory names
        """

        # self.out_dir = self.out_dir.replace('\\', '/')

        # Open the samples.
        self.shp_info = vector_tools.vopen(self.points_file)

        if 'POINT' not in self.shp_info.shp_geom_name:

            # Convert polygon to points.
            self.points_file = self.convert2points()

            # Close the polygon shapefile
            self.shp_info.close()

            self.shp_info = None

            self.d_name_points, f_name_points = os.path.split(self.points_file)
            self.f_base_points = os.path.splitext(f_name_points)[0]

            # Open the samples
            self.shp_info = vector_tools.vopen(self.points_file)

        self.n_feas = self.shp_info.n_feas

        self.lyr = self.shp_info.lyr

        self.get_class_count()

        if isinstance(self.append_name, str):

            self.data_file = os.path.join(self.out_dir,
                                          '{POINTS}__{RASTER}_{BASE}_SAMPLES.txt'.format(POINTS=self.f_base_points,
                                                                                         RASTER=self.f_base_rst,
                                                                                         BASE=self.append_name))

            # information samples file
            self.n_samps = os.path.join(self.out_dir,
                                        '{POINTS}__{RASTER}_{BASE}_INFO.txt'.format(POINTS=self.f_base_points,
                                                                                    RASTER=self.f_base_rst,
                                                                                    BASE=self.append_name))

        else:

            self.data_file = os.path.join(self.out_dir,
                                          '{POINTS}__{RASTER}_SAMPLES.txt'.format(POINTS=self.f_base_points,
                                                                                  RASTER=self.f_base_rst))

            # information samples file
            self.n_samps = os.path.join(self.out_dir,
                                        '{POINTS}__{RASTER}_INFO.txt'.format(POINTS=self.f_base_points,
                                                                             RASTER=self.f_base_rst))

        # create array of zeros for the class counter
        # self.count_arr = np.zeros(len(self.n_classes), dtype='uint8')
        self.count_dict = dict()

        for nc in self.class_list:
            self.count_dict[nc] = 0

    def convert2points(self):

        """
        Converts polygons to points

        Returns:
            Name of converted points file
        """

        out_points = os.path.join(self.d_name_points, '{}_points.shp'.format(self.f_base_points))

        if not os.path.isfile(out_points):

            poly_to_points(self.points_file,
                           out_points,
                           self.image_file,
                           class_id=self.class_id,
                           field_type=self.field_type,
                           use_extent=self.use_extent)

        return out_points

    def get_class_count(self):

        """
        Gets the class counts
        """

        try:

            self.class_list = [self.shp_info.lyr.GetFeature(n).GetField(self.class_id)
                               for n in range(0, self.shp_info.n_feas)]

        except:

            logger.error('  Field <{}> does not exist or there is a feature issue.\n'.format(self.class_id))
            raise IOError

        if 0 in self.class_list:
            self.zs = True
        else:
            self.zs = False

        self.class_list = np.unique(np.array(self.class_list))

        self.n_classes = len(self.class_list)

    def sample(self):

        if self.verbose > 0:
            logger.info('  Sampling {} ...'.format(self.f_name_rst))

        # Open the image.
        with raster_tools.ropen(self.image_file) as self.m_info:

            # Check if any of the bands are corrupted.
            if self.check_corrupted_bands:

                self.m_info.check_corrupted_bands()

                if self.m_info.corrupted_bands:

                    logger.info()
                    logger.info('The following bands appear to be corrupted:')
                    logger.info(', '.join(self.m_info.corrupted_bands))

                    raise ValueError

            headers = self.write_headers()

            self.fill_dictionary()

            # Return if no samples were within
            #   the raster frame.
            if len(self.coords_offsets) == 0:
                self.finish()

            # Sample each image layer
            value_array = self.sample_image()

            self.write2file(value_array,
                            headers)

        self.shp_info.close()

        self.shp_info = None

        self.m_info = None

        self.finish()

    def write_headers(self):

        """
        Writes text headers
        """

        headers = ['Id', 'X', 'Y']

        # Then <image name.band position> format.
        [headers.append('{}.{:d}'.format(self.f_base_rst, b)) for b in range(1, self.m_info.bands+1)]

        headers.append('response')

        return headers

    def write2file(self,
                   value_array,
                   headers):

        """
        Writes samples to file
        """

        df = pd.DataFrame(value_array, columns=headers)
        df.to_csv(self.data_file, sep=',', index=False)

    def fill_dictionary(self):

        """
        Creates a dictionary where for each feature,
            feature 1 = [x coordinate, y coordinate, x offset, y offset, class label]
            ...
            feature n = [x coordinate, y coordinate, x offset, y offset, class label]
        """

        # Dictionary to store sampled data.
        self.coords_offsets = dict()

        if self.neighbors:
            self.updater = 5
        else:
            self.updater = 1

        def get_xy(fea):

            """
            Gets the x, y coordinate of the point
            """

            geometry = fea.GetGeometryRef()

            # Get X,Y coordinates.
            return geometry.GetX(), geometry.GetY()

        # Iterate over each point feature
        #   in the vector file.
        for n in range(0, self.n_feas):

            # Get the feature object.
            feature = self.shp_info.lyr.GetFeature(n)

            # Get the current point.
            x, y = get_xy(feature)

            # Get the class label.
            pt_id = feature.GetField(self.class_id)

            # Check if the sample points fall
            #   within the [current] raster boundary.
            if vector_tools.xy_within_image(x, y, self.m_info):

                # Get x, y coordinates and offsets.
                x, y, x_off, y_off = vector_tools.get_xy_offsets(image_info=self.m_info,
                                                                 x=x,
                                                                 y=y)

                # Update the counter array with the current label.
                self.count_dict[int(pt_id)] += self.updater

                x = float('{:.6f}'.format(x))
                y = float('{:.6f}'.format(y))

                # Add x, y coordinates, image offset indices,
                #   and class value to the dictionary.
                self.coords_offsets[n] = [x, y, x_off, y_off, pt_id]

            feature.Destroy()
            feature = None

    def sample_image(self):

        """
        The main image sampler
        """

        # Convert position items to a list.
        c_list = list(iteritems(self.coords_offsets))

        # Get the number of sample points.
        feature_length = len(self.coords_offsets)

        if self.n_jobs != 0:

            # Sample the image
            value_arr = Parallel(n_jobs=self.n_jobs)(delayed(_sample_parallel)(f_bd,
                                                                               self.image_file,
                                                                               c_list,
                                                                               self.accuracy,
                                                                               feature_length)
                                                     for f_bd in range(1, self.m_info.bands+1))

            # Transpose the data to [samples x image layers].
            value_arr = np.array(value_arr, dtype='float32').T

            # Check for coordinates with no data.
            idx = np.where(value_arr.mean(axis=1) != -999)[0]

            # The order is the same as the point labels
            #   because we iterate over the sorted (by
            #   feature position) dictionary items
            #   in both cases.

            # nx2 coordinate array
            xy_coordinates = np.zeros((feature_length, 3), dtype='float32')

            # 1d of n length labels array
            labels = np.zeros(feature_length, dtype='float32')

            # Sort by feature index position and
            #   get the x, y coordinates.
            for vi, values in enumerate(sorted(c_list)):

                # values[0] = feature index position
                # values[1] = list of coordinate data

                x = values[1][0]
                y = values[1][1]

                # Transform the x,y coordinates.
                if isinstance(self.transform_xy_proj, int) or isinstance(self.transform_xy_proj, str):

                    grid_envelope = dict(left=x,
                                         right=x,
                                         top=y,
                                         bottom=y)

                    ptr = vector_tools.TransformExtent(grid_envelope,
                                                       self.m_info.projection,
                                                       to_epsg=self.transform_xy_proj)

                    x = ptr.left
                    y = ptr.top

                # Fill index + x & y coordinates
                xy_coordinates[vi] = [vi, x, y]

                # Fill labels
                labels[vi] = values[1][4]

            if np.any(idx):

                # Remove coordinates with no data.
                value_arr = value_arr[idx]
                xy_coordinates = xy_coordinates[idx]
                labels = labels[idx]

            try:

                # Combine all the x,y coordinates,
                #   data, and sample value.
                value_arr = np.c_[xy_coordinates,
                                  value_arr,
                                  labels]

            except ArrayOffsetError:
                raise ArrayOffsetError('Check the projections and extents of the datasets.')

        else:

            # Create the array to write values to and
            #   add three to columns -- one for the class
            #   label, two for the x, y coordinates.
            neighbor_offsets = [[0, -1], [1, 0], [0, 1], [-1, 0]]

            value_arr = np.zeros((feature_length*self.updater, self.m_info.bands+4), dtype='float32')

            logger.info('\nSampling {:,d} samples from {:d} image layers ...\n'.format(feature_length, self.m_info.bands))

            ctr, pbar = _iteration_parameters_values(self.m_info.bands, feature_length)

            to_delete = []

            # Iterate over each band.
            for f_bd in range(1, self.m_info.bands+1):

                band = self.m_info.datasource.GetRasterBand(f_bd)

                point_iter = 0      # necessary because of neighbors

                # Iterate over each point.
                #   values = [x, y, x_off, y_off, pt_id]

                # Sort by feature index position.
                for vi, values in enumerate(sorted(c_list)):

                    # Get the image offset indices.
                    x_off = values[1][2]
                    y_off = values[1][3]

                    if (x_off-1 > self.m_info.cols) or (y_off-1 > self.m_info.rows):
                        raise ArrayOffsetError('Check the projections and extents of the datasets.')

                    # Get the image value.
                    value = band.ReadAsArray(x_off, y_off, 1, 1)

                    if isinstance(value, np.ndarray):
                        value = np.float32(value[0, 0])
                    else:
                        continue

                    if not self.accuracy:
                        value = float(('{:.4f}'.format(value)))
                    else:
                        value = int(value)

                    # Update value the list.
                    value_arr[point_iter, 0] = vi               # Index id
                    value_arr[point_iter, 1:3] = values[1][:2]  # x,y coordinates
                    value_arr[point_iter, f_bd+2] = value       # raster values
                    value_arr[point_iter, -1] = values[1][4]    # class label

                    if self.neighbors:

                        """
                        | |1| |
                        |4|x|2|
                        | |3| |
                                            1         2       3       4
                        neighbor_offsets = [[0, -1], [1, 0], [0, 1], [-1, 0]]
                        """

                        for noff in range(1, self.updater):

                            if (x_off + neighbor_offsets[noff-1][0] >= self.m_info.cols) or \
                                    (y_off + neighbor_offsets[noff-1][1] >= self.m_info.rows):

                                to_delete.append(point_iter + noff)
                                self.count_arr[self.n_classes.index(values[1][4])] -= 1

                                continue

                            else:

                                value = band.ReadAsArray(x_off+neighbor_offsets[noff-1][0],
                                                         y_off+neighbor_offsets[noff-1][1], 1, 1).astype(np.float32)[0, 0]

                            if not self.accuracy:
                                value = float(('{:.4f}'.format(value)))
                            else:
                                value = int(value)

                            # Write to array.
                            value_arr[point_iter+noff, 0] = \
                                values[1][0] + (neighbor_offsets[noff-1][0] * self.m_info.cellY)

                            value_arr[point_iter+noff, 1] = \
                                values[1][1] + (neighbor_offsets[noff-1][1] * -self.m_info.cellY)

                            value_arr[point_iter+noff, f_bd+1] = value
                            value_arr[point_iter+noff, -1] = values[1][4]

                    point_iter += self.updater

                    pbar.update(ctr)
                    ctr += 1

                band.FlushCache()
                band = None

            pbar.finish()

            if to_delete:
                value_arr = np.delete(value_arr, np.array(to_delete), axis=0)

        value_arr[np.isnan(value_arr) | np.isinf(value_arr)] = 0.

        return value_arr

    def finish(self):

        with open(self.n_samps, 'w') as n_sample_writer:

            self.class_sum = sum(itervalues(self.count_dict))

            # Write the number of samples from
            #   the counter array.
            for nc in self.class_list:

                n_sample_writer.write('Class {:d}: {:,d}\n'.format(int(nc),
                                                                   int(self.count_dict[nc])))

            # write the total number of samples
            n_sample_writer.write('Total: {:,d}'.format(self.class_sum))

        if self.class_sum == 0:

            if os.path.isfile(self.data_file):

                try:
                    os.remove(self.data_file)
                except:
                    pass

            if os.path.isfile(self.n_samps):

                try:
                    os.remove(self.n_samps)
                except:
                    pass

        else:

            if self.accuracy:

                # Output confusion matrix text file.
                if isinstance(self.append_name, str):

                    error_file = os.path.join(self.out_dir,
                                              '{POINTS}__{RASTER}_{BASE}_ACCURACY.txt'.format(POINTS=self.f_base_points,
                                                                                              RASTER=self.f_base_rst,
                                                                                              BASE=self.append_name))

                else:

                    error_file = os.path.join(self.out_dir,
                                              '{POINTS}__{RASTER}_ACCURACY.txt'.format(POINTS=self.f_base_points,
                                                                                 RASTER=self.f_base_rst))

                emat = error_matrix()
                emat.get_stats(po_text=self.data_file, header=True)
                emat.write_stats(error_file)


def sample_raster(points,
                  image,
                  out_dir=None,
                  option=1,
                  class_id='Id',
                  accuracy=False,
                  field_type='int',
                  use_extent=True,
                  sql_expression_field='Id',
                  sql_expression_attr=None,
                  neighbors=False,
                  search_ext=None,
                  n_jobs=0):
    
    """
    Samples an image, or imagery, using a point, or points, shapefile.

    Args:
        points (str): Points shapefile or directory containing point shapefiles.
        image (str): Image or directory containing imagery.
        out_dir (Optional[str]): The directory to save text files. Default is None, or the same directory as 
            the points shapefile(s).
        option (Optional[int]): Default is 1.
            Options:
                1 :: One point shapefile    ---> One raster file
                2 :: One point shapefile    ---> Many raster files
                3 :: Many point shapefiles  ---> Many raster files
        class_id (Optional[str]): Shapefile field id containing class values. Default is 'Id'.
        accuracy (Optional[bool]): Whether to compute accuracy from `image`. Default is False.
        field_type (Optional[str]): The field type of `class_id`. Default is 'int'.
        use_extent (Optional[bool]): Whether to use the extent of `image` for `poly_to_points`. Default is True.
        sql_expression_field (Optional[str]): Default is 'Id'.
        sql_expression_attr (Optional[str]): Default is [].
        neighbors (Optional[bool]): Whether to sample neighboring pixels. Default is False.
        search_ext (Optional[str list]): A list of file extensions to search. Default is ['tif'].
        n_jobs (Optional[int]): The number of parallel jobs. Default is 0.

    Returns:
        None, writes results to ``out_dir``.
    """

    if not sql_expression_attr:
        sql_expression_attr = list()

    if not search_ext:
        search_ext = ['tif']

    # 1:1
    if option == 1:

        si = SampleImage(points,
                         image,
                         out_dir,
                         class_id,
                         accuracy=accuracy,
                         n_jobs=n_jobs,
                         field_type=field_type,
                         use_extent=use_extent,
                         neighbors=neighbors,
                         sql_expression_attr=sql_expression_attr,
                         sql_expression_field=sql_expression_field)

        si.sample()

    # 1:--
    elif option == 2:

        search_ext = ['*.{}'.format(se) for se in search_ext]

        image_list = list()
        for se in search_ext:
            [image_list.append(fn) for fn in fnmatch.filter(os.listdir(image), se)]

        for im in image_list:

            im_ = os.path.join(image, im)

            si = SampleImage(points,
                             im_,
                             out_dir,
                             class_id,
                             accuracy=accuracy,
                             n_jobs=n_jobs,
                             field_type=field_type,
                             use_extent=use_extent,
                             neighbors=neighbors,
                             sql_expression_attr=sql_expression_attr,
                             sql_expression_field=sql_expression_field)

            si.sample()

    # --:1
    elif option == 3:

        point_list = fnmatch.filter(os.listdir(points), '*.shp')

        for pt in point_list:

            pt_ = os.path.join(points, pt)

            si = SampleImage(pt_,
                             image,
                             out_dir,
                             class_id,
                             accuracy=accuracy,
                             n_jobs=n_jobs,
                             field_type=field_type,
                             use_extent=use_extent,
                             neighbors=neighbors,
                             sql_expression_attr=sql_expression_attr,
                             sql_expression_field=sql_expression_field)

            si.sample()


def _options():
    
    sys.exit("""\

    1 :: One point shapefile    ---> One raster file
    2 :: One point shapefile    ---> Many raster files
    3 :: Many point shapefiles  ---> Many raster files

    """)


def _examples():
    
    sys.exit("""\

    # Sample some_image.tif with pts.shp, returning one set of sample data
    sample_raster -s /pts.shp -i /some_image.tif -o /out_dir

    # Sample all rasters in /some_dir with pts.shp, returning one set of sample data
    sample_raster -s /pts.shp -i /some_dir -opt 2

    # Sample all rasters in /some_dir with all shapefiles in /some_dir_pts, returning sample data for each raster
    sample_raster -s /some_dir_pts -i /some_dir --option 3

    # Query the <trees> and <shrubs> fields in <polys.shp> prior to sampling
    sample_raster -s /polys.shp -c CLASS -i /image.tif -o /out_dir --sql_field name --sql_attr trees shrubs

    # compute the accuracy of some_image.tif
    sample_raster -s /pts.shp -i /some_image.tif --accuracy

    """)


def main():

    parser = argparse.ArgumentParser(description='Sample raster(s) with shapefile(s)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-s', '--shapefile', dest='shapefile', help='The shapefile to sample with', default=None)
    parser.add_argument('-i', '--input', dest='input', help='The input image to sample', default=None)
    parser.add_argument('-c', '--classid', dest='classid', help='The field class id name', default='Id')
    parser.add_argument('-f', '--fieldtype', dest='fieldtype', help='The field type of the class field', default='int')
    parser.add_argument('-o', '--output', dest='output', help='Output directory or base name of text extension',
                        default=None)
    parser.add_argument('-opt', '--option', dest='option', help='The option to use', default=1, type=int,
                        choices=[1, 2, 3])
    parser.add_argument('-n', '--neighbors', dest='neighbors', help='Whether to use neighboring pixels',
                        action='store_true')
    parser.add_argument('-a', '--accuracy', dest='accuracy', help='Whether to compute accuracy', action='store_true')
    parser.add_argument('-j', '--n_jobs', dest='n_jobs', help='Number of parallel jobs', default=0, type=int)
    parser.add_argument('--sql_attr', dest='sql_attr', help='The SQL field attributes', default=[], nargs='+')
    parser.add_argument('--sql_field', dest='sql_field', help='The SQL class field', default='Id')
    parser.add_argument('--options', dest='options', help='Whether to show sampling options', action='store_true')

    args = parser.parse_args()

    if args.options:
        _options()

    if args.examples:
        _examples()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    sample_raster(args.shapefile, args.input, out_dir=args.output, option=args.option, class_id=args.classid,
                  accuracy=args.accuracy, field_type=args.fieldtype, neighbors=args.neighbors,
                  n_jobs=args.n_jobs, sql_expression_attr=args.sql_attr, sql_expression_field=args.sql_field)

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()
