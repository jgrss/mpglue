#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 12/20/2012
"""

from __future__ import division

from builtins import int

import os
import time
import argparse

from .raster_to_points import raster_to_points
from ..errors import logger
from .. import raster_tools
from .. import vector_tools
from ..helpers import _iteration_parameters

# GDAL
try:
    from osgeo import ogr
except ImportError:
    raise ImportError('GDAL did not load')
    
# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy did not load')


def _add_points_from_raster(out_shp,
                            class_id,
                            field_type,
                            in_rst,
                            no_data_value,
                            projection,
                            skip,
                            be_quiet,
                            block_size_rows,
                            block_size_cols):

    # Create the new shapefile
    mpc = vector_tools.create_vector(out_shp,
                                     field_names=[class_id],
                                     projection=projection,
                                     field_type=field_type)

    # create the geometry
    pt_geom = ogr.Geometry(ogr.wkbPoint)

    with raster_tools.ropen(in_rst) as m_info:

        if not be_quiet:
            ctr, pbar = _iteration_parameters(m_info.rows, m_info.cols, block_size_rows, block_size_cols)

        for i in range(0, m_info.rows, block_size_rows):

            top = m_info.top - (i * m_info.cellY)

            n_rows = raster_tools.n_rows_cols(i, block_size_rows, m_info.rows)

            for j in range(0, m_info.cols, block_size_cols):

                left = m_info.left + (j * m_info.cellY)

                n_cols = raster_tools.n_rows_cols(j, block_size_cols, m_info.cols)

                # Read the current block.
                block = m_info.read(bands2open=1,
                                    i=i,
                                    j=j,
                                    rows=n_rows,
                                    cols=n_cols)

                block_dtype = block.dtype

                # Create the points.
                if block.max() != no_data_value:

                    for i2 in range(0, n_rows, skip):

                        for j2 in range(0, n_cols, skip):

                            point_value = block[i2, j2]

                            if point_value == no_data_value:
                                continue

                            try:

                                if 'float' in block_dtype.name or 'double' in block_dtype.name:
                                    point_value = float('{:.2f}'.format(point_value))
                                else:
                                    point_value = int(point_value)

                            except:
                                continue

                            # if int(str(point_value)[str(point_value).find('.')+1]) == 0:
                            #     point_value = int(point_value)
                            # else:
                            #     point_value = float('{:.2f}'.format(point_value))

                            if point_value != no_data_value:

                                top_shift = (top - (i2 * m_info.cellY)) - (m_info.cellY / 2.0)
                                left_shift = (left + (j2 * m_info.cellY)) + (m_info.cellY / 2.0)

                                # left_shift = left + ((m_info.cellY * skip) - (m_info.cellY / 2.))
                                # top_shift = top - ((m_info.cellY * skip) - (m_info.cellY / 2.))

                                # create a point at left, top
                                vector_tools.add_point(left_shift, top_shift, mpc, class_id, point_value)

                        #     left += (m_info.cellY * skip)
                        #
                        # top -= (m_info.cellY * skip)

                if not be_quiet:

                    pbar.update(ctr)
                    ctr += 1

    if not be_quiet:
        pbar.finish()

    m_info = None

    pt_geom.Destroy()
    mpc.close()

    # Cleanup
    if os.path.isfile(in_rst):

        try:
            os.remove(in_rst)
        except:
            pass

    
def poly_to_points(input_polygon,
                output_points,
                target_image,
                class_id='Id',
                cell_size=None,
                field_type='int',
                use_extent=True,
                no_data_value=-1,
                storage='int16',
                be_quiet=False,
                skip_factor=1,
                all_touched=True,
                block_size_rows=1024,
                block_size_cols=1024):

    """
    Converts polygons to points.

    Args:    
        input_polygon (str): Path, name, and extension of polygon vector to compute.
        output_points (str): Path, name, and extension of output vector points.
        target_image (str): Path, name, and extension of image to align to.
        class_id (Optional[str]): The field id in ``poly`` to get class values from. Default is 'Id'.
        cell_size (Optional[float]): The cell size for point spacing. Default is None, or cell size of ``target_image``.
        field_type (Optional[str]): The output field data type. Default is 'int'.
        use_extent (Optional[bool]): Whether to use the extent of `target_image`. Default is True.
        no_data_value (Optional[str]): The output no data value. Default is -1.
        storage (Optional[str]): The output image data storage type. Default is 'int16'.
        be_quiet (Optional[bool]): Whether to be quiet and do not print. Default is False.
        skip_factor (Optional[int]): The within-polygon point skip factor. Default is 1.
            E.g.,
                `skip_factor`=1 would create a point at every pixel that has its centroid inside a polygon.
                `skip_factor`=2 would create a point at every other pixel that has its centroid inside a polygon.
        all_touched (Optional[bool]): Whether to rasterize all pixels touched by the vector. Otherwise,
            only include pixels that have their centroids inside of the polygon. Default is True.
        block_size_rows (Optional[int]): The processing row block size, in pixels. Default is 1024.
        block_size_cols (Optional[int]): The processing column block size, in pixels. Default is 1024.

    Examples:
        >>> from mpglue.classification.poly_to_points import poly_to_points
        >>>
        >>> poly_to_points('/polygons.shp',
        >>>             '/points.shp',
        >>>             '/target_image.tif')

    Returns:
        None, writes to ``output_points``.
    """

    d_name, f_name = os.path.split(output_points)
    f_base, f_ext = os.path.splitext(f_name)
        
    rasterized_polygons = os.path.join(d_name, '{}.tif'.format(f_base))

    if os.path.isfile(rasterized_polygons):

        try:
            os.remove(rasterized_polygons)
        except:
            logger.warning('  Could not delete the output raster. Will attempt to overwrite it.')

    with raster_tools.ropen(target_image) as m_info:

        m_info.update_info(storage=storage)

        if not isinstance(cell_size, float):
            cell_size = m_info.cellY

        # Check if the shapefile is UTM North or South. gdal_rasterize has trouble with UTM South
        # if 'S' in vct_info.proj.GetAttrValue('PROJCS')[-1]: # GetUTMZone()
        #     sys.exit('\nERROR!! The shapefile should be projected to UTM North (even for the Southern Hemisphere).\n')

        if not be_quiet:
            logger.info('  Rasterizing {} ...'.format(input_polygon))

        # Rasterize the polygon.
        if use_extent:

            raster_tools.rasterize_vector(input_polygon,
                                          rasterized_polygons,
                                          burn_id=class_id,
                                          cell_size=cell_size,
                                          top=m_info.top,
                                          bottom=m_info.bottom,
                                          left=m_info.left,
                                          right=m_info.right,
                                          projection=m_info.projection,
                                          initial_value=no_data_value,
                                          storage=m_info.storage,
                                          all_touched=all_touched)

            # com = 'gdal_rasterize -init %d -a %s -te %f %f %f %f -tr %f %f -ot Float32 %s %s' % \
            #       (no_data_value, class_id, m_info.left, m_info.bottom, m_info.right, m_info.top, cell_size, cell_size, \
            #        input_polygon, rasterized_polygons)
        else:

            raster_tools.rasterize_vector(input_polygon,
                                          rasterized_polygons,
                                          burn_id=class_id,
                                          cell_size=cell_size,
                                          projection=m_info.projection,
                                          initial_value=no_data_value,
                                          storage=m_info.storage,
                                          all_touched=all_touched)

            # com = 'gdal_rasterize -init %d -a %s -tr %f %f -ot Float32 %s %s' % \
            #       (no_data_value, class_id, cell_size, cell_size, input_polygon, rasterized_polygons)

        # subprocess.call(com, shell=True)

    m_info = None

    if not be_quiet:
        logger.info('  Converting {} to points ...'.format(rasterized_polygons))

    raster_to_points(rasterized_polygons,
                     output_points,
                     column=class_id,
                     no_data_value=no_data_value,
                     block_size=block_size_rows)

    # Get information from the input polygon.
    # with vector_tools.vopen(input_polygon) as v_info:
    #
    #     # Create the points from
    #     #   the rasterized polygon.
    #     _add_points_from_raster(output_points,
    #                             class_id,
    #                             field_type,
    #                             rasterized_polygons,
    #                             no_data_value,
    #                             v_info.projection,
    #                             skip_factor,
    #                             be_quiet,
    #                             block_size_rows,
    #                             block_size_cols)
    #
    # v_info = None


def _examples():
    return


def main():

    parser = argparse.ArgumentParser(description='Converts polygons to points',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-s', '--shapefile', dest='shapefile', help='The shapefile to sample with', default=None)
    parser.add_argument('-py', '--poly', dest='poly', help='The input polygon to convert', default=None)
    parser.add_argument('-pt', '--points', dest='points', help='The output points', default=None)
    parser.add_argument('-bi', '--base-image', dest='base_image', help='The base image to grid to', default=None)
    parser.add_argument('-id', '--class-id', dest='class_id', help='The polygon class id to rasterize', default='Id')
    parser.add_argument('-cs', '--cell-size', dest='cell_size', help='The cell size to grid to',
                        default=None, type=float)
    parser.add_argument('-ft', '--field-type', dest='field_type', help='The class field type', default='int')

    args = parser.parse_args()

    if args.examples:
        _examples()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    poly_to_points(args.poly,
                   args.points,
                   args.base_image,
                   args.class_id,
                   cell_size=args.cell_size,
                   field_type=args.field_type)

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                (time.asctime(time.localtime(time.time())), (time.time()-start_time)))


if __name__ == '__main__':
    main()
