#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 12/20/2012
"""

import os
import sys
import time
import argparse

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


def _add_points_from_raster(out_shp, class_id, field_type, in_rst,
                            epsg=None, projection=None, no_data_value=-1, skip=1,
                            be_quiet=False):

    # Create the new shapefile
    mpc = vector_tools.create_vector(out_shp,
                                     field_names=[class_id],
                                     projection=projection,
                                     field_type=field_type)

    # create the geometry
    pt_geom = ogr.Geometry(ogr.wkbPoint)

    m_info = raster_tools.ropen(in_rst)
    m_info.update_info(storage='byte')

    # band = m_info.datasource.GetRasterBand(1)

    block_size_rows = 512
    block_size_cols = 512

    print '\nConverting to points ...\n'

    if not be_quiet:
        ctr, pbar = _iteration_parameters(m_info.rows, m_info.cols, block_size_rows, block_size_cols)

    for i in xrange(0, m_info.rows, block_size_rows):

        top = m_info.top - (i * m_info.cellY)

        n_rows = raster_tools.n_rows_cols(i, block_size_rows, m_info.rows)

        for j in xrange(0, m_info.cols, block_size_cols):

            left = m_info.left + (j * m_info.cellY)

            n_cols = raster_tools.n_rows_cols(j, block_size_cols, m_info.cols)

            block = m_info.read(bands2open=1,
                                   i=i, j=j,
                                   rows=n_rows, cols=n_cols,
                                   d_type='int16')

            # block = np.float32(band.ReadAsArray(j, i, n_cols, n_rows))

            # blk_mean = block.mean()
            # blk_mean = float('%.2f' % blk_mean)

            if block.max() != no_data_value:

                for i2 in xrange(0, n_rows, skip):

                    for j2 in xrange(0, n_cols, skip):

                        val = block[i2, j2]

                        if int(val) == no_data_value:
                            continue

                        if int(str(val)[str(val).find('.')+1]) == 0:
                            val = int(val)
                        else:
                            val = float('{:.2f}'.format(val))

                        if val != no_data_value:

                            top_shift = (top - (i2 * m_info.cellY)) - (m_info.cellY / 2.)
                            left_shift = (left + (j2 * m_info.cellY)) + (m_info.cellY / 2.)

                            # left_shift = left + ((m_info.cellY * skip) - (m_info.cellY / 2.))
                            # top_shift = top - ((m_info.cellY * skip) - (m_info.cellY / 2.))

                            # create a point at left, top
                            vector_tools.add_point(left_shift, top_shift, mpc, class_id, val)

                    #     left += (m_info.cellY * skip)
                    #
                    # top -= (m_info.cellY * skip)

            if not be_quiet:
                pbar.update(ctr)
                ctr += 1

    if not be_quiet:
        pbar.finish()

    # band = None
    m_info.close()

    pt_geom.Destroy()
    mpc.close()

    if os.path.isfile(in_rst):

        try:
            os.remove(in_rst)
        except:
            pass

    
def poly2points(poly, out_shp, targ_img, class_id='Id', cell_size=None,
                field_type='int', use_extent=True, no_data_value=-1):

    """
    Converts polygons to points.

    Args:    
        poly (str): Path, name, and extension of polygon vector to compute.
        out_shp (str): Path, name, and extension of output vector points.
        targ_img (str): Path, name, and extension of image to align to.
        class_id (Optional[str]): The field id in ``poly`` to get class values from. Default is 'Id'.
        cell_size (Optional[float]): The cell size for point spacing. Default is None, or cell size of ``targ_img``.
        field_type
        use_extent
        no_data_value

    Examples:
        >>> from mappy.sample import poly2points
        >>> poly2points('C:/someDir/somePoly.shp', 'C:/someDir/somePts.shp')
    
        Command line usage
        ------------------
        .. mappy\sample\poly2points.py -i C:\someDir\somePoly.shp -o C:\someOutDir\somePts.shp

    Returns:
        None, writes to ``out_shp``.
    """

    d_name, f_name = os.path.split(out_shp)
    f_base, f_ext = os.path.splitext(f_name)
        
    out_rst = os.path.join(d_name, '{}.tif'.format(f_base))

    if os.path.isfile(out_rst):

        try:
            os.remove(out_rst)
        except:
            sys.exit('ERROR!! Could not delete the output raster.')
            
    m_info = raster_tools.ropen(targ_img)

    m_info.update_info(storage='int16')

    if not isinstance(cell_size, float):
        cell_size = m_info.cellY

    # Check if the shapefile is UTM North or South. gdal_rasterize has trouble with UTM South
    # if 'S' in vct_info.proj.GetAttrValue('PROJCS')[-1]: # GetUTMZone()
    #     sys.exit('\nERROR!! The shapefile should be projected to UTM North (even for the Southern Hemisphere).\n')

    print '\nRasterizing {} ...\n'.format(f_name)

    if use_extent:

        raster_tools.rasterize_vector(poly, out_rst,
                                      burn_id=class_id,
                                      cell_size=cell_size,
                                      top=m_info.top, bottom=m_info.bottom,
                                      left=m_info.left, right=m_info.right,
                                      projection=m_info.projection,
                                      initial_value=no_data_value,
                                      storage=m_info.storage)

        # com = 'gdal_rasterize -init %d -a %s -te %f %f %f %f -tr %f %f -ot Float32 %s %s' % \
        #       (no_data_value, class_id, m_info.left, m_info.bottom, m_info.right, m_info.top, cell_size, cell_size, \
        #        poly, out_rst)
    else:
        raster_tools.rasterize_vector(poly, out_rst,
                                      burn_id=class_id,
                                      cell_size=cell_size,
                                      projection=m_info.projection,
                                      initial_value=no_data_value,
                                      storage=m_info.storage)

        # com = 'gdal_rasterize -init %d -a %s -tr %f %f -ot Float32 %s %s' % \
        #       (no_data_value, class_id, cell_size, cell_size, poly, out_rst)

    # subprocess.call(com, shell=True)

    m_info.close()

    # get vector info
    with vector_tools.vopen(poly) as v_info:

        _add_points_from_raster(out_shp, class_id, field_type, out_rst,
                                no_data_value=no_data_value,
                                projection=v_info.projection)


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

    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    poly2points(args.poly, args.points, args.base_image, args.class_id,
                cell_size=args.cell_size, field_type=args.field_type)

    print('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
          (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()
