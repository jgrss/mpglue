#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 7/31/2013
"""

from future.utils import viewitems

import os
import sys
import time
import argparse
import ast

# MapPy
from ..errors import logger
from .. import raster_tools

# GDAL
try:
    from osgeo.gdalconst import *
except ImportError:
    raise ImportError('GDAL did not load')

# NumPy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy did not load')


def recode_func(im, recode_dict=None):

    """
    The image block recode function

    Args:
        im (list of ndarrays)
        recode_dict (dict)
    """

    out_img = im[0]

    # Iterate over each polygon.
    for poly_id, reclass_dict in viewitems(recode_dict):

        # Iterate over the recode rules and
        #   update the image.
        for from_key, to_key in viewitems(reclass_dict):
            out_img[(im[1] == poly_id) & (out_img == from_key)] = to_key

    return out_img


def recode(input_poly, input_image, output_image, recode_dict, class_id='Id'):

    """
    Recodes a thematic image given a vector polygon and a set of rules

    Args:
         input_poly
         input_image
         output_image
         recode_dict
         class_id
    """
    
    d_name, f_name = os.path.split(input_poly)
    f_base, f_ext = os.path.splitext(f_name)    
    
    out_vector_image = os.path.join(d_name, '{}.tif'.format(f_base))

    if os.path.isfile(out_vector_image):
        
        try:
            os.remove(out_vector_image)
        except:
            raise OSError('Could not delete the output raster.')

    with raster_tools.ropen(input_image) as i_info:

        i_info.update_info(storage='int16')

        # get vector info
        # with vector_tools.vopen(input_poly) as v_info:
        #
        #     # Check if the shapefile is UTM North or South. gdal_rasterize has trouble with UTM South
        #     if 'S' in v_info.projection.GetAttrValue('PROJCS')[-1]: # GetUTMZone()
        #         raise ValueError('\nThe shapefile should be projected to UTM North (even for the Southern Hemisphere).\n')

        if not os.path.isfile(out_vector_image):

            logger.info('\nRasterizing {} ...\n'.format(f_name))

            raster_tools.rasterize_vector(input_poly, out_vector_image,
                                          burn_id=class_id,
                                          cell_size=i_info.cellY,
                                          top=i_info.top, bottom=i_info.bottom,
                                          left=i_info.left, right=i_info.right,
                                          projection=i_info.projection,
                                          initial_value=-1,
                                          storage=i_info.storage)

        with raster_tools.ropen(out_vector_image) as v_info:

            o_info = i_info.copy()
            o_info.update_info(storage='byte')

            bp = raster_tools.BlockFunc(recode_func, [i_info, v_info], output_image, o_info,
                                        y_offset=[0, 0], x_offset=[0, 0],
                                        print_statement='\nRecoding {} ...\n'.format(input_image),
                                        recode_dict=recode_dict)

            bp.run()

        v_info = None
        o_info = None

    i_info = None

    if os.path.isfile(out_vector_image):

        try:
            os.remove(out_vector_image)
        except:
            pass

        
def _examples():

    sys.exit("""\

    # In polygon 1, reclassify 6 to 5; in polygon 2, reclassify 2 to 5 and 3 to 5
    recode -p /polygon.shp -i /thematic_map.tif -o /recoded_map.tif --rules "{1: {6:5}, 2: {2:5, 3:5}}"

    """)


def main():

    parser = argparse.ArgumentParser(description='Reclassifies a thematic map',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-p', '--poly', dest='poly', help='The input polygon recode regions', default=None)
    parser.add_argument('-i', '--input', dest='input', help='The input image to recode', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output recoded image', default=None)
    parser.add_argument('-r', '--rules', dest='recode_rules', help='The recode rules', default=None)
    parser.add_argument('-c', '--class-id', dest='class_id', help='The field class id name', default='Id')

    args = parser.parse_args()

    if args.examples:
        _examples()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    recode(args.poly, args.input, args.output, ast.literal_eval(args.recode_rules),
           class_id=args.class_id)

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                (time.asctime(time.localtime(time.time())), (time.time()-start_time)))


if __name__ == '__main__':
    main()
