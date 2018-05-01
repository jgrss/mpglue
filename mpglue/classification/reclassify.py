#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 11/14/2011
""" 

from future.utils import viewitems

import sys
import time
import ast
import argparse

# MapPy
from ..errors import logger
from .. import raster_tools


def reclassify_func(im, recode_dict=None):

    """
    The image block reclassification function
    """

    out_img = im[0].copy()

    # reclassify image
    for key, value in sorted(viewitems(recode_dict)):
        out_img[(im[0] == key)] = value

    return out_img


def reclassify(input_image, output_image, recode_dict):

    """
    Reclassifies a thematic image

    Args:
        input_image (str): Path, name, and extension of image to reclassify.
        output_image (str): Path, name, and extension of output image.
        recode_dict (dict): Dictionary of values to reclassify.

    Examples:
        >>> # Reclassify class 1 to 3, class 2 to 3, class 4 to 3, and so on ...
        >>> from mpglue.classification import reclassify
        >>>
        >>> recode_dict  = {1:3, 2:3, 4:3, 5:3, 9:4, -128:255}
        >>> reclassify('/image_to_reclassify.tif', '/out_image.tif', recode_dict)

    Returns:
        None, writes reclassified images to ``output_image``.
    """

    # Get image information and reclassify.
    with raster_tools.ropen(input_image) as i_info:

        o_info = i_info.copy()

        o_info.update_info(bands=1,
                           storage='byte')

        bp = raster_tools.BlockFunc(reclassify_func,
                                    [i_info],
                                    output_image,
                                    o_info,
                                    print_statement='\nReclassifying {} ...\n'.format(input_image),
                                    recode_dict=recode_dict)

        bp.run()

    del i_info


def _examples():

    sys.exit("""\

    # Reclassify class 1 to 1, 2 to 1, and 3 to 2
    reclassify -i /some_image.tif -o /output_image.tif -r "{2:1,3:2}"

    """)


def main():

    parser = argparse.ArgumentParser(description='Reclassifies a thematic map',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--input', dest='input', help='The input image to adjust', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output adjusted image', default=None)
    parser.add_argument('-r', '--reclassify', dest='reclassify', help='The reclassification class dictionary',
                        default=None)

    args = parser.parse_args()

    if args.examples:
        _examples()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    reclassify(args.input, args.output, ast.literal_eval(args.reclassify))

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                (time.asctime(time.localtime(time.time())), (time.time()-start_time)))


if __name__ == '__main__':
    main()
