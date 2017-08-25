#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 2/25/2014
"""

import os
import sys
import time
import argparse

# MapPy
from ..errors import logger
from .. import raster_tools, vector_tools

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


def unique_class_func(im, unique_classes=None):

    unique_classes_blk = np.unique(im[0])

    if len(unique_classes_blk) > 1:

        for cl in unique_classes_blk:

            if cl not in unique_classes:
                unique_classes.append(cl)

    return None, sorted(unique_classes)


def change_func(im, unique_classes=None, change_combos=None):

    out_arr = np.zeros(im[0].shape, dtype='uint8')

    # For each class, get the total pixels that changed.
    for cl_b in unique_classes:

        for cl in unique_classes:

            # Get the class Id.
            cl_id = change_combos['{:d}->{:d}'.format(cl_b, cl)][0]

            # Get the pixels that changed for the current class combination.
            change_pixs = np.where((im[0] == cl_b) & (im[1] == cl), 1, 0)

            # Update the change area dictionary.
            change_combos['{:d}->{:d}'.format(cl_b, cl)][1] += change_pixs.sum()

            # Update the output array with the change Id value.
            out_arr[change_pixs == 1] = cl_id

    return out_arr, change_combos


def change(img_1, img_2, out_img=None, out_report=None,
           boundary_file=None, mask_file=None, be_quiet=False):

    """
    Args:
        img_1 (str): Image for time 1.
        img_2 (str): Image for time 2.
        out_img (Optional[str]): The name of the output change image. Default is None.
        out_report (Optional[str]): The name of the output change text report. Default is None.
        boundary_file (Optional[str]): A file to use for block intersection. Default is None.
            Skip blocks that do not intersect ``boundary_file``.
        mask_file (Optional[str]): An file to use for block masking. Default is None.
            Recode blocks to binary 1 and 0 that intersect ``mask_file``.
        be_quiet (Optional[bool]): Whether to be quiet and do not print progress status. Default is False.

    Returns:
        None, writes to ``out_img`` or ``out_report``.
    """

    if isinstance(out_img, str):
        write_array = True
    else:
        write_array = False

    i_info_1 = raster_tools.ropen(img_1)
    i_info_2 = raster_tools.ropen(img_2)

    # get minimum overlapping extent
    overlap_info = raster_tools.GetMinExtent(i_info_1, i_info_2)

    # set the output image
    o_info = overlap_info.copy()
    # o_info.update_info(left=m_info.left, right=m_info.right, top=m_info.top, bottom=m_info.bottom, rows=rows, cols=cols)

    __, __, x_off_1, y_off_1 = vector_tools.get_xy_offsets(image_info=i_info_1,
                                                           x=overlap_info.left,
                                                           y=overlap_info.top,
                                                           check_position=False)

    __, __, x_off_2, y_off_2 = vector_tools.get_xy_offsets(image_info=i_info_2,
                                                           x=overlap_info.left,
                                                           y=overlap_info.top,
                                                           check_position=False)

    # get unique classes
    unique_classes = list()

    bp = raster_tools.BlockFunc(unique_class_func, [i_info_1], None, o_info,
                                proc_info=overlap_info,
                                y_offset=[y_off_1],
                                x_offset=[x_off_1],
                                out_attributes=['unique_classes_list'],
                                print_statement='\nGetting unique classes ...\n',
                                write_array=False,
                                be_quiet=be_quiet,
                                boundary_file=boundary_file,
                                unique_classes=unique_classes)

    bp.run()

    unique_classes = bp.unique_classes_list

    # get change combinations
    change_combos = dict()

    cl_id = 1
    for cl_b in unique_classes:

        for cl in unique_classes:

            change_combos['{:d}->{:d}'.format(cl_b, cl)] = [cl_id, 0]  # class Id, count

            cl_id += 1

    i_info_1.close()
    i_info_1 = raster_tools.ropen(img_1)

    bp = raster_tools.BlockFunc(change_func, [i_info_1, i_info_2], out_img, o_info,
                                proc_info=overlap_info,
                                y_offset=[y_off_1, y_off_2],
                                x_offset=[x_off_1, x_off_2],
                                out_attributes=['change_combos'],
                                print_statement='\nGetting change ...\n',
                                write_array=write_array,
                                be_quiet=be_quiet,
                                boundary_file=boundary_file,
                                mask_file=mask_file,
                                unique_classes=unique_classes,
                                change_combos=change_combos)

    bp.run()

    # Write the change combination report.
    if isinstance(out_report, str):

        df = pd.DataFrame(bp.change_combos).transpose()

        df.columns = ['Id', 'Count']
        df.index.name = 'Combo'

        df.to_csv(out_report, sep=',')


def _examples():

    sys.exit("""\

    # Get the change between two images and write to a raster and CSV
    change.py -im1 /image_1.tif -im2 /image_2.tif -o /out_image.tif -r /out_report.csv

    # Write the report to a CSV only.
    change.py -im1 /image_1.tif -im2 /image_2.tif -r /out_report.csv

    """)


def main():

    parser = argparse.ArgumentParser(description='Thematic map change statistics',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-im1', '--input1', dest='input1', help='The first image (time 1)', default=None)
    parser.add_argument('-im2', '--input2', dest='input2', help='The second image (time 2)', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output image', default=None)
    parser.add_argument('-r', '--report', dest='report', help='The output report', default=None)

    args = parser.parse_args()

    if args.examples:
        _examples()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    change(args.input1, args.input2, out_img=args.output, out_report=args.report)

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()
