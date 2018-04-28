#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 7/26/2011
"""

from __future__ import division
from future.utils import viewitems
from builtins import int

import sys
import time
import argparse
import itertools
from joblib import Parallel, delayed

from ..errors import logger
from .. import raster_tools
from ..helpers import overwrite_file, get_block_chunks
from ._moving_window import moving_window

try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')


def do_stat(chunk_array, block_chunk, window_size, chunk_size, statistic, ignore_value):

    ip = block_chunk[3]
    jp = block_chunk[7]

    return moving_window(chunk_array,
                         statistic=statistic,
                         window_size=window_size,
                         ignore_value=ignore_value)[ip:chunk_size, jp:chunk_size]


def piece(stat_chunks, block_chunks, fill_array):

    # Put it back together.
    for filled_image, block_chunk in itertools.izip(stat_chunks, block_chunks):

        fis = block_chunk[0]
        fjs = block_chunk[4]

        fi = filled_image.shape[0]
        fj = filled_image.shape[1]

        if (fi == 0) or (fj == 0):
            continue

        fill_array[fis:fis+fi, fjs:fjs+fj] = filled_image

    return fill_array


class Parameters(object):

    def __init__(self):
        self.window_size = 3


def focal_statistics(in_image, out_image, band=1, overwrite=False, chunk_size=512, n_jobs=0, **kwargs):

    """
    Computes focal (moving window) statistics.

    Args:
        in_image (str): The input image.
        out_image (str): The output image.
        band (int or int list). The band to process. Default is 1.
        overwrite (Optional[bool]): Whether to overwrite an existing file. Default is False.
        chunk_size (Optional[int]): The block chunk size for joblib. Default is 512.
        n_jobs (Optional[int]): The number of parallel jobs. If -1 or greater than 1, the entire image will be
            processed in memory. Otherwise, chunks are written in tiles. Default is 0.

    Returns:
        None, writes to ``out_image``.

    Examples:
        >>> from mappy.classifiers.post import focal_statistics
        >>>
        >>> # class majority filter
        >>> focal_statistics('/image.tif', '/output.tif',
        >>>                  statistic='majority', window_size=5)
        >>>
        >>> # or with an array, call the function directly
        >>> from mappy.classifiers.post import moving_window
        >>> from mappy import raster_tools
        >>>
        >>> in_array = raster_tools.read('/image.tif')
        >>> array = moving_window(in_array, statistic='majority', window_size=5)
    """

    parameters = Parameters()

    for k, v in viewitems(kwargs):
        setattr(parameters, k, v)

    i_info = raster_tools.ropen(in_image)

    o_info = i_info.copy()

    o_info.bands = 1

    if parameters.statistic == 'majority':
        o_info.storage = 'byte'
    else:
        o_info.storage = 'float32'

    if overwrite:
        overwrite_file(out_image)

    out_rst = raster_tools.create_raster(out_image, o_info)

    if (n_jobs == -1) or (n_jobs > 1):

        block_chunks = get_block_chunks(i_info.rows, i_info.cols, chunk_size, window_size)

        logger.info('  Processing tiles ...')

        chunk_stats = Parallel(n_jobs=n_jobs,
                               max_nbytes=None)(delayed(do_stat)(i_info.read(bands2open=band,
                                                                             i=block_chunk[1],
                                                                             j=block_chunk[5],
                                                                             rows=block_chunk[8],
                                                                             cols=block_chunk[9],
                                                                             d_type='float32'),
                                                                 block_chunk, parameters.window_size, chunk_size,
                                                                 parameters.statistic, parameters.ignore_value)
                                                for block_chunk in block_chunks)

        # Put it back together.
        fill_array = piece(chunk_stats, block_chunks, i_info.read(bands2open=band, d_type='float32'))

        out_rst.write_array(fill_array, band=band)

    else:

        out_rst.get_band(1)

        half_window = parameters.window_size / 2

        block_rows, block_cols = raster_tools.block_dimensions(i_info.rows, i_info.cols,
                                                               row_block_size=parameters.window_size*256,
                                                               col_block_size=parameters.window_size*256)

        ttl_blks = int((np.ceil(float(i_info.rows) / float(block_rows))) *
                       (np.ceil(float(i_info.cols) / float(block_cols))))

        ttl_blks_ct = 0

        for i in range(0, i_info.rows, block_rows-half_window-1):

            n_rows = raster_tools.n_rows_cols(i, block_rows+half_window+1, i_info.rows)

            for j in range(0, i_info.cols, block_cols-half_window-1):

                n_cols = raster_tools.n_rows_cols(j, block_cols+half_window+1, i_info.cols)

                if ttl_blks_ct % 20 == 0:

                    tile_count = ttl_blks_ct + 19

                    if tile_count > ttl_blks:
                        tile_count = ttl_blks

                    logger.info('  Processing tiles {:d} -- {:d} of {:d} ...'.format(ttl_blks_ct, tile_count, ttl_blks))

                out_array = moving_window(i_info.read(bands2open=band, i=i, j=j, rows=n_rows, cols=n_cols,
                                                      d_type='float32'), **kwargs)

                out_rst.write_array(out_array[half_window:-half_window, half_window:-half_window],
                                    i=i+half_window, j=j+half_window)

                ttl_blks_ct += 1

    i_info.close()

    out_rst.close_all()

    out_rst = None


def _examples():

    sys.exit("""\

    # Calculate the majority value within a 5x5 pixel window.
    focal_statistics.py -i /image.tif -o /output.tif -s majority -w 5

    # Calculate the mean value within a 3x3 pixel window, ignoring zeros.
    focal_statistics.py -i /image.tif -o /output.tif -s mean -w 3 -iv 0

    # Calculate the mean value within a 3x3 pixel window, running 8 jobs in parallel.
    focal_statistics.py -i /image.tif -o /output.tif -s mean -w 3 -j 8

    """)


def main():

    parser = argparse.ArgumentParser(description='Moving window statistics',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--input', dest='input', help='The input image', default=None)
    parser.add_argument('-o', '--output', dest='output', help='The output image', default=None)
    parser.add_argument('-b', '--band', dest='band', help='The band to process', default=1, type=int)
    parser.add_argument('-s', '--statistic', dest='statistic', help='The statistic', default='mean',
                        choices=['mean', 'min', 'max', 'median', 'majority', 'fill', 'percent', 'sum'])
    parser.add_argument('-w', '--window_size', dest='window_size', help='The window size', default=3, type=int)
    parser.add_argument('-iv', '--ignore_value', dest='ignore_value', help='A value to ignore', default=None, type=int)
    parser.add_argument('--overwrite', dest='overwrite', help='Whether to overwrite an existing file',
                        action='store_true')
    parser.add_argument('--resample', dest='resample', help='Whether to resample to the kernel size',
                        action='store_true')
    parser.add_argument('-j', '--n_jobs', dest='n_jobs',
                        help='The number of parallel jobs. If 0 or 1, chunks are written in tiles.',
                        default=0, type=int)

    args = parser.parse_args()

    if args.examples:
        _examples()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    focal_statistics(args.input, args.output, band=args.band, statistic=args.statistic,
                     window_size=args.window_size, ignore_value=args.ignore_value,
                     overwrite=args.overwrite, resample=args.resample, n_jobs=args.n_jobs)

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                (time.asctime(time.localtime(time.time())), (time.time()-start_time)))

if __name__ == '__main__':
    main()
