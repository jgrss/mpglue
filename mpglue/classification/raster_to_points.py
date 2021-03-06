#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 12/6/2018
"""

from __future__ import division, print_function

import os
import sys
import argparse

if sys.version_info.major != 3:
    from itertools import izip as zip

from ..errors import logger
from .. import raster_tools, vector_tools

# fiona
try:
    import fiona
except:
    logger.error('  Fiona must be installed')
    raise ImportError

# shapely
try:
    from shapely.geometry import Point, mapping
    import shapely.speedups
    shapely.speedups.enable()
except:
    logger.error('  shapely must be installed')
    raise ImportError

# NumPy
try:
    import numpy as np
except:
    logger.error('  NumPy must be installed')
    raise ImportError


def _get_n_blocks(rows, cols, block_size):

    n_blocks = 0

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            n_blocks += 1

    return n_blocks


def raster_to_points(values,
                     points,
                     column='value',
                     no_data_value=0,
                     skip=0,
                     sample_frac=None,
                     sample_map=None,
                     block_size=512,
                     overwrite=False,
                     verbose=0):

    """
    Converts a raster to points. Points are created in cell centers if
    a cell value != `no_data_value`.

    Args:
        values (str): The raster values.
        points (str): The vector points.
        column (Optional[str]): The class id column name. Default is 'value'.
        no_data_value (Optional[int,float]): The raster no data value.
            Default is 0.
        skip (Optional[int]): A cell skip factor.
        sample_frac (Optional[float]): A random sub-sample fraction.
        sample_map (Optional[dict]): A random sub-sample fraction dictionary map.
        block_size (Optional[int]): The processing block size. Default is 512.
        overwrite (Optional[bool]): Whether to overwrite existing points file.
            Default is False.
        verbose (Optional[int]): The verbosity level. Default is 0.

    Returns:
        None, writes to `points`.
    """

    if not os.path.isfile(values):

        logger.error('  The values file does not exist.')
        raise OSError

    if os.path.isfile(points):

        if overwrite:
            vector_tools.delete_vector(points)
        else:

            logger.warning('  The points file already exists.')
            return

    d_name = os.path.dirname(points)

    if not os.path.isdir(d_name):
        os.makedirs(d_name)

    with raster_tools.ropen(values) as src:

        hcell = src.cellY / 2.0

        if verbose > 0:

            n_block = 1
            n_blocks = _get_n_blocks(src.rows, src.cols, block_size)

        for i in range(0, src.rows, block_size):

            top = src.top - (i * src.cellY)

            n_rows = raster_tools.n_rows_cols(i, block_size, src.rows)

            for j in range(0, src.cols, block_size):

                left = src.left + (j * src.cellY)

                n_cols = raster_tools.n_rows_cols(j, block_size, src.cols)

                if verbose > 0:

                    logger.info('  Block {:,d} of {:,d} ...'.format(n_block, n_blocks))
                    n_block += 1

                # Read the current block.
                block = src.read(bands=1,
                                 i=i,
                                 j=j,
                                 rows=n_rows,
                                 cols=n_cols)

                block[np.isnan(block) | np.isinf(block)] = no_data_value

                # Create the points
                if block.max() != no_data_value:

                    # Block indices of points
                    idx = np.where(block != no_data_value)

                    # Point coordinates
                    x_shift = ((left + (idx[1] * src.cellY)) + hcell).tolist()
                    y_shift = ((top - (idx[0] * src.cellY)) - hcell).tolist()

                    block_data = block[idx].tolist()

                    if skip > 0:

                        x_shift = x_shift[::skip]
                        y_shift = y_shift[::skip]
                        block_data = block_data[::skip]

                    if isinstance(sample_frac, float) or isinstance(sample_map, dict):

                        # Sample proportional for each class
                        x_shift_t = list()
                        y_shift_t = list()
                        block_data_t = list()

                        for code in np.unique(np.array(block_data)).tolist():

                            # Indices of current code value
                            code_idx = np.where(np.array(block_data) == code)[0]

                            n = code_idx.shape[0]

                            if n > 0:

                                if isinstance(sample_frac, float):
                                    sample = sample_frac
                                else:
                                    sample = sample_map[int(code)]

                                idx = np.array(sorted(np.random.choice(range(0, n),
                                                                       size=int(sample * n),
                                                                       replace=False)), dtype='int64')

                                x_shift_t += np.array(x_shift)[code_idx[idx]].tolist()
                                y_shift_t += np.array(y_shift)[code_idx[idx]].tolist()
                                block_data_t += np.array(block_data)[code_idx[idx]].tolist()

                        x_shift = x_shift_t
                        y_shift = y_shift_t
                        block_data = block_data_t

                    schema = {'geometry': 'Point',
                              'properties': {column: 'int'}}

                    mode = 'w' if not os.path.isfile(points) else 'a'

                    if verbose > 0:
                        logger.info('  Writing points to file ...')

                    with fiona.open(points,
                                    mode,
                                    driver='ESRI Shapefile',
                                    schema=schema,
                                    crs_wkt=src.projection) as output:

                        for x, y, v in zip(x_shift, y_shift, block_data):

                            output.write({'geometry': mapping(Point(x, y)),
                                          'properties': {column: v}})

    src = None


def main():

    parser = argparse.ArgumentParser(description='Creates points from a raster',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-v', '--values', dest='values', help='The values raster', default=None)
    parser.add_argument('-p', '--points', dest='points', help='The output points', default=None)
    parser.add_argument('--column', dest='column', help='The value column name', default='value')
    parser.add_argument('--no-data', dest='no_data_value', help='The no data raster value', default=0.0, type=float)
    parser.add_argument('--skip', dest='skip', help='A cell skip factor', default=0, type=int)
    parser.add_argument('--sample-frac', dest='sample_frac', help='A random sub-sample fraction', default=None, type=float)
    parser.add_argument('--block-size', dest='block_size', help='The processing block size', default=512, type=int)
    parser.add_argument('--overwrite', dest='overwrite', help='Whether to overwrite existing points file', action='store_true')
    parser.add_argument('--verbose', dest='verbose', help='The verbosity level', default=0, type=int)

    args = parser.parse_args()

    raster_to_points(args.values,
                     args.points,
                     column=args.column,
                     no_data_value=args.no_data_value,
                     skip=args.skip,
                     sample_frac=args.sample_frac,
                     block_size=args.block_size,
                     overwrite=args.overwrite,
                     verbose=args.verbose)


if __name__ == '__main__':
    main()
