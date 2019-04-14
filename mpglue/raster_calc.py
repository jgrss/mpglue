#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 8/8/2012
"""    

from __future__ import division
from future.utils import viewitems
from builtins import dict

import os
import sys
import time
import argparse
from copy import copy

try:

    from .errors import logger
    from . import raster_tools, vector_tools
    from .helpers import _iteration_parameters, overwrite_file

except:

    from mpglue.errors import logger
    from mpglue import raster_tools, vector_tools
    from mpglue.helpers import _iteration_parameters, overwrite_file

# Numpy
try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed')

# Numexpr
try:
    import numexpr as ne
except ImportError:
    raise ImportError('Numexpr must be installed')


def raster_calc(output,
                equation=None,
                out_type='byte',
                extent=None,
                overwrite=False,
                be_quiet=False,
                out_no_data=0,
                row_block_size=2000,
                col_block_size=2000,
                apply_all_bands=False,
                **kwargs):

    """
    Raster calculator

    Args:
        output (str): The output image.
        equation (Optional[str]): The equation to calculate.
        out_type (Optional[str]): The output raster storage type. Default is 'byte'.
        extent (Optional[str]): An image or instance of ``mappy.ropen`` to use for the output extent. Default is None.
        overwrite (Optional[bool]): Whether to overwrite an existing IDW image. Default is False.
        be_quiet (Optional[bool]): Whether to be quiet and do not report progress. Default is False.
        out_no_data (Optional[int]): The output no data value. Default is 0.
        row_block_size (Optional[int]): The row block chunk size. Default is 2000.
        col_block_size (Optional[int]): The column block chunk size. Default is 2000.
        apply_all_bands (Optional[bool]): Whether to apply the equation to all bands. Default is False.
        **kwargs (str): The rasters to compute. E.g., A='/some_raster1.tif', F='/some_raster2.tif'.
            Band positions default to 1 unless given as [A]_band.

    Examples:
        >>> from mpglue.raster_calc import raster_calc
        >>>
        >>> # Multiply image A x image B
        >>> raster_calc('/output.tif',
        >>>             equation='A * B',
        >>>             A='/some_raster1.tif',
        >>>             B='some_raster2.tif')
        >>>
        >>> # Reads as...
        >>> # Where image A equals 1 AND image B is greater than 5,
        >>> #   THEN write image A, OTHERWISE write 0
        >>> raster_calc('/output.tif',
        >>>             equation='where((A == 1) & (B > 5), A, 0)',
        >>>             A='/some_raster1.tif',
        >>>             B='some_raster2.tif')
        >>>
        >>> # Use different bands from the same image. The letter given for the
        >>> #   image must be the same for the band, followed by _band.
        >>> # E.g., for raster 'n', the corresponding band would be 'n_band'. For
        >>> #   raster 'r', the corresponding band would be 'r_band', etc.
        >>> raster_calc('/output.tif',
        >>>             equation='(n - r) / (n + r)',
        >>>             n='/some_raster.tif',
        >>>             n_band=4,
        >>>             r='/some_raster.tif',
        >>>             r_band=3)

    Returns:
        None, writes to ``output``.
    """

    # Set the image dictionary
    image_dict = dict()
    info_dict = dict()
    info_list = list()
    band_dict = dict()

    temp_files = list()

    if isinstance(extent, str):

        ot_info = raster_tools.ropen(extent)

        temp_dict = copy(kwargs)

        for kw, vw in viewitems(kwargs):

            if isinstance(vw, str):

                d_name, f_name = os.path.split(vw)
                f_base, __ = os.path.splitext(f_name)

                vw_sub = os.path.join(d_name, '{}_temp.vrt'.format(f_base))

                raster_tools.translate(vw,
                                       vw_sub,
                                       format='VRT',
                                       projWin=[ot_info.left,
                                                ot_info.top,
                                                ot_info.right,
                                                ot_info.bottom])

                temp_files.append(vw_sub)

                temp_dict[kw] = vw_sub

        kwargs = temp_dict

    for kw, vw in viewitems(kwargs):

        if '_band' not in kw:
            band_dict['{}_band'.format(kw)] = 1

        if isinstance(vw, str):

            image_dict[kw] = vw

            exec('i_info_{} = raster_tools.ropen(r"{}")'.format(kw, vw))
            exec('info_dict["{}"] = i_info_{}'.format(kw, kw))
            exec('info_list.append(i_info_{})'.format(kw))

        if isinstance(vw, int):
            band_dict[kw] = vw

    for key, value in viewitems(image_dict):
        equation = equation.replace(key, 'marrvar_{}'.format(key))

    # Check for NumPy functions.
    # for np_func in dir(np):
    #
    #     if 'np.' + np_func in equation:
    #
    #         equation = 'np.{}'.format(equation)
    #         break

    for kw, vw in viewitems(info_dict):

        o_info = copy(vw)
        break

    n_bands = 1 if not apply_all_bands else o_info.bands

    if isinstance(extent, raster_tools.ropen):

        # Set the extent from an object.
        overlap_info = extent

    elif isinstance(extent, str):

        # Set the extent from an existing image.
        overlap_info = raster_tools.ropen(extent)

    else:

        # Check overlapping extent
        overlap_info = info_list[0].copy()

        for i_ in range(1, len(info_list)):

            # Get the minimum overlapping extent
            # from all input images.
            overlap_info = raster_tools.GetMinExtent(overlap_info, info_list[i_])

    o_info.update_info(left=overlap_info.left,
                       right=overlap_info.right,
                       top=overlap_info.top,
                       bottom=overlap_info.bottom,
                       rows=overlap_info.rows,
                       cols=overlap_info.cols,
                       storage=out_type,
                       bands=n_bands)

    if overwrite:
        overwrite_file(output)

    out_rst = raster_tools.create_raster(output, o_info)

    if n_bands == 1:
        out_rst.get_band(1)

    block_rows, block_cols = raster_tools.block_dimensions(o_info.rows, o_info.cols,
                                                           row_block_size=row_block_size,
                                                           col_block_size=col_block_size)

    if not be_quiet:
        ctr, pbar = _iteration_parameters(o_info.rows, o_info.cols, block_rows, block_cols)

    # Iterate over the minimum overlapping extent.
    for i in range(0, o_info.rows, block_rows):

        n_rows = raster_tools.n_rows_cols(i, block_rows, o_info.rows)

        for j in range(0, o_info.cols, block_cols):

            n_cols = raster_tools.n_rows_cols(j, block_cols, o_info.cols)

            # For each image, get the offset and
            # convert bands in the equation to ndarrays.
            for key, value in viewitems(image_dict):

                # exec 'x_off, y_off = vector_tools.get_xy_offsets3(overlap_info, i_info_{})'.format(key)
                x_off, y_off = vector_tools.get_xy_offsets(image_info=info_dict[key],
                                                           x=overlap_info.left,
                                                           y=overlap_info.top,
                                                           check_position=False)[2:]

                exec('marrvar_{KEY} = info_dict["{KEY}"].read(bands2open=band_dict["{KEY}_band"], i=i+y_off, j=j+x_off, rows=n_rows, cols=n_cols, d_type="float32")'.format(KEY=key))

            if '&&' in equation:

                out_array = np.empty((n_bands, n_rows, n_cols), dtype='float32')

                for eqidx, equation_ in enumerate(equation.split('&&')):

                    if 'nan_to_num' in equation_:

                        if not equation_.startswith('np.'):
                            equation_ = 'np.' + equation_

                        equation_ = 'out_array[eqidx] = {}'.format(equation_)
                        exec(equation_)

                    else:
                        out_array[eqidx] = ne.evaluate(equation_)

            else:

                if 'nan_to_num' in equation:

                    equation_ = 'out_array = {}'.format(equation)
                    exec(equation_)

                else:
                    out_array = ne.evaluate(equation)

            # Set the output no data values.
            out_array[np.isnan(out_array) | np.isinf(out_array)] = out_no_data

            if n_bands == 1:

                out_rst.write_array(out_array,
                                    i=i,
                                    j=j)

            else:

                for lidx in range(0, n_bands):

                    out_rst.write_array(out_array[lidx],
                                        i=i,
                                        j=j,
                                        band=lidx+1)

            if not be_quiet:

                pbar.update(ctr)
                ctr += 1

    if not be_quiet:
        pbar.finish()

    # Close the input image.
    for key, value in viewitems(info_dict):
        info_dict[key].close()

    # close the output drivers
    out_rst.close_all()

    out_rst = None

    # Cleanup
    for temp_file in temp_files:

        if os.path.isfile(temp_file):
            os.remove(temp_file)


def _examples():

    sys.exit("""\

    # Write out class 2 from image A
    raster_calc.py -A /image.tif -o /output.tif -eq "where(A==2, 1, 0)"

    # Write the intersection of images A and B as 1s
    raster_calc.py -A /image_a.tif -B /image_b.tif -o /output.tif -eq "where((A==1) & (B==1), 1, 0)"
    
    # Convert an RGB image to grayscale
    raster-calc -A rgb.tif -B rgb.tif -C rgb.tif -A_band 1 -B_band 2 -C_band 3 -o grayscale.tif -ot float32 -eq "A*0.2989 + B*0.5870 + C*0.1140"

    """)


def main():

    parser = argparse.ArgumentParser(description='Raster calculator (*The number of input images is limitless if used as a Python function.)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-A', dest='A', help='Add an image as A')
    parser.add_argument('-B', dest='B', help='Add an image as B')
    parser.add_argument('-C', dest='C', help='Add an image as C')
    parser.add_argument('-D', dest='D', help='Add an image as D')
    parser.add_argument('-E', dest='E', help='Add an image as E')
    parser.add_argument('-F', dest='F', help='Add an image as F')
    parser.add_argument('-G', dest='G', help='Add an image as G')
    parser.add_argument('-A_band', dest='A_band', help='A band position', default=1, type=int)
    parser.add_argument('-B_band', dest='B_band', help='B band position', default=1, type=int)
    parser.add_argument('-C_band', dest='C_band', help='C band position', default=1, type=int)
    parser.add_argument('-D_band', dest='D_band', help='D band position', default=1, type=int)
    parser.add_argument('-E_band', dest='E_band', help='E band position', default=1, type=int)
    parser.add_argument('-F_band', dest='F_band', help='F band position', default=1, type=int)
    parser.add_argument('-G_band', dest='G_band', help='G band position', default=1, type=int)
    parser.add_argument('-o', '--output', dest='output', help='The output image', default=None)
    parser.add_argument('-eq', '--equation', dest='equation', help='The equation', default='', type=str)
    parser.add_argument('-ot', '--out_type', dest='out_type', help='The output type', default='byte')
    parser.add_argument('--extent', dest='extent', help='An image with the desired output extent', default=None)
    parser.add_argument('--apply-all-bands', dest='apply_all_bands',
                        help='Whether to apply the equation to all bands', action='store_true')
    parser.add_argument('--overwrite', dest='overwrite',
                        help='Whether to overwrite an existing image', action='store_true')
    parser.add_argument('--be-quiet', dest='be_quiet',
                        help='Whether to be quiet and do not print to screen', action='store_true')

    args = parser.parse_args()

    if args.examples:
        _examples()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    raster_calc(args.output,
                equation=args.equation,
                out_type=args.out_type,
                extent=args.extent,
                apply_all_bands=args.apply_all_bands,
                overwrite=args.overwrite,
                be_quiet=args.be_quiet,
                A=args.A,
                B=args.B,
                C=args.C,
                D=args.D,
                E=args.E,
                F=args.F,
                G=args.G,
                A_band=args.A_band,
                B_band=args.B_band,
                C_band=args.C_band,
                D_band=args.D_band,
                E_band=args.E_band,
                F_band=args.F_band,
                G_band=args.G_band)

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                (time.asctime(time.localtime(time.time())), (time.time()-start_time)))


if __name__ == '__main__':
    main()
