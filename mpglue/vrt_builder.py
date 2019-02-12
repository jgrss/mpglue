#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 4/29/2015
"""

from __future__ import division
from future.utils import iteritems, viewitems
from builtins import int

import os
import sys
import fnmatch
import time
import argparse
from collections import OrderedDict

from .errors import logger
from . import raster_tools, vector_tools

FORCE_TYPE_DICT = dict(byte='Byte',
                       float32='Float32',
                       float64='Float64',
                       uint16='UInt16',
                       uint32='Uint32',
                       uint64='Uint64')


class VRTBuilder(object):

    """
    A class to manage and build a VRT file from a dictionary list of raster files.
    """

    def __init__(self):

        self.time_stamp = time.asctime(time.localtime(time.time()))

    def get_xml_base(self):

        """
        holds the XML base strings
        """

        gap_1 = ' '
        gap_2 = '  '
        gap_3 = '   '

        self.xml_base = '<VRTDataset rasterXSize="full_rasterXSize" rasterYSize="full_rasterYSize">\n \
                        {}<SRS>image_SRS</SRS>\n \
                        {}<GeoTransform>image_GeoTransform</GeoTransform>\n'.format(gap_1, gap_1)

        self.xml_band_header = '{}<VRTRasterBand dataType="image_dataType" band="image_band">\n \
                                {}<NoDataValue>0</NoDataValue>\n \
                                {}<ColorInterp>Gray</ColorInterp>\n'.format(gap_1, gap_2, gap_2)

        self.xml_band = '{}<ComplexSource>\n \
                        {}<SourceFilename relativeToVRT="0">image_SourceFilename</SourceFilename>\n \
                        {}<SourceBand>image_SourceBand</SourceBand>\n \
                        {}<SourceProperties RasterXSize="image_RasterXSize" RasterYSize="image_RasterYSize" DataType="image_dataType" BlockXSize="image_BlockXSize" BlockYSize="image_BlockYSize" />\n \
                        {}<SrcRect xOff="0" yOff="0" xSize="image_RasterXSize" ySize="image_RasterYSize" />\n \
                        {}<DstRect xOff="full_xOff" yOff="full_yOff" xSize="image_RasterXSize" ySize="image_RasterYSize" />\n \
                        {}<NODATA>0</NODATA>\n \
                        {}</ComplexSource>\n{}'.format(gap_2, gap_3, gap_3, gap_3, gap_3, gap_3, gap_3, gap_2, gap_2)

        self.band_end = '</VRTRasterBand>\n'

        self.xml_end = '</VRTDataset>'

    def get_full_extent(self, in_dict, separate, search_extension, use_image_min=False):

        """Gets extent information for all images"""

        if isinstance(in_dict, str):

            if not os.path.isdir(in_dict):

                logger.error('The input string should be an existing directory.')
                raise TypeError

            # List the images.
            image_search_list = sorted(fnmatch.filter(os.listdir(in_dict), '*{}'.format(search_extension)))

            image_search_list = [os.path.join(in_dict, im) for im in image_search_list]

            # Store as a dictionary.
            if separate:

                in_dict = dict(zip(['{:05d}'.format(ci) for ci in range(1, len(image_search_list)+1)],
                                   [[im] for im in image_search_list]))

            else:
                in_dict = {'00001': image_search_list}

        elif isinstance(in_dict, list):

            # Store as a dictionary.
            if separate:
                
                in_dict = dict(zip(['{:05d}'.format(ci) for ci in range(1, len(in_dict)+1)],
                                   [[im] for im in in_dict]))

            else:
                in_dict = {'00001': in_dict}

        else:

            if not isinstance(in_dict, dict):

                logger.error('The input parameter must be a dictionary, list, or directory.')
                raise TypeError

        # Ensure order by sorting by `OrderedDict` items.
        self.in_dict = OrderedDict(sorted(list(iteritems(in_dict))))

        # Use only the first list
        #   for the extent.
        image_list = self.in_dict[list(self.in_dict)[0]]

        if not isinstance(image_list, list):

            logger.error('An image list must be given.')
            raise TypeError

        if not image_list:

            logger.error('A full image list must be given.')
            raise ValueError

        # Get the minimum and maximum
        #   extent of all images.

        if use_image_min:

            self.left = -100000000.0
            self.right = 100000000.0
            self.top = 100000000.0
            self.bottom = -100000000.0

        else:

            self.left = 100000000.0
            self.right = -100000000.0
            self.top = -100000000.0
            self.bottom = 100000000.0

        for im, image in enumerate(image_list):

            with raster_tools.ropen(image) as i_info:

                if im == 0:

                    self.cell_size = i_info.cellY
                    self.cellY = i_info.cellY
                    self.cellX = i_info.cellX
                    self.projection = i_info.projection
                    self.storage = i_info.storage
                    self.geo_transform = list(i_info.geo_transform)

                if use_image_min:

                    self.left = max(i_info.left, self.left)
                    self.right = min(i_info.right, self.right)
                    self.top = min(i_info.top, self.top)
                    self.bottom = max(i_info.bottom, self.bottom)

                else:

                    self.left = min(i_info.left, self.left)
                    self.right = max(i_info.right, self.right)
                    self.top = max(i_info.top, self.top)
                    self.bottom = min(i_info.bottom, self.bottom)

            i_info = None

        self.geo_transform[0] = self.left
        self.geo_transform[3] = self.top

        self.geo_transform = str(self.geo_transform)

        if (self.left < 0) and (self.right < 0):
            self.columns = int(round(abs(abs(self.right) - abs(self.left)) / self.cell_size))
        elif (self.left >= 0) and (self.right >= 0):
            self.columns = int(round(abs(abs(self.right) - abs(self.left)) / self.cell_size))
        elif (self.left < 0) and (self.right >= 0):
            self.columns = int(round((abs(self.right) + abs(self.left)) / self.cell_size))

        if (self.top < 0) and (self.bottom < 0):
            self.rows = int(round(abs(abs(self.top) - abs(self.bottom)) / self.cell_size))
        elif (self.top >= 0) and (self.bottom >= 0):
            self.rows = int(round(abs(abs(self.top) - abs(self.bottom)) / self.cell_size))
        elif (self.top >= 0) and (self.bottom < 0):
            self.rows = int(round((abs(self.top) + abs(self.bottom)) / self.cell_size))

    def replace_main(self, no_data=None):

        self.xml_base = self.xml_base.replace('full_rasterXSize', str(self.columns))
        self.xml_base = self.xml_base.replace('full_rasterYSize', str(self.rows))

        self.xml_base = self.xml_base.replace('image_SRS', self.projection)

        self.geo_transform = self.geo_transform.replace('[', '')
        self.geo_transform = self.geo_transform.replace(']', '')

        self.xml_base = self.xml_base.replace('image_GeoTransform', self.geo_transform)

        if isinstance(no_data, int):
            self.xml_band_header = self.xml_band_header.replace('<NoDataValue>0', '<NoDataValue>{:d}'.format(no_data))
            self.xml_band = self.xml_band.replace('<NODATA>0', '<NODATA>{:d}'.format(no_data))
        elif isinstance(no_data, float):
            self.xml_band_header = self.xml_band_header.replace('<NoDataValue>0', '<NoDataValue>{:f}'.format(no_data))
            self.xml_band = self.xml_band.replace('<NODATA>0', '<NODATA>{:f}'.format(no_data))

    def add_bands(self,
                  bands2include=None,
                  start_band=1,
                  force_type=None,
                  subset=False,
                  base_name=None,
                  be_quiet=False,
                  relative_path=False,
                  out_vrt=None):

        """
        in_dict (OrderedDict): An ordered dictionary. The main bands should be first, followed by ancillary data.
            Example:
                in_dict = {'1': ['image1.tif', 'image2.tif'],
                          '2': ['dem_tile1.tif', 'dem_tile2.tif', ..., 'dem_tileN.tif']}

        Args:
            bands2include (Optional[list]): An empty list results in all bands. Default is [].
            start_band (Optional[int]): The starting band position.
            force_type (Optional[str]): Used to force mixed data types. Default is None, or no forcing.
            subset (Optional[bool]): Whether to subset ancillary data to main data. Default is False.
            base_name (Optional[str])
            be_quiet (Optional[str]): Whether to be quiet and do not print to screen. Default is False.
            relative_path (Optional[bool]): Whether to use relative path names. Default is False.
            out_vrt (Optional[str]): The output VRT file. Default is None.
        """

        new_dict = OrderedDict()
        for k, v in viewitems(self.in_dict):
            new_dict['{:05d}'.format(int(k))] = v

        self.in_dict = OrderedDict(sorted(list(iteritems(new_dict)), key=lambda tv: tv[0]))

        self.base_name = base_name

        self._band_count()

        band_counter = 1

        for bdk, bd in viewitems(self.band_dict):

            if not be_quiet:

                sys.stdout.write('\rBuilding list {} of {:05d} ...'.format(bdk, len(self.band_dict)))
                sys.stdout.flush()

            image_list = self.in_dict[bdk]

            for bdi in range(start_band, bd+start_band):

                band_found = False

                if bands2include:

                    for band2include in bands2include:

                        if band2include == bdi:

                            band_found = True

                            break

                else:
                    band_found = True

                if not band_found:
                    continue

                if isinstance(force_type, str):
                    self.xml_band_header_ = self.xml_band_header.replace('image_dataType',
                                                                         FORCE_TYPE_DICT[force_type.lower()])
                else:
                    self.xml_band_header_ = self.xml_band_header.replace('image_dataType', self.storage)

                self.xml_band_header_ = self.xml_band_header_.replace('image_band', str(band_counter))

                # Add to the XML string.
                self.xml_base = '{}{}'.format(self.xml_base, self.xml_band_header_)

                for image in image_list:

                    i_info = raster_tools.ropen(image)

                    # Check if the image is outside the current frame.
                    if i_info.outside(self):

                        i_info.close()
                        i_info = None

                        continue

                    # Subset the current image to
                    #   the correct extent.
                    if subset:

                        if bdk != '00001':

                            image, sub_directory = self._subset(i_info, image)

                            if not image:

                                i_info.close()
                                i_info = None

                                continue

                            i_info.close()
                            i_info = None

                            i_info = raster_tools.ropen(image)

                    __, __, x_offset, y_offset = vector_tools.get_xy_offsets(image_info=self,
                                                                             xy_info=i_info,
                                                                             round_offset=True,
                                                                             check_position=False)

                    # Set the image name.
                    if relative_path:

                        image_dir, image_name = os.path.split(image)
                        vrt_dir = os.path.split(out_vrt)[0]

                        relative_image = os.path.join(os.path.relpath(image_dir, vrt_dir), image_name)

                        self.xml_band_ = self.xml_band.replace('image_SourceFilename', relative_image)
                        self.xml_band_ = self.xml_band_.replace('relativeToVRT="0"', 'relativeToVRT="1"')

                    else:
                        self.xml_band_ = self.xml_band.replace('image_SourceFilename', image)

                    if isinstance(force_type, str):
                        self.xml_band_ = self.xml_band_.replace('image_dataType', FORCE_TYPE_DICT[force_type.lower()])
                    else:
                        self.xml_band_ = self.xml_band_.replace('image_dataType', str(i_info.storage))

                    vrt_text_list = ['image_SourceBand',
                                     'image_RasterXSize',
                                     'image_RasterYSize',
                                     'image_BlockXSize',
                                     'image_BlockYSize',
                                     'full_xOff',
                                     'full_yOff']

                    image_info_list = [bdi,
                                       i_info.cols,
                                       i_info.rows,
                                       i_info.block_x,
                                       i_info.block_y,
                                       x_offset,
                                       y_offset]

                    for rep, wit in zip(*[vrt_text_list, image_info_list]):
                        self.xml_band_ = self.xml_band_.replace(rep, str(wit))

                    i_info.close()
                    i_info = None

                    # add to the XML string
                    self.xml_base = '{}{}'.format(self.xml_base, self.xml_band_)

                band_counter += 1

                # add to the XML string
                self.xml_base = '{}{}'.format(self.xml_base, self.band_end)

        # add to the XML string
        self.xml_base = '{}{}'.format(self.xml_base, self.xml_end)

        if not be_quiet:

            logger.info('')
            logger.info('  Finished processing VRT file')

    # def delete_subs(self, sub_directory):
    #
    #     sub_files = fnmatch.filter(os.listdir(sub_directory), '*.tif')
    #
    #     for sub_file in sub_files:
    #
    #         full_file = '%s/%s' % (sub_directory, sub_file)
    #
    #         if os.path.isfile(full_file):
    #
    #             os.remove(full_file)
    #
    #     shutil.rmtree(sub_directory)

    def _subset(self, image_info, image_name):

        d_name, f_name = os.path.split(image_name)
        f_base, f_ext = os.path.splitext(f_name)

        if isinstance(self.base_name, str):
            sub_directory = os.path.join(d_name, self.base_name, 'subs')
        else:
            sub_directory = os.path.join(d_name, 'subs')

        if not os.path.isdir(sub_directory):
            os.makedirs(sub_directory)

        out_sub = os.path.join(sub_directory, '{}_sub.tif'.format(f_base))

        if not os.path.isfile(out_sub):

            if image_info.left < self.left:
                sub_left = self.left
            else:
                sub_left = image_info.left

            if image_info.right > self.right:
                sub_right = self.right
            else:
                sub_right = image_info.right

            if image_info.top > self.top:
                sub_top = self.top
            else:
                sub_top = image_info.top

            if image_info.bottom < self.bottom:
                sub_bottom = self.bottom
            else:
                sub_bottom = image_info.bottom

            if (sub_right - sub_left < image_info.cellY) or (sub_top - sub_bottom < image_info.cellY):
                return None, sub_directory
            else:

                # minX, minY, maxX, maxY
                raster_tools.warp(image_name, out_sub,
                                  outputBounds=[sub_left, sub_bottom, sub_right, sub_top],
                                  multithread=True,
                                  warpMemoryLimit=256,
                                  creationOptions=['COMPRESS=YES'])

                # -te [xmin ymin xmax ymax]
                # com = 'gdalwarp -q -te {:f} {:f} {:f} {:f} -multi -wo NUM_THREADS=ALL_CPUS \
                # -wm 256 --config GDAL_CACHEMAX 256 \
                # -wo USE_OPENCL=TRUE -co COMPRESS=DEFLATE -co TILED=YES {} {}'.format(sub_left, sub_bottom, sub_right,
                #                                                                      sub_top, image_name, out_sub)
                #
                # subprocess.call(com, shell=True)

        return out_sub, sub_directory

    def _band_count(self):

        self.band_dict = dict()

        # get first image from each list
        for k, v in viewitems(self.in_dict):

            vi_ = v[0]

            if 'HDF4_EOS:EOS_GRID:' in vi_:
                n_bands = 1
            else:

                with raster_tools.ropen(v[0]) as i_info:
                    n_bands = i_info.bands

                i_info = None

            self.band_dict[k] = n_bands

        self.band_dict = OrderedDict(sorted(list(iteritems(self.band_dict)), key=lambda t: t[0]))


def vrt_builder(in_dict,
                out_vrt,
                bands2include=None,
                start_band=1,
                force_type=None,
                subset=False,
                base_name=None,
                no_data=None,
                be_quiet=False,
                overwrite=False,
                separate=False,
                search_extension='.tif',
                relative_path=False,
                use_image_min=False):

    """
    Builds a VRT file, accepting raster files with different band counts.
    
    Args:
        in_dict (dict or list or str): The input dictionary, image list, or directory of images to add to a VRT file.

            dict format:
                {'1': [<list of images>],
                 '2': [<list of images>],
                 ...}

            where,
                each dictionary key is a separate layer in the VRT output.

            Note:
                If `in_dict`=`str`, it is assumed to be a directory and the file names will be sorted.
                If `in_dict`=`list`, the file names are assumed to be sorted.

        out_vrt (str): The output VRT file.
        bands2include (Optional[list]): A list of bands to include. Default is None, or all bands.
        force_type (Optional[str]): Force the output storage type. Default is None.
        subset (Optional[bool]): Whether to subset ``in_dict`` >= '2' to '1'. Default is False.
        base_name (Optional[str]): A base name to prepend to the /subs directory. Default is None.
        separate (Optional[bool]): Whether to keep layers separate. Otherwise, mosaic as one layer. Default is False.
            Note:
                Only applied when `in_dict`=`str`.
        search_extension (Optional[str]): The image search extension when `in_dict`=`str`. Default is '.tif'.
        no_data (Optional[int or float]): The output no data flag. Default is None.
        be_quiet (Optional[bool]): Whether to be quiet and do not print progress. Default is False.
        overwrite (Optional[bool]): Whether to overwrite `out_vrt`, if it exists. Default is False.
        relative_path (Optional[bool]): Whether to use relative path names. Default is False.

    Examples:
        >>> # Stack image bands from a dictionary.
        >>> vrt_builder({'1': ['image1.tif'],
        >>>             '2': ['image2.tif'],
        >>>             '3': ['image3.tif']},
        >>>             '/out_vrt.vrt')
        >>>
        >>> # Stack GeoTiffs from a directory.
        >>> vrt_builder('/image_dir',
        >>>             '/out_vrt.vrt',
        >>>             separate=True,
        >>>             search_extension='.tif')
        >>>
        >>> # Mosaic GeoTiffs from a list.
        >>> vrt_builder(['/image1.tif', '/image2.tif'],
        >>>             '/out_vrt.vrt',
        >>>             separate=False)
        >>>
        >>> # Stack multiple sources and subset.
        >>> vrt_builder({'1': ['image1.tif', 'imag2.tif'],
        >>>             '2': ['srtm1.tif', 'srtm2.tif', 'srtm3.tif']},
        >>>             '/out_vrt.vrt',
        >>>             bands2include=[1, 3, 4],
        >>>             force_type='float32',
        >>>             subset=True)
    """

    # VRT builder class
    vb = VRTBuilder()

    # Prepare the VRT XML strings.
    vb.get_xml_base()

    # Get the image extent information.
    vb.get_full_extent(in_dict, separate, search_extension, use_image_min=use_image_min)

    # Update the main part of the XML string.
    vb.replace_main(no_data=no_data)

    # Add the data to the XML string.
    vb.add_bands(bands2include=bands2include,
                 start_band=start_band,
                 force_type=force_type,
                 subset=subset,
                 base_name=base_name,
                 be_quiet=be_quiet,
                 relative_path=relative_path,
                 out_vrt=out_vrt)

    d_name = os.path.split(out_vrt)[0]

    if not os.path.isdir(d_name):
        os.makedirs(d_name)

    if overwrite:

        if os.path.isfile(out_vrt):
            os.remove(out_vrt)

    # Write the VRT to file.
    with open(out_vrt, 'w') as xml_writer:
        xml_writer.writelines(vb.xml_base)


def _examples():

    sys.exit("""\

    vrt_builder.py -i /image1.tif /image2.tif -o output.vrt -f float32

    """)


def main():

    parser = argparse.ArgumentParser(description='Build a VRT file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--examples', dest='examples', action='store_true', help='Show usage examples and exit')
    parser.add_argument('-i', '--inputs', dest='inputs', help='The input image list', default=[], nargs='+')
    parser.add_argument('-o', '--output', dest='output', help='The output VRT image', default=None)
    parser.add_argument('-b', '--bands2include', dest='bands2include', help='A list of bands to include',
                        default=[], type=int, nargs='+')
    parser.add_argument('-f', '--force-type', dest='force_type', help='Force the output storage type', default=None)
    parser.add_argument('-s', '--subset', dest='subset', help='Whether to subset the dictionary', action='store_true')
    parser.add_argument('-n', '--base-name', dest='base_name', help='A base name to prepend to the /subs directory',
                        default=None)

    args = parser.parse_args()

    if args.examples:
        _examples()

    logger.info('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    start_time = time.time()

    vrt_builder({'00001': args.inputs}, args.output,
                bands2include=args.bands2include, force_type=args.force_type,
                subset=args.subset, base_name=args.base_name)

    logger.info('\nEnd data & time -- (%s)\nTotal processing time -- (%.2gs)\n' %
                (time.asctime(time.localtime(time.time())), (time.time()-start_time)))


if __name__ == '__main__':
    main()
