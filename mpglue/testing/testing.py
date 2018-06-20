#!/usr/bin/env python

import unittest

from mpglue import raster_tools
from mpglue.data import landsat_gtiff, landsat_vrt

import numpy as np


def _test_array(image, dtype='float64'):

    with raster_tools.ropen(image) as l_info:

        image_array = l_info.read(bands2open=-1,
                                  d_type=dtype)

    l_info = None

    return image_array


def _test_object(image):

    with raster_tools.ropen(image) as l_info:

        bands = l_info.bands
        rows = l_info.rows
        cols = l_info.cols

    l_info = None

    return bands, rows, cols


class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def test_isarray_gtiff(self):
        """Test the array read"""
        self.assertIsInstance(_test_array(landsat_gtiff), np.ndarray)

    def test_isarray_vrt(self):
        """Test the array read"""
        self.assertIsInstance(_test_array(landsat_vrt), np.ndarray)

    def test_nbands_gtiff_array(self):
        """Test the array read"""
        self.assertEqual(_test_array(landsat_gtiff).shape[0], 6)

    def test_nbands_vrt_array(self):
        """Test the array read"""
        self.assertEqual(_test_array(landsat_vrt).shape[0], 2)

    def test_nrows_gtiff_array(self):
        """Test the array read row size"""
        self.assertEqual(_test_array(landsat_gtiff).shape[1], 224)

    def test_nrows_vrt_array(self):
        """Test the array read row size"""
        self.assertEqual(_test_array(landsat_vrt).shape[1], 224)

    def test_ncols_gtiff_array(self):
        """Test the array read column size"""
        self.assertEqual(_test_array(landsat_gtiff).shape[2], 235)

    def test_ncols_vrt_array(self):
        """Test the array read column size"""
        self.assertEqual(_test_array(landsat_vrt).shape[2], 235)

    def test_dtype_float64_gtiff(self):
        """Test the array read data type"""
        self.assertEqual(_test_array(landsat_gtiff, dtype='float64').dtype, 'float64')

    def test_dtype_float64_vrt(self):
        """Test the array read data type"""
        self.assertEqual(_test_array(landsat_vrt, dtype='float64').dtype, 'float64')

    def test_dtype_byte_gtiff(self):
        """Test the array read data type"""
        self.assertEqual(_test_array(landsat_gtiff, dtype='byte').dtype, 'uint8')

    def test_dtype_byte_vrt(self):
        """Test the array read data type"""
        self.assertEqual(_test_array(landsat_vrt, dtype='byte').dtype, 'uint8')

    def test_nbands_gtiff_object(self):
        """Test the object band count"""
        self.assertEqual(_test_object(landsat_gtiff)[0], 6)

    def test_nbands_vrt_object(self):
        """Test the object band count"""
        self.assertEqual(_test_object(landsat_vrt)[0], 2)

    def test_nrows_gtiff_object(self):
        """Test the object row size"""
        self.assertEqual(_test_object(landsat_gtiff)[1], 224)

    def test_nrows_vrt_object(self):
        """Test the object row size"""
        self.assertEqual(_test_object(landsat_vrt)[1], 224)

    def test_ncols_gtiff_object(self):
        """Test the object column size"""
        self.assertEqual(_test_object(landsat_gtiff)[2], 235)

    def test_ncols_vrt_object(self):
        """Test the object column size"""
        self.assertEqual(_test_object(landsat_vrt)[2], 235)


if __name__ == '__main__':
    unittest.main()
