#!/usr/bin/env python

import sys
import unittest

import mpglue as gl
from mpglue.data import landsat_gtiff, landsat_vrt

import numpy as np


def _test_band_count(image):

    with gl.raster_tools.ropen(image) as l_info:
        n_bands = l_info.bands

    l_info = None

    return n_bands


def _test_array(image):

    with gl.raster_tools.ropen(image) as l_info:

        image_array = l_info.read(bands2open=[1, 2, 3],
                                  d_type='float64')

    l_info = None

    return image_array


class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def test_nbands_gtiff(self):
        """Test the image band count"""
        self.assertEquals(_test_band_count(landsat_gtiff), 6)

    def test_nbands_vrt(self):
        """Test the image band count"""
        self.assertEquals(_test_band_count(landsat_vrt), 2)

    def test_isarray_gtiff(self):
        """Test the array read"""
        self.assertIsInstance(_test_array(landsat_gtiff), np.ndarray)

    def test_isarray_vrt(self):
        """Test the array read"""
        self.assertIsInstance(_test_array(landsat_vrt), np.ndarray)

    def test_nbands_gtiff_array(self):
        """Test the array read"""
        self.assertEquals(_test_array(landsat_gtiff).shape[0], 3)

    def test_nbands_vrt_array(self):
        """Test the array read"""
        self.assertEquals(_test_array(landsat_vrt).shape[0], 3)

    def test_dtype_gtiff(self):
        """Test the array read"""
        self.assertEquals(_test_array(landsat_gtiff).dtype, 'float64')

    def test_dtype_vrt(self):
        """Test the array read"""
        self.assertEquals(_test_array(landsat_vrt).dtype, 'float64')


if __name__ == '__main__':
    unittest.main()
