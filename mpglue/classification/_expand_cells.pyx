#!/usr/bin/env python

import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE_uint8 = np.uint8
ctypedef np.uint8_t DTYPE_uint8_t

# Pymorph
try:
    from pymorph import sedisk
except ImportError:
    raise ImportError('Pymorph must be installed')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_uint8_t[:, :] _fill_block(DTYPE_uint8_t[:, :] image_block,
                                     DTYPE_uint8_t[:, :] kernel_block,
                                     int rs, int cs):

    cdef:
        Py_ssize_t ii, jj

    for ii in xrange(0, rs):

        for jj in xrange(0, rs):

            if image_block[ii, jj] == 0:
                image_block[ii, jj] = kernel_block[ii, jj]

    return image_block


@cython.boundscheck(False)   
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_uint8_t, ndim=2] expand(DTYPE_uint8_t[:, :] image_array,
                                              unsigned int window_size,
                                              int rows, int cols, str mode,
                                              int iters):

    """
    Args:
        image_array (ndarray)
        window_size (Optional[int]):
        mode (Optional[str]): 'square' or 'circle'

    Returns:
        ndarray
    """

    cdef:
        Py_ssize_t i, j, iter
        unsigned int half_window = int(window_size / 2)
        DTYPE_uint8_t[:, :] kernel
        DTYPE_uint8_t[:, :] out_array = image_array.copy()
        int row_dims = rows - (half_window*2)
        int col_dims = cols - (half_window*2)

    if mode == 'square':
        kernel = np.ones((window_size, window_size), dtype='uint8')
    elif mode == 'circle':

        if window_size == 3:

            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype='uint8')

        else:
            kernel = np.uint8(sedisk(half_window) * 1)

    for iter in xrange(0, iters):

        for i in xrange(0, row_dims):

            for j in xrange(0, col_dims):

                if image_array[i+half_window, j+half_window] == 1:

                    out_array[i:i+window_size, j:j+window_size] = _fill_block(out_array[i:i+window_size,
                                                                                        j:j+window_size],
                                                                              kernel, window_size, window_size)

        image_array = out_array.copy()

    return np.uint8(out_array)


@cython.boundscheck(False)
@cython.wraparound(False)
def expand_cells(np.ndarray image_array, unsigned int window_size=3, str mode='square', int iters=1):

    """
    Args:
        image_array (2d array): The image array to process.
        window_size (Optional[int]): The window kernel size to use. Default is 3.
        mode (Optional[str]): The expand mode to use. Default is 'square'. Choices are ['square', 'circle'].
    """

    cdef:
        int rows = image_array.shape[0]
        int cols = image_array.shape[1]

    if mode not in ['square', 'circle']:
        raise ValueError('The expand mode was not recognized.')

    return expand(np.uint8(np.ascontiguousarray(image_array)), window_size, rows, cols, mode, iters)