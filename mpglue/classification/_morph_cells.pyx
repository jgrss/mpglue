# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cython
cimport cython

import numpy as np
cimport numpy as np

DTYPE_uint8 = np.uint8
ctypedef np.uint8_t DTYPE_uint8_t

DTYPE_intp = np.intp
ctypedef np.intp_t DTYPE_intp_t

# Pymorph
try:
    from pymorph import sedisk
except:
    pass


# Define a function pointer to a metric.
ctypedef DTYPE_uint8_t[:, :] (*metric_ptr)(DTYPE_uint8_t[:, :], DTYPE_uint8_t[:, :], DTYPE_intp_t) nogil


cdef DTYPE_uint8_t[:, :] _expand_block(DTYPE_uint8_t[:, :] image_block,
                                       DTYPE_uint8_t[:, :] kernel_block,
                                       int ws) nogil:

    cdef:
        Py_ssize_t ii, jj

    for ii in range(0, ws):
        for jj in range(0, ws):

            if (image_block[ii, jj] == 0) and (kernel_block[ii, jj] == 1):
                image_block[ii, jj] = 1

    return image_block


cdef DTYPE_uint8_t[:, :] _shrink_block(DTYPE_uint8_t[:, :] image_block,
                                       DTYPE_uint8_t[:, :] kernel_block,
                                       int ws) nogil:

    cdef:
        Py_ssize_t ii, jj
        int block_sum = 0

    for ii in range(0, ws):
        for jj in range(0, ws):
            if image_block[ii, jj] == 1:
                block_sum += 1

    if block_sum < ws*2:

        for ii in range(0, ws):
            for jj in range(0, ws):

                if (image_block[ii, jj] == 1) and (kernel_block[ii, jj] == 1):
                    image_block[ii, jj] = 0

    return image_block


cdef np.ndarray[DTYPE_uint8_t, ndim=2] _morph_cells(DTYPE_uint8_t[:, :] image_array,
                                                    int window_size,
                                                    int rows,
                                                    int cols,
                                                    str mode,
                                                    str se,
                                                    int iters):

    """
    Returns:
        2d array
    """

    cdef:
        Py_ssize_t i, j, iteration
        unsigned int half_window = int(window_size / 2)
        DTYPE_uint8_t[:, :] kernel
        DTYPE_uint8_t[:, :] out_array = image_array.copy()
        int row_dims = rows - (half_window*2)
        int col_dims = cols - (half_window*2)
        metric_ptr function

    if mode == 'expand':
        function = &_expand_block
    else:
        function = &_shrink_block

    if se == 'square':
        kernel = np.ones((window_size, window_size), dtype='uint8')
    elif se == 'circle':

        if window_size == 3:

            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype='uint8')

        else:
            kernel = np.uint8(sedisk(half_window) * 1)

    for iteration in range(0, iters):

        for i in range(0, row_dims):

            for j in range(0, col_dims):

                if image_array[i+half_window, j+half_window] == 1:

                    out_array[i:i+window_size, j:j+window_size] = function(out_array[i:i+window_size,
                                                                                     j:j+window_size],
                                                                           kernel,
                                                                           window_size)

        if iters > 1:
            image_array = out_array.copy()

    return np.uint8(out_array)


def morph_cells(np.ndarray image_array, int window_size=3, str mode='expand', str se='square', int iters=1):

    """
    Args:
        image_array (2d array): The image array to process.
        window_size (Optional[int]): The window kernel size to use. Default is 3.
        mode (Optional[str]): The cell mode to use. Choices are ['expand', 'shrink']. Default is 'expand'.
        se (Optional[str]): The expand mode to use. Choices are ['square', 'circle']. Default is 'square'.
        iters (Optional[int]): The number of cell iterations. Default is 1.
    """

    cdef:
        int rows = image_array.shape[0]
        int cols = image_array.shape[1]

    if mode not in ['expand', 'shrink']:
        raise ValueError('The cell mode was not recognized.')

    if se not in ['square', 'circle']:
        raise ValueError('The structuring element was not recognized.')

    return _morph_cells(np.uint8(np.ascontiguousarray(image_array)), window_size, rows, cols, mode, se, iters)
