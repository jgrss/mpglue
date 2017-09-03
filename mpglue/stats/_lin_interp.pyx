# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

"""
@author: Jordan Graesser
"""

from __future__ import division

import cython
cimport cython

# from libc.math cimport isnan
# from cython.parallel import prange, parallel

import numpy as np
cimport numpy as np

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t

DTYPE_long = np.uint64
ctypedef np.uint64_t DTYPE_long_t

# cdef extern from 'numpy/npy_math.h':
#     bint npy_isnan(DTYPE_float32_t x)


cdef inline DTYPE_float32_t _interp(int x1, int x2, int x3, DTYPE_float32_t y1, DTYPE_float32_t y3) nogil:
    return ((float((x2 - x1)) * (y3 - y1)) / float((x3 - x1))) + y1


cdef DTYPE_float32_t _get_max(DTYPE_float32_t[:] in_row, int cols) nogil:

    cdef:
        Py_ssize_t a
        DTYPE_float32_t m = in_row[0]

    for a in range(1, cols):

        if in_row[a] > m:

            m = in_row[a]

    return m


cdef int _get_x1(int x2, DTYPE_float32_t[:] in_row, DTYPE_float32_t value2fill) nogil:

    cdef:
        Py_ssize_t x1_iter
        int x1 = -9999

    for x1_iter in range(1, x2+1):

        if in_row[x2-x1_iter] != value2fill:

            x1 = x2 - x1_iter

            break

    return x1


cdef int _get_x3(int x2, DTYPE_float32_t[:] in_row, int dims, DTYPE_float32_t value2fill) nogil:

    cdef:
        Py_ssize_t x3_iter
        int x3 = -9999
        int x3_range = dims - x2

    for x3_iter in range(1, x3_range):

        if in_row[x2+x3_iter] != value2fill:

            x3 = x2 + x3_iter

            break

    return x3


cdef tuple _find_indices(DTYPE_float32_t[:] in_row, int dims, DTYPE_float32_t value2fill):

    cdef:
        Py_ssize_t fi
        Py_ssize_t counter = 0
        DTYPE_float32_t[:] index_positions = in_row.copy()

    with nogil:

        for fi in range(0, dims):

            if in_row[fi] == value2fill:

                index_positions[counter] = fi

                counter += 1

    return index_positions, counter


cdef void _fill_ends(DTYPE_float32_t[:] in_block,
                     int dims,
                     DTYPE_float32_t value2fill) nogil:

    cdef:
        Py_ssize_t ib, ic
        DTYPE_float32_t valid_data

    if in_block[0] == value2fill:

        # Find the first valid data point
        for ib in range(1, dims):

            if in_block[ib] != value2fill:
                valid_data = in_block[ib]
                break

        # Fill the ends up to the valid data point.
        for ic in range(0, ib):
            in_block[ic] = valid_data

    if in_block[dims-1] == value2fill:

        # Find the first non-zero data
        for ib in range(dims-2, 0, -1):

            if in_block[ib] != value2fill:
                valid_data = in_block[ib]
                break

        # Fill the ends up to the valid data point.
        for ic in range(ib+1, dims):
            in_block[ic] = valid_data


cdef void _interp_loop_1d(DTYPE_float32_t[:] in_row,
                          int dims,
                          DTYPE_float32_t value2fill):

    cdef:
        Py_ssize_t x2_idx
        Py_ssize_t len_idx
        DTYPE_float32_t[:] idx
        int x1, x2, x3

    idx, len_idx = _find_indices(in_row, dims, value2fill)

    if len_idx != 0:

        with nogil:

            # check for data
            if _get_max(in_row, dims) > 0:

                for x2_idx in range(0, len_idx):

                    x2 = <int>(idx[x2_idx])

                    # get x1
                    x1 = _get_x1(x2, in_row, value2fill)

                    # get x3
                    x3 = _get_x3(x2, in_row, dims, value2fill)

                    if (x1 != -9999) and (x3 != -9999):
                        in_row[x2] = _interp(x1, x2, x3, in_row[x1], in_row[x3])

        _fill_ends(in_row, dims, value2fill)


cdef np.ndarray[DTYPE_float32_t, ndim=2, mode='c'] _interp_loop(DTYPE_float32_t[:, :] in_block,
                                                                int rows, 
                                                                int dims,
                                                                DTYPE_float32_t value2fill):

    cdef:
        Py_ssize_t i    #, x2_idx
        #Py_ssize_t len_idx
        DTYPE_float32_t[:] in_row
        #DTYPE_float32_t[:] idx
        #int x1, x2, x3

    for i in range(0, rows):

        # get the current row
        in_row = in_block[i, :]

        # idx, len_idx = _find_indices(in_row, dims, value2fill)
        #
        # if len_idx == 0:
        #     continue

        _interp_loop_1d(in_row,
                        dims,
                        value2fill)

        # check for data
        # if _get_max(in_row, dims) > 0:

            # for x2_idx in range(0, len_idx):
            # 
            #     x2 = int(idx[x2_idx])
            # 
            #     # get x1
            #     x1 = _get_x1(x2, in_row, value2fill)
            # 
            #     # get x3
            #     x3 = _get_x3(x2, in_row, dims, value2fill)
            # 
            #     if (x1 != -9999) and (x3 != -9999):
            #         in_row[x2] = _interp(x1, x2, x3, in_row[x1], in_row[x3])

        # in_block[i] = _fill_ends(in_row, dims, value2fill)
        in_block[i] = in_row

    return np.asarray(in_block)


def lin_interp(np.ndarray input_array, DTYPE_float32_t value2fill):

    """
    Linearly interpolates between 'no data' points

    Args:
        nd_array (ndarray): The 2d array to interpolate. The expected dimensions are
            [(rows x columns) x dimensions]. Zeros are considered 'no data'.

    Returns:
        Interpolated version of ``nd_array``.
    """

    cdef:
        int rows = input_array.shape[0]
        int dims = input_array.shape[1]

    return _interp_loop(np.float32(np.ascontiguousarray(input_array)), rows, dims, value2fill)


def lin_interp_1d(np.ndarray input_array, float value2fill):

    """
    Linearly interpolates between 'no data' points

    Args:
        arr (ndarray): The 1d array to interpolate. The expected dimensions are
            (row x dimensions). Zeros are considered 'no data'.

    Returns:
        Interpolated version of ``arr``.
    """

    cdef:
        int dims = int(len(input_array))
        DTYPE_float32_t[:] in1d = np.float32(input_array)

    _interp_loop_1d(in1d, dims, value2fill)

    return np.float32(in1d)
