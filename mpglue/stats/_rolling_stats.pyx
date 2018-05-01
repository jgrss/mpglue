# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

"""
@author: Jordan Graesser
"""

import cython
cimport cython

import numpy as np
cimport numpy as np

from cython.parallel import prange
from libc.math cimport pow

DTYPE_uint64 = np.uint64
ctypedef np.uint64_t DTYPE_uint64_t

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t

cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(DTYPE_float32_t x) nogil


cdef extern from 'numpy/npy_math.h':
    bint npy_isinf(DTYPE_float32_t x) nogil


# cdef inline DTYPE_float32_t get_mean(np.ndarray[DTYPE_float32_t, ndim=1] row_arr, int window_size):
#
#     cdef int v
#     cdef DTYPE_float32_t the_sum = 0.
#
#     for v in range(0, window_size):
#
#         the_sum += row_arr[v]
#
#     return the_sum / float(window_size)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef int _get_max_val(np.ndarray[DTYPE_float32_t, ndim=1] arr, int n_cols):
#
#     cdef:
#         int cl, mx_idx
#         DTYPE_float32_t mx = 0.
#
#     for cl in range(0, n_cols):
#
#         if arr[cl] > mx:
#
#             mx = arr[cl]
#             mx_idx = cl
#
#     return mx_idx
#
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef int _get_min_val(np.ndarray[DTYPE_float32_t, ndim=1] arr, int n_cols):
#
#     cdef:
#         int cl, mn_idx
#         DTYPE_float32_t mn = 10000.
#
#     for cl in range(0, n_cols):
#
#         if arr[cl] < mn:
#
#             mn = arr[cl]
#             mn_idx = cl
#
#     return mn_idx


cdef DTYPE_float32_t _get_weighted_mean(DTYPE_float32_t[:] array_1d,
                                        int array_length,
                                        DTYPE_float32_t[:] weights,
                                        DTYPE_float32_t weights_sum,
                                        DTYPE_float32_t ignore_value) nogil:

    cdef:
        Py_ssize_t jj
        DTYPE_float32_t array_sum = (array_1d[0] * weights[0])

    for jj in range(1, array_length):

        if array_1d[jj] == ignore_value:
            weights_sum -= weights[jj]
        else:
            array_sum += (array_1d[jj] * weights[jj])

    return array_sum / weights_sum


cdef DTYPE_float32_t _get_mean(DTYPE_float32_t[:] array_1d,
                               DTYPE_float32_t ignore_value) nogil:

    cdef:
        Py_ssize_t jj
        int array_length = 0
        DTYPE_float32_t array_sum = array_1d[0]

    for jj in range(1, array_length):

        if array_1d[jj] != ignore_value:

            array_sum += array_1d[jj]
            array_length += 1

    return array_sum / array_length


cdef DTYPE_float32_t _get_sum(DTYPE_float32_t[:] array_1d, int array_length) nogil:

    cdef:
        Py_ssize_t jj
        DTYPE_float32_t array_sum = array_1d[0]

    for jj in range(1, array_length):
        array_sum += array_1d[jj]

    return array_sum


cdef void _get_min(DTYPE_float32_t[:] val1, DTYPE_float32_t[:] val2, int cols) nogil:

    cdef:
        Py_ssize_t c

    for c in range(0, cols):

        if val2[c] < val1[c]:
            val1[c] = val2[c]


cdef DTYPE_float32_t _get_max_value(DTYPE_float32_t[:] array1d, int cols) nogil:

    cdef:
        Py_ssize_t c
        DTYPE_float32_t max_value = array1d[0]

    for c in range(1, cols):

        if array1d[c] > max_value:
            max_value = array1d[c]

    return max_value


cdef void _get_max(DTYPE_float32_t[:] val1, DTYPE_float32_t[:] val2, int cols) nogil:

    cdef:
        Py_ssize_t c

    for c in range(0, cols):

        if val2[c] > val1[c]:
            val1[c] = val2[c]


cdef void _least_squares(DTYPE_float32_t[:] x,
                         DTYPE_float32_t[:, :] y,
                         int n,
                         int cols,
                         DTYPE_float32_t[:] y_avg,
                         DTYPE_float32_t[:] cov_xy_) nogil:

    cdef:
        DTYPE_float32_t temp
        Py_ssize_t ii, jj
        DTYPE_float32_t x_avg = 0.
        DTYPE_float32_t var_x = 0.

    for ii in range(0, n):

        x_avg += x[ii]

        # for jj in prange(0, cols, nogil=True, num_threads=cols, schedule='static'):
        for jj in range(0, cols):
            y_avg[jj] += y[ii, jj]

    x_avg /= n

    # for jj in prange(0, cols, nogil=True, num_threads=cols, schedule='static'):
    for jj in range(0, cols):
        y_avg[jj] /= n

    for ii in range(0, n):

        temp = x[ii] - x_avg
        var_x += pow(temp, 2.)

        # for jj in prange(0, cols, nogil=True, num_threads=cols, schedule='static'):
        for jj in range(0, cols):
            cov_xy_[jj] += temp * (y[ii, jj] - y_avg[jj])

    # for jj in prange(0, cols, nogil=True, num_threads=cols, schedule='static'):
    for jj in range(0, cols):
        cov_xy_[jj] /= var_x


# cdef class Vector(object):
#
#     cdef float *data
#     cdef public int n_ax0
#
#     def __init__(Vector self, int n_ax0):
#
#         self.data = <float*> malloc (sizeof(float) * n_ax0)
#         self.n_ax0 = n_ax0
#
#     def __dealloc__(Vector self):
#         free(self.data)


cdef void _replace_nans(DTYPE_float32_t[:] nan_array, int replace_value, int cols) nogil:

    cdef:
        Py_ssize_t j
        DTYPE_float32_t nan_value

    # for j in prange(0, cols, nogil=True, num_threads=cols, schedule='static'):
    for j in range(0, cols):

        nan_value = nan_array[j]

        if npy_isnan(nan_value) or npy_isinf(nan_value):
            nan_array[j] = 0.


cdef tuple _rolling_least_squares(DTYPE_float32_t[:, :] image_array, int window_size):

    cdef:
        Py_ssize_t w
        unsigned int rows = image_array.shape[0]
        unsigned int cols = image_array.shape[1]
        # np.ndarray[DTYPE_float32_t, ndim=1, mode='c'] X = np.arange(window_size, 'float32')
        # np.ndarray[DTYPE_float32_t, ndim=1, mode='c'] slopes_max = np.zeros(cols, dtype='float32')
        # np.ndarray[DTYPE_float32_t, ndim=1, mode='c'] slopes_min = np.multiply(np.ones(cols, dtype='float32'), 1000)
        # np.ndarray[DTYPE_float32_t, ndim=1, mode='c'] slopes

        # DTYPE_float32_t[:] X = cython.view.array(shape=(10, 2), itemsize=sizeof(int), format="i")
        # cdef Vector X = Vector(cols)

        DTYPE_float32_t[:] X = np.arange(window_size, dtype='float32')
        DTYPE_float32_t[:] slopes_max = np.ones(cols, dtype='float32') * -100000.
        DTYPE_float32_t[:] slopes_min = np.ones(cols, dtype='float32') * 100000.
        DTYPE_float32_t[:] slopes
        DTYPE_float32_t[:] y_avg = np.zeros(cols, dtype='float32')
        DTYPE_float32_t[:] y_avgc
        DTYPE_float32_t[:] cov_xy = np.zeros(cols, dtype='float32')

    with nogil:

        for w in range(0, rows-window_size):

            slopes = cov_xy[:]
            y_avgc = y_avg[:]

            _least_squares(X, image_array[w:w+window_size, :], window_size, cols, y_avgc, slopes)

            _get_min(slopes_min, slopes, cols)
            _get_max(slopes_max, slopes, cols)

    _replace_nans(slopes_min, 0, cols)
    _replace_nans(slopes_max, 0, cols)

    # slopes_min[np.isnan(slopes_min) | np.isinf(slopes_min)] = 0
    # slopes_max[np.isnan(slopes_max) | np.isinf(slopes_max)] = 0

    return np.float32(slopes_min), np.float32(slopes_max)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef DTYPE_float32_t get_median(np.ndarray[DTYPE_float32_t, ndim=1] the_row, DTYPE_float32_t mxv, DTYPE_float32_t mnv, int window_size):
#
#     cdef:
#         unsigned int v
#         DTYPE_float32_t med_val
#
#     for v in range(0, window_size):
#
#         if (v != mxv) and (v != mnv):
#
#             med_val = the_row[v]
#
#             break
#
#     return med_val


cdef DTYPE_float32_t[:, :] _rolling_median(DTYPE_float32_t[:, :] arr, int window_size):

    cdef:
        Py_ssize_t i, j
        unsigned int rows = arr.shape[0]
        unsigned int cols = arr.shape[1]
        unsigned int window_half1 = window_size / 2
        unsigned int window_half2 = (window_size / 2) + 1
        DTYPE_float32_t[:] the_row
        DTYPE_float32_t[:, :] results = np.zeros((rows, cols), dtype='float32')

    ## iterate over the array by rows
    for i in range(0, rows):

        the_row = arr[i]

        for j in range(0, cols-window_half1):

            the_row[j+window_size-window_half2] = sorted(the_row[j:j+window_size])[window_half1]

        results[i] = the_row

    return results


cdef np.ndarray[DTYPE_float32_t, ndim=3, mode='c'] _rolling_window(np.ndarray[DTYPE_float32_t, ndim=2, mode='c'] a,
                                                                   int window):

    cdef:
        tuple shape, strides

    shape = (<object> a).shape[:1] + ((<object> a).shape[1] - window + 1, window)
    strides = (<object> a).strides + ((<object> a).strides[1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


cdef np.ndarray[DTYPE_float32_t, ndim=2, mode='c'] _rolling_median_v(np.ndarray[DTYPE_float32_t, ndim=2, mode='c'] arr, int window_size):

    cdef:
        int cols = arr.shape[1]
        int window_half1 = window_size / 2
        np.ndarray[DTYPE_float32_t, ndim=3] arr_strides, arr_strides_sorted
        np.ndarray[DTYPE_float32_t, ndim=2] results

    # get window strides
    arr_strides = _rolling_window(arr, window_size)

    # sort on the last axis
    arr_strides_sorted = np.sort(arr_strides)

    # get the center value
    results = np.sort(arr_strides_sorted)[:, :, window_half1]

    # put back into original array
    arr[:, 1:cols-window_half1] = results

    return arr


cdef void _convolve1d(DTYPE_float32_t[:] array2process,
                      DTYPE_float32_t[:] array2write,
                      int cols,
                      int window_size,
                      int window_half,
                      DTYPE_float32_t[:] weights,
                      bint update_weights,
                      int iterations) nogil:

    cdef:
        Py_ssize_t j, n_iter, ws

    for n_iter in range(1, iterations+1):

        for j in range(0, cols-window_size):

            for ws in range(0, window_size):
                array2write[j+window_half] += array2process[j+ws] * weights[ws]

        if update_weights:

            weights[0] = 1. / (n_iter + 2.)
            weights[1] = n_iter / (n_iter + 2.)
            weights[3] = 1. / (n_iter + 2.)


cdef void _rolling_mean1d(DTYPE_float32_t[:] array2process,
                          int cols,
                          int window_size,
                          int window_half,
                          bint do_weights,
                          DTYPE_float32_t[:] weights,
                          DTYPE_float32_t weights_sum,
                          DTYPE_float32_t ignore_value,
                          bint apply_under,
                          DTYPE_float32_t apply_under_value,
                          bint apply_over,
                          DTYPE_float32_t apply_over_value,
                          int iterations) nogil:

    cdef:
        Py_ssize_t j, n_iter

    for n_iter in range(1, iterations+1):

        # Get the rolling mean.
        for j in range(0, cols-window_size):

            if apply_under:

                if array2process[j+window_half] < ignore_value + .01:
                    continue

                if array2process[j+window_half] > apply_under_value:
                    continue

                if _get_max_value(array2process[j:j+window_size], window_size) > apply_under_value:
                    continue

            if apply_over:

                if array2process[j+window_half] < apply_over_value:
                    continue

            if do_weights:

                array2process[j+window_half] = _get_weighted_mean(array2process[j:j+window_size],
                                                                  window_size,
                                                                  weights,
                                                                  weights_sum,
                                                                  ignore_value)

            else:

                array2process[j+window_half] = _get_mean(array2process[j:j+window_size],
                                                         ignore_value)


cdef DTYPE_float32_t[:, :] _rolling_stats(DTYPE_float32_t[:, :] arr,
                                          int window_size,
                                          int window_half,
                                          DTYPE_float32_t[:] weights,
                                          bint do_weights,
                                          bint update_weights,
                                          DTYPE_float32_t weights_sum,
                                          DTYPE_float32_t ignore_value,
                                          bint apply_under,
                                          DTYPE_float32_t apply_under_value,
                                          bint apply_over,
                                          DTYPE_float32_t apply_over_value,
                                          int iterations,
                                          str stat):

    cdef:
        Py_ssize_t i
        unsigned int rows = arr.shape[0]
        unsigned int cols = arr.shape[1]
        DTYPE_float32_t[:] the_row, out_row
        DTYPE_float32_t[:, :] results = np.zeros((rows, cols), dtype='float32')
        DTYPE_float32_t[:] zzzzeros = np.zeros(cols, dtype='float32')

    # Iterate over the array by rows.
    for i in range(0, rows):

        # Get the current row
        the_row = arr[i, :]

        if stat == 'convolve':

            out_row = zzzzeros.copy()

            with nogil:

                _convolve1d(the_row,
                            out_row,
                            cols,
                            window_size,
                            window_half,
                            weights,
                            update_weights,
                            iterations)

            results[i, :] = out_row[:]

        elif stat == 'mean':

            with nogil:

                _rolling_mean1d(the_row,
                                cols,
                                window_size,
                                window_half,
                                do_weights,
                                weights,
                                weights_sum,
                                ignore_value,
                                apply_under,
                                apply_under_value,
                                apply_over,
                                apply_over_value,
                                iterations)

            results[i, :] = the_row[:]

    return results


def rolling_stats(np.ndarray image_array,
                  str stat='mean',
                  int window_size=3,
                  window_weights=None,
                  bint update_weights=False,
                  bint is_1d=False,
                  int cols=0,
                  float ignore_value=-999.,
                  bint apply_under=False,
                  float apply_under_value=-999.,
                  bint apply_over=False,
                  float apply_over_value=-999.,
                  int iterations=1):

    """
    Computes rolling statistics
    
    Args:
        image_array (ndarray): The expected shape is (image dimensions x samples), or (image dimensions x 
            (rows x columns)). To reshape the array to expected format, follow:
            If image shape = (dimensions, rows, columns):
                image.reshape(dimensions, rows*columns)
                image.T.reshape(rows*columns, dimensions) -->
                    image.reshape(columns, rows, dimensions).T.reshape(dimensions, rows*columns)
        stat (Optional[str]): The statistic to compute. Default is 'median'.
            Choices are ['mean', 'median', 'slope', 'convolve'].
        window_size (Optional[int]): The window size. Default is 3.
        window_weights (Optiona[1d array])
        update_weights (Optional[bool])
        is_1d (Optional[bool])
        cols (Optional[int])
        ignore_value (Optional[float])
        apply_under (Optional[bool])
        apply_under_value (Optional[float])
        apply_over (Optional[bool])
        apply_over_value (Optional[float])
        iterations (Optional[int])
    """

    cdef:
        DTYPE_float32_t[:] weights
        bint do_weights
        DTYPE_float32_t weights_sum
        cdef DTYPE_float32_t[:] array2process
        int window_half = int(window_size / 2.)

    if isinstance(window_weights, np.ndarray):

        weights = np.float32(window_weights)
        weights_sum = _get_sum(weights, window_size)
        do_weights = True

    else:

        weights = np.zeros(1, dtype='float32')
        weights_sum = 0.
        do_weights = False

    if cols == 0:
        cols = image_array.shape[1]

    if is_1d:

        _rolling_mean1d(image_array,
                        cols,
                        window_size,
                        window_half,
                        do_weights,
                        weights,
                        weights_sum,
                        ignore_value,
                        apply_under,
                        apply_under_value,
                        apply_over,
                        apply_over_value,
                        iterations)

        return np.float32(image_array)

    else:

        if stat in ['convolve', 'mean']:

            return np.float32(_rolling_stats(np.float32(image_array),
                                             window_size,
                                             window_half,
                                             weights,
                                             do_weights,
                                             update_weights,
                                             weights_sum,
                                             ignore_value,
                                             apply_under,
                                             apply_under_value,
                                             apply_over,
                                             apply_over_value,
                                             iterations,
                                             stat))

        elif stat == 'median':

            return _rolling_median(np.float32(image_array), window_size)

        elif stat == 'slope':

            return _rolling_least_squares(np.float32(image_array), window_size)
