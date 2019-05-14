from ..errors import logger
from ..stats._rolling_stats import rolling_stats

import numpy as np


def _nan_reshape(Xd, no_data):

    Xd[np.isnan(Xd) | np.isinf(Xd)] = no_data

    return Xd[:, np.newaxis]


def _mean(X, axis=1, no_data=0):

    """
    Computes the mean
    """

    X_ = X.mean(axis=axis)

    return _nan_reshape(X_, no_data)


def _cv(X, axis=1, no_data=0):

    """
    Computes the coefficient of variation
    """

    X_ = X.std(axis=axis) / X.mean(axis=axis)

    return _nan_reshape(X_, no_data)


def _five_cumulative(X, axis=1, no_data=0):

    """
    Computes the value at the cumulative 5th percentile
    """

    x_range = list(range(0, X.shape[1]+1))

    # Cumulative sum
    X_ = X.cumsum(axis=axis)

    # Position at the nth percentile
    pct_idx = int(np.ceil(np.percentile(x_range, 5)))

    # Values at the nth percentile
    X_ = X_[:, pct_idx]

    return _nan_reshape(X_, no_data)


def _twenty_five_cumulative(X, axis=1, no_data=0):

    """
    Computes the value at the cumulative 25th percentile
    """

    x_range = list(range(0, X.shape[1]+1))

    # Cumulative sum
    X_ = X.cumsum(axis=axis)

    # Position at the nth percentile
    pct_idx = int(np.ceil(np.percentile(x_range, 25)))

    # Values at the nth percentile
    X_ = X_[:, pct_idx]

    return _nan_reshape(X_, no_data)


def _fifty_cumulative(X, axis=1, no_data=0):

    """
    Computes the value at the cumulative 50th percentile
    """

    x_range = list(range(0, X.shape[1]+1))

    # Cumulative sum
    X_ = X.cumsum(axis=axis)

    # Position at the nth percentile
    pct_idx = int(np.ceil(np.percentile(x_range, 50)))

    # Values at the nth percentile
    X_ = X_[:, pct_idx]

    return _nan_reshape(X_, no_data)


def _seventy_five_cumulative(X, axis=1, no_data=0):

    """
    Computes the value at the cumulative 75th percentile
    """

    x_range = list(range(0, X.shape[1]+1))

    # Cumulative sum
    X_ = X.cumsum(axis=axis)

    # Position at the nth percentile
    pct_idx = int(np.ceil(np.percentile(x_range, 75)))

    # Values at the nth percentile
    X_ = X_[:, pct_idx]

    return _nan_reshape(X_, no_data)


def _ninety_five_cumulative(X, axis=1, no_data=0):

    """
    Computes the value at the cumulative 95th percentile
    """

    x_range = list(range(0, X.shape[1]+1))

    # Cumulative sum
    X_ = X.cumsum(axis=axis)

    # Position at the nth percentile
    pct_idx = int(np.ceil(np.percentile(x_range, 95)))

    # Values at the nth percentile
    X_ = X_[:, pct_idx]

    return _nan_reshape(X_, no_data)


def _five(X, axis=1, no_data=0):

    """
    Computes the 5th percentile
    """

    X_ = np.percentile(X, 5, axis=axis)

    return _nan_reshape(X_, no_data)


def _twenty_five(X, axis=1, no_data=0):

    """
    Computes the 25th percentile
    """

    X_ = np.percentile(X, 25, axis=axis)

    return _nan_reshape(X_, no_data)


def _fifty(X, axis=1, no_data=0):

    """
    Computes the 50th percentile (median)
    """

    X_ = np.percentile(X, 50, axis=axis)

    return _nan_reshape(X_, no_data)


def _seventy_five(X, axis=1, no_data=0):

    """
    Computes the 75th percentile
    """

    X_ = np.percentile(X, 75, axis=axis)

    return _nan_reshape(X_, no_data)


def _ninety_five(X, axis=1, no_data=0):

    """
    Computes the 95th percentile
    """

    X_ = np.percentile(X, 95, axis=axis)

    return _nan_reshape(X_, no_data)


def _slopes(X, axis=None, no_data=0):

    """
    Computes min. and max. moving slope
    """

    # Reshape to [dims x samples].
    X_min, X_max = rolling_stats(X.T,
                                 stat='slope',
                                 window_size=15)

    return np.hstack((_nan_reshape(X_min, no_data),
                      _nan_reshape(X_min, no_data)))


def _max_diff(X, axis=1, no_data=0):

    """
    Computes the maximum difference
    """

    X_ = np.abs(np.diff(X, n=2, axis=axis)).max(axis=axis)

    return _nan_reshape(X_, no_data)


class TimeSeriesFeatures(object):

    def __init__(self):

        self.ts_funcs = None

        self._func_dict = dict(mean=_mean,
                               cv=_cv,
                               five=_five,
                               twenty_five=_twenty_five,
                               fifty=_fifty,
                               seventy_five=_seventy_five,
                               ninety_five=_ninety_five,
                               five_cumulative=_five_cumulative,
                               twenty_five_cumulative=_twenty_five_cumulative,
                               fifty_cumulative=_fifty_cumulative,
                               seventy_five_cumulative=_seventy_five_cumulative,
                               ninety_five_cumulative=_ninety_five_cumulative,
                               slopes=_slopes)

    def add_features(self, feature_list):

        """
        Adds features

        Args:
            feature_list (str list): A list of features to add.

                Choices are:
                    'cv'
                    'mean'
                    'five'
                    'twenty_five'
                    'fifty'
                    'seventy_five'
                    'ninety_five'
                    'five_cumulative'
                    'twenty_five_cumulative'
                    'fifty_cumulative'
                    'seventy_five_cumulative'
                    'ninety_five_cumulative'
                    'slopes'
        """

        self.ts_funcs = [(feature_name, self._func_dict[feature_name]) for feature_name in feature_list
                         if feature_name in self._func_dict]

    def apply_features(self, X=None, ts_indices=None, append_features=True, **kwargs):

        """
        Applies features to an array

        Args:
            X (Optional[2d array]): The array to add features to.
            ts_indices (Optional[1-d array like]): A list of indices to index the time series. Default is None.
            append_features (Optional[bool]): Whether to append features to `X`. Default is True.
        """

        if not isinstance(self.ts_funcs, list):
            logger.error('  The features must be added with `add_features`.')

        if not append_features:
            Xnew = None

        for ts_func in self.ts_funcs:

            if isinstance(ts_indices, np.ndarray) or isinstance(ts_indices, list):

                if append_features:

                    X = np.hstack((X,
                                   ts_func[1](X[:, ts_indices],
                                              **kwargs)))

                else:

                    if isinstance(Xnew, np.ndarray):

                        Xnew = np.hstack((Xnew,
                                          ts_func[1](X[:, ts_indices],
                                                     **kwargs)))

                    else:

                        Xnew = ts_func[1](X[:, ts_indices],
                                          **kwargs)

            else:

                if append_features:

                    X = np.hstack((X,
                                   ts_func[1](X,
                                              **kwargs)))

                else:

                    if isinstance(Xnew, np.ndarray):

                        Xnew = np.hstack((Xnew,
                                          ts_func[1](X,
                                                     **kwargs)))

                    else:

                        Xnew = ts_func[1](X,
                                          **kwargs)

        if append_features:
            return X
        else:
            return Xnew
