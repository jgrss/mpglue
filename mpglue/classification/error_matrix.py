#!/usr/bin/env python

"""
@author: Jordan Graesser
Date Created: 9/24/2011
"""

from __future__ import division

from builtins import int

import os
import time
import logging
from copy import copy
from six import string_types
import platform

from .. import raster_tools

try:
    import numpy as np
except ImportError:
    raise ImportError('NumPy must be installed.')

# Scikit-learn
try:
    from sklearn import metrics
except ImportError:
    raise ImportError('Scikits-learn must be installed.')

# Ndimage
try:
    from scipy.ndimage.measurements import label as lab_img
    import scipy.stats as st
except ImportError:
    raise ImportError('Ndimage must be installed')

# Scikit-image
try:
    from skimage.measure import label, regionprops
except ImportError:
    raise ImportError('Scikit-image must be installed')

# Matplotlib
try:
    import matplotlib as mpl

    if (os.environ.get('DISPLAY', '') == '') or (platform.system() == 'Darwin'):
        mpl.use('Agg')

    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Matplotlib must be installed')

import warnings
warnings.filterwarnings('ignore')


class error_matrix(object):

    """
    Computes accuracy statistics

    Args:
        po_text (str): Predicted and observed labels as a text file,
            where (predicted, observed) are the last two columns.
        po_array (ndarray): Predicted and observed labels as an array,
            where (predicted, observed) are the last two columns.
        header (Optional[bool]): Whether ``file`` or ``predicted_observed`` contains a header. Default is False.
        class_list (Optional[list])
        discrete (Optional[bool])
        e_matrix (Optional[ndarray])

    Attributes:
        n_classes (int): Number of unique classes.
        class_list (list): List of unique classes.
        e_matrix (ndarray): Error matrix.
        accuracy (float): Overall accuracy.
        report
        f_scores (float)
        f_beta (float)
        hamming (float)
        kappa_score (float)
        mae (float)
        mse (float)
        rmse (float)
        r_squared (float)

    Examples:
        >>> from mpglue.classification import error_matrix
        >>>
        >>> emat = error_matrix()
        >>>
        >>> # Get an accuracy report from an array
        >>> emat.get_stats(po_array=test_array)
        >>> print emat.accuracy
        >>>
        >>> # Get an accuracy report from a text file
        >>> emat.get_stats(po_text='/test_samples.txt')
        >>>
        >>> # Write statistics to file
        >>> emat.write_stats('/accuracy_report.txt')

    Returns:
        None, writes to ``files`` if given.

    Reference:
        Overall accuracy:
            where,
                Oacc = D / (N * 100)
                    where,
                        D = total number of correctly labeled samples (on the diagonal)
                        N = total number of samples in the matrix
        Kappa:
            : Measure of agreement
        F1-score:
            where,
                F-measure = 2 * ((Producer's * User's) / (Producer's + User's))
        Producer's accuracy:
            Omission error
                -- "excluding an area from the category in which it truly belongs"
                    - Congalton and Green (1999)

              # correctly classified observed samples for class N
            = ----------------------------------------------------- x 100
              total # of observed (column-wise) samples for class N

        User's accuracy:
            Commission error
                -- "including an area into a category when it does not belong to that category"
                    - Congalton and Green (1999)

              # correctly classified predicted samples for class N
            = ---------------------------------------------------- x 100
              total # of predicted (row-wise) samples for class N

        RMSE:
            where,
                (Square root of (Sum of (x - y)^2) / N)

        |========================================================================|
        |                ********************                                    |
        |                * Confusion Matrix *                                    |
        |                ********************                                    |
        |                                                                        |
        |                   Observed/                                            |
        |                   Reference                                            |
        |                  ---------------                                       |
        |                  C0   C1   C2   ..   Cn  | Column totals | User's (%)  |
        |                  ----------------------- | ------------- | ----------- |
        |  Predicted/| C0 |(#)                     | sum(C0 row)   | %           |
        |  Labeled   | C1 |     (#)                | sum(C1 row)   | %           |
        |            | C2 |          (#)           | sum(C2 row)   | %           |
        |            | .. |               ..       | ..            | %           |
        |            | Cn |                   (#)  | sum(Cn row)   | %           |
        |       Row totals|                        | (TOTAL)       |             |
        |   Producer's (%)| %    %    %   ..   %   |               | (Overall %) |
        |========================================================================|
    """

    def __init__(self):
        self.time_stamp = time.asctime(time.localtime(time.time()))

    def get_stats(self,
                  po_text=None,
                  po_array=None,
                  header=False,
                  class_list=None,
                  discrete=True,
                  e_matrix=None):

        self.discrete = discrete

        if isinstance(e_matrix, np.ndarray):

            self.e_matrix = e_matrix

            self.n_classes = self.e_matrix.shape[0]
            self.class_list = np.arange(1, self.n_classes+1)
            self.n_samples = self.e_matrix.sum()

            # Reverse the error matrix
            self.X, self.y = self.error_matrix2xy()

        else:

            if isinstance(po_text, str):

                samples = np.genfromtxt(po_text, delimiter=',').astype(int)

            else:

                try:
                    samples = po_array.copy()
                except ValueError:
                    raise ValueError('Observed and predicted labels must be passed.')

            if header:
                hdr_idx = 1
            else:
                hdr_idx = 0

            # observed (true)
            self.y = np.int16(np.float32(np.asarray(samples[hdr_idx:, -1]).ravel()))

            # predicted
            self.X = np.int16(np.float32(np.asarray(samples[hdr_idx:, -2]).ravel()))

            self.n_samps = len(self.y)

            if not class_list:

                # Get unique class values
                class_list1 = np.unique(self.X)
                class_list2 = np.unique(self.y)

                self.merge_lists(class_list1, class_list2)

            else:
                self.class_list = class_list

            self.n_classes = len(self.class_list)
            self.class_list = sorted(self.class_list)
            self.n_samples = self.y.shape[0]

            # Create the error matrix
            self.e_matrix = np.zeros((self.n_classes, self.n_classes), dtype='int16')

            # Add to error matrix
            for predicted, observed in zip(self.X, self.y):

                self.e_matrix[self.class_list.index(predicted),
                              self.class_list.index(observed)] += 1

        if self.discrete:

            # Producer's and User's accuracy
            self.producers_accuracy()
            self.users_accuracy()

            # Overall accuracy
            self.accuracy = metrics.accuracy_score(self.y, self.X) * 100.0

            # Statistics report
            self.report = metrics.classification_report(self.y, self.X)

            # Get f scores for each class
            self.f_scores = metrics.f1_score(self.y, self.X, average=None)

            # get the weighted f beta score
            try:

                self.f_beta = metrics.fbeta_score(self.y, self.X,
                                                  beta=0.5,
                                                  labels=self.class_list,
                                                  pos_label=self.class_list[1])

            except:
                self.f_beta = None

            # get the hamming loss score
            self.hamming = metrics.hamming_loss(self.y, self.X)

            # get the Kappa score
            self.kappa(self.y, self.X)

        else:

            # get the mean absolute error
            self.mae = metrics.mean_absolute_error(self.y, self.X)

            # get the mean square error
            self.mse = metrics.mean_squared_error(self.y, self.X)

            # get the median absolute error
            self.medae = metrics.median_absolute_error(self.y, self.X)

            # get the root mean squared error
            self.rmse = np.sqrt(self.mse)

            # get the r squared
            self.r_squared = metrics.r2_score(self.y, self.X)

    def error_matrix2xy(self):

        """
        Reverses the error matrix to predictions and observations
        """

        observed = list()
        predicted = list()

        for ei in range(0, self.e_matrix.shape[0]):

            for ej in range(0, self.e_matrix.shape[1]):

                n_ = self.e_matrix[ei, ej]

                for n in range(0, n_):

                    observed.append(ei+1)
                    predicted.append(ej+1)

        return np.int16(np.array(observed)), np.int16(np.array(predicted))

    def sample_size(self, class_area, users_accuracy, standard_error=0.01):

        """
        Calculates the sample size given a target standard error

        Args:
            class_area (list): A list of class areas.
            users_accuracy (list): A list of class user accuracies.
            standard_error (Optional[float]): The target standard error.

        References:
            Olofsson et al. (2014) Good practices for estimating area and assessing
                accuracy of land change. Remote Sensing of Environment 148, 442-57.

        Example:
            # Example 5
            >>> emat = error_matrix()
            >>>
            >>> class_area = [18000, 13500, 288000, 580500]
            >>>
            >>> # Estimates of user's accuracy
            >>> users_accuracy = [0.7, 0.6, 0.9, 0.95]
            >>>
            >>> emat.sample_size(class_area, users_accuracy, standard_error=0.01)
            >>> emat.sample_n --> 641
        """

        if not isinstance(class_area, np.ndarray):
            class_area = np.array(class_area, dtype='float32')

        if not isinstance(users_accuracy, np.ndarray):
            users_accuracy = np.array(users_accuracy, dtype='float32')

        # Class area proportions
        class_area_prop = class_area / class_area.sum()

        # Estimated user's accuracy variance
        # Eq. 6
        users_variance = users_accuracy * (1.0 - users_accuracy)

        # Stratum standard deviation
        users_standard_deviation = np.sqrt(users_variance)

        self.sample_n = int(round(((class_area_prop * users_standard_deviation).sum() / standard_error)**2))

        self.class_proportions = np.int64(self.sample_n * class_area_prop)

    def sample_bias(self, class_area, conf=0.95):

        """
        Calculates the area adjusted sampling bias

        Args:
            class_area (list): A list of class areas.
            conf (Optional[float]): The confidence level.

        References:
            Olofsson et al. (2013) Making better use of accuracy data in land
                change studies: Estimating accuracy and area and quantifying
                uncertainty using stratified estimation.
                Remote Sensing of Environment 129, 122-131.

        Example:
            >>> emat = error_matrix()
            >>> class_area = [22353, 1122543, 610228]
            >>> emat.e_matrix = np.array([[97, 0, 3], [3, 279, 18], [2, 1, 97]], dtype='float32')
            >>> emat.sample_bias(class_area)
            >>>
            >>> # Final land change error with margin of error (95% conf.)
            >>> emat.stratified_area_estimate +- emat.margin_of_error
        """

        e_matrix_float = np.float32(self.e_matrix)

        if not isinstance(class_area, np.ndarray):
            class_area = np.array(class_area, dtype='float32')

        total_area = class_area.sum()

        # Calculate the map area weights.
        self.class_area_prop = class_area / total_area

        emat_row_sum = e_matrix_float.sum(axis=1)
        emat_col_sum = e_matrix_float.sum(axis=0)

        # Estimate the class proportions.
        e_matrix_pr = np.array([self.class_area_prop * (e_matrix_float[:, ci] / emat_row_sum)
                                for ci in range(e_matrix_float.shape[1])], dtype='float32')

        # User and producer weights
        # Equation 9
        prd_weights = e_matrix_pr.sum(axis=0)
        usr_weights = e_matrix_pr.sum(axis=1)

        # Equation 10 (stratified error-adjusted area estimate)
        self.stratified_area_estimate = usr_weights * total_area

        self.area_difference = self.stratified_area_estimate - class_area

        self.standard_error_prop = list()

        # The estimated standard error of the estimated area proportion
        # Eq. 3
        for ci in range(e_matrix_float.shape[0]):

            a = self.class_area_prop**2
            b = e_matrix_float[:, ci] / emat_row_sum
            c = 1.0 - (e_matrix_float[:, ci] / emat_row_sum)

            self.standard_error_prop.append(((a * ((b * c) / (emat_row_sum - 1.0))).sum() ** 0.5))

        self.standard_error_prop = np.array(self.standard_error_prop, dtype='float32')

        # The standard error of the error-adjusted estimated area
        # Eq. 4
        self.standard_error_area = self.standard_error_prop * total_area

        # Margin of error (confidence interval)
        # Eq. 5
        self.margin_of_error = self.standard_error_area * st.norm.interval(conf, loc=0, scale=1)[1]

        # Equation 14
        self.stratified_users = np.diagonal(e_matrix_pr) / prd_weights

        # Equation 15
        self.stratified_producers = np.diagonal(e_matrix_pr) / usr_weights

        # Equation 16
        self.stratified_overall = np.diagonal(e_matrix_pr).sum()

    def producers_accuracy(self):

        """
        Producer's accuracy
        """

        self.producers = np.zeros(self.n_classes, dtype='float32')

        producer_sums = self.e_matrix.sum(axis=0)

        for pr_j in range(0, self.n_classes):
            self.producers[pr_j] = (self.e_matrix[pr_j, pr_j] / float(producer_sums[pr_j])) * 100.0

        self.producers[np.isnan(self.producers) | np.isinf(self.producers)] = 0.0

    def users_accuracy(self):

        """
        User's accuracy
        """

        self.users = np.zeros(self.n_classes, dtype='float32')

        user_sums = self.e_matrix.sum(axis=1)

        for pr_i in range(0, self.n_classes):
            self.users[pr_i] = (self.e_matrix[pr_i, pr_i] / float(user_sums[pr_i])) * 100.0

        self.users[np.isnan(self.users) | np.isinf(self.users)] = 0.0

    def merge_lists(self, list1, list2):

        self.class_list = list(copy(list1))

        for value2 in list2:

            if value2 not in list1:
                self.class_list.append(value2)

    def kappa(self, y_true, y_pred, weights=None, allow_off_by_one=False):

        """
        Calculates the kappa inter-rater agreement between two the gold standard
        and the predicted ratings. Potential values range from -1 (representing
        complete disagreement) to 1 (representing complete agreement).  A kappa
        value of 0 is expected if all agreement is due to chance.

        In the course of calculating kappa, all items in ``y_true`` and ``y_pred`` will
        first be converted to floats and then rounded to integers.

        It is assumed that ``y_true`` and ``y_pred`` contain the complete range of possible
        ratings.

        This function contains a combination of code from yorchopolis's kappa-stats
        and Ben Hamner's Metrics projects on Github.

        Args:
            y_true
            y_pred
            weights (Optional[str or numpy array]): Specifies the weight matrix for the calculation. Choices
                are [None :: unweighted-kappa, 'quadratic' :: quadratic-weighted kappa, 'linear' ::
                linear-weighted kappa, two-dimensional numpy array :: a custom matrix of weights. Each weight
                corresponds to the :math:`w_{ij}` values in the wikipedia description of how to calculate
                weighted Cohen's kappa.]
            allow_off_by_one (Optional[bool]): If true, ratings that are off by one are counted as equal, and
                all other differences are reduced by one. For example, 1 and 2 will be considered to be
                equal, whereas 1 and 3 will have a difference of 1 for when building the weights matrix.

        Reference:
            Authors: SciKit-Learn Laboratory
            https://skll.readthedocs.org/en/latest/_modules/skll/metrics.html
        """

        logger = logging.getLogger(__name__)

        # Ensure that the lists are both the same length
        assert(len(y_true) == len(y_pred))

        # This rather crazy looking typecast is intended to work as follows:
        # If an input is an int, the operations will have no effect.
        # If it is a float, it will be rounded and then converted to an int
        # because the ml_metrics package requires ints.
        # If it is a str like "1", then it will be converted to a (rounded) int.
        # If it is a str that can't be typecast, then the user is
        # given a hopefully useful error message.
        try:
            y_true = [int(round(float(y))) for y in y_true]
            y_pred = [int(round(float(y))) for y in y_pred]
        except ValueError as e:
            logger.error("For kappa, the labels should be integers or strings " +
                         "that can be converted to ints (E.g., '4.0' or '3').")
            raise e

        # Figure out normalized expected values
        min_rating = np.minimum(np.min(y_true), np.min(y_pred))
        max_rating = np.maximum(np.max(y_true), np.max(y_pred))

        # shift the values so that the lowest value is 0
        # (to support scales that include negative values)
        y_true = [y - min_rating for y in y_true]
        y_pred = [y - min_rating for y in y_pred]

        # Build the observed/confusion matrix
        num_ratings = max_rating - min_rating + 1
        obsv = metrics.confusion_matrix(y_true, y_pred, labels=list(range(num_ratings)))
        num_scored_items = float(len(y_true))

        # Build weight array if weren't passed one
        if isinstance(weights, string_types):
            wt_scheme = weights
            weights = None
        else:
            wt_scheme = ''
        if weights is None:
            weights = np.empty((num_ratings, num_ratings))

            for i in range(num_ratings):
                for j in range(num_ratings):
                    diff = np.abs(i - j)
                    if allow_off_by_one and diff:
                        diff -= 1
                    if wt_scheme == 'linear':
                        weights[i, j] = diff
                    elif wt_scheme == 'quadratic':
                        weights[i, j] = diff ** 2
                    elif not wt_scheme:  # unweighted
                        weights[i, j] = bool(diff)
                    else:
                        raise ValueError(('Invalid weight scheme specified for ' +
                                          'kappa: {}').format(wt_scheme))

        hist_true = np.bincount(y_true, minlength=num_ratings)
        hist_true = hist_true[: num_ratings] / num_scored_items
        hist_pred = np.bincount(y_pred, minlength=num_ratings)
        hist_pred = hist_pred[: num_ratings] / num_scored_items
        expected = np.outer(hist_true, hist_pred)

        # Normalize observed array
        obsv = obsv / num_scored_items

        # If all weights are zero, that means no disagreements matter.
        self.kappa_score = 1.

        if np.count_nonzero(weights):
            self.kappa_score -= (np.sum(np.sum(weights * obsv)) / np.sum(np.sum(weights * expected)))

    def write_stats(self, out_report):

        """
        Args:
            out_report (str): The file to write to.
        """

        with open(out_report, 'a') as write_txt:

            if self.discrete:

                # write statistics to text file
                write_txt.write('============\n')
                write_txt.write('Error Matrix\n')
                write_txt.write('============\n\n')

                write_txt.write('                Observed\n')
                write_txt.write('                --------\n')
                write_txt.write('                C {0:5d}'.format(self.class_list[0]))

                for c in range(1, self.n_classes):
                    write_txt.write('   C {0:5d}'.format(self.class_list[c]))

                write_txt.write('   Total   User(%)\n')

                write_txt.write('                -------   ')

                for c in range(0, self.n_classes-1):
                    write_txt.write('-------   ')

                write_txt.write('-----   -------\n')

                for i in range(0, self.n_classes):

                    if i == 0:
                        write_txt.write('Predicted C{:03d}| ('.format(self.class_list[0]))
                    else:
                        write_txt.write('          C{:03d}| '.format(self.class_list[i]))

                    for j in range(0, self.n_classes):

                        spacer = 10 - len(str(int(self.e_matrix[i, j])))

                        if i == j and i != 0:
                            write_txt.write('(')
                        if i == j:
                            write_txt.write(str(int(self.e_matrix[i, j])) + ')')

                            for s in range(0, spacer-2):
                                write_txt.write(' ')
                        else:
                            write_txt.write(str(int(self.e_matrix[i, j])))

                            for s in range(0, spacer):
                                write_txt.write(' ')

                    write_txt.write('{:d}'.format(self.e_matrix[i, :].sum()))

                    spacer = len(str(self.e_matrix[i, :].sum()))

                    if spacer == 1: spacer = 7
                    elif spacer == 2: spacer = 6
                    elif spacer == 3: spacer = 5
                    elif spacer == 4: spacer = 4
                    elif spacer == 5: spacer = 3
                    elif spacer == 6: spacer = 2
                    elif spacer == 7: spacer = 1

                    for s in range(0, spacer):
                        write_txt.write(' ')

                    # User's accuracy
                    write_txt.write('{:.2f}\n'.format(self.users[i]))

                write_txt.write('         Total| ')

                for j in range(0, self.n_classes):

                    spacer = 10 - len(str(int(self.e_matrix[:, j].sum())))

                    write_txt.write('{:d}'.format(self.e_matrix[:, j].sum()))

                    for s in range(0, spacer):
                        write_txt.write(' ')

                write_txt.write('({:d})\n'.format(self.e_matrix.sum(axis=0).sum()))

                # Producer's accuracy
                write_txt.write('   Producer(%)| ')

                for pr_j in self.producers:

                    pr_jf = '{:.2f}'.format(pr_j)

                    spacer = 10 - len(pr_jf)

                    write_txt.write(pr_jf)

                    for s in range(0, spacer):
                        write_txt.write(' ')

                if len(pr_jf) == 1:
                    sp = 17
                elif len(pr_jf) == 2:
                    sp = 16
                elif len(pr_jf) == 3:
                    sp = 15
                elif len(pr_jf) == 4:
                    sp = 14
                elif len(pr_jf) == 5:
                    sp = 13
                elif len(pr_jf) == 6:
                    sp = 12
                elif len(pr_jf) == 7:
                    sp = 11
                elif len(pr_jf) == 8:
                    sp = 10
                elif len(pr_jf) == 9:
                    sp = 9

                for s in range(0, sp-spacer):
                    write_txt.write(' ')

                write_txt.write('({:.2f}%)\n'.format(self.accuracy))

                write_txt.write('\nSamples: {:,d}\n'.format(self.n_samps))
                write_txt.write('\n==========\n')
                write_txt.write('Statistics\n')
                write_txt.write('==========\n')
                write_txt.write('\nOverall Accuracy (%): {:.2f}\n'.format(self.accuracy))
                write_txt.write('Kappa: {:.2f}\n'.format(self.kappa_score))

                if isinstance(self.f_beta, float) or isinstance(self.f_beta, np.ndarray):

                    if isinstance(self.f_beta, float):
                        write_txt.write('F-beta: {:.2f}\n'.format(self.f_beta))
                    else:

                        for fi, fb in enumerate(self.f_beta):

                            if fi == (len(self.f_beta)-1):
                                write_txt.write('F-beta: {:.2f}\n'.format(fb))
                            else:
                                write_txt.write('F-beta: {:.2f},'.format(fb))

                write_txt.write('Hamming loss: {:.2f}\n'.format(self.hamming))

                write_txt.write('\n============\n')
                write_txt.write('Class report\n')
                write_txt.write('============\n')
                write_txt.write('\n{}'.format(self.report))

            else:

                write_txt.write('=====================\n')
                write_txt.write('Regression statistics\n')
                write_txt.write('=====================\n\n')
                write_txt.write('Mean Absolute Error: {:.4f}\n'.format(self.mae))
                write_txt.write('Median Absolute Error: {:.4f}\n'.format(self.medae))
                write_txt.write('Mean Squared Error: {:.4f}\n'.format(self.mse))
                write_txt.write('Root Mean Squared Error: {:.4f}\n'.format(self.rmse))
                write_txt.write('R squared: {:.4f}\n'.format(self.r_squared))


class object_accuracy(object):

    """
    Assesses object accuracy measures.

    Args:
        reference_array (ndarray)
        predicted_array (ndarray)

    Methods:
        error_array, which is a (5 x rows x columns) array, where the error layers are ...
            1: over-segmentation
            2: under-segmentation
            3: fragmentation
            4: shape error
            5: offset (Euclidean distance (in map units) of object centroids, not found in Persello et al. (2010))

    Reference:
        Persello C and Bruzzone L (2010) A Novel Protocol for Accuracy Assessment in Classification of Very
            High Resolution Images. IEEE Transactions on Geoscience and Remote Sensing, 48(3).

    Examples:
        >>> from mappy.sample import object_accuracy
        >>>
        >>> oi = object_accuracy(reference_array, predicted_array)
        >>> oi.label_objects()
        >>> oi.iterate_objects()
        >>>
        >>> # write to file
        >>> o_info = copy(i_info)
        >>> o_info.storage = 'float32'
        >>> o_info.bands = 3
        >>> oi.write_stats('/out_object_accuracy.tif', o_info)
    """

    def __init__(self,
                 reference_array,
                 predicted_array,
                 image_id=None,
                 objects_labeled=True):

        # predicted_array[predicted_array > 0] = 1
        # reference_array[reference_array > 0] = 1

        self.image_id = image_id
        self.objects_labeled = objects_labeled

        self.unique_object_ids = np.unique(reference_array)

        self.rows = predicted_array.shape[0]
        self.cols = predicted_array.shape[1]

        self.reference_array = reference_array
        self.predicted_array = predicted_array

        # convert the predictions to binary
        # self.predicted_array[self.predicted_array > 0] = 1
        # print np.unique(self.predicted_array)
        # sys.exit()

        # over_segmentation = band 1
        # under_segmentation = band 2
        # fragmentation = band 3
        # shape error = band 4
        # offset error = band 5
        self.error_array = np.zeros((6, self.rows, self.cols), dtype='float32') + 999

    def label_objects(self):

        # Label the objects of the predicted array.
        if self.objects_labeled:
            self.predicted_objects = self.predicted_array
        else:
            self.predicted_objects, __ = lab_img(self.predicted_array)

        # plt.subplot(121)
        # plt.imshow(self.predicted_objects)
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(self.reference_array+self.predicted_objects)
        # plt.axis('off')
        # plt.show()
        # sys.exit()

        # get the object properties of the labeled reference array
        # self.props = regionprops(self.reference_objects)

    def iterate_ids(self):

        """
        Iterate over objects, where each object has a unique id
        """

        self.ids = list()
        self.over = list()
        self.under = list()
        self.frag = list()
        self.shape = list()
        self.dist = list()
        self.area_reference = list()
        self.area_predicted = list()
        self.relative = list()

        # iterate over each object
        for uoi in self.unique_object_ids:

            # get the current object (binary)
            if uoi == 0:
                continue

            self.reference_sub = np.where(self.reference_array == uoi, 1, 0)

            # This is the reference object sum, which is
            #   O_i in Persello et al. (2010)
            self.reference_object_area = self.reference_sub.sum()

            # Get the object properties of the current object.
            self.props = regionprops(self.reference_sub)

            # Bounding box of the current object (min row, min col, max row, max col)
            object_properties = [(prop.bbox, prop.eccentricity, prop.area) for prop in self.props][0]

            bbox = object_properties[0]
            self.reference_eccentricity = object_properties[1]
            self.reference_area = object_properties[2]

            # subset the object (50 is for padding)
            self.get_padding(bbox, 50)

            self.reference_sub = self.reference_sub[bbox[0]-self.row_min:bbox[2]+50, bbox[1]-self.col_min:bbox[3]+50]
            self.predicted_sub = self.predicted_objects[bbox[0]-self.row_min:bbox[2]+50, bbox[1]-self.col_min:bbox[3]+50]

            # Get the object properties of the current object
            self.props_ = regionprops(self.reference_sub)

            # bounding box of the current object (min row, min col, max row, max col)
            self.reference_centroid = [prop_.centroid for prop_ in self.props_][0]

            # Get predicted object with the maximum number
            #   of overlapping points in the reference object.
            unique_labels = np.unique(self.predicted_sub)

            self.fragments = np.where(self.reference_sub == 1, self.predicted_sub, 0)

            self.max_sum = 0

            # Iterate over each fragment in the predicted array. Here
            #   we iterate over each fragment rather than just doing
            #   an intersection because we are interested in the full
            #   fragment that touches the reference object.
            for unique_label in unique_labels:

                if unique_label == 0:
                    continue

                # Get the pixel count where predicted and
                #   the reference object overlap.
                self.overlap_sum = np.where((self.predicted_sub == unique_label) &
                                            (self.reference_sub == 1), 1, 0).sum()

                # Take the object with the highest
                #   number of overlapping pixels.
                if self.overlap_sum > self.max_sum:

                    # This is the predicted object sum, which is
                    #   M_i in Persello et al. (2010).
                    self.predicted_sub_ = np.where(self.predicted_sub == unique_label, 1, 0)
                    self.predicted_object_area = self.predicted_sub_.sum()

                    # Get the object properties of the current predicted object
                    pprops = regionprops(self.predicted_sub_)

                    predicted_object_properties = [(pprop.eccentricity,
                                                    pprop.centroid,
                                                    pprop.area) for pprop in pprops][0]

                    # Get the eccentricity of the predicted object.
                    self.predicted_eccentricity = predicted_object_properties[0]

                    # Get the centroid of the predicted object.
                    self.predicted_centroid = predicted_object_properties[1]

                    # Get the area of the predicted object.
                    self.predicted_area = predicted_object_properties[2]

                    # The object label of the highest overlapping object.
                    self.max_label = copy(unique_label)

                    # This is the union of O_i and M_i in Persello et al. (2010).
                    self.max_sum = copy(self.overlap_sum)

            if self.max_sum == 0:
                continue

            # now, <max_label> has the most overlapping pixels with
            # the reference object, so we can get statistics for it.
            stat_over = self.over_segmentation()
            stat_under = self.under_segmentation()
            stat_frag = self.fragmentation()
            stat_shape = self.shape_error()
            stat_off = self.offset_error()
            stat_rel = self.relative_error()

            # print
            # print self.reference_eccentricity, self.predicted_eccentricity
            # print self.reference_centroid, self.predicted_centroid
            # print self.reference_area, self.predicted_object_area
            # print self.max_label
            # print self.max_sum
            # print stat_rel
            # plt.subplot(121)
            # plt.imshow(self.reference_sub)
            # plt.axis('off')
            # plt.subplot(122)
            # plt.imshow(self.predicted_sub_+self.reference_sub)
            # plt.axis('off')
            # plt.show()
            # sys.exit()

            self.error_array[0][self.reference_array == uoi] = stat_over
            self.error_array[1][self.reference_array == uoi] = stat_under
            self.error_array[2][self.reference_array == uoi] = stat_frag
            self.error_array[3][self.reference_array == uoi] = stat_shape
            self.error_array[4][self.reference_array == uoi] = stat_off
            self.error_array[5][self.reference_array == uoi] = stat_rel

            self.ids.append(uoi)
            self.over.append(stat_over)
            self.under.append(stat_under)
            self.frag.append(stat_frag)
            self.shape.append(stat_shape)
            self.dist.append(stat_off)
            self.area_reference.append(self.reference_area)
            self.area_predicted.append(self.predicted_area)
            self.relative.append(stat_rel)

        self.error_array[self.error_array == 999] = 0

    def iterate_objects(self):

        """
        Iterate over objects, where each object equals 1 are clearly separated
        """

        # iterate over each object
        for prop in self.props:

            # bounding box of the current object (min row, min col, max row, max col)
            bbox = prop.bbox

            # get the current object (binary)
            self.reference_sub = np.where(self.reference_objects == prop.label, 1, 0)

            # This is the reference object sum, which is
            # O_i in Persello et al. (2010)
            self.reference_object_area = self.reference_sub.sum()

            # subset the object (50 is for padding)
            self.get_padding(bbox, 50)

            self.reference_sub = self.reference_sub[bbox[0]-self.row_min:bbox[2]+50, bbox[1]-self.col_min:bbox[3]+50]
            self.predicted_sub = self.predicted_objects[bbox[0]-self.row_min:bbox[2]+50, bbox[1]-self.col_min:bbox[3]+50]

            # get predicted object with the maximum number of overlapping points in the reference object
            unique_labels = np.unique(self.predicted_sub)

            self.max_sum = 0
            for unique_label in unique_labels:

                # Get the pixel count where predicted and the reference object overlap.
                self.overlap_sum = np.where((self.predicted_sub == unique_label) &
                                            (self.reference_sub == 1), 1, 0).sum()

                if self.overlap_sum >= self.max_sum:

                    # This is the predicted object sum, which is
                    # M_i in Persello et al. (2010)
                    self.predicted_object_area = np.where(self.predicted_sub == unique_label, 1, 0).sum()

                    # The object label of the highest overlapping object.
                    self.max_label = copy(unique_label)

                    # This is the union of O_i and M_i in Persello et al. (2010)
                    self.max_sum = copy(self.overlap_sum)

            # now, <max_label> has the most overlapping pixels with the reference object, so we can get
            # statistics for it
            self.error_array[0][self.reference_objects == prop.label] = self.over_segmentation()
            self.error_array[1][self.reference_objects == prop.label] = self.under_segmentation()
            self.error_array[2][self.reference_objects == prop.label] = self.fragmentation()
            self.error_array[3][self.reference_objects == prop.label] = self.shape_error()
            self.error_array[4][self.reference_objects == prop.label] = self.offset_error()

    def relative_error(self):

        """
        Relative object error

        Equation:
            (extracted object area - reference object area) / reference object area * 100%
        """

        return ((self.predicted_object_area - self.reference_object_area) / float(self.reference_object_area)) * 100.

    def over_segmentation(self):

        """
        Over-segmentation

        0-1 range:
            0 = perfect agreement
            1 = high over-segmentation
        """

        # Get the union of the reference object and
        #   the highest overlapping object.
        # union(reference object, highest overlap) / reference object sum (object of interest is 1, so we can sum)
        return 1. - (float(self.max_sum) / self.reference_object_area)

    def under_segmentation(self):

        """
        Under-segmentation

        0-1 range:
            0 = perfect agreement
            1 = high under-segmentation
        """

        # union(reference object, highest overlap) / sum of object with highest overlap
        return 1. - (float(self.max_sum) / self.predicted_object_area)

    def fragmentation(self):

        """
        0-1 range
            0 being the optimum case (i.e., only one region is overlapping with the reference object)
            1 is where all the pixels belong to different regions
        """

        # number of regions - 1 / reference total - 1
        n_fragments = np.unique(self.fragments)

        if 0 in n_fragments:
            r_i = len(n_fragments) - 1
        else:
            r_i = len(n_fragments)

        return (r_i - 1.) / (self.reference_object_area - 1.)

    def shape_error(self):
        return abs(self.reference_eccentricity - self.predicted_eccentricity)

    def offset_error(self):

        """
        Returns the euclidean distance of two centroids
        """

        return np.sqrt((self.predicted_centroid[0] - self.reference_centroid[0])**2. +
                       (self.predicted_centroid[1] - self.reference_centroid[1])**2.)

    def get_padding(self, bbox, pad):

        if (bbox[0] - pad) < 0:
            self.row_min = 0
        else:
            self.row_min = bbox[0] - pad

        if (bbox[1] - pad) < 0:
            self.col_min = 0
        else:
            self.col_min = bbox[1] - pad

    def write_report(self, out_report):

        with open(out_report, 'w') as ro:

            ro.write('ID,UNQ,PIX_REF,PIX_PRED,OVER,UNDER,FRAG,SHAPE,OFFSET,RELATIVE\n')

            for unq, ar, ap, ov, un, fr, sh, di, rl in zip(self.ids,
                                                           self.area_reference,
                                                           self.area_predicted,
                                                           self.over,
                                                           self.under,
                                                           self.frag,
                                                           self.shape,
                                                           self.dist,
                                                           self.relative):

                ro.write('{},{:d},{:d},{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(self.image_id,
                                                                                                int(unq),
                                                                                                int(ar),
                                                                                                int(ap),
                                                                                                ov,
                                                                                                un,
                                                                                                fr,
                                                                                                sh,
                                                                                                di,
                                                                                                rl))

    def write_stats(self, out_image, o_info):

        raster_tools.write2raster(self.error_array,
                                  out_image,
                                  o_info=o_info,
                                  compress='none',
                                  tile=False,
                                  flush_final=True)
