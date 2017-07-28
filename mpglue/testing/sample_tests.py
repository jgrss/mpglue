#!/usr/bin/env python

import unittest

import mpglue as gl
import numpy as np
import pandas as pd


sample_data = 'data/samples.csv'
clear_data = 'data/clear_observations.csv'
weights_data = 'data/weights.csv'

clear_df = pd.read_csv(clear_data)
weights_df = pd.read_csv(weights_data)

cl = gl.classification()

cl.split_samples(sample_data,
                 perc_samp=.9,
                 perc_samp_each=0,
                 scale_data=False,
                 class_subs=None,
                 norm_struct=True,
                 labs_type='int',
                 recode_dict=None,
                 classes2remove=None,
                 sample_weight=weights_df.weights.values,
                 ignore_feas=None,
                 use_xy=False,
                 stratified=False,
                 spacing=1000.,
                 x_label='X',
                 y_label='Y',
                 response_label='response',
                 clear_observations=clear_df.clear.values,
                 min_observations=0)

print(cl.p_vars)
print('')
print(cl.sample_weight)
print('')
print(cl.train_clear)
print('')


class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def test_instance_pvars(self):
        self.assertIsInstance(cl.p_vars, np.ndarray)

    def test_nsamps(self):
        self.assertEqual(cl.n_samps, 35)

    def test_nfeas(self):
        self.assertEqual(cl.n_feas, 4)

    def test_labels1(self):
        self.assertEqual(len(cl.labels), cl.n_samps)

    def test_labels2(self):
        self.assertEqual(len(cl.labels), cl.p_vars.shape[0])

    def test_weights(self):
        self.assertEqual(len(cl.sample_weight), cl.p_vars.shape[0])

    def test_clear(self):
        self.assertEqual(len(cl.train_clear), cl.p_vars.shape[0])

    def test_n_classes(self):
        self.assertEqual(cl.n_classes, 4)

    def test_classes(self):
        self.assertEqual(cl.classes, [0, 1, 2, 3])

    def test_scaled(self):
        self.assertEqual(cl.scaled, False)

    def test_test(self):
        self.assertEqual(cl.p_vars_test.shape[0], len(cl.labels_test))
        self.assertEqual(cl.p_vars_test.shape[0], len(cl.test_clear))

if __name__ == '__main__':
    unittest.main()
