# from __future__ import print_function
# from keras.layers import Dense, Activation
# from keras.models import Sequential
# from sklearn import preprocessing
# import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import logging
import tensorflow as tf
import numpy as np

import os
import re

from utils import *

def cnn_features_nn():
    logging.info('Loading CNN codes...')
    data_train = np.load('cnncodes_train.npz')
    data_test = np.load('cnncodes_test.npz')
    x_train, y_train = data_train['cnn_codes'], data_train['y']
    x_test, y_test, paths_test = data_test['cnn_codes'], data_test['y'], data_test['paths']

    # Scale data (training set) to 0 mean and unit standard deviation.
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)

    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
      x_train)

    hu = [64, 64, 64]
    regressor = tf.contrib.learn.DNNRegressor(
      feature_columns=feature_columns, optimizer=tf.train.AdamOptimizer,
      hidden_units=hu, label_dimension=2)

    # Fit
    regressor.fit(x_train, y_train, steps=5000, batch_size=64)

    # Transform
    x_transformed = scaler.transform(x_test)

    # Predict and score on the training data
    y_preds = list(regressor.predict(x_train, as_iterable=True))
    score = metrics.mean_squared_error(y_preds, y_train)

    print('Training MSE: {0:f}'.format(score))

    # Predict and score on the test data
    y_preds = list(regressor.predict(x_transformed, as_iterable=True))
    score = metrics.mean_squared_error(y_preds, y_test)

    print('Test MSE: {0:f}'.format(score))

    # Plot the first 50 predicitions
    for i in range(50):
        import re
        m = re.search('_([0-9]+_[0-9]+).', paths_test[i])
        if m:
            img_id = m.group(1)
        else:
            img_id = 'UNK'
        plt.scatter(y_test[i,0], y_test[i,1], c='b')
        plt.text(y_test[i,0], y_test[i,1], img_id, size='8')
        plt.scatter(y_preds[i][0], y_preds[i][1], c='r')
        plt.text(y_preds[i][0], y_preds[i][1], img_id, size = '8')

    plt.savefig('res_test.jpeg')

    return score

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    modes = ['cnn_features_nn']
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=modes[0], choices=modes,
                      help='Operation to perform. Possible values: {}'.format(', '.join(modes)))
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    if args.mode == 'cnn_features_nn':
        acc = cnn_features_nn()
        logging.info('Regression: {:.1f}%'.format(acc * 100))
    else:
        logging.warning('Uknown mode. Possible values: {}'.format(', '.join(modes)))