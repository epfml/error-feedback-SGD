"""
Utility functions to load the datasets used in the convex optimization cases.
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


def load_abalone(test_size=0.2):
    column_names = ['sex', 'length', 'diameter', 'height', 'whole weight',
                    'shucked weight', 'viscera weight', 'shell weight', 'rings']
    train_columns = ['diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight', 'shell weight', 'rings']
    df = pd.read_csv('data/abalone.csv', names=column_names)
    df_train, df_test = train_test_split(df, test_size=test_size)
    x_train, x_test = df_train[train_columns].values, df_test[train_columns].values
    y_train, y_test = df_train[['length']].values, df_test[['length']].values

    return x_train, y_train, x_test, y_test


def load_gisette():
    x, y = sklearn.datasets.load_svmlight_file('data/gisette_scale.bz2')
    x = x[:, :1500]
    y = y.reshape(y.shape[0], 1)
    x_train, y_train, x_test, y_test = x[:1000].toarray(), y[:1000], x[1000:2000].toarray(), y[1000:2000]
    return x_train, y_train, x_test, y_test


def load_e2006():
    x_train, y_train = sklearn.datasets.load_svmlight_file('data/E2006.train.bz2')
    x_test, y_test = sklearn.datasets.load_svmlight_file('data/E2006.test.bz2')
    y_train, y_test = y_train.reshape(y_train.shape[0], 1), y_test.reshape(y_test.shape[0], 1)
    x_train, y_train, x_test, y_test = x_train[:1000], y_train[:1000], x_test[:1000], y_test[:1000]
    x_train, x_test = x_train[:, :1500].toarray(), x_test[:, :1500].toarray()
    return x_train, y_train, x_test, y_test


def load_artificial(n=1000, p=0.5):
    y = 2 * np.random.binomial(1, p, size=2*n) - 1
    x = np.zeros((2*n, 6 * 2*n))
    x[:, 0] = y
    for i in range(2*n):
        x[i, 1:3] = 1
        x[i, 3 + 5*i:4 + 5*i + 2*(1 - y[i])] = 1
    y = y.reshape((2*n, 1))
    perm = np.random.permutation(2*n)
    x = x[perm]
    y = y[perm]
    return x[:n], y[:n], x[n:], y[n:]
