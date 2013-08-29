#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

figure = plt.figure()

X, y = make_circles(n_samples=100)

features_X = X**2

clf = SVC(kernel='linear')
clf.fit(features_X, y)

x_ = np.linspace(min(X[:, 0]), max(X[:, 0]), 10)
y_ = np.linspace(min(X[:, 1]), max(X[:, 1]), 10)

X_, Y_ = np.meshgrid(x_, y_)
Z = np.empty(X_.shape)

for (i, j), val in np.ndenumerate(X_):
    x1 = val**2
    x2 = Y_[i, j]**2
    p = clf.decision_function([x1, x2])
    Z[i, j] = p[0]

levels = [-1.0, 0, 1, 0]
colors = 'k'
linestyles = ['dashed', 'solid', 'dashed']

plt.subplot(1, 2, 1)
plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(X[:, 0], X[:, 1], c=y, label='Actual Data')
plt.title('Linearly Non-separable')
plt.contour(X_, Y_, Z, colors=colors, levels=levels, linestyles=linestyles)
plt.scatter(np.sqrt(clf.support_vectors_[:, 0]),
            np.sqrt(clf.support_vectors_[:, 1]), s=80,
            facecolor='none', label='Support Vectors')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.xlabel('X1**2')
plt.ylabel('X2**2')
plt.scatter(features_X[:, 0], features_X[:, 1], c=y,
            label='Data Transformed as X**2')
plt.title('Linearly Separable')
plt.contour(X_**2, Y_**2, Z, colors=colors, level=levels,
            linestyles=linestyles)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
            facecolors='none', label='Support Vectors')
plt.legend(loc='best')

plt.show()
