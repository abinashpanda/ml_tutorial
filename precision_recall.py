#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve, auc
import random
from matplotlib import animation

# Iris dataset present in sklearn
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Converting this problem into binary classification
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape

# Shuffling data elements
p = range(n_samples)
random.shuffle(p)
X, y = X[p], y[p]

# Adding some noisy features
np.random.seed(0)
X = np.c_[X, np.random.randn(n_samples, 200*n_features)]

half = int(n_samples/2)

clf = svm.SVC(kernel='linear', probability=True, random_state=0)
clf.fit(X[: half], y[: half])
probab_ = clf.predict_proba(X[half:])

precision, recall, thresholds = precision_recall_curve(y[half:],
                                                       probab_[:, 1])

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
plot_data, = ax.plot([], [], lw=2)
threshold_text = ax.text(0.4, 0.2, '')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')


def setup_plot():
    plot_data.set_data([], [])
    threshold_text.set_text('')
    return plot_data, threshold_text


def update(i):
    plot_data.set_data(recall[: i], precision[: i])
    threshold_text.set_text('Threshold = ' + str(thresholds[i]))
    return plot_data,

anim = animation.FuncAnimation(fig, update, init_func=setup_plot,
                               frames=len(thresholds), repeat=False,
                               blit=False)

#plt.plot(recall, precision)
plt.show()
