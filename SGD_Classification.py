#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

def plot_sgd_classifier(num_samples, clt_std):
	#generation of data
	X, y = make_blobs(n_samples=num_samples, centers=2, cluster_std=clt_std)

	#fitting of data using logistic regression
	clf = SGDClassifier(loss='log', alpha=0.01)
	clf.fit(X, y)

	#plotting of data
	x_ = np.linspace(min(X[:,0]), max(X[:,0]), 10)
	y_ = np.linspace(min(X[:,1]), max(X[:,1]), 10)

	X_, Y_ = np.meshgrid(x_, y_)
	Z = np.empty(X_.shape)

	for (i, j), val in np.ndenumerate(X_):
		x1 = val
		x2 = Y_[i, j]
		conf_score = clf.decision_function([x1, x2])
		Z[i, j] = conf_score[0]

	levels = [-1.0, 0, 1.0]
	colors = 'k'
	linestyles = ['dashed', 'solid', 'dashed']

	ax = plt.axes()
	ax.contour(X_, Y_, Z, colors=colors, levels=levels, linestyles=linestyles, labels='Boundary')	
	ax.scatter(X[:,0], X[:,1], c=y)

if __name__=='__main__':
	print 'Enter num of samples'
	num_samples = input()
	print 'Enter cluster deviation'
	clt_std = input()
	plot_sgd_classifier(num_samples, clt_std)
	plt.show()
