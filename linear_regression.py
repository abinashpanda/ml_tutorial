#!/usr/bin/env python

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge

import numpy as np
import matplotlib.pyplot as plt

def data_generator(num_samples):
	X = np.random.rand(num_samples, 1)*10
	y = np.sin(X) + np.random.randn(num_samples, 1)*0.1
	return X, y

def data_generator_noiseless(num_samples):
	X = np.random.rand(num_samples, 1)*10
	y = np.sin(X)
	return X, y

def feature_generator(X, order):
	feature_matrix = []
	for val in map(float, X):
		feature_matrix.append([val**power for power in range(order+1)])
	feature_matrix = np.array(feature_matrix)
	return feature_matrix

def plot_(clf, feature_generator, order, range_X_min, range_X_max, color, label_):
	X_plot = np.arange(range_X_min, range_X_max, 0.01)
	feature_matrix = feature_generator(X_plot, order)
	y_plot = clf.predict(feature_matrix)
	return plt.plot(X_plot, y_plot, color, label=label_)

if __name__ == '__main__':
	
	print 'Enter num of samples'
	num_samples = input()
	print 'Order of the polynomial'
	order = input()

	#X, y = data_generator_noiseless(num_samples)
	X, y = data_generator(num_samples)

	clf = Ridge(alpha=0.5)
	clf.fit(feature_generator(X, order), y)
	
	clf_biased = LinearRegression()
	clf_biased.fit(feature_generator(X, order), y)

	#clf_bayesian = BayesianRidge()
	#clf_bayesian.fit(feature_generator(X, order), y)

	plt.scatter(X, y, c='r', label='Actual Data')
	plt.xlabel('X')
	plt.ylabel('y')
	plt.title('Linear Regression')
	plot_(clf, feature_generator, order, np.min(X), np.max(X), 'm', 'Ridge Regression')
	plot_(clf_biased, feature_generator, order, np.min(X), np.max(X), 'c', 'Linear Regression without Regularization')
	#plot_(clf_bayesian, feature_generator, order, np.min(X), np.max(X), 'k', 'Bayesian Ridge Regression')	
	plt.plot(np.arange(np.min(X), np.max(X), 0.01), np.sin(np.arange(np.min(X), np.max(X), 0.01)), 'g', label='Target Function')
	plt.legend(loc='best')
	plt.show()
