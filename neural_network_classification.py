#!/usr/bin/env python

"""Naval Fate.

Usage:
  neural_network_classification.py num_samples <samples> max_iters <iters> hidden_layers <layers>...
  neural_network_classification.py (-h | --help)

Options:
  -h --help     Show this screen.

"""

from docopt import docopt

from sklearn.datasets import make_blobs

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer 

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

arguments = docopt(__doc__)

num_hidden_layers = map(int, arguments['<layers>'])
max_iters = int(arguments['<iters>'])

X1, y1 = make_blobs(n_samples=int(arguments['<samples>'])/2, centers=2, \
				  cluster_std=0.6)
X2, y2 = make_blobs(n_samples=int(arguments['<samples>'])/2, centers=2, \
				  cluster_std=0.6)

X = np.concatenate((X1, X2))
y = np.concatenate((y1, y2))

m,n = X.shape

dataset = ClassificationDataSet(n, 1, nb_classes=2) 
for i in range(m):
	dataset.addSample(X[i], y[i])

tst_data, trn_data = dataset.splitWithProportion(0.25)

tst_data._convertToOneOfMany()
trn_data._convertToOneOfMany()

layers = [trn_data.indim]
layers += num_hidden_layers
layers += [trn_data.outdim]

neural_network = buildNetwork(*layers, outclass=SoftmaxLayer)
trainer = BackpropTrainer(neural_network, dataset=trn_data, verbose=False, \
						  weightdecay=0.01, momentum=0.1)


fig = plt.figure()
fig.set_size_inches(15, 15)

ax_data = plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Input Data') 

ax = plt.subplot(1, 2, 2)
scat = ax.scatter(tst_data['input'][:, 0], tst_data['input'][:, 1], s=60)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Classification of Data using Neural Network')

iter_text = ax.text(0, 0, '')

def setup_plot():
	iter_text.set_text('')
	return scat,

def update(i):
	trainer.trainEpochs(1)
	out = neural_network.activateOnDataset(tst_data)
	out = out.argmax(axis=1)
	scat.set_array(out)
	iter_text.set_text('Iteration = '+str(i+1))
	return scat,

anim = animation.FuncAnimation(fig, update, init_func=setup_plot,
							   frames=max_iters, repeat=False )

plt.show()
