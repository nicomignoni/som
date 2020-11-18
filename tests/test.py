from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

from mapping import SOM

dataset = load_iris()
train = dataset.data

# Reducing the dimensionality of the train set from
# 4 to 2 with PCA
pca       = PCA(n_components=2)
train_pca = pca.fit_transform(train)

parameters = {'n_points'  : 500,
              'alpha0'    : 0.5,
              't_alpha'   : 25,
              'sigma0'    : 2,
              't_sigma'   : 25,
              'epochs'    : 300,
              'seed'      : 124,
              'scale'     : True,
              'shuffle'   : True,
              'history'   : True}

# Load and train the model
model = SOM()
model.set_params(parameters)
model.fit(train_pca)

weights = model.weights

# Plot the train dataset and the weights
fig, ax = plt.subplots()
fig.suptitle("Train set (PCA-reduced) and weights")
t = ax.scatter(train_pca[:,0], train_pca[:,1])
w = ax.scatter(weights[:, 0], weights[:, 1])
fig.legend((t, w), ("Train", "Weights"))
plt.show()
