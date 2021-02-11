# Kohonen's Self-Organizing Map (SOM)

[![PyPI version shields.io](https://img.shields.io/pypi/v/kohonen-som.svg)](https://pypi.python.org/pypi/kohonen-som/)

![gif](https://github.com/nicomignoni/SOM/blob/master/animation/animation.gif)

## Installation
```
pip install kohonen-som
```

## Background

The original [paper](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1990-Kohonen-PIEEE.pdf) written by Teuvo Kohonen in 1990 was one of the first neural network model capable of unsupervised learning.

Out of the different implementations of the algorithm, this one follows almost entirely the original paper. The update function is defined as


<div align="center"><img src="https://latex.codecogs.com/gif.latex?w_%7Bij%7D%28t%29%20%3D%20w_%7Bij%7D%28t%29%20&plus;%20%5Calpha%28t%29%20%5Ccdot%20h%28t%29%20%5Ccdot%20%7C%7Cx_%7Bci%7D%20-%20w_%7Bij%7D%28t%29%7C%7C"></div>

where

<div align="center"><img src=https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cbegin%7Balign*%7D%20%5Cdisplaystyle%20%5Calpha%28t%29%20%3D%20%5Calpha_0%20%5Ccdot%20%5Ctext%7Bexp%7D%5Cleft%28%20-%5Cfrac%7Bt%7D%7B%5Ctau_%7B%5Calpha%7D%7D%20%5Cright%29%2C%20%5C%20h%28t%29%20%3D%20%5Ctext%7Bexp%7D%5Cleft%28-%5Cfrac%7B%7C%7Cw_%7Bc%7D%20-%20w_%7Bi%7D%7C%7C%5E2%7D%7B2%5Csigma%28t%29%5E2%7D%20%5Cright%29%2C%20%5C%20%5Csigma%28t%29%20%3D%20%5Csigma_0%20%5Ccdot%5Ctext%7Bexp%7D%20%5Cleft%28%20-%5Cfrac%7Bt%7D%7B%5Ctau_%7B%5Csigma%7D%7D%20%5Cright%29%20%5Cend%7Balign*%7D></div>

and ![equation](https://latex.codecogs.com/gif.latex?%5Cinline%20t) is the current epoch.

Also, each neuron is connected to all the other ones, hence the map is a ![equation](https://latex.codecogs.com/gif.latex?K_p) complete graph, where ![equation](https://latex.codecogs.com/gif.latex?p) is the number of neurons. 

## Example
```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

from som.mapping import SOM

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

weights = model.get_weights()

# Plot the train dataset and the weights
fig, ax = plt.subplots()
fig.suptitle("Train set (PCA-reduced) and weights")
t = ax.scatter(train_pca[:,0], train_pca[:,1])
w = ax.scatter(weights[:, 0], weights[:, 1])
fig.legend((t, w), ("Train", "Weights"))
plt.show()


```
