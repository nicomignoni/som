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

$$
w_{ij}(t) = w_{ij}(t) + \alpha(t)h(t) \|x_{ci} - w_{ij}(t)\|
$$

where

$$
\alpha(t) = \alpha_0 \exp\left( -\frac{t}{t_{\alpha}}\right), \ h(t) = \exp\left( -\frac{\|w_c - w_i\|^2}{2\sigma^2(t)} \right), \ \sigma(t) = \sigma_0 \exp\left( -\frac{t}{t_{\sigma}} \right) 
$$

and $t$ is the current epoch.

Also, each neuron is connected to all the other ones, hence the map is a $K_p$ complete graph, where $p$ is the number of neurons. 

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
