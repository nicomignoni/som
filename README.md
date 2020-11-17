# Kohonen's Self-Organizing Map (SOM)

The original [paper](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1990-Kohonen-PIEEE.pdf) written by Teuvo Kohonen in 1990 was one of the first neural network model capable on unsupervised learning, training a set on initially randomized weight vectors.

Out of the different implementation of the algorithm, this one follows almost entirely the original paper. The update function is defined as


<div align="center"><img src="https://latex.codecogs.com/gif.latex?w_%7Bij%7D%28t%29%20%3D%20w_%7Bij%7D%28t%29%20&plus;%20%5Calpha%28t%29%20%5Ccdot%20h%28t%29%20%5Ccdot%20%7C%7Cx_%7Bci%7D%20-%20w_%7Bij%7D%28t%29%7C%7C"></div>


where ![equation](https://latex.codecogs.com/gif.latex?w_%7Bij%7D) is the ![equation](https://latex.codecogs.com/gif.latex?ij)-th entry of the weight matrix ![equation](https://latex.codecogs.com/gif.latex?W%28t%29), ![equation](https://latex.codecogs.com/gif.latex?t) is the current epoch, ![equation](https://latex.codecogs.com/gif.latex?%5Calpha%28t%29) the learning rate, ![equation](https://latex.codecogs.com/gif.latex?h%28t%29) is the neighborhood function, ![equation](https://latex.codecogs.com/gif.latex?x_c) the current randomly selected training vector.
