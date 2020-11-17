# Kohonen's Self-Organizing Map (SOM)

The original [paper](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1990-Kohonen-PIEEE.pdf) written by Teuvo Kohonen in 1990 was one of the first neural network model capable on unsupervised learning, training a set on initially randomized weight vectors.

Out of the different implementation of the algorithm, this one follows almost entirely the original paper. The update function is defined as


<div align="center"><img src="https://latex.codecogs.com/gif.latex?w_%7Bij%7D%28t%29%20%3D%20w_%7Bij%7D%28t%29%20&plus;%20%5Calpha%28t%29%20%5Ccdot%20h%28t%29%20%5Ccdot%20%7C%7Cx_%7Bci%7D%20-%20w_%7Bij%7D%28t%29%7C%7C"></div>

where

<div align="center"><img src="https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cbegin%7Balign*%7D%20%5Cdisplaystyle%20%5Calpha%28t%29%20%3D%20%5Calpha_0%20%5Ccdot%20%5Ctext%7Bexp%7D%5Cleft%28%20-%5Cfrac%7Bt%7D%7B%5Ctau_%7B%5Calpha%7D%7D%20%5Cright%29%2C%20%5Cquad%20h%28t%29%20%3D%20%5Ctext%7Bexp%7D%5Cleft%28-%5Cfrac%7B%7C%7Cw_%7Bc%7D%20-%20w_%7Bi%7D%7C%7C%5E2%7D%7B2%5Csigma%28t%29%5E2%7D%20%5Cright%29%2C%20%5C%5C%20%5Csigma%28t%29%20%3D%20%5Csigma_0%20%5Ccdot%5Ctext%7Bexp%7D%20%5Cleft%28%20-%5Cfrac%7Bt%7D%7B%5Ctau_%7B%5Csigma%7D%7D%20%5Cright%29%20%5Cend%7Balign*%7D"></div>
