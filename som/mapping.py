import torch
from math import exp
from tqdm import trange

class SOM:
    def __init__(self, n_points=500, alpha0=0.5, t_alpha=25,
                 sigma0=2, t_sigma=25, epochs=300, seed=42,
                 shuffle=True, scale=True, history=False):
        
        self.params = {'n_points' : n_points, # The number of points/weights
                       'alpha0'   : alpha0,   # Initial learning rate
                       't_alpha'  : t_alpha,  # Exponential reduction constant (alpha)
                       'sigma0'   : sigma0,   # Initial neighborhood strenght 
                       't_sigma'  : t_sigma,  # Exponential reduction constant (sigma)
                       'epochs'   : epochs,   # Number of iteration
                       'seed'     : seed,     # Random seed
                       'scale'    : scale,    # Make max(W) = max(X) and min(W) = min(X)
                       'history'  : history,  # Save all the Ws (debugging)
                       'shuffle'  : shuffle}  # Shuffle the train set
        
        if self.params['seed']:
            torch.manual_seed(seed)
    
    def fit(self, X):
        n_samples, n_attributes = X.shape

        # Initialize the points randomly (weights)
        self.W = torch.rand((self.params['n_points'], n_attributes),
                             dtype=torch.double)

        # From numpy conversion
        X = torch.from_numpy(X).type(torch.double)
        
        # Shuffling
        if self.params['shuffle']:
            indices = torch.randperm(n_samples)
            X = X[indices, :]

        # Scaling W in the same range as X
        if self.params['scale']:
            self.W = self.W*(torch.max(X) - torch.min(X)) + torch.min(X)

        # Record each W for each t (debugging)
        if self.params['history']:
            self.history = self.W.reshape(1, self.params['n_points'], n_attributes)      

        # The training loop
        for t in trange(self.params['epochs']):
            x       = X[t % n_samples, :] # The current sampled x
            x_dists = x - self.W          # Distances from x to W
                    
            # Find the winning point
            distances = torch.pow((x_dists), 2).sum(axis=1) # [n_points x 1]
            winner    = torch.argmin(distances)

            # Lateral distance between neurons
            lat_dist = torch.pow((self.W - self.W[winner, :]), 2).sum(axis=1) # [n_points x 1]

            # Update the learning rate
            alpha = self.params['alpha0']*exp(-t/self.params['t_alpha']) # [scalar]

            # Update the neighborhood size
            sigma = self.params['sigma0']*exp(-t/self.params['t_sigma']) # [scalar]

            # Evaluate the topological neighborhood
            h = torch.exp(-lat_dist/(2*sigma**2)).reshape((self.params['n_points'], 1)) # [n_points x 1]

            # Update W
            self.W += alpha*h*(x_dists)

            if self.params['history']:
                self.history = torch.cat((self.history,
                                          self.W.reshape(1, self.params['n_points'], n_attributes)),
                                         axis=0)

    def adjacency_matrix(self, M):
        M = torch.from_numpy(M)
        n_samples, n_attributes = M.shape
        
        # Broadcast the M matrix to a tensor of the shape
        # (n_sample, (n_samples, n_attributes))

        tensor    = M.repeat(n_samples, 1, 1)              # Make n_samples copy of M
        M_flat    = M.reshape(n_samples, 1, n_attributes)  # Each row of M or each copy in tensor
        distances = torch.pow((tensor - M_flat), 2).sum(axis=2).sqrt()
        return distances
        
    def set_params(self, params):
        self.params = {param: value for param, value in params.items()}

    def get_params(self):
        return self.params
  
    def get_weights(self):
        return self.W
