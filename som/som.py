import numpy as np
import matplotlib.pyplot as plt
import time
from math import exp, log

class SOM:
    def __init__(self, x, y, train_set, epochs, learning_rate):
        self._w = np.random.rand(x, y, len(train_set[0]))

        # Normalize train data
        for i in range(train_set.shape[0]):
            train_set[i]/max(train_set[i])
            
        self._train_set = train_set
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._sigma = max(x, y)/2
        self._lambda = self._epochs/log(self._sigma)
        self._step = 0
    
    def distance(self, vector_1, vector_2):
        return np.linalg.norm(vector_1 - vector_2)

    def train(self):
        while self._step <= self._epochs:
            # Update the learning rate
            learning_rate = self._learning_rate * exp(-self._step/self._lambda)

            # Update the radius  
            sigma = self._sigma * exp(-self._step/self._lambda)

            start = time.time()
            
            for vector in self._train_set:
                
                # Evaluate the distance between the input vector and every weight node
                vector_weight_distances = np.apply_along_axis(lambda x: self.distance(x,vector),
                                                              2, self._w)
                
                # Select the winner position (the node closer to the vector)
                winner_pos = np.unravel_index(np.argmin(vector_weight_distances),
                                              vector_weight_distances.shape)

                # Evaluate the distance between the winner and every weight node
                winner = self._w[winner_pos, :]
                winner_weight_distances = np.apply_along_axis(lambda x: self.distance(x,winner),
                                                              2, self._w)

                # Evaluate the neighborhood function
                h = learning_rate * np.exp(-winner_weight_distances**2/(2*sigma**2))

                # Update weights
                h_tensor = np.repeat(h[:,:,np.newaxis], len(vector), axis=2)
                vector_tensor = np.tile(vector, list(self._w.shape[:2])+[1])
                self._w += np.multiply(h_tensor, (vector_tensor - self._w))
                
            end = time.time()

            print('Epoch {} completed, time elapsed {} sec'.format(self._step + 1, '%.2f'%(end-start)))

            # Next step
            self._step += 1

    def predictions(self):
        # Each cell of the pred matrix represent how many time that node has benn the winner node
        # with respect to the entire dataset.
        pred_matrix = np.zeros(self._w.shape[:2])
        for vector in self._train_set:
            # Evaluate the distance between the input vector and every weight node
            vector_weight_distances = np.apply_along_axis(lambda x: self.distance(x,vector),
                                                          2, self._w)
            
            # Select the winner position (the node closer to the vector)
            winner_pos = np.unravel_index(np.argmin(vector_weight_distances),
                                              vector_weight_distances.shape)
            
            # Increase by 1 the cell in the pred_matrix corrisponding to the winner position
            pred_matrix[winner_pos] += 1

        plt.imshow(pred_matrix, interpolation='nearest')
        plt.colorbar()
        plt.title('Prediction Matrix')
        plt.show()

            
    










    
