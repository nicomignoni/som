'''Mock test.'''
import unittest

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from som.mapping import SOM

class SOMTest(unittest.TestCase):

    def test_mapping(self):
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

        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()

