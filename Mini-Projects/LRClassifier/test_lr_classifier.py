from lr_classifier import sigmoid
import numpy as np

#print(type(a))
def test_sigmoid():
    a = np.array(range(5))
    np.testing.assert_allclose(sigmoid(a), np.array([0.5, 0.73105858, 0.88079708, 0.95257413, 0.98201379]))
