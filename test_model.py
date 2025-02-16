import unittest
from src.inference import predict

class TestModel(unittest.TestCase):
    def test_prediction(self):
        sample_input = [5.1, 3.5, 1.4, 0.2]
        prediction = predict(sample_input)
        self.assertIn(prediction, [0, 1, 2])  # Expected output classes

if __name__ == '__main__':
    unittest.main()
