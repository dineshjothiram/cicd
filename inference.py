import numpy as np
from model import Model

# Load trained model
model = Model.load_model()

# Predict function
def predict(features):
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]

if __name__ == "__main__":
    sample_features = [5.1, 3.5, 1.4, 0.2]  # Example input
    prediction = predict(sample_features)
    print(f"Predicted Class: {prediction}")
