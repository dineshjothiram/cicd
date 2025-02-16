from model import Model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=45)

# Train model
ml_model = Model()
ml_model.train(X_train, y_train)

# Evaluate and save
accuracy = ml_model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
ml_model.save_model()
