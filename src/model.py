import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Model:
    def __init__(self, n_estimators=100, random_state=42):
        """Initialize the Random Forest model."""
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def save_model(self, file_path="model.pkl"):
        """Save the trained model to a file."""
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load_model(file_path="model.pkl"):
        """Load a saved model from a file."""
        with open(file_path, "rb") as f:
            return pickle.load(f)

# Run training if script is executed directly
if __name__ == "__main__":
    # Load dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Train and evaluate model
    ml_model = Model()
    ml_model.train(X_train, y_train)
    accuracy = ml_model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    # Save trained model
    ml_model.save_model()
