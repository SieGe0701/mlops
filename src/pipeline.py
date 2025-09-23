import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Example: Load data
def load_data(path):
    return pd.read_csv(path)

# Example: Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Example: Evaluate model
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

if __name__ == "__main__":
    # Dummy pipeline usage
    print("This is a template for your ML pipeline.")
