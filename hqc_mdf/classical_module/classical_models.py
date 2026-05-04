from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import joblib
import os

def train_classical_models(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates multiple classical models for benchmarking.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    
    detailed_results = {}
    
    if not os.path.exists("models"):
        os.makedirs("models")

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test).tolist() # Convert to list for easier handling
        end_time = time.time()
        
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, zero_division=0)
        rec = recall_score(y_test, predictions, zero_division=0)
        exec_time = end_time - start_time
        
        detailed_results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Execution Time": exec_time,
            "Predictions": predictions
        }
        
        # Save model
        model_path = f"models/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)
        
    return detailed_results, models
