import os
import joblib  # Or pickle

def load_model():
    path = os.path.join(os.path.dirname(__file__), 'bearing_fault_model.pkl')
    return joblib.load(path)
def load_scaler():
    path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    return joblib.load(path)
