from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import joblib

def evaluate_model(data_path, model_path):
    # Load data
    df = pd.read_parquet(data_path)
    X = pd.DataFrame(df['features'].tolist(), index=df.index)
    y = df['label']

    # Load model
    model = XGBClassifier()
    model.load_model(model_path)

    # Predict
    y_pred = model.predict(X)
    print("Classification Report:")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate_model("data/preprocessed_data.parquet", "models/xgboost_model.json")
