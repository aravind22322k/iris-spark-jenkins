import numpy as np
import xgboost as xgb
import pandas as pd

def evaluate_model(data_path, model_path):
    # Load the data
    df = pd.read_parquet(data_path)
    
    # Check the type of the 'features' column to understand the structure
    print(df["features"].head())  # Print the first few rows of the features column

    # Extract the 'values' from the dictionaries in the 'features' column
    X = np.array(df["features"].apply(lambda x: x['values']).tolist(), dtype=np.float32)

    # Extract the labels (assumes the label column is named 'label')
    y = df["label"].values

    # Load the trained model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Predict on the test data
    y_pred = model.predict(X)

    # Evaluate the model (you can add your evaluation metric, e.g., accuracy)
    accuracy = (y_pred == y).mean()
    print(f"Model accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model("data/preprocessed_data.parquet", "models/xgboost_model.json")
