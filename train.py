from pyspark.sql import SparkSession
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model(input_path, model_output_path):
    spark = SparkSession.builder \
        .appName("Model Training") \
        .getOrCreate()

    data = spark.read.parquet(input_path)
    df = data.toPandas()

    X = pd.DataFrame(df['features'].tolist(), index=df.index)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2f}")

    model.save_model(model_output_path)

if __name__ == "__main__":
    train_model("data/preprocessed_data.parquet", "models/xgboost_model.json")
