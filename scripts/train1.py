import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.ml import feature

# Create a Spark session
spark = SparkSession.builder.appName("XGBoost Training").getOrCreate()

# Load the preprocessed data
df = spark.read.parquet("data/preprocessed_data.parquet")

# Convert the 'features' column to a list of arrays
from pyspark.ml.linalg import Vectors

def to_array(v):
    if isinstance(v, Vectors):
        return v.toArray()
    return v

# Convert Spark dataframe to Pandas dataframe
pandas_df = df.toPandas()

# Apply the conversion to the 'features' column
X = pandas_df["features"].apply(to_array).to_list()

# Get the labels
y = pandas_df["label"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Save the trained model
model.save_model("models/xgboost_model.json")
