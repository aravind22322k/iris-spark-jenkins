from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler

def preprocess_data(input_path, output_path):
    spark = SparkSession.builder \
        .appName("Data Preprocessing") \
        .getOrCreate()

    # Load the data
    data = spark.read.parquet(input_path)

    # Update the StringIndexer to use the correct column "class"
    indexer = StringIndexer(inputCol="class", outputCol="label")
    data = indexer.fit(data).transform(data)

    # Assemble the feature vector
    assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        outputCol="features"
    )
    data = assembler.transform(data)

    # Select only the necessary columns
    data = data.select("features", "label")
    
    # Show the preprocessed data
    print("Data Preprocessed:")
    data.show()

    # Write the preprocessed data to Parquet
    data.write.parquet(output_path)

if __name__ == "__main__":
    preprocess_data("data/raw_data.parquet", "data/preprocessed_data.parquet")
