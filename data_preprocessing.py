from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler

def preprocess_data(input_path, output_path):
    spark = SparkSession.builder \
        .appName("Data Preprocessing") \
        .getOrCreate()

    data = spark.read.parquet(input_path)

    indexer = StringIndexer(inputCol="species", outputCol="label")
    data = indexer.fit(data).transform(data)

    assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        outputCol="features"
    )
    data = assembler.transform(data)

    data = data.select("features", "label")
    print("Data Preprocessed:")
    data.show()
    data.write.parquet(output_path)

if __name__ == "__main__":
    preprocess_data("data/raw_data.parquet", "data/preprocessed_data.parquet")
