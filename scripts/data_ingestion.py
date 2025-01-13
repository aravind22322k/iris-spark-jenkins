from pyspark.sql import SparkSession

def ingest_data(input_path):
    spark = SparkSession.builder \
        .appName("Data Ingestion") \
        .getOrCreate()

    data = spark.read.csv(input_path, header=True, inferSchema=True)
    print("Data Ingested:")
    data.show()
    return data

if __name__ == "__main__":
    input_path = "data/iris.csv"  # Replace with the actual path
    data = ingest_data(input_path)
    data.write.parquet("data/raw_data.parquet")
