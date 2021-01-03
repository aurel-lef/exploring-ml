import sys
from pyspark.sql import SparkSession

if len(sys.argv) != 2:
  raise Exception("argument <datasetPath> is required")

datasetFilePath = sys.argv[1]


spark = SparkSession.builder \
    .appName("test") \
    .getOrCreate()

#path = "C:/Users/aurel/dev/repositories/exploring-ml/kaggle-titanic-dataset/data/train.csv"
spark.read.csv(datasetFilePath, header=True, inferSchema = True).show()