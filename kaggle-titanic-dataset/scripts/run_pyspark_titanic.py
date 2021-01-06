from titanicVotingClassifier import IOHandler, FeatureExtractor, VotingClassifier
from pyspark.sql import SparkSession
import sys

print("python version = {}".format(sys.version))

if len(sys.argv) != 4:
  raise Exception("arguments <trainDatasetPath>, <testDatasetPath>, <predictionSubmissionPath> are required")

trainDatasetPath = sys.argv[1]
testDatasetPath = sys.argv[2]
predictionSubmissionPath = sys.argv[3]

spark = SparkSession.builder \
        .appName("titanic-votingclassifier-gcp-dataproc") \
        .getOrCreate()

def quiet_logs(spark):
    logger = spark.sparkContext._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    
quiet_logs(spark)

# data acquisition
train_df = IOHandler.read(spark, trainDatasetPath)
test_df = IOHandler.read(spark, testDatasetPath)

print("data acquisition:")
train_df.printSchema()
train_df.show(5)
test_df.show(5)

# feature extraction
featureExtractor = FeatureExtractor().fit(train_df)
train = featureExtractor.transform(train_df).cache()
test = featureExtractor.transform(test_df).cache()

print("feature engineering:")
train.show(5)
test.show(5)

# model training
print("ensemble classifier training:")
votingClf = VotingClassifier().fit(train)
print("prediction with majority vote of ensemble {mlp, rf, gbt}:")
final_prediction = votingClf.transform(test).cache()
final_prediction.show()
print("store prediction csv into google storage")
IOHandler.write(final_prediction, predictionSubmissionPath)
