from titanicVotingClassifier import IOHandler, FeatureExtractor, VotingClassifier
from pyspark.sql import SparkSession
import sys

if len(sys.argv) != 5:
  raise Exception("""arguments <trainDatasetPath>, 
                  <testDatasetPath>, 
                  <predictionSubmissionPath>,
                  <tuning> are required""")

trainDatasetPath = sys.argv[1]
testDatasetPath = sys.argv[2]
predictionSubmissionPath = sys.argv[3]
tuning = sys.argv[4]

spark = ( SparkSession.builder
         .appName("titanic-votingclassifier-gcp-dataproc")
        .getOrCreate() )
        
spark.sparkContext.setLogLevel("ERROR")

# data acquisition
train_df = IOHandler.read(spark, trainDatasetPath).cache()
test_df = IOHandler.read(spark, testDatasetPath).cache()

print("data acquisition:")
train_df.printSchema()
train_df.show(5)
test_df.show(5)

# feature extraction
featureExtractor = FeatureExtractor().fit(train_df)
train = featureExtractor.transform(train_df).repartition(200).cache()
test = featureExtractor.transform(test_df).cache()
train_df.unpersist()
test_df.unpersist()

print("feature engineering:")
train.show(5)
test.show(5)

# model training
print("ensemble classifier training:")
votingClf = VotingClassifier().fit(train, {"tuning": tuning})
print("prediction with majority vote of ensemble {mlp, rf, gbt}:")
final_prediction = votingClf.transform(test).cache()
final_prediction.show()
print("store prediction csv into google storage")
IOHandler.write(final_prediction, predictionSubmissionPath)
