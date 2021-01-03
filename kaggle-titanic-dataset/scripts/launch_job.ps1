.\scripts\titanic_pyspark_hadoop_gcp.ps1 `
-project_id "explore-ml-pyspark" `
-region "us-central1" `
-bucket "gs://explore-ml-files" `
-trainDatasetLocalPath .\data\train.csv `
-testDatasetLocalPath .\data\test.csv `
-cluster "pyspark-test" `
-scriptPath C:\Users\aurel\dev\repositories\exploring-ml\kaggle-titanic-dataset\scripts\pyspark-test.py