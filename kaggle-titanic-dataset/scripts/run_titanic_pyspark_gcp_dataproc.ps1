.\scripts\titanic_pyspark_gcp_dataproc.ps1 `
-projectId "explore-ml-pyspark" `
-region "us-central1" `
-bucket "gs://explore-ml-files" `
-trainDatasetLocalPath .\data\train.csv `
-testDatasetLocalPath .\data\test.csv `
-cluster "pyspark-test" `
-scriptPath .\scripts\run_pyspark_titanic.py `
-modulePaths .\scripts\titanicVotingClassifier.py `
-hyperparameterTuning $true 