# Titanic dataset

Yes, famous dataset!  
Great opportunity to make a deep dive in feature engineering and supervised model training.  

## Notebook exploration  

I opted here for an majority vote ensemble of fitted classifiers : { Random Forest, Gradient Boosted Tree, MultiLayer Perceptron}, both in [scikit-learn](./notebooks/titanic-scikit-learn.ipynb) and [Pyspark](./notebooks/titanic-pyspark.ipynb) notebooks.  

## Train and predict on cloud ressources  

Once I finalized the pyspark analysis, I also ran it on a Spark/hadoop cluster on google cloud dataproc. 

The below script will:
- deploy the train/test datasets in google storage
- create a dataproc cluster of N workers using a preview image [Python 3.8, Miniconda3 4.9.0, Spark 3.0.1]
- set "protective" criteria of shutdown cluster (max-idle/max-age) to avoid cloud money waste
- launch the job spark, that is the training of the majority vote ensemble followed by the prediction of the test set stored in gs
- deletion of the cluster


```powershell
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
``` 




