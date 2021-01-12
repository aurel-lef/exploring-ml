from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *

class IOHandler:
    
    @staticmethod
    def read(spark, path):
        return spark.read.csv(
            path = path, 
            header = True,
            inferSchema = True
        )
    @staticmethod
    def write(df, path):
        return (
            df
            .select(["PassengerId", "Survived"])
            .coalesce(1)
            .orderBy("PassengerId")
            .write.csv(
                path,
                mode = "overwrite",
                header = True
            )
        )

import re
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler, Imputer
from pyspark.ml import Pipeline, Transformer, Estimator, Model, PipelineModel
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import ParamGridBuilder

# Accompanied = binary feature = "has either sibling, spouse, parent or child"
# stateless transformation => use a transformer
class ExtractAccompaniedFeature(Transformer):
    def transform(self, dataset, params=None):
        return dataset.withColumn(
            "Accompanied", 
            (dataset.SibSp + dataset.Parch >= 1).cast(IntegerType()) 
        )

# Imput missing Embarked with most frequent value
# stateful transformation => use an estimator
class HandleMissingEmbarked(Estimator):
    def fit(self, dataset, params=None):
        mostFrequentValue = (dataset.groupby("Embarked")
                             .count()
                             .orderBy("count", ascending=False)
                             .first()
                             .Embarked
                            )
        return HandleMissingEmbarkedModel(mostFrequentValue)
        
class HandleMissingEmbarkedModel(Model):
    
    def __init__(self, mostFrequentValue):
        self.mostFrequentValue = mostFrequentValue
        
    def transform(self, dataset, params=None):
        return dataset.fillna(self.mostFrequentValue, "Embarked")

# TItle regex processing
@udf(returnType=StringType())
def replace_title(s):
    mrs_pattern = "(Mme\.|Ms\.|Countess\.|Lady\.)"
    miss_pattern = "(Mlle\.)"
    mr_pattern = "(Don\.|Major\.|Sir\.|Col\.|Capt\.)"
    if re.search(mrs_pattern, s):
        return re.sub(mrs_pattern, "Mrs.", s)
    if re.search(miss_pattern, s):
        return re.sub(miss_pattern, "Miss.", s)
    if re.search(mr_pattern, s):
        return re.sub(mr_pattern, "Mr.", s)
    return s


@udf
def replace_empty(s):
    if s == "":
        return "No-Title"
    return s

# extraction of Title feature from Name
class ExtractTitle(Transformer):
    def transform(self, dataset, params=None):
        titles_extract_pattern = r'(Mr\.|Mrs\.|Miss\.|Master\.|Dr\.|Rev\.)'
        return ( dataset.withColumn("Title", regexp_extract("Name", titles_extract_pattern, 1))
                .withColumn("Title", replace_empty("Title"))
               )
    
# Imputing missing "Age" from regression of Pclass and Sex
class HandleMissingAge(Estimator):
    def __init__(self):
        vect = VectorAssembler(
            inputCols = ["Pclass_encoded", "Sex_encoded"], 
            outputCol='features_class_sex'
        )
        
        lr = LinearRegression(
            featuresCol="features_class_sex",
            labelCol='Age',
            predictionCol='Age_imputed',
            regParam = 0.3
        )

        self.pipe = Pipeline(
            stages = [
                vect,
                lr
            ])
        self._params = None
        self._paramMap = ParamGridBuilder().build()


    def fit(self, dataset, params=None):
        dataset_without_missing = dataset.where(col("Age").isNotNull())
        ageRegressor = self.pipe.fit(dataset_without_missing)
        return HandleMissingAgeModel(ageRegressor)

    
class HandleMissingAgeModel(Model):
    
    def __init__(self, ageRegressor):
        self.ageRegressor = ageRegressor
        
    def transform(self, dataset, params=None):
        null_age_df = dataset.where(col("Age").isNull())
        not_null_age_df = dataset.where(col("Age").isNotNull())
              
        null_age_df = (
            self.ageRegressor
            .transform(null_age_df)
            .drop("features_class_sex")
            .cache()
        )
            
        return null_age_df.union(
            not_null_age_df.withColumn("Age_imputed", col("Age"))
        )
    
class FeatureExtractor(Estimator):
        
    def fit(self, dataset, params=None):
        extract_features = Pipeline(
            stages = [
                ExtractAccompaniedFeature(),
                ExtractTitle(),
                HandleMissingEmbarked(),
                
                # need to get Dummy categorisation of Pclass & Sex for Age Imputation
                StringIndexer(inputCol = "Pclass", outputCol='Pclass_indexed', handleInvalid='keep'),
                StringIndexer(inputCol = "Sex", outputCol='Sex_indexed', handleInvalid='keep'),
                OneHotEncoder(inputCol = "Pclass_indexed", outputCol='Pclass_encoded', handleInvalid='keep'),
                OneHotEncoder(inputCol = "Sex_indexed", outputCol='Sex_encoded', handleInvalid='keep'),
                
                HandleMissingAge(),
                Imputer(inputCol = "Fare", outputCol='Fare_imputed')       
        ])
        pipelineModel = extract_features.fit(dataset)
        return FeatureExtractorModel(pipelineModel)

class FeatureExtractorModel(PipelineModel):
    
    def __init__(self, pipelineModel):
        self.pipelineModel = pipelineModel
        
    def transform(self, dataset, params=None):
        keep_features = ["PassengerId", "Survived", "Pclass_encoded", "Title", "Sex_encoded", 
                 "Age_imputed", "Fare_imputed", "Embarked", "Accompanied"]
        return ( self.pipelineModel.transform(dataset)
                .transform(lambda df: df.select([col for col in df.columns if col in keep_features]))
        )

from pyspark.ml.classification import GBTClassifier, RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise Exception('Boolean value expected.')
        
class VotingClassifier(Estimator):
    
    def fit(self, dataset, params=None):
        preprocess = Pipeline(
            stages = [
                StringIndexer(inputCol = "Embarked", outputCol='Embarked_indexed', handleInvalid='keep'),
                StringIndexer(inputCol = "Title", outputCol='Title_indexed', handleInvalid='keep'),
                OneHotEncoder(inputCol = "Embarked_indexed", outputCol='Embarked_encoded', handleInvalid='keep'),
                OneHotEncoder(inputCol = "Accompanied", outputCol='Accompanied_encoded', handleInvalid='keep'),
                OneHotEncoder(inputCol = "Title_indexed", outputCol='Title_encoded', handleInvalid='keep'),
                VectorAssembler(inputCols = ["Fare_imputed"], outputCol='Fare_vect'),
                StandardScaler(withMean = True, inputCol = "Fare_vect", outputCol='Fare_std'),
                VectorAssembler(inputCols = ["Age_imputed"], outputCol='Age_vect'),
                StandardScaler(withMean = True, inputCol = "Age_vect", outputCol='Age_std'),        
        ])
        
        inputCols = [
            "Pclass_encoded", 
            "Sex_encoded", 
            "Embarked_encoded", 
            "Accompanied_encoded", 
            "Title_encoded",
            "Age_std", 
            "Fare_std"]
        
        gbt = GBTClassifier(
            featuresCol='features',
            labelCol = "Survived")
        
        rf = RandomForestClassifier(
            featuresCol='features',
            labelCol = "Survived")
        
        va = VectorAssembler(
            inputCols = inputCols,
            outputCol='features'
        )
        
        # here we have 23 inputs features corresponding to the 7 features in inputCols which are already encoded
        # can be checked with: 
        # transformed = pipe_preprocessing.fit(train_df).transform(train_df)
        # va.transform(transformed).schema["features"].metadata["ml_attr"]
        layers = [23, 100, 2]
        
        mlp = MultilayerPerceptronClassifier(
            featuresCol='features',
            labelCol = "Survived",
            solver = 'gd',
            layers=layers
        )
        
        mlp_paramGrid = (
            ParamGridBuilder()
            .addGrid(mlp.layers, [[23, 100, 2], [23, 50, 50, 2]])
            # .addGrid(mlp.maxIter, [100, 150, 200])
            .addGrid(mlp.stepSize, [0.01, 0.03, 0.05])
            #.addGrid(mlp.solver, ['l-bfgs', 'gd'])
            # .addGrid(mlp.blockSize, [32, 64, 128])
            .build()
        )

        rf_paramGrid = (
            ParamGridBuilder()
            # .addGrid(rf.numTrees, [20, 50, 100, 200])
            .addGrid(rf.maxDepth, list(range(3,10)))
            .addGrid(rf.featureSubsetStrategy, ['0.25', '0.5', '0.75', '1.0'])
            .build()
        )

        gbt_paramGrid = (
            ParamGridBuilder()
            .addGrid(gbt.stepSize, [0.05, 0.1, 0.3])
            .addGrid(gbt.maxDepth, list(range(3,10)))
            # .addGrid(gbt.maxIter, [20, 50, 100, 150, 200, 300])
            .build()
        )
        
        paramGrids = [ParamGridBuilder().build()]*3 # no parameter tuning
        if (params and ("tuning" in params.keys()) and str2bool(params["tuning"])):
            paramGrids = [ gbt_paramGrid, rf_paramGrid, mlp_paramGrid]
        
        trainedModels = dict()
        for classifier, classifier_name, paramGrid in zip( 
                [ gbt, rf , mlp], 
                [ "gbt", "rf", "mlp"],
                paramGrids
                ):

            clf = Pipeline(
                stages = [
                preprocess,
                va,
                classifier
            ])
        
            # Yes, accuracy is rather a poor evaluation metric
            # but it is the metric defined in the Titanic kaggle competition
            # the only way to get an accuracy evaluator seems to use MulticlassClassificationEvaluator
            evaluator = MulticlassClassificationEvaluator(labelCol="Survived", metricName="accuracy")
        
            cv = CrossValidator(
                estimator=clf, 
                estimatorParamMaps=paramGrid, 
                evaluator=evaluator, 
                numFolds=3,
                parallelism = 2)
        
            cv_model = cv.fit(dataset)
            trainedModels[classifier_name] = cv_model.bestModel
            print(f"[{classifier_name}] avg accuracy:\n{cv_model.avgMetrics[0]}\n")
        return VotingClassifierModel(trainedModels)

import numpy as np

@udf(returnType = IntegerType())
def majority_vote_prediction(predictions):
    return int(np.mean(predictions) >= 0.5)

@udf(returnType = DoubleType())
def majority_vote_proba(predictions, probas):
    prediction = int(np.mean(predictions) >= 0.5)
    if prediction == 0:
        # high probability for the class [Survived == 0]
        return float(np.max(probas))
    else:
        return float(np.min(probas))
    
class VotingClassifierModel(Model):
    def __init__(self, fittedModelsDict):
        self.fittedModelsDict = fittedModelsDict
    
    def transform(self, dataset, params=None):
        all_predictions_df = None
        for model_name, model in self.fittedModelsDict.items():
            prediction = (model.transform(dataset)
                          .select("PassengerId", "prediction", "probability")
                          .withColumnRenamed("prediction", f"{model_name}_prediction")
                          .withColumnRenamed("probability", f"{model_name}_probability")
                         )
            # merge all model predictions in a single dataframe
            if not all_predictions_df:
                all_predictions_df = prediction
            else:
                all_predictions_df = all_predictions_df.join(prediction, "PassengerId")
        
        
        predictionCols = [col for col in all_predictions_df.columns if "_prediction" in col]
        probabilityCols = [col for col in all_predictions_df.columns if "_probability" in col]
        return (
            all_predictions_df
            .withColumn(
                "prediction", 
                majority_vote_prediction(array([col(c) for c in predictionCols])))
            .withColumn(
                "probability", 
                majority_vote_proba(array([col(c) for c in predictionCols]), array([col(c) for c in probabilityCols])))
            .withColumnRenamed("prediction", "Survived")
        )