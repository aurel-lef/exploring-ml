# this code has been produced in the "titanic-scikit-learn.ipynb" notebook


from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
import re

class AbstractModel(ABC):

    @abstractmethod
    def fit(self, train_df: pd.DataFrame):
        ...

    @abstractmethod
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        ...





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


def extract_titles(df):
    titles_extract_pattern = r'(?P<Name>Mr\.|Mrs\.|Miss\.|Master\.|Dr\.|Rev\.)'
    df["Title"] = (df.Name
                   .transform(replace_title)
                   .str
                   .extract(titles_extract_pattern)
                   .fillna("No-Title.")
                   )  ## in case we find other Name patterns in the test set
    return df


def extract_accompanied_feature(arr):
    return (
        np.where(arr.sum(axis=1) >= 1, 1, 0)
            .reshape(-1, 1)
    )


preprocess = Pipeline([

    # extracting title from Name
    ("title", FunctionTransformer(extract_titles)),

    # Will need Pclass and Sex Features in order to regress Age missing values
    # encode them first
    ("encode_class_sex", ColumnTransformer([
        # keep only useful features, that is dropping Survived, PassengerId, Ticket, Cabin
        ("passthrough", "passthrough", ["Title", "SibSp", "Parch", "Fare", "Embarked", "Age"]),
        # encode Pclass and Sex
        ("encode_class_sex", OneHotEncoder(handle_unknown="ignore"), ["Pclass", "Sex"])
    ], n_jobs=-1)),

    # new column order : "Title", "SibSp", "Parch", "Fare", "Embarked","Age", "Pclass", "Sex"

    ("union", ColumnTransformer([

        # keep Title
        ("passthrough", "passthrough", [0]),

        # creating binary feature from SibSp and Parch
        ("accompanied", FunctionTransformer(extract_accompanied_feature), [1, 2]),

        # handling missing NA values for Embarked feature (col 4) with 'most-frequent' values
        # Note: SimpleImputer only works with 2D arrays, so workaround by adding Fare (col 3)
        ("embarked", SimpleImputer(strategy='most_frequent'), [3, 4]),

        # Age (col 5): will be regressed from Pclass and Sex features col[6:]
        ("age", IterativeImputer(), list(range(5, 11)))

    ], n_jobs=-1)),
    # new column order : "Title", "Accompanied", "Fare", "Embarked","Age", "Pclass", "Sex"

    ("norm_and_encode", ColumnTransformer([

        ("norm", StandardScaler(), [2, 4]),

        # only Title and Accompanied are not yet encoded
        ("encode", OneHotEncoder(handle_unknown="ignore"), [0, 1, 3]),

        ("passthrough", "passthrough", list(range(5, 10))),

    ], n_jobs=-1))
])


class Model(AbstractModel):

    def __init__(self, estimator):
        self.__trainedClassifier = None
        self.__estimator = estimator

    @property
    def is_trained(self) -> bool:
        return self.__trainedClassifier is not None

    def fit(self, train_df: pd.DataFrame):
        self.__trainedClassifier = (
            make_pipeline(preprocess, self.__estimator)
            .fit(train_df.drop("Survived", axis=1), train_df.Survived)
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        return self.__trainedClassifier.predict(test_df)
