from pandas import read_csv
from sklearn.dummy import DummyClassifier
from app.model import Model


def test_train():
    model = Model(DummyClassifier(strategy="stratified"))
    train_df = read_csv('../../data/train.csv')
    test_df = read_csv('../../data/test.csv')
    predictions = model.fit(train_df).predict(test_df)
    assert abs(predictions.mean() - train_df.Survived.mean()) <= 0.2
