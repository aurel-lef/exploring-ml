from abc import ABC, abstractmethod

class Model(ABC):
    
    @abstractmethod
    def train(self, trainFile):
        """ train model. trainFile is a csv file, kaggle csv format with header """
    
    @abstractmethod
    def predict(self, test):
        """ make prediction, test is a csv file, kaggle csv format with header """