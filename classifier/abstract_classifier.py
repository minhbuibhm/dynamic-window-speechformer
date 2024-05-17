from abc import ABC, abstractmethod

class AbstractClassifier(ABC):
    @abstractmethod     
    def train(self, x, y):
        pass
    
    @abstractmethod
    def predict(self, x):
        pass