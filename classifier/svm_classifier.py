from classifier.abstract_classifier import AbstractClassifier
from sklearn.svm import SVC

class SvmClassifier(AbstractClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.model = SVC(kernel='rbf')
        
    def train(self, x, y) -> None:
        self.model.fit(x, y)
    
    def predict(self, x):
        return self.model.predict(x)