from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from utils.utils import *
import pickle
import os


class SVM(object):
    def __init__(self, checkpoint_path=''):
        self.checkpoint_path = checkpoint_path
        self.l_svm = LinearSVC()
        self.experiment = os.path.join('checkpoints', 'SVM')
        if not os.path.isfile(self.checkpoint_path):
            print("Invalid checkpoints. Check your checkpoint_path or retrain model")
            if not yes_or_no("Do you want to retrain SVM model"):
                exit()
        else:
            with open(self.checkpoint_path, 'rb') as f:
                self.l_svm = pickle.load(f)

    def feed_forward(self, train_data, train_labels):
        X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels,
                                                          test_size=0.2, shuffle=False)
        self.l_svm.fit(X_train, y_train)
        print("Finish training. Save model to {}".format(self.experiment))
        pickle.dump(self.l_svm, open(os.path.join(self.experiment, 'model.pkl'), "wb"))

    def evaluate(self, test_data, test_labels):
        pred = self.l_svm.predict(test_data)
        return accuracy_score(test_labels, pred), classification_report(test_labels, pred)

    def inference(self, features: list):
        pred = self.l_svm.predict(features)
        y_margins = self.l_svm.decision_function(features)
        confidences = sigmoid_v(y_margins)
        confidences = np.amax(confidences, axis=1)
        return pred, confidences


