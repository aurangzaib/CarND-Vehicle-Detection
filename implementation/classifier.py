import glob
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from configuration import Configuration
from helper import Helper

hyper_params = Configuration().__dict__


class Classifier:
    @staticmethod
    def get_trained_classifier(use_pre_trained=False):
        if use_pre_trained:
            data = pickle.load(open('trained_classifier.p', 'rb'))
            print("classifier trained.")
            return data["clf"], data["x_scaler"]

        # glob for cars and not cars
        not_cars = glob.glob(hyper_params["training_not_cars"])
        cars = glob.glob(hyper_params["training_cars"])

        # files for cars and not cars
        not_cars_files = [img_file for img_file in not_cars]
        cars_files = [img_file for img_file in cars]

        # features for cars and not cars
        car_features = Helper.extract_features(cars_files)
        not_cars_features = Helper.extract_features(not_cars_files)

        # append the feature vertically -- i.e. grow in rows with rows constant
        features = np.vstack((car_features, not_cars_features)).astype(np.float64)

        # normalize the features
        scaler = StandardScaler().fit(features)
        features = scaler.transform(features)

        # labels
        labels = np.hstack((np.ones(len(cars_files)), np.zeros(len(not_cars_files))))

        # split dataset
        features, labels = shuffle(features, labels)

        # initialize SVM with optimized params using GridSearchCV
        # best params --> kernel='rbf', C=10
        # but makes the classifier slow
        clf = SVC()

        # train the classifier
        clf.fit(features, labels)

        print("classifier trained.")

        pickle.dump({"clf": clf, "x_scaler": scaler}, open('trained_classifier.p', 'wb'))
        return clf, scaler
