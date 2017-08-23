import glob
import pickle
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from configuration import Configuration
from feature_extraction import FeatureExtraction

config = Configuration().__dict__


class Classifier:
    @staticmethod
    def normalize_features(features):
        scaler = StandardScaler().fit(features)
        features = scaler.transform(features)
        return features, scaler

    @staticmethod
    def get_trained_classifier(use_pre_trained=False):
        if use_pre_trained:
            data = pickle.load(open(config["classifier"], 'rb'))
            print("classifier trained.")
            return data["clf"], data["x_scaler"]

        # glob for cars and not cars
        not_cars = glob.glob(config["training_not_cars"])
        cars = glob.glob(config["training_cars"])

        # files for cars and not cars
        not_cars_files = [img_file for img_file in not_cars]
        cars_files = [img_file for img_file in cars]

        # features for cars and not cars
        car_features = FeatureExtraction.extract_features(cars_files)
        not_cars_features = FeatureExtraction.extract_features(not_cars_files)

        # append the feature vertically -- i.e. grow in rows with rows constant
        features = np.vstack((car_features, not_cars_features)).astype(np.float64)

        # normalize the features
        features, scaler = Classifier.normalize_features(features)

        # labels
        labels = np.hstack((np.ones(len(cars_files)), np.zeros(len(not_cars_files))))

        # shuffle dataset
        features, labels = shuffle(features, labels)

        # initialize SVM with optimized params using GridSearchCV
        clf = SVC(kernel='rbf', C=0.001)

        # train the classifier
        clf.fit(features, labels)

        print("classifier trained.")

        pickle.dump({"clf": clf, "x_scaler": scaler}, open(config["classifier"], 'wb'))
        return clf, scaler

    @staticmethod
    def evaluate_classifier_parameters():
        # glob for cars and not cars
        not_cars = glob.glob(config["training_not_cars_small"])
        cars = glob.glob(config["training_cars_small"])

        # files for cars and not cars
        not_cars_files = [img_file for img_file in not_cars]
        cars_files = [img_file for img_file in cars]

        print("cars: {}, not-cars: {}".format(len(cars_files), len(not_cars_files)))

        # features for cars and not cars
        car_features = FeatureExtraction.extract_features(cars_files)
        not_cars_features = FeatureExtraction.extract_features(not_cars_files)

        # append the feature vertically -- i.e. grow in rows with rows constant
        features = np.vstack((car_features, not_cars_features)).astype(np.float64)

        # normalize the features
        features, scaler = Classifier.normalize_features(features)

        # labels
        labels = np.hstack((np.ones(len(cars_files)), np.zeros(len(not_cars_files))))

        # shuffle dataset
        features, labels = shuffle(features, labels)

        # split dataset for training and testing
        x_train, x_test, y_train, y_test = train_test_split(features,
                                                            labels,
                                                            test_size=0.2,
                                                            random_state=42)

        clf = SVC(kernel='rbf', C=0.001)

        t_start = int(time.time())
        # train the classifier
        clf.fit(x_train, y_train)
        t_end = int(time.time())

        t_start_test = int(time.time())
        y_predicted = clf.predict(x_test)
        t_end_test = int(time.time())
        score = accuracy_score(y_test, y_predicted)

        print("train time: {}s, test time: {}s, accuracy: {}%".format(t_end - t_start,
                                                                      t_end_test - t_start_test,
                                                                      score * 100))
