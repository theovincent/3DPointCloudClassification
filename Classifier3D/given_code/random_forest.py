import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from Classifier3D.utils.ply_file import read_ply
from Classifier3D.features_extraction.extractor import compute_features

from os import listdir
from os.path import exists, join


class FeaturesExtractor:
    """
    Class that computes features from point clouds
    """

    def __init__(self, radius, num_per_class):
        # Neighborhood radius
        self.radius = radius

        # Number of training points per class
        self.num_per_class = num_per_class

    def extract_training(self, path):
        """
        This method extracts features/labels of a subset of the training points. It ensures a balanced choice between
        classes.
        :param path: path where the ply files are located.
        :return: features and labels
        """

        # Get all the ply files in data folder
        ply_files = [f for f in listdir(path) if f.endswith(".ply")]

        # Initiate arrays
        training_features = np.empty((0, 4))
        training_labels = np.empty((0,))

        # Loop over each training cloud
        for i, file in enumerate(ply_files):

            # Load Training cloud
            cloud_ply = read_ply(join(path, file))
            points = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T
            labels = cloud_ply["class"]

            # Initiate training indices array
            training_inds = np.empty(0, dtype=np.int32)

            # Loop over each class to choose training points
            for label, name in self.label_names.items():

                # Do not include class 0 in training
                if label == 0:
                    continue

                # Collect all indices of the current class
                label_inds = np.where(labels == label)[0]

                # If you have not enough indices, just take all of them
                if len(label_inds) <= self.num_per_class:
                    training_inds = np.hstack((training_inds, label_inds))
                else:
                    random_choice = np.random.choice(len(label_inds), self.num_per_class, replace=False)
                    training_inds = np.hstack((training_inds, label_inds[random_choice]))

            # Gather chosen points
            training_points = points[training_inds, :]

            # Compute features for the points of the chosen indices and place them in a [N, 4] matrix
            vert, line, plan, sphe = compute_features(training_points, points, self.radius)
            features = np.vstack((vert.ravel(), line.ravel(), plan.ravel(), sphe.ravel())).T

            # Concatenate features / labels of all clouds
            training_features = np.vstack((training_features, features))
            training_labels = np.hstack((training_labels, labels[training_inds]))

        return training_features, training_labels

    def extract_test(self, path):
        """
        This method extract features of all the test points.
        :param path: path where the ply files are located.
        :return: features
        """

        # Get all the ply files in data folder
        ply_files = [f for f in listdir(path) if f.endswith(".ply")]

        # Initiate arrays
        test_features = np.empty((0, 4))

        # Loop over each training cloud
        for i, file in enumerate(ply_files):

            # Load Training cloud
            cloud_ply = read_ply(join(path, file))
            points = np.vstack((cloud_ply["x"], cloud_ply["y"], cloud_ply["z"])).T

            # Compute features only one time and save them for further use
            #
            #   WARNING : This will save you some time but do not forget to delete your features file if you change
            #             your features. Otherwise you will not compute them and use the previous ones
            #

            # Name the feature file after the ply file.
            feature_file = file[:-4] + "_features.npy"
            feature_file = join(path, feature_file)

            # If the file exists load the previously computed features
            if exists(join(path, feature_file)):
                features = np.load(feature_file)

            # If the file does not exist, compute the features (very long) and save them for future use
            else:

                vert, line, plan, sphe = compute_features(points, points, self.radius)
                features = np.vstack((vert.ravel(), line.ravel(), plan.ravel(), sphe.ravel())).T
                np.save(feature_file, features)

            # Concatenate features of several clouds
            # (For this minichallenge this is useless as the test set contains only one cloud)
            test_features = np.vstack((test_features, features))

        return test_features


if __name__ == "__main__":
    # Path of the training and test files
    training_path = "../data/MiniChallenge/training"
    test_path = "../data/MiniChallenge/test"
    radius = 0.5
    num_per_class = 500

    # Create a feature extractor
    f_extractor = FeaturesExtractor(radius=radius, num_per_class=num_per_class)

    # Collect training features and labels
    training_features, training_labels = f_extractor.extract_training(training_path)

    # Train a random forest classifier
    clf = RandomForestClassifier()
    clf.fit(training_features, training_labels)

    # Collect test features
    test_features = f_extractor.extract_test(test_path)

    # Test the random forest on our features
    predictions = clf.predict(test_features)

    np.savetxt("MiniDijon9.txt", predictions, fmt="%d")
