import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import graycomatrix, graycoprops


class DamageDetector:
    def __init__(self):
        # Create a basic Decision Tree classifier
        self.classifier = DecisionTreeClassifier()

        # Dummy training data (for demonstration purposes)
        self.X_train = [
            [1.0, 2.0, 3.0, 4.0, 5.0],  # Feature vector 1
            [2.0, 3.0, 4.0, 5.0, 6.0],  # Feature vector 2
            # Add more training feature vectors as needed
        ]
        # Dummy labels (0: Low Severity, 1: High Severity)
        self.y_train = [0, 1]

    def extract_features(self, image_path):
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Compute GLCM
        glcm = graycomatrix(
            img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)

        # Extract some features from GLCM
        contrast = graycoprops(glcm, prop='contrast')[0, 0]
        dissimilarity = graycoprops(glcm, prop='dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, prop='homogeneity')[0, 0]
        energy = graycoprops(glcm, prop='energy')[0, 0]
        correlation = graycoprops(glcm, prop='correlation')[0, 0]

        # Return the five extracted features as a list
        return [contrast, dissimilarity, homogeneity, energy, correlation]

    def fit_classifier(self):
        # Fit the classifier with training data
        self.classifier.fit(self.X_train, self.y_train)

    def predict_severity(self, image_path):
        self.fit_classifier()
        # Extract features
        features = self.extract_features(image_path)

        # Use our classifier to predict severity
        severity = self.classifier.predict([features])[0]
        return severity
