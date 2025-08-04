# x-ray fracture recovery time prediction
# uses a cnn model to analyze x-ray images and tabular data to predict or classify recovery time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class SimpleFracturePredictor:
    def __init__(self):
        self.cnn_model = None
        self.tabular_model = None
        self.label_encoders = {}
        self.image_size = (224, 224)
        self.scaler = StandardScaler()

    def create_cnn_model(self):
        # create a cnn model using mobilenetv2 as the base
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*self.image_size, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation="linear")(x)  # regression output
        self.cnn_model = Model(inputs=base_model.input, outputs=predictions)
        self.cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        print("cnn model created.")

    def preprocess_images(self, image_paths):
        # preprocess images for cnn input
        images = []
        for path in image_paths:
            img = Image.open(path).resize(self.image_size)
            img = np.array(img) / 255.0  # normalize pixel values
            images.append(img)
        return np.array(images)

    def train_cnn_model(self, image_paths, recovery_times):
        # train the cnn model on x-ray images
        images = self.preprocess_images(image_paths)
        X_train, X_val, y_train, y_val = train_test_split(images, recovery_times, test_size=0.2, random_state=42)
        history = self.cnn_model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32
        )
        print("cnn model training complete.")
        return history

    def train_tabular_model(self, tabular_data, recovery_times):
        # train a random forest model on tabular data
        X_train, X_val, y_train, y_val = train_test_split(tabular_data, recovery_times, test_size=0.2, random_state=42)
        self.tabular_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.tabular_model.fit(X_train, y_train)
        y_pred = self.tabular_model.predict(X_val)
        print(f"tabular model r²: {r2_score(y_val, y_pred):.3f}")

    def predict_combined(self, image_paths, tabular_data):
        # combine predictions from cnn and tabular models
        cnn_predictions = self.cnn_model.predict(self.preprocess_images(image_paths))
        tabular_predictions = self.tabular_model.predict(tabular_data)
        combined_predictions = (cnn_predictions.flatten() + tabular_predictions) / 2
        return combined_predictions

    def classify_recovery_time(self, recovery_times):
        # classify recovery times into categories
        bins = [0, 6, 12, np.inf]
        labels = ["short", "medium", "long"]
        return pd.cut(recovery_times, bins=bins, labels=labels)

    def plot_results(self, y_true, y_pred):
        # plot actual vs predicted recovery times
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel("actual recovery time (weeks)")
        plt.ylabel("predicted recovery time (weeks)")
        plt.title("actual vs predicted recovery time")
        plt.show()


def main():
    # Main function to run the fracture prediction project
    print("X-ray fracture recovery time prediction")
    print("=" * 50)

    # Initialize predictor
    predictor = SimpleFracturePredictor()
    predictor.create_cnn_model()

    # Example synthetic data (replace with real data)
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]  # Replace with actual paths
    tabular_data = pd.DataFrame({
        "age": [25, 40],
        "bone_type": ["femur", "tibia"],
        "severity": ["high", "medium"],
        "treatment": ["surgery", "cast"]
    })
    recovery_times = np.array([12, 8])  # Replace with actual recovery times

    # Encode categorical features
    for col in ["bone_type", "severity", "treatment"]:
        predictor.label_encoders[col] = LabelEncoder()
        tabular_data[col] = predictor.label_encoders[col].fit_transform(tabular_data[col])

    # Scale numerical features
    tabular_data = predictor.scaler.fit_transform(tabular_data)

    # Train models
    predictor.train_cnn_model(image_paths, recovery_times)
    predictor.train_tabular_model(tabular_data, recovery_times)

    # Predict and evaluate
    combined_predictions = predictor.predict_combined(image_paths, tabular_data)
    print("Combined Predictions:", combined_predictions)

    # Plot results
    predictor.plot_results(recovery_times, combined_predictions)


#main()