import os
import pickle
import numpy as np
import pandas as pd
from data_preparation import DataPreparation

class TestModel:
    def __init__(self, base_path, model_path, output_path):
        self.base_path = base_path
        self.model_path = model_path
        self.output_path = output_path

    def load_models(self):
        """Load trained models from a file."""
        with open(self.model_path, 'rb') as f:
            models = pickle.load(f)
        return models

    def align_features(self, train_features, test_data):
        """Ensure test_data has the same features as the training data."""
        print("=== Debugging align_features ===")
        print(f"Columns before alignment: {test_data.columns.tolist()}")

        # Ensure 'building' and 'num_date_time' columns are always retained
        essential_columns = {'building', 'num_date_time'}
        train_features = set(train_features) | essential_columns

        # Identify missing and extra columns
        missing_cols = train_features - set(test_data.columns)
        extra_cols = set(test_data.columns) - train_features

        print(f"Missing columns to add: {missing_cols}")
        print(f"Extra columns to drop: {extra_cols - essential_columns}")  # Exclude essential columns

        # Add missing columns with default value
        for col in missing_cols:
            test_data[col] = 0

        # Remove extra columns (excluding essential columns)
        test_data = test_data.drop(columns=extra_cols - essential_columns, errors='ignore')

        print(f"Columns after alignment: {test_data.columns.tolist()}")
        print("=== End of Debugging align_features ===")

        return test_data

    def predict(self):
        # Step 1: Data Preparation
        data_prep = DataPreparation(self.base_path)
        test_data, _ = data_prep.prepare()

        # Ensure column names are clean
        test_data.columns = test_data.columns.str.strip()
        test_data.columns = test_data.columns.str.replace(' ', '_')

        # Debugging: Print test_data structure
        print("Columns in test_data:", test_data.columns)
        print("First few rows of test_data:", test_data.head())
        print(f"Is test_data empty? {test_data.empty}")
        print("Unique values in 'building' column:", test_data['building'].unique())

        # Step 2: Load Models
        models = self.load_models()

        # Align features of test_data with training data
        train_features = ['temp', 'prec', 'wind', 'hum', 'holiday', 'dow_hour_mean', 'holiday_mean',
                          'holiday_std', 'hour_mean', 'hour_std', 'type', 'all_area', 'cool_area', 'sun']
        test_data = self.align_features(train_features, test_data)

        # Debugging: Check columns after alignment
        print("Columns in test_data after alignment:", test_data.columns)

        # Step 3: Make Predictions
        predictions = []
        if 'building' in test_data.columns and not test_data.empty:
            for building in test_data['building'].unique():
                print(f"Predicting for Building {building}...")

                # Filter data for the current building
                building_data = test_data[test_data['building'] == building]
                X_test = building_data.drop(columns=['building', 'num_date_time'], errors='ignore')

                # Predict with the corresponding model
                if building in models:
                    model = models[building]
                    y_pred = model.predict(X_test)
                    predictions.append(pd.DataFrame({
                        'num_date_time': building_data['num_date_time'],
                        'building': building,
                        'prediction': y_pred
                    }))
                else:
                    print(f"Warning: No model found for Building {building}. Skipping...")
        else:
            print("Error: Test data is empty or 'building' column not found.")

        # Combine all predictions
        if predictions:
            predictions_df = pd.concat(predictions, axis=0)

            # Step 4: Save Predictions
            predictions_df.to_csv(self.output_path, index=False)
            print(f"Predictions saved to {self.output_path}.")
        else:
            print("No predictions made due to missing 'building' column or empty data.")


if __name__ == "__main__":
    base_path = './data'
    model_path = './models/xgb_models_pool.pkl'
    output_path = './predictions/test_predictions_pool.csv'

    tester = TestModel(base_path, model_path, output_path)
    tester.predict()
