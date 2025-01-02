import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from data_preparation import DataPreparation
# from feature_engineering import FeatureEngineering

# Weighted MSE as a global function
def weighted_mse_fixed(label, pred, alpha=1):
    residual = (label - pred).astype("float")
    grad = np.where(residual > 0, -2 * alpha * residual, -2 * residual)
    hess = np.where(residual > 0, 2 * alpha, 2.0)
    return grad, hess

# Wrapper function for XGBoost
def weighted_mse(alpha=1):
    def func(label, pred):
        return weighted_mse_fixed(label, pred, alpha=alpha)
    return func

class TrainModel:
    def __init__(self, base_path, save_path):
        self.base_path = base_path
        self.save_path = save_path

    def smape(self, y_true, y_pred):
        return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

    def train(self):
        # Step 1: Data Preparation
        data_prep = DataPreparation(self.base_path)
        train_data, _ = data_prep.prepare()

        # : Feature Engineering
        # feature_engineer = FeatureEngineering()
        # train_data = feature_engineer.engineer_features(train_data)

        # train
        models = {}
        from sklearn.model_selection import train_test_split
        from xgboost import XGBRegressor

        for building in train_data['building'].unique():
            print(f"Training model for Building {building}...")

            # Filter data for the current building
            building_data = train_data[train_data['building'] == building]

            # Drop columns that are not useful for training
            X = building_data.drop(columns=['target', 'building', 'num_date_time', 'date_time', 'date'],
                                   errors='ignore')
            y = building_data['target']

            # Ensure X contains only numeric columns
            X = X.select_dtypes(include=['int64', 'float64', 'bool'])

            # Train-Test Split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize the model
            model = XGBRegressor(
                colsample_bytree=0.8,
                eta=0.01,
                max_depth=5,
                min_child_weight=6,
                n_estimators=10,
                subsample=0.9,
                early_stopping_rounds=10,
                eval_metric='rmse'  # Example: Root Mean Squared Error
            )

            # Train the model
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )

            print(f"Finished training for Building {building}")

            # Evaluate the model
            y_val_pred = model.predict(X_val)
            val_smape = self.smape(y_val, y_val_pred)
            print(f"Building {building} SMAPE: {val_smape:.2f}%")

            # Save the model
            models[building] = model

        # Save all models to file
        with open(self.save_path, 'wb') as f:
            pickle.dump(models, f)

        print(f"Models saved to {self.save_path}.")

if __name__ == "__main__":
    base_path = './data'
    save_path = './models/xgb_models_pool.pkl'

    trainer = TrainModel(base_path, save_path)
    trainer.train()


