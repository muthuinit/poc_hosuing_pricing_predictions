import argparse
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from google.cloud import storage
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to the dataset in GCS")
parser.add_argument("--model_dir", type=str, help="Directory to save the model")
args = parser.parse_args()

# Load dataset
def load_data(data_path):
    try:
        if data_path.startswith("gs://"):
            from google.cloud import storage
            import io

            client = storage.Client()
            bucket_name, blob_name = data_path[5:].split("/", 1)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            data = blob.download_as_text()
            df = pd.read_csv(io.StringIO(data))
        else:
            df = pd.read_csv(data_path)
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# Train model
def train_model(df):
    try:
        Q1, Q3 = df["price"].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df_cleaned = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)].copy()

        if "area" not in df_cleaned.columns or (df_cleaned["area"] <= 0).any():
            raise ValueError("The 'area' column is missing or contains invalid values (<= 0).")

        df_cleaned["log_area"] = np.log(df_cleaned["area"])
        features = ["log_area", "bedrooms", "bathrooms", "stories", "parking",
                    "mainroad", "guestroom", "basement", "hotwaterheating",
                    "airconditioning", "prefarea", "furnishingstatus"]
        target = df_cleaned["price"]

        X = df_cleaned[features]
        y = target

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["log_area", "bathrooms", "bedrooms", "stories", "parking"]),
                ("cat", OneHotEncoder(), ["mainroad", "guestroom", "basement", "hotwaterheating",
                                          "airconditioning", "prefarea", "furnishingstatus"]),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", XGBRegressor(objective="reg:squarederror", random_state=42))
            ]
        )

        param_grid = {
            "regressor__n_estimators": [100, 200, 500],
            "regressor__learning_rate": [0.01, 0.05, 0.1],
            "regressor__max_depth": [3, 5, 7],
            "regressor__subsample": [0.8, 1.0],
            "regressor__colsample_bytree": [0.8, 1.0]
        }

        kf = KFold(n_splits=min(5, len(df_cleaned)), shuffle=True, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring="neg_mean_squared_error", verbose=1)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        logger.info(f"Best hyperparameters: {grid_search.best_params_}")
        logger.info(f"Best score (neg MSE): {grid_search.best_score_}")

        return best_model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

# Save model to GCS
def save_model(model, model_dir):
    try:
        local_model_path = "model.joblib"
        joblib.dump(model, local_model_path, protocol=4)

        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local model file not found: {local_model_path}")
        logger.info(f"Model file exists: {local_model_path}, size: {os.path.getsize(local_model_path)} bytes")

        if not model_dir.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {model_dir}. Must start with 'gs://'.")

        model_dir = model_dir[5:]
        bucket_name, *blob_path = model_dir.split("/", 1)
        blob_path = blob_path[0] if blob_path else "model.joblib"

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        if not bucket.exists():
            raise ValueError(f"GCS bucket does not exist: {bucket_name}")

        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_model_path)
        logger.info(f"Model successfully uploaded to gs://{bucket_name}/{blob_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

# Main function
def main():
    try:
        df = load_data(args.data_path)
        model = train_model(df)
        save_model(model, args.model_dir)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
