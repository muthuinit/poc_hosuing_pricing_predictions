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

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to the dataset in GCS")
parser.add_argument("--model_dir", type=str, help="Directory to save the model")
args = parser.parse_args()

# Load dataset
def load_data(data_path):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    return df

# Train model
def train_model(df):
    # Remove outliers using IQR
    Q1, Q3 = df["price"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_cleaned = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)].copy()

    # Feature engineering
    df_cleaned["log_area"] = np.log(df_cleaned["area"])
    features = ["log_area", "bedrooms", "bathrooms", "stories", "parking",
                "mainroad", "guestroom", "basement", "hotwaterheating",
                "airconditioning", "prefarea", "furnishingstatus"]
    target = df_cleaned["price"]

    X = df_cleaned[features]
    y = target

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["log_area", "bathrooms", "bedrooms", "stories", "parking"]),
            ("cat", OneHotEncoder(), ["mainroad", "guestroom", "basement", "hotwaterheating",
                                      "airconditioning", "prefarea", "furnishingstatus"]),
        ]
    )

    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(objective="reg:squarederror", random_state=42))
        ]
    )

    # Hyperparameter tuning
    param_grid = {
        "regressor__n_estimators": [100, 200, 500],
        "regressor__learning_rate": [0.01, 0.05, 0.1],
        "regressor__max_depth": [3, 5, 7],
        "regressor__subsample": [0.8, 1.0],
        "regressor__colsample_bytree": [0.8, 1.0]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring="neg_mean_squared_error", verbose=1)
    grid_search.fit(X, y)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best score (neg MSE): {grid_search.best_score_}")

    return best_model

# Save model to GCS
def save_model(model, model_dir):
    joblib.dump(model, "model.joblib")
    client = storage.Client()
    bucket = client.bucket(model_dir.split("/")[2])
    blob = bucket.blob("/".join(model_dir.split("/")[3:]) + "/model.joblib")
    blob.upload_from_filename("model.joblib")

# Main function
def main():
    df = load_data(args.data_path)
    model = train_model(df)
    save_model(model, args.model_dir)

if __name__ == "__main__":
    main()