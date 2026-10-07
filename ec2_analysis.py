"""Analysis used by the Streamlit app, following the supplied EC2 activity."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

COST_COLUMNS = [
    "On Demand", "Linux Reserved cost", "Linux Spot Minimum cost",
    "Windows On Demand cost", "Windows Reserved cost",
]
FEATURES = ["Instance Memory", "vCPUs"]
REQUIRED_COLUMNS = ["Name", "API Name", *FEATURES,
                    "Instance Storage", "Network Performance", *COST_COLUMNS]


def validate_schema(data):
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if data.empty:
        raise ValueError("The CSV contains no instance rows.")


def clean_data(raw):
    validate_schema(raw)
    data = raw.copy()
    for column in COST_COLUMNS:
        text = data[column].astype("string").str.replace(r"[$,\s]|hourly", "", regex=True)
        data[column] = pd.to_numeric(text, errors="coerce")
    memory = data["Instance Memory"].astype("string").str.replace(
        r"\s*GiB\s*", "", regex=True).str.replace(",", "", regex=False)
    data["Instance Memory"] = pd.to_numeric(memory, errors="coerce")
    cpu = data["vCPUs"].astype("string").str.extract(r"(\d+)", expand=False)
    data["vCPUs"] = pd.to_numeric(cpu, errors="coerce")
    # Unavailable prices remain missing; they are never treated as free instances.
    data[COST_COLUMNS + FEATURES] = data[COST_COLUMNS + FEATURES].replace(
        [np.inf, -np.inf], np.nan)
    return data


def detect_outliers(data, column="On Demand"):
    q1, q3 = data[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    rows = data.loc[(data[column] < lower) | (data[column] > upper)].copy()
    return rows, {"Q1": q1, "Q3": q3, "IQR": iqr,
                  "Lower bound": lower, "Upper bound": upper}


def reserved_comparison(data):
    comparison = data[["Name", "API Name", "On Demand", "Linux Reserved cost"]].dropna().copy()
    comparison["Savings USD/hour"] = comparison["On Demand"] - comparison["Linux Reserved cost"]
    comparison["Savings percent"] = 100 * comparison["Savings USD/hour"] / comparison["On Demand"].replace(0, np.nan)
    return comparison.sort_values("On Demand")


def filter_instance_family(data, family):
    """Use the activity's Name prefix rule (T3 also includes T3A)."""
    return data.loc[data["Name"].astype("string").str.startswith(family, na=False)].copy()


def train_regression(data):
    cleaned = data.dropna(subset=["On Demand", *FEATURES]).copy()
    if len(cleaned) < 5:
        raise ValueError("At least five rows with numeric cost, memory and vCPUs are required.")
    X, y = cleaned[FEATURES], cleaned["On Demand"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    predicted = model.predict(X_test)
    mse = mean_squared_error(y_test, predicted)
    metrics = {"MAE": mean_absolute_error(y_test, predicted), "MSE": mse,
               "RMSE": float(np.sqrt(mse)), "R²": r2_score(y_test, predicted)}
    predictions = cleaned.loc[X_test.index, ["Name", "API Name", *FEATURES]].copy()
    predictions["Actual USD/hour"] = y_test
    predictions["Predicted USD/hour"] = predicted
    predictions["Residual USD/hour"] = y_test - predicted
    return {"model": model, "cleaned": cleaned, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test, "predictions": predictions,
            "metrics": metrics, "dropped": len(data) - len(cleaned)}
