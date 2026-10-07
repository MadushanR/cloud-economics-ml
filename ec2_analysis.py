"""Analysis used by the Streamlit app, following the supplied EC2 activity."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold

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


def select_linear_fit(X, y):
    """Choose linear-fit settings using only training-fold validation prices."""
    candidates = []
    folds = list(KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42).split(X))
    for intercept in [True, False]:
        for power in [0, 1, 2]:
            errors = []
            for train_idx, valid_idx in folds:
                fold_X, fold_y = X.iloc[train_idx], y.iloc[train_idx]
                q1, q3 = fold_y.quantile([0.25, 0.75])
                keep = fold_y.between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
                fold_X, fold_y = fold_X.loc[keep], fold_y.loc[keep]
                weights = (fold_y.median() / fold_y) ** power
                model = LinearRegression(positive=True, fit_intercept=intercept)
                model.fit(fold_X, fold_y, sample_weight=weights)
                predicted = model.predict(X.iloc[valid_idx])
                errors.extend(np.abs(predicted - y.iloc[valid_idx]) / y.iloc[valid_idx])
            candidates.append({"Fit intercept": intercept, "Weight power": power,
                               "CV mean absolute percentage error": float(np.mean(errors) * 100)})
    comparison = pd.DataFrame(candidates).sort_values("CV mean absolute percentage error", ignore_index=True)
    return comparison.iloc[0].to_dict(), comparison


def regression_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    percent_error = 100 * np.abs(np.asarray(predicted) - np.asarray(actual)) / np.asarray(actual)
    return {"MAE": mean_absolute_error(actual, predicted), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R²": r2_score(actual, predicted),
            "Median absolute percentage error": float(np.median(percent_error)),
            "Mean absolute percentage error": float(np.mean(percent_error)),
            "Within 20 percent": int(np.count_nonzero(percent_error <= 20))}


def train_regression(data):
    cleaned = data.dropna(subset=["On Demand", *FEATURES]).copy()
    cleaned = cleaned.loc[(cleaned["On Demand"] > 0) & (cleaned[FEATURES] > 0).all(axis=1)]
    if len(cleaned) < 10:
        raise ValueError("At least ten rows with positive numeric cost, memory and vCPUs are required.")
    X, y = cleaned[FEATURES], cleaned["On Demand"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    training_before = len(X_train)
    # Selection precedes the final IQR filter; each CV fold learns its own bounds.
    selected, cv_results = select_linear_fit(X_train, y_train)
    q1, q3 = y_train.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    keep = y_train.between(lower, upper)
    excluded = cleaned.loc[y_train.index[~keep]].copy()
    X_train, y_train = X_train.loc[keep], y_train.loc[keep]
    if len(X_train) < 2:
        raise ValueError("Too few training rows remain after IQR outlier filtering.")
    baseline = LinearRegression().fit(X_train, y_train)
    weights = (y_train.median() / y_train) ** int(selected["Weight power"])
    model = LinearRegression(positive=True, fit_intercept=bool(selected["Fit intercept"]))
    model.fit(X_train, y_train, sample_weight=weights)
    predicted = model.predict(X_test)
    baseline_predicted = baseline.predict(X_test)
    predictions = cleaned.loc[X_test.index, ["Name", "API Name", *FEATURES]].copy()
    predictions["Actual USD/hour"] = y_test
    predictions["Predicted USD/hour"] = predicted
    predictions["Original fit USD/hour"] = baseline_predicted
    predictions["Residual USD/hour"] = y_test - predicted
    predictions["Absolute error USD/hour"] = np.abs(y_test - predicted)
    predictions["Absolute error percent"] = 100 * np.abs(y_test - predicted) / y_test
    predictions["Within retained training price range"] = y_test.between(lower, upper)
    return {"model": model, "baseline_model": baseline, "cleaned": cleaned,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "predictions": predictions, "metrics": regression_metrics(y_test, predicted),
            "baseline_metrics": regression_metrics(y_test, baseline_predicted),
            "selection": selected, "cv_results": cv_results,
            "dropped": len(data) - len(cleaned), "training_before": training_before,
            "training_outliers": excluded, "outliers_removed": len(excluded),
            "outlier_bounds": {"Q1": q1, "Q3": q3, "IQR": iqr,
                               "Lower bound": lower, "Upper bound": upper}}
