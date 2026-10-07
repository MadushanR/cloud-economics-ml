"""Generate the independent, step-by-step activity notebook."""
import json
import textwrap
from pathlib import Path

cells = []
SELECTION_CODE = 'def select_linear_fit(X, y):\n    """Choose linear-fit settings using only training-fold validation prices."""\n    candidates = []\n    folds = list(KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42).split(X))\n    for intercept in [True, False]:\n        for power in [0, 1, 2]:\n            errors = []\n            for train_idx, valid_idx in folds:\n                fold_X, fold_y = X.iloc[train_idx], y.iloc[train_idx]\n                q1, q3 = fold_y.quantile([0.25, 0.75])\n                keep = fold_y.between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))\n                fold_X, fold_y = fold_X.loc[keep], fold_y.loc[keep]\n                weights = (fold_y.median() / fold_y) ** power\n                model = LinearRegression(positive=True, fit_intercept=intercept)\n                model.fit(fold_X, fold_y, sample_weight=weights)\n                predicted = model.predict(X.iloc[valid_idx])\n                errors.extend(np.abs(predicted - y.iloc[valid_idx]) / y.iloc[valid_idx])\n            candidates.append({"Fit intercept": intercept, "Weight power": power,\n                               "CV mean absolute percentage error": float(np.mean(errors) * 100)})\n    comparison = pd.DataFrame(candidates).sort_values("CV mean absolute percentage error", ignore_index=True)\n    return comparison.iloc[0].to_dict(), comparison\n\n\n'
METRICS_CODE = 'def regression_metrics(actual, predicted):\n    mse = mean_squared_error(actual, predicted)\n    percent_error = 100 * np.abs(np.asarray(predicted) - np.asarray(actual)) / np.asarray(actual)\n    return {"MAE": mean_absolute_error(actual, predicted), "MSE": mse,\n            "RMSE": float(np.sqrt(mse)), "R²": r2_score(actual, predicted),\n            "Median absolute percentage error": float(np.median(percent_error)),\n            "Mean absolute percentage error": float(np.mean(percent_error)),\n            "Within 20 percent": int(np.count_nonzero(percent_error <= 20))}\n\n\n'


def markdown(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": textwrap.dedent(source).strip() + "\n"})


markdown("""
    # Analyzing Amazon EC2 Instance Costs
    This notebook follows both parts of **Activity – Analysing EC2 Instances** using
    the supplied `ec2dataset.csv` (861 rows, 11 columns). Run cells from top to bottom
    with **Shift+Enter**. It is independent of the Streamlit app: no app files or helper
    modules are imported. Keep the notebook beside the CSV, or edit `file_path` below.

    Part 1 explores prices, missing values, IQR outliers, reserved savings and instance
    families. Part 2 trains the specified linear regression with memory and vCPUs,
    using an 80/20 split and `random_state=42`. As requested, on-demand cost outliers
    are excluded from training with training-only 1.5 × IQR bounds. The test set remains intact.

    All prices describe the supplied dataset. Its AWS region, capture date and reservation
    terms are unspecified. Missing prices remain missing, rather than being treated as zero.
""")
markdown("## Part 1 Step 1 — Install dependencies\nThis cell checks the active notebook environment and installs only missing packages. Internet access is needed if a package must be installed.")
code("""
    import importlib.util
    import subprocess
    import sys
    packages = {"pandas": "pandas>=2.2,<4", "numpy": "numpy>=1.26,<3",
                "matplotlib": "matplotlib>=3.8,<4", "seaborn": "seaborn>=0.13,<0.14",
                "sklearn": "scikit-learn>=1.5,<2"}
    missing = [package for module, package in packages.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    print("All analysis dependencies are available in", sys.executable)
""")
markdown("## Part 1 Step 2 — Load and explore the dataset\nChange the path if the CSV is stored elsewhere. The preview and information show original values before cleaning.")
code("""
    %matplotlib inline
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display
    sns.set_theme(style="whitegrid")
    file_path = Path("ec2dataset.csv")
    if not file_path.exists():
        raise FileNotFoundError("Set file_path to the full path of your ec2dataset.csv.")
    raw_data = pd.read_csv(file_path)
    data = raw_data.copy()
    cost_columns = ["On Demand", "Linux Reserved cost", "Linux Spot Minimum cost",
                    "Windows On Demand cost", "Windows Reserved cost"]
    features = ["Instance Memory", "vCPUs"]
    required = ["Name", "API Name", *features, "Instance Storage", "Network Performance", *cost_columns]
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))
    print(f"Rows: {len(data)}, columns: {len(data.columns)} (activity describes 861 × 11)")
    data.info()
    display(data.head())
""")
markdown("## Part 1 Step 3 — Clean the cost columns\nRemove currency symbols, commas, whitespace and the word hourly. Unavailable values become NaN. The original data is retained in `raw_data`.")
code("""
    for column in cost_columns:
        numeric_text = data[column].astype("string").str.replace(r"[$,\\s]|hourly", "", regex=True)
        data[column] = pd.to_numeric(numeric_text, errors="coerce")
    data[cost_columns] = data[cost_columns].replace([np.inf, -np.inf], np.nan)
    display(data[cost_columns].isna().sum().rename("Missing rows").to_frame())
    display(data.head())
""")
markdown("## Part 1 Step 4 — Summarize costs\nCount excludes missing prices. All other statistics are calculated from the available prices in each column.")
code("""
    cost_summary = data[cost_columns].describe()
    display(cost_summary)
""")
markdown("## Part 1 Step 5 — Visualize pricing distributions")
code("""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=data[cost_columns], palette="Set2", ax=ax)
    ax.set(title="Cost Comparison of Amazon EC2 Instances (Hourly)", ylabel="Cost (USD/hour)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
""")
markdown("## Part 1 Step 6 — Identify potential on-demand outliers\nUse the activity's 1.5 × IQR rule. A flag is a statistical observation; it does not establish that a price is incorrect.")
code("""
    def detect_outliers(column):
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    q1, q3 = data["On Demand"].quantile([0.25, 0.75])
    iqr = q3 - q1
    print(f"Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
    print(f"Bounds: {q1 - 1.5 * iqr:.4f} to {q3 + 1.5 * iqr:.4f} USD/hour")
    outliers_on_demand = detect_outliers("On Demand")
    print(f"Potential on-demand outliers: {len(outliers_on_demand)}")
    display(outliers_on_demand)
""")
markdown("## Part 1 Step 7 — Compare reserved and on-demand costs\nThe first table reproduces the activity's ten cheapest instances with both prices. Savings are calculated for each matched instance, with zero on-demand values excluded from percentage division.")
code("""
    cost_comparison = data[["Name", "On Demand", "Linux Reserved cost"]].dropna().sort_values("On Demand").copy()
    display(cost_comparison.head(10))
    cost_comparison["Savings USD/hour"] = cost_comparison["On Demand"] - cost_comparison["Linux Reserved cost"]
    cost_comparison["Savings percent"] = 100 * cost_comparison["Savings USD/hour"] / cost_comparison["On Demand"].replace(0, np.nan)
    display(cost_comparison.head(10))
    print(f"Median reserved savings: {cost_comparison['Savings percent'].median():.2f}% across {len(cost_comparison)} comparable rows")
""")
markdown("### Compare costs for specific capacity needs\nEdit the minimum memory and CPU values. CPU counts and memory indicate capacity, while architecture, burst limits, storage and networking also affect actual workload performance.")
code("""
    memory_gib = pd.to_numeric(data["Instance Memory"].astype("string").str.replace(r"\\s*GiB\\s*", "", regex=True).str.replace(",", "", regex=False), errors="coerce")
    cpu_count = pd.to_numeric(data["vCPUs"].astype("string").str.extract(r"(\\d+)", expand=False), errors="coerce")
    minimum_memory_gib = 4.0
    minimum_vcpus = 2
    suitable_instances = data.loc[(memory_gib >= minimum_memory_gib) & (cpu_count >= minimum_vcpus)].dropna(subset=["On Demand"]).sort_values("On Demand")
    display(suitable_instances.head(10))
""")
markdown("### Conclusions from the analysis\nThese statements are calculated from your CSV, rather than assumed AWS pricing trends.")
code("""
    priced = data.dropna(subset=["On Demand"])
    if not priced.empty:
        lowest = priced.loc[priced["On Demand"].idxmin()]
        print(f"Lowest listed Linux on-demand price: {lowest['API Name']} at ${lowest['On Demand']:.4f}/hour.")
        print(f"The IQR rule flags {len(outliers_on_demand)} of {len(priced)} available on-demand prices.")
    print(f"The median matched Linux reserved discount is {cost_comparison['Savings percent'].median():.2f}%.")
    print("Compare workload requirements and reservation terms before selecting an instance.")
    print("Large or specialized instances may legitimately appear as IQR outliers.")
""")
markdown("## Part 1 Step 8 — Filter and compare instance families\nThis preserves the activity's Name prefix filter. Its T3 selection includes T3A instances. An additional exact API family comparison below separates T3 and T3A.")
code("""
    def filter_instance_family(family):
        return data[data["Name"].astype("string").str.startswith(family, na=False)]

    t2_instances = filter_instance_family("T2")
    t3_instances = filter_instance_family("T3")
    t2_summary = t2_instances[cost_columns].describe()
    t3_summary = t3_instances[cost_columns].describe()
    print(f"T2 prefix: {len(t2_instances)} instances")
    display(t2_summary)
    print(f"T3 prefix (including T3A): {len(t3_instances)} instances")
    display(t3_summary)
""")
code("""
    for title, subset, palette in [("T2 Instances", t2_instances, "Blues"),
                                   ("T3 Prefix Instances (including T3A)", t3_instances, "Greens")]:
        if subset.empty:
            print(f"No rows for {title}")
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=subset[cost_columns], palette=palette, showmeans=True, ax=ax)
        ax.set(title=f"Cost Distribution for {title}", ylabel="Cost (USD/hour)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
""")
code("""
    comparison = pd.concat([t2_instances[["Name", "On Demand", "Linux Reserved cost"]],
                            t3_instances[["Name", "On Demand", "Linux Reserved cost"]]])
    comparison_sorted = comparison.dropna().sort_values("On Demand")
    display(comparison_sorted.head(10))
    exact_families = data["API Name"].astype("string").str.split(".", regex=False).str[0].str.upper()
    exact_comparison = data.loc[exact_families.isin(["T2", "T3"])].copy()
    print("Exact T2 and T3 API families (excluding T3A)")
    display(exact_comparison[["Name", "API Name", "On Demand", "Linux Reserved cost"]].sort_values("On Demand").head(10))
""")
markdown("## Part 2 Step 1 — Regression dependencies\nThe setup cell already checks and installs scikit-learn along with the plotting and data libraries.")
code("""
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
""")
markdown("## Part 2 Step 2 — Reload and clean costs\nReloading keeps the regression data preparation independent of Part 1 transformations.")
code("""
    data = pd.read_csv(file_path)
    for column in cost_columns:
        text = data[column].astype("string").str.replace(r"[$,\\s]|hourly", "", regex=True)
        data[column] = pd.to_numeric(text, errors="coerce")
    data[cost_columns] = data[cost_columns].replace([np.inf, -np.inf], np.nan)
    display(data[cost_columns].head())
""")
markdown("## Part 2 Step 3 — Engineer numeric features\nStrip GiB from memory and extract the CPU count from descriptions that can contain burst-duration text.")
code("""
    data["Instance Memory"] = pd.to_numeric(data["Instance Memory"].astype("string").str.replace(r"\\s*GiB\\s*", "", regex=True).str.replace(",", "", regex=False), errors="coerce")
    data["vCPUs"] = pd.to_numeric(data["vCPUs"].astype("string").str.extract(r"(\\d+)", expand=False), errors="coerce")
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    display(data[features].head())
""")
markdown("## Part 2 Step 4 — Handle missing values\nDrop rows missing On Demand, Instance Memory or vCPUs, or with nonpositive target or feature values. Positive values are required for relative-error weighting. Missing prices in unused columns do not remove a row from training.")
code("""
    data_cleaned = data.dropna(subset=["On Demand", *features]).copy()
    data_cleaned = data_cleaned.loc[(data_cleaned["On Demand"] > 0) & (data_cleaned[features] > 0).all(axis=1)]
    print(f"Retained {len(data_cleaned)} rows; dropped {len(data) - len(data_cleaned)} rows")
    display(data_cleaned.isna().sum().rename("Missing rows").to_frame())
    assert data_cleaned[["On Demand", *features]].notna().all().all()
    if len(data_cleaned) < 10:
        raise ValueError("At least ten positive complete rows are required for regression.")
""")
markdown("## Part 2 Step 5 — Split data into 80% training and 20% testing\nThe features contain only memory and vCPUs. The split uses the assignment's random_state=42.")
code("""
    X = data_cleaned[features]
    y = data_cleaned["On Demand"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}, testing samples: {len(X_test)}")
""")
markdown("### Choose linear-fit settings using training-only validation\nKeep memory and vCPUs as the only features. Compare weight powers 0, 1 and 2, with and without an intercept, using mean absolute percentage error. Each of the five training validation folds learns its own IQR bounds. No test prices are used to select settings. Nonnegative coefficients constrain the slopes; the intercept can still be negative.")
code("FEATURES = features\n" + SELECTION_CODE + "\nselected, cv_results = select_linear_fit(X_train, y_train)\ndisplay(cv_results)\nprint('Selected:', selected)")
markdown("### Filter training outliers before fitting\nCalculate the 1.5 × IQR bounds using only the training target prices. Exclude outliers from training and preserve every test row. This avoids using test data to select training rows. Expensive instances can be legitimate; the filter narrows the model's training coverage.")
code("""
    training_rows_before = len(X_train)
    train_q1, train_q3 = y_train.quantile([0.25, 0.75])
    train_iqr = train_q3 - train_q1
    train_lower = train_q1 - 1.5 * train_iqr
    train_upper = train_q3 + 1.5 * train_iqr
    training_inliers = y_train.between(train_lower, train_upper)
    excluded_training_outliers = data_cleaned.loc[y_train.index[~training_inliers]].copy()
    X_train = X_train.loc[training_inliers]
    y_train = y_train.loc[training_inliers]
    if len(X_train) < 2:
        raise ValueError("Too few training rows remain after outlier filtering.")
    print(f"Training IQR bounds: {train_lower:.6f} to {train_upper:.6f} USD/hour")
    print(f"Training rows: {training_rows_before} before filtering, {len(X_train)} retained, {len(excluded_training_outliers)} excluded")
    print(f"Test rows unchanged: {len(X_test)}")
    display(excluded_training_outliers)
    filtered_training_data = data_cleaned.loc[X_train.index].copy()
    display(filtered_training_data.head())
""")
markdown("## Part 2 Step 6 — Train linear regression\nThe intercept is the fitted price at zero memory and zero CPUs; it is not an actual purchasable instance. Each coefficient describes the fitted price change per feature unit, holding the other feature constant.")
code("""
    original_model = LinearRegression().fit(X_train, y_train)
    weights = (y_train.median() / y_train) ** int(selected["Weight power"])
    model = LinearRegression(positive=True, fit_intercept=bool(selected["Fit intercept"]))
    model.fit(X_train, y_train, sample_weight=weights)
    print(f"Intercept: {model.intercept_}")
    coefficients = pd.Series(model.coef_, index=features, name="Coefficient")
    display(coefficients.to_frame())
""")
markdown("## Part 2 Step 7 — Evaluate on the held-out test set\nMAE and RMSE use USD/hour. MSE uses squared USD/hour. R² is included as an additional fit measure and is unitless.")
code("""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r_squared = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R²: {r_squared:.6f}")
    predictions = data_cleaned.loc[X_test.index, ["Name", "API Name", *features]].copy()
    predictions["Actual USD/hour"] = y_test
    predictions["Predicted USD/hour"] = y_pred
    predictions["Original fit USD/hour"] = original_model.predict(X_test)
    predictions["Residual USD/hour"] = y_test - y_pred
    predictions["Absolute error USD/hour"] = np.abs(y_test - y_pred)
    predictions["Absolute error percent"] = 100 * np.abs(y_test - y_pred) / y_test
    display(predictions.head(10))
""")
markdown("### Compare the selected fit with the original on actual held-out prices\nBoth fits use the same filtered training rows and all 169 test rows. Within 20 percent is an illustrative count, not a correctness guarantee. Inspect the actual prices and largest errors as well as averages.")
code(METRICS_CODE + "\ncomparison_metrics = pd.DataFrame([regression_metrics(y_test, original_model.predict(X_test)), regression_metrics(y_test, y_pred)], index=['Original unweighted fit', 'Selected weighted fit'])\ndisplay(comparison_metrics)\ndisplay(predictions.sort_values('Actual USD/hour').head(10))\ndisplay(predictions.nlargest(10, 'Absolute error percent'))")
markdown("## Part 2 Step 8 — Plot actual versus predicted prices\nThe dashed red line indicates perfect predictions. Negative model outputs are shown as fitted, rather than silently changed.")
code("""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.7, color="b")
    lower = min(y_test.min(), y_pred.min())
    upper = max(y_test.max(), y_pred.max())
    ax.plot([lower, upper], [lower, upper], "r--", label="Perfect prediction")
    ax.set(title="Actual vs Predicted On-Demand Costs", xlabel="Actual On-Demand Cost (USD/hour)", ylabel="Predicted On-Demand Cost (USD/hour)")
    ax.legend()
    plt.tight_layout()
    plt.show()
""")
markdown("## Part 2 Step 9 — Predict a new configuration\nThe default is the activity's 4 GiB, 2 vCPU example. Edit these two values and rerun this cell for another configuration.")
code("""
    new_memory_gib = 4.0
    new_vcpus = 2
    if new_memory_gib <= 0 or new_vcpus < 1 or int(new_vcpus) != new_vcpus:
        raise ValueError("Memory must be positive and vCPUs must be a positive integer.")
    new_instance = pd.DataFrame([[new_memory_gib, new_vcpus]], columns=features)
    predicted_cost = float(model.predict(new_instance)[0])
    print(f"Predicted On-Demand Cost for {new_memory_gib:g} GiB, {new_vcpus} vCPUs: ${predicted_cost:.4f}/hour")
    if predicted_cost < 0:
        print("This negative output is not a valid price. The fitted model is unsuitable for this configuration.")
    if any(new_instance[col].iloc[0] < X_train[col].min() or new_instance[col].iloc[0] > X_train[col].max() for col in features):
        print("This configuration is outside the training range; the prediction is an extrapolation.")
""")
markdown("""
    ### Interpreting the regression
    Memory and vCPUs capture only part of the variation in EC2 pricing. The activity's
    linear model omits instance family, architecture, region and storage/network attributes.
    The selected weighted fit uses only memory and vCPUs, preserving the linear model.
    Multiple families can share those two inputs yet have different prices, so this model
    cannot predict a unique correct price for every instance. Training excludes cost outliers selected using training-only IQR bounds. Evaluation
    retains all test instances, so errors also measure predictions for expensive instances
    outside the retained training prices. Use the held-out errors to judge its fit. Predictions are
    educational model estimates rather than AWS quotes. A negative prediction indicates
    that the fitted relationship is unsuitable for that configuration.
""")
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
             "language_info": {"name": "python", "version": "3.14.3"}}, "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells):
    cell["id"] = f"ec2-step-{index:03d}"
Path("EC2_Analysis.ipynb").write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Created EC2_Analysis.ipynb with {sum(c['cell_type'] == 'code' for c in cells)} code cells.")
