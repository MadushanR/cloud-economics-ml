"""Streamlit implementation of both parts of the EC2 analysis activity."""
from pathlib import Path
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from ec2_analysis import (COST_COLUMNS, FEATURES, clean_data, detect_outliers,
                          reserved_comparison, filter_instance_family, train_regression)

st.set_page_config(page_title="EC2 Instance Cost Analysis", page_icon="☁️", layout="wide")
sns.set_theme(style="whitegrid")


def show_boxplot(data, title, palette="Set2"):
    if data.empty or not data[COST_COLUMNS].notna().any().any():
        st.info("No numeric prices are available for this selection.")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=data[COST_COLUMNS], palette=palette, showmeans=True, ax=ax)
    ax.set(title=title, ylabel="Cost (USD/hour)")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def download_csv(data, label, filename, key):
    st.download_button(label, data.to_csv(index=False).encode("utf-8"), filename,
                       mime="text/csv", key=key)


st.title("EC2 Instance Cost Analysis")
st.write("Explore the five pricing options, compare instance families, and predict Linux on-demand hourly cost from memory and vCPUs.")
st.caption("Prices come from the supplied CSV. Its region, capture date and reservation terms are unspecified; these results describe that dataset.")
uploaded = st.sidebar.file_uploader("Use another EC2 comparison CSV", type="csv")
default_path = Path(__file__).parent / "ec2dataset.csv"
try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        source_name = uploaded.name
    elif default_path.exists():
        raw = pd.read_csv(default_path)
        source_name = default_path.name
    else:
        st.info("Upload the EC2 dataset using the sidebar to begin.")
        st.stop()
    data = clean_data(raw)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

st.sidebar.caption(f"Source: {source_name}")
st.sidebar.caption("Analysis and regression use all rows. Family and workload filters apply only to their comparison tables.")
cols = st.columns(3)
cols[0].metric("Instance rows", len(raw))
cols[1].metric("Source columns", len(raw.columns))
cols[2].metric("Available on-demand prices", int(data["On Demand"].notna().sum()))
if raw.shape != (861, 11):
    st.info(f"The activity describes 861 rows and 11 columns; this CSV has {len(raw)} rows and {len(raw.columns)} columns.")
explore, analysis, families, regression = st.tabs([
    "1. Load and clean", "2. Costs and outliers", "3. Instance families", "4. Regression and prediction"])

with explore:
    st.subheader("Dataset preview")
    st.dataframe(raw.head(10), hide_index=True)
    buffer = io.StringIO()
    raw.info(buf=buffer)
    with st.expander("Dataset information"):
        st.code(buffer.getvalue(), language="text")
    st.subheader("Missing values after conversion")
    st.write("Currency symbols and hourly labels are removed. Memory is expressed in GiB and the leading vCPU count is extracted from CPU descriptions.")
    st.dataframe(data[COST_COLUMNS + FEATURES].isna().sum().rename("Missing rows").to_frame())
    st.subheader("Cleaned dataset")
    st.dataframe(data, hide_index=True)
    download_csv(data, "Download cleaned dataset", "ec2_cleaned.csv", "cleaned")

with analysis:
    st.subheader("Cost summary statistics")
    st.dataframe(data[COST_COLUMNS].describe())
    show_boxplot(data, "Cost comparison of Amazon EC2 instances")
    st.subheader("Potential cost outliers using the IQR method")
    outlier_column = st.selectbox("Pricing option", COST_COLUMNS)
    outliers, bounds = detect_outliers(data, outlier_column)
    st.write("Rows below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR are flagged. A flag identifies an unusually priced row, not necessarily an error.")
    st.dataframe(pd.Series(bounds, name="USD/hour").to_frame())
    st.write(f"{len(outliers)} potential outliers in {outlier_column}.")
    st.dataframe(outliers.sort_values(outlier_column), hide_index=True)
    download_csv(outliers, "Download outliers", "ec2_outliers.csv", "outliers")
    st.subheader("Reserved versus on-demand costs")
    comparison = reserved_comparison(data)
    st.dataframe(comparison.head(10), hide_index=True)
    if not comparison.empty:
        st.write(f"Across {len(comparison)} rows with both Linux prices, the median reserved discount is {comparison['Savings percent'].median():.1f}%. The table shows the ten lowest on-demand prices with a reserved price available.")
    download_csv(comparison, "Download all reserved comparisons", "ec2_reserved_comparison.csv", "reserved")
    st.subheader("Lowest-cost instances meeting your requirements")
    left, right = st.columns(2)
    min_memory = left.number_input("Minimum memory (GiB)", min_value=0.0, value=4.0, step=0.5)
    min_cpus = right.number_input("Minimum vCPUs", min_value=1, value=2, step=1)
    suitable = data.loc[(data["Instance Memory"] >= min_memory) & (data["vCPUs"] >= min_cpus)].dropna(subset=["On Demand"]).sort_values("On Demand")
    st.dataframe(suitable.head(10), hide_index=True)
    st.caption("Memory and CPU counts describe capacity. Workload performance also depends on CPU architecture, burst limits, storage and networking.")
    st.subheader("Conclusions from this dataset")
    available = data.dropna(subset=["On Demand"])
    if not available.empty:
        cheapest = available.loc[available["On Demand"].idxmin()]
        st.write(f"The lowest listed Linux on-demand price is {cheapest['API Name']} at ${cheapest['On Demand']:.4f}/hour. Compare instances that meet your workload requirements before choosing by price.")
        od_outliers, _ = detect_outliers(data)
        st.write(f"The on-demand IQR rule flags {len(od_outliers)} of {len(available)} priced rows. Larger or specialized instances can legitimately fall above this threshold.")

with families:
    st.subheader("T2 and T3 comparisons from the activity")
    st.caption("The activity filters Name by prefix: T3 includes T3A. The exact family selector below separates those families.")
    t2 = filter_instance_family(data, "T2")
    t3 = filter_instance_family(data, "T3")
    for label, subset, palette in [("T2", t2, "Blues"), ("T3 prefix (including T3A)", t3, "Greens")]:
        st.write(f"{label}: {len(subset)} instances")
        st.dataframe(subset[COST_COLUMNS].describe())
        show_boxplot(subset, f"Cost distribution for {label} instances", palette)
    st.write("Ten lowest on-demand prices across the T2 and T3 prefix selections")
    st.dataframe(reserved_comparison(pd.concat([t2, t3])).head(10), hide_index=True)
    st.subheader("Compare exact instance families")
    family = data["API Name"].astype("string").str.split(".", regex=False).str[0].str.upper()
    choices = sorted(family.dropna().unique().tolist())
    selected = st.multiselect("Instance families", choices, default=[f for f in ["T2", "T3"] if f in choices])
    chosen = data.loc[family.isin(selected)]
    st.dataframe(chosen, hide_index=True)
    show_boxplot(chosen, "Cost distribution for selected families")

with regression:
    st.subheader("Weighted linear regression using memory and vCPUs")
    st.write("Rows missing the target or either feature are dropped. An 80/20 split uses random_state=42. On-demand cost outliers are then excluded from training using 1.5 × IQR bounds calculated on the training split only. The test set is kept intact. Positive prices and capacity values are required for relative-error weighting.")
    try:
        result = train_regression(data)
    except ValueError as error:
        st.warning(str(error))
    else:
        st.write(f"Training: {len(result['X_train'])} rows · Testing: {len(result['X_test'])} rows · Dropped: {result['dropped']} rows")
        st.write(f"Training rows before filtering: {result['training_before']} · Training outliers excluded: {result['outliers_removed']}")
        with st.expander("Training outlier bounds and excluded instances"):
            st.dataframe(pd.Series(result["outlier_bounds"], name="USD/hour").to_frame())
            st.dataframe(result["training_outliers"], hide_index=True)
        download_csv(result["training_outliers"], "Download excluded training outliers", "ec2_training_outliers.csv", "training_outliers")
        training_data = result["cleaned"].loc[result["X_train"].index]
        download_csv(training_data, "Download filtered training dataset", "ec2_training_cleaned.csv", "training_cleaned")
        st.caption("Outliers can be legitimate expensive instances. Excluding them narrows the training coverage; the metrics below still evaluate every held-out test row, including expensive instances.")
        st.write("The fit settings are selected using five-fold cross-validation within the training split. Each fold learns its own IQR bounds. The model still uses only memory and vCPUs.")
        st.write(f"Selected weight power: {int(result['selection']['Weight power'])} · Fit intercept: {bool(result['selection']['Fit intercept'])}")
        with st.expander("Training cross-validation results"):
            st.dataframe(result["cv_results"], hide_index=True)
        st.caption("Weights equal (median training price / instance price) raised to the selected power. Larger powers give lower-priced instances more influence. Nonnegative coefficients do not constrain the intercept.")
        model = result["model"]
        st.dataframe(pd.DataFrame({"Term": ["Intercept", *FEATURES], "Coefficient": [model.intercept_, *model.coef_]}), hide_index=True)
        metric_cols = st.columns(4)
        for col, (label, value) in zip(metric_cols, list(result["metrics"].items())[:4]):
            col.metric(label, f"{value:.4f}")
        st.caption("MAE and RMSE are in USD/hour. MSE is in squared USD/hour; R² is unitless. Metrics use only the held-out test set.")
        st.subheader("Accuracy against actual held-out prices")
        accuracy = pd.DataFrame([result["baseline_metrics"], result["metrics"]], index=["Original unweighted fit", "Selected weighted fit"])
        st.dataframe(accuracy)
        st.caption("Percentage errors use each instance's actual price. Within 20 percent is an illustrative count, not a guarantee of acceptable accuracy. Both fits use the same training rows and untouched test rows.")
        predictions = result["predictions"]
        st.write(f"{result['metrics']['Within 20 percent']} of {len(predictions)} held-out predictions are within 20% of their actual prices. Median absolute percentage error: {result['metrics']['Median absolute percentage error']:.1f}%.")
        st.write("Lowest-priced held-out instances")
        st.dataframe(predictions.sort_values("Actual USD/hour").head(10), hide_index=True)
        st.write("Largest percentage errors")
        st.dataframe(predictions.nlargest(10, "Absolute error percent"), hide_index=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(predictions["Actual USD/hour"], predictions["Predicted USD/hour"], alpha=0.7)
        bounds = [float(predictions[["Actual USD/hour", "Predicted USD/hour"]].min().min()),
                  float(predictions[["Actual USD/hour", "Predicted USD/hour"]].max().max())]
        ax.plot(bounds, bounds, "r--", label="Perfect prediction")
        ax.set(title="Actual vs predicted on-demand costs", xlabel="Actual USD/hour", ylabel="Predicted USD/hour")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(predictions, hide_index=True)
        download_csv(predictions, "Download test predictions", "ec2_test_predictions.csv", "predictions")
        st.subheader("Predict a new instance configuration")
        with st.form("predict"):
            memory = st.number_input("Memory (GiB)", min_value=0.001, value=4.0, step=0.5)
            cpus = st.number_input("vCPUs", min_value=1, value=2, step=1)
            submitted = st.form_submit_button("Predict hourly cost")
        if submitted:
            new_instance = pd.DataFrame([[memory, cpus]], columns=FEATURES)
            prediction = float(model.predict(new_instance)[0])
            st.metric("Predicted Linux on-demand USD/hour", f"${prediction:.4f}")
            if prediction < 0:
                st.warning("The fitted linear model predicts a negative price for this configuration. This is not a valid price; it indicates poor model fit in this part of the data.")
            outside = any(new_instance[col].iloc[0] < result["X_train"][col].min() or new_instance[col].iloc[0] > result["X_train"][col].max() for col in FEATURES)
            if outside:
                st.warning("This configuration lies outside the training range for memory or vCPUs, so the estimate extrapolates beyond the training data.")
        matching = result["cleaned"].loc[(result["cleaned"]["Instance Memory"] == memory) & (result["cleaned"]["vCPUs"] == cpus)]
        if not matching.empty:
            st.write(f"The dataset has {len(matching)} instances with this memory and vCPU combination, with prices from ${matching['On Demand'].min():.4f} to ${matching['On Demand'].max():.4f}/hour. A two-feature model gives all of them the same estimate.")
            with st.expander("Actual dataset prices for this configuration"):
                st.dataframe(matching[["API Name", "Instance Memory", "vCPUs", "On Demand"]], hide_index=True)
        st.caption("This educational model omits instance family, architecture, region and other pricing factors. Predictions describe a fitted relationship in the CSV, not an AWS price quote.")
