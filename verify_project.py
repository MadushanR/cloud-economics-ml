"""Check the supplied dataset, app interactions and independent notebook."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from streamlit.testing.v1 import AppTest
from ec2_analysis import (COST_COLUMNS, FEATURES, clean_data, detect_outliers,
                          reserved_comparison, filter_instance_family, train_regression)

root = Path(__file__).resolve().parent
os.chdir(root)
verification = root / "verification"
verification.mkdir(exist_ok=True)
raw = pd.read_csv("ec2dataset.csv")
assert raw.shape == (861, 11)
data = clean_data(raw)
assert data[COST_COLUMNS].isna().sum().tolist() == [20, 24, 39, 263, 266]
assert data[FEATURES].notna().all().all()
assert np.allclose(data.loc[0, FEATURES].astype(float), [0.5, 2])
assert len(detect_outliers(data)[0]) == 66
assert len(filter_instance_family(data, "T2")) == 7
assert len(filter_instance_family(data, "T3")) == 14
result = train_regression(data)
assert (len(result["X_train"]), len(result["X_test"]), result["dropped"]) == (621, 169, 20)
assert not set(result["X_train"].index) & set(result["X_test"].index)
assert np.isfinite(list(result["metrics"].values())).all()
assert list(result["model"].feature_names_in_) == FEATURES
assert result["training_before"] == 672 and result["outliers_removed"] == 51
assert not set(result["training_outliers"].index) & set(result["X_test"].index)
assert result["y_train"].between(result["outlier_bounds"]["Lower bound"], result["outlier_bounds"]["Upper bound"]).all()
assert float(result["model"].predict(pd.DataFrame([[4, 2]], columns=FEATURES))[0]) > 0
# Changing only held-out targets must not change training bounds, rows or fit.
changed_test = data.copy()
changed_test.loc[result["X_test"].index, "On Demand"] = 1000000.0
changed_result = train_regression(changed_test)
assert changed_result["outlier_bounds"] == result["outlier_bounds"]
assert changed_result["X_train"].index.equals(result["X_train"].index)
assert np.allclose(changed_result["model"].coef_, result["model"].coef_)
assert np.isclose(changed_result["model"].intercept_, result["model"].intercept_)
assert changed_result["selection"] == result["selection"]
assert result["metrics"]["MAE"] < result["baseline_metrics"]["MAE"]
assert result["metrics"]["Median absolute percentage error"] < result["baseline_metrics"]["Median absolute percentage error"]
assert result["metrics"]["Within 20 percent"] > result["baseline_metrics"]["Within 20 percent"]
assert result["selection"]["Weight power"] == 1
assert result["model"].predict(pd.DataFrame([[0.001, 1]], columns=FEATURES))[0] > 0
actual_prices = data.loc[result["X_test"].index, "On Demand"]
assert np.allclose(result["predictions"]["Actual USD/hour"], actual_prices)
assert np.allclose(result["predictions"]["Predicted USD/hour"], result["model"].predict(result["X_test"]))
comparison = reserved_comparison(data)
assert comparison["On Demand"].is_monotonic_increasing
try:
    clean_data(raw.drop(columns=["On Demand"]))
    raise AssertionError("Missing columns were accepted")
except ValueError as error:
    assert "On Demand" in str(error)
edge = raw.head(2).copy()
edge.loc[0, "On Demand"] = "$1,234.50 hourly"
edge.loc[1, "On Demand"] = "unavailable"
assert clean_data(edge).loc[0, "On Demand"] == 1234.5
assert pd.isna(clean_data(edge).loc[1, "On Demand"])
assert len(raw.columns) == 11 and raw.loc[0, "Instance Memory"] == "0.5 GiB"
print("Dataset and analysis checks passed.", flush=True)

at = AppTest.from_file(str(root / "app.py"), default_timeout=90).run()
assert not at.exception, str(at.exception)
print("Streamlit initial render passed.", flush=True)
assert at.metric[0].value == "861"
at.selectbox[0].select("Windows On Demand cost").run()
assert not at.exception, str(at.exception)
print("Streamlit alternate price selection passed.", flush=True)
at.multiselect[0].set_value([]).run()
assert not at.exception, str(at.exception)
print("Streamlit empty family selection passed.", flush=True)
at.multiselect[0].set_value(["T2", "T3"]).run()
at.button[0].click().run()
assert not at.exception, str(at.exception)
assert not any("negative price" in warning.value for warning in at.warning)
assert any(metric.label == "Predicted Linux on-demand USD/hour" and metric.value == "$0.0701" for metric in at.metric)
at.number_input[2].set_value(100000.0)
at.button[0].click().run()
assert not at.exception, str(at.exception)
assert any("outside the training range" in warning.value for warning in at.warning)
print("Streamlit loading, cost selection, family selection and prediction checks passed.", flush=True)

kernel_root = verification / "kernels"
kernel_path = kernel_root / "ec2-project"
kernel_path.mkdir(parents=True, exist_ok=True)
(kernel_path / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "EC2 Project Verification", "language": "python"}), encoding="utf-8")
manager = KernelManager(kernel_name="ec2-project", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
notebook = nbformat.read("EC2_Analysis.ipynb", as_version=4)
nbformat.validate(notebook)
client = NotebookClient(notebook, km=manager, timeout=120,
                        resources={"metadata": {"path": str(root)}})
client.execute()
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert len(code_cells) == 24
assert all(cell.execution_count is not None for cell in code_cells)
assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
assert sum("image/png" in output.get("data", {}) for cell in code_cells for output in cell.outputs) == 4
# Check the independent notebook and app fitted the same filtered data.
prediction_cell = code_cells[-1]
assert any("$0.0701/hour" in str(output.get("text", "")) for output in prediction_cell.outputs)
nbformat.write(notebook, "EC2_Analysis.ipynb")
summary = {"dataset_shape": list(raw.shape), "missing_prices": data[COST_COLUMNS].isna().sum().to_dict(),
           "on_demand_outliers": 66, "training_rows": len(result["X_train"]), "test_rows": len(result["X_test"]),
           "training_outliers_removed": result["outliers_removed"], "outlier_bounds": result["outlier_bounds"],
           "dropped_rows": 20, "metrics": result["metrics"], "baseline_metrics": result["baseline_metrics"], "selection": result["selection"],
           "example_prediction": float(result["model"].predict(pd.DataFrame([[4, 2]], columns=FEATURES))[0]),
           "notebook_code_cells_executed": len(code_cells), "streamlit_checks": "passed"}
(verification / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("All 24 notebook code cells executed successfully; saved notebook with outputs.", flush=True)
