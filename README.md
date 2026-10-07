# EC2 Instance Cost Analysis

This project implements both parts of **Activity – Analysing EC2 Instances.docx** using the supplied `ec2dataset.csv`, containing 861 rows and 11 columns. It provides a Streamlit app and an independent Jupyter notebook with separate cells for each activity step.

## Files

- `app.py`: Streamlit entry point.
- `ec2_analysis.py`: app cleaning, comparisons, IQR detection and regression.
- `ec2dataset.csv`: the original supplied dataset, unchanged.
- `EC2_Analysis.ipynb`: independent notebook; run its cells from top to bottom with Shift+Enter.
- `requirements.txt`: pinned app dependencies.
- `requirements-notebook.txt`: notebook tools and app dependencies.

## Tasks remaining for public deployment

These steps require your personal accounts. The app has not been published yet.

1. Sign in to the GitHub account that can push to [MadushanR/cloud-economics-ml](https://github.com/MadushanR/cloud-economics-ml).
2. Open PowerShell in `C:\Users\madus\Downloads\cloud-economics-ml` and run `git push -u origin main`. The local repository is already connected to GitHub as `origin`. Complete GitHub's sign-in prompt if one appears.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using GitHub and connect your GitHub account when prompted.
4. Click **Create app**, select **Yup, I have an app**, and select **MadushanR/cloud-economics-ml**, branch **main**, with main file path **`app.py`**.
5. In **Advanced settings**, select **Python 3.14** to match the tested local environment. This project does not require secrets.
6. Click **Deploy**. Confirm that the app loads and the four tabs display data. In the app's sharing settings, confirm that access is public, then copy the generated `streamlit.app` link.

If Python 3.14 is not offered in your Streamlit account, report the versions offered before changing the dependency versions. If a deployment fails, share the build error rather than any passwords or account tokens.

Deployment steps follow [Streamlit's deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). Dependency placement follows [Streamlit's dependency documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Push this repository to GitHub

The local repository is on `main` and already has `origin` set to `https://github.com/MadushanR/cloud-economics-ml.git`. The initial commit includes the app, dataset, notebook and dependency files. Open PowerShell and run:

```powershell
cd "C:\Users\madus\Downloads\cloud-economics-ml"
git push -u origin main
```

Complete GitHub's sign-in prompt if one appears. Then select this repository and its `main` branch in Streamlit Community Cloud, with `app.py` as the entry point. The local `.venv` and verification outputs are excluded from Git; Streamlit installs the dependencies from `requirements.txt`.

## Run locally on this computer

An isolated `.venv` environment is already prepared in this project. Double-click **Open Notebook.cmd** to open JupyterLab or **Run App.cmd** to open the app. Alternatively, open PowerShell in the project folder and use the commands below.

For the Streamlit app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

For JupyterLab:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab EC2_Analysis.ipynb
```

Select a Python kernel from this project's environment and run cells with **Shift+Enter**. The notebook has 22 code cells and does not import the app or its helper module. Keep `ec2dataset.csv` beside it, or edit `file_path` in the loading cell. The first cell installs any missing analysis libraries in the active kernel environment.

On a different computer, install Python 3.14, create a virtual environment, and install `requirements-notebook.txt` before launching JupyterLab. For the app alone, install `requirements.txt`.

## Coverage of the activity

| Activity step | Streamlit location | Notebook |
|---|---|---|
| Part 1 Step 1 Dependencies | `requirements.txt` | Dependency check and installation cell |
| Part 1 Step 2 Load, info, head | Load and clean tab | Original preview and `info()` |
| Part 1 Step 3 Cost cleaning and missing counts | Load and clean tab | Cost conversion and missing counts |
| Part 1 Step 4 Summary statistics | Costs and outliers tab | `describe()` |
| Part 1 Step 5 Pricing boxplot | Costs and outliers tab | Five-column boxplot |
| Part 1 Step 6 IQR outliers | Costs and outliers tab | On-demand IQR bounds and rows |
| Part 1 Step 7 Conclusions, reserved savings and workload costs | Costs and outliers tab | Matched comparison, savings, workload filtering and conclusions |
| Part 1 Step 8 T2/T3 summaries, boxplots and comparisons | Instance families tab | Prefix filters, both boxplots and combined comparison |
| Part 2 Steps 1–4 Dependencies, cost cleaning, features, missing rows | Regression and prediction tab | Separate preparation cells and training-only IQR outlier removal |
| Part 2 Step 5 80/20 split with seed 42 | Regression and prediction tab | `train_test_split` cell |
| Part 2 Step 6 Linear regression, coefficients and intercept | Regression and prediction tab | Training and coefficient cell |
| Part 2 Step 7 MAE, MSE and RMSE | Regression and prediction tab | Held-out evaluation cell |
| Part 2 Step 8 Actual versus predicted plot | Regression and prediction tab | Scatter plot cell |
| Part 2 Step 9 Prediction for 4 GiB, 2 vCPUs | Prediction form default values | Editable prediction cell |

The activity's `Name.startswith('T3')` filter includes T3A. Both deliverables preserve this behavior and explicitly label it; an additional exact API-family comparison excludes T3A when choosing T3.

## Reading the results

Prices are historical values from the supplied CSV. The dataset does not specify its region, capture date or reservation terms. Missing prices remain missing. IQR flags indicate statistical outliers rather than confirmed data errors.

The required regression uses only memory and vCPUs. It excludes instance family, architecture and other pricing factors. The model excludes on-demand cost outliers from training using the 1.5 × IQR rule, with bounds calculated from the training targets only after the 80/20 split. For the supplied CSV, 51 of 672 training rows are excluded and 621 remain. All 169 test rows are retained. The 4 GiB, 2 vCPU example predicts approximately $0.2355/hour. Filtering narrows the training coverage and does not guarantee lower errors on the full test set or nonnegative predictions for every possible configuration. Negative outputs, if any, are flagged as invalid prices. Neither fitted estimates nor the capacity-based cheapest-instance table establish actual workload performance.
