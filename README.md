# AI-Based Machine Fault Detection with Limited Failure Data

A machine-learning research project focused on **Remaining Useful Life (RUL) prediction** using NASA C-MAPSS FD001. The current implementation investigates how limited labeled engine trajectories affect prediction accuracy and evaluates machine-learning models and calibration methods under limited failure data.

## Research Objectives

- Build machine-learning baselines for RUL prediction.
- Investigate how the amount of labeled training data affects prediction accuracy.
- Compare Random Forest and Extra Trees regressors.
- Explore pseudo-labeling as a semi-supervised learning baseline.
- Evaluate feature selection and RUL target capping.
- Study calibration-based bias correction under simulated partial observation.
- Document experiments, limitations, and reproducible results.

## Dataset

This project uses the **NASA C-MAPSS FD001 Turbofan Engine Degradation Simulation Dataset**, provided by the NASA Ames Prognostics Center of Excellence (PCoE).

- **Source:** [NASA C-MAPSS Dataset — Zenodo](https://doi.org/10.5281/zenodo.15346912)
- **Dataset version:** v2
- **Creator:** National Aeronautics and Space Administration (NASA)
- **Publisher:** NASA Diagnostics and Prognostics Group
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Archive:** `CMAPSSData.zip`

The FD001 subset contains simulated turbofan engine run-to-failure data under a single operating condition and a single fault mode. It includes three operational settings and 21 sensor measurements used to study engine degradation and Remaining Useful Life (RUL) prediction.

**Dataset files used in this project:**
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

The original dataset is not included in this repository. Download it from the official Zenodo record and place the required files in the appropriate project data directories.

**Integrity verification:** The downloaded archive was verified using MD5. Its checksum matches the checksum published for Zenodo version v2:

`79a22f36e80606c69d0e9e4da5bb2b7a`

## Technologies Used

- Python 3.12
- NumPy
- pandas
- scikit-learn
- Joblib
- Streamlit
- ReportLab
- Git and GitHub

## Methodology

1. Load and prepare the NASA C-MAPSS FD001 training data.
2. Calculate training Remaining Useful Life (RUL) as the number of cycles remaining until the end of each training trajectory.
3. Split validation data by engine ID to prevent the same engine from appearing in both training and validation sets.
4. Compare baseline regressors and investigate limited-label training scenarios using 20, 40, 60, and 80 training engines.
5. Compare Random Forest and Extra Trees regressors.
6. Investigate feature selection, RUL target capping, and pseudo-labeling as additional experiments.
7. Evaluate calibration-based bias correction under simulated partial observation.
8. Train the final Random Forest and Extra Trees models using all 100 training engines.
9. Evaluate the final models on the official FD001 test dataset using MAE, RMSE, and the NASA scoring function.
10. Save the trained models, configuration metadata, predictions, and evaluation reports for reproducibility.
11. Use a Streamlit dashboard to display predictions and compare model performance.

## Preliminary Official Test Results

Both Random Forest and Extra Trees were trained using all 100 FD001 training engines and evaluated on the 100 official test engines.

| Model | Training Engines | Test Engines | MAE | RMSE | NASA Score |
|---|---:|---:|---:|---:|---:|
| Random Forest | 100 | 100 | 25.05 | 33.96 | 48,627.96 |
| Extra Trees | 100 | 100 | **23.54** | **31.98** | **28,214.11** |

Extra Trees achieved lower MAE, RMSE, and NASA score than Random Forest on this official test evaluation.

**Note:** Lower values indicate better performance for all three metrics. These results represent one evaluation on the official FD001 test set.

### Understanding the Evaluation Metrics

- **MAE (Mean Absolute Error):** Measures the average absolute difference between predicted and actual Remaining Useful Life (RUL), in operating cycles. An MAE of 23.54 means the average absolute prediction error is approximately 23.54 cycles.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more heavily than MAE. A lower RMSE indicates better performance under this metric.
- **NASA Score:** Uses an asymmetric scoring function to penalize prediction errors differently depending on their direction. This implementation penalizes underprediction using an exponential scale of 13 cycles and overprediction using an exponential scale of 10 cycles. The penalties are accumulated across the evaluated test engines, so a lower score indicates better performance under this scoring function.

The NASA score is a penalty score, not a percentage or an error measured in cycles. Its magnitude depends on the dataset, the number of evaluated engines, and the scoring definition. Comparisons should therefore use the same evaluation dataset and scoring function.

Extra Trees performs better than Random Forest on all three metrics in this experiment. These are experimental results on FD001 and are **not a claim of superiority over published research**.

## Limited-Data Experiments

The project investigates how the availability of labeled engine trajectories affects Remaining Useful Life (RUL) prediction. The experiments include:

- Training with different numbers of labeled engines.
- Comparing Random Forest and Extra Trees.
- Removing constant and near-constant sensor features.
- Comparing capped and uncapped RUL targets.
- Investigating a simple pseudo-labeling approach.

### Multi-Split Limited-Data Results

A Random Forest model was evaluated using 20, 40, 60, and 80 labeled training engines. The experiment used three engine-level train-validation splits and five subset-selection repeats for each engine count within each split, producing 15 recorded runs per training size. For the 80-engine setting, all available training engines were selected in every repeat, so the five repeats within each split used identical training subsets.

| Training engines | Runs | Mean MAE (cycles) | Mean RMSE (cycles) | MAE standard deviation | RMSE standard deviation |
|---:|---:|---:|---:|---:|---:|
| 20 | 15 | 31.09 | 42.15 | 2.67 | 3.36 |
| 40 | 15 | 28.68 | 39.20 | 2.56 | 2.93 |
| 60 | 15 | 27.99 | 38.21 | 2.17 | 2.65 |
| 80 | 15 | 27.87 | 37.96 | 2.12 | 2.59 |

Lower MAE and RMSE indicate better predictive performance.

### Interpretation and Limitations

- Increasing the number of labeled training engines from 20 to 80 reduced mean MAE by approximately 10.4% and mean RMSE by approximately 9.9%.
- The largest improvement occurred between 20 and 40 training engines. The gains became smaller at higher training sizes.
- The standard deviations decreased as training size increased, indicating lower variation across the recorded runs.
- The 15 runs per training size are based on three validation splits, with five subset selections per split. They should not be interpreted as 15 independent validation splits.
- For the 80-engine setting, the five repeats within each split produce identical results because they use the same training subset, validation set, and model random seed. Therefore, this setting has three distinct split-level evaluations rather than 15 distinct training-subset evaluations.
- The metrics are calculated across validation time-step observations, so engines with more observations may contribute more to the overall scores.
- These are exploratory validation results for the current Random Forest configuration. They do not establish that the same trend will hold for other models or datasets, and they are separate from the official NASA FD001 test-set results.

Other experiments found that pseudo-labeling improvements were small and inconsistent. Further evaluation with more robust validation procedures is needed before drawing stronger conclusions.

The multi-split run-level results and summary are saved in `reports/limited_data_multisplit_runs.csv` and `reports/limited_data_multisplit_summary.csv`, respectively.

## Repeated Bias-Correction Experiment

A calibration-based bias-correction experiment was evaluated over 10 trials using separate training, calibration, and evaluation engine groups.

Simulated cutoff cycles between 50% and 90% of each engine's lifetime were used to approximate partially observed engines.

### Results

| Metric | Original Extra Trees | Bias-Corrected Extra Trees |
|---|---:|---:|
| Mean MAE (cycles) | 29.05 | 26.00 |
| Mean RMSE (cycles) | 37.64 | 31.23 |

- MAE improved in 7 of 10 trials.
- RMSE improved in 10 of 10 trials.
- Average MAE reduction: 3.05 cycles (approximately 10.5%).
- Average RMSE reduction: 6.41 cycles (approximately 17.0%).

### Interpretation and Limitations

The results suggest that calibration-based bias correction may improve RUL predictions under this simulated evaluation setup. However, the MAE improvement was not consistent across all trials.

The evaluation uses simulated cutoff points from training trajectories that run to failure. This does not perfectly reproduce the official FD001 test distribution.

These results are preliminary and must not be interpreted as an improvement in official test performance. The official test results reported above remain unchanged, and bias correction has not been adopted as the final model.

## Saved Models

The project saves the trained Random Forest and Extra Trees models for later use and reproducibility.

The `models/` directory contains:

| File | Description |
|---|---|
| `random_forest_fd001.joblib` | Trained Random Forest RUL regression model |
| `extra_trees_fd001.joblib` | Trained Extra Trees RUL regression model |
| `feature_columns.json` | Ordered list of the 24 input features used by the models |
| `model_metadata.json` | Dataset information, training configuration, model filenames, and software versions |

**Note:** The trained `.joblib` model files are not included in this GitHub repository because of their large size. They must be available in the local `models/` directory before the dashboard can make predictions.

To generate the model files, run the official test evaluation script from the repository root. **Warning:** This script retrains both models and overwrites the existing model artifacts and metadata in the `models/` directory. Run it only when you intentionally want to retrain the models.

The JSON files document the model inputs and training configuration but cannot replace the trained model files.

### Model Configuration

| Parameter | Random Forest | Extra Trees |
|---|---:|---:|
| Number of estimators | 100 | 100 |
| Minimum samples per leaf | 2 | 2 |
| Random state | 42 | 42 |
| `n_jobs` | 2 | 2 |

### Loading a Saved Model

Run the following example from the repository root after installing the dependencies and generating or obtaining the saved model files:

```python
import json
import joblib
import numpy as np
import pandas as pd

# Load the saved Extra Trees model
model = joblib.load("models/extra_trees_fd001.joblib")

# Load the ordered feature list
with open("models/feature_columns.json", "r", encoding="utf-8") as file:
    feature_columns = json.load(file)

# Load processed official test data
test_data = pd.read_csv("data/processed/test_FD001_official.csv")

# Select features in the required order
X_test = test_data[feature_columns]

# Generate RUL predictions
predictions = model.predict(X_test)

# Prevent negative RUL predictions
predictions = np.maximum(predictions, 0)

print(predictions)
```

**Note:** The file `data/processed/test_FD001_official.csv` is generated locally and is not included in this repository. Prepare the official test data using the project's preprocessing workflow before running this example.

The example uses the processed official test dataset. For new input data, provide the same 24 features in the same order and use the same preprocessing expected by the trained model.

The saved model artifacts and metadata are intended to support model reuse. They do not represent a deployed real-time aircraft monitoring system.

## Project Structure

```text
Predictive-Maintenance-AI/
│
├── app.py
│
├── models/
│   ├── feature_columns.json
│   └── model_metadata.json
│
├── reports/
│   └── Experiment and evaluation CSV reports
│
├── src/
│   └── predictive_maintenance/
│       ├── __init__.py
│       ├── config.py
│       ├── prepare_data.py
│       ├── prepare_official_test.py
│       ├── train_baseline.py
│       ├── compare_models.py
│       ├── compare_tree_models.py
│       ├── compare_error_groups.py
│       ├── evaluate_official_test.py
│       ├── limited_data_experiment.py
│       ├── limited_data_experiment_backup.py
│       ├── limited_data_experiment_multisplit_backup.py
│       ├── multi_split_experiment.py
│       ├── per_engine_evaluation.py
│       ├── semi_supervised_experiment.py
│       ├── semi_supervised_experiment_backup.py
│       ├── feature_selection_experiment.py
│       ├── rul_capping_experiment.py
│       ├── independent_bias_calibration.py
│       ├── repeated_bias_calibration.py
│       └── validation_bias_experiment.py
│
├── tests/
│   └── test_config.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

The repository contains experiment scripts and CSV reports covering model evaluation, limited-data experiments, calibration, feature selection, and supporting analyses.

**Note:** Raw and processed dataset files and trained `.joblib` model files are excluded from GitHub through `.gitignore`. These files must be obtained or generated separately. The saved model artifacts are required to run the Streamlit dashboard without retraining the models.

## Installation and Setup

### Prerequisites

- Python 3.12
- Git
- The NASA C-MAPSS FD001 dataset

### 1. Clone the Repository

```bash
git clone https://github.com/haqnawaz82/Predictive-Maintenance-AI.git
cd Predictive-Maintenance-AI
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download the FD001 dataset from the [NASA C-MAPSS dataset record](https://doi.org/10.5281/zenodo.15346912).

Place the required raw dataset files in `data/raw/` using the filenames expected by the project's data-preparation scripts. Prepare the processed training and official test data before running the experiment or evaluation modules.

### 5. Run the Python Modules (Optional)

From the repository root, set the Python source path.

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
```

On macOS or Linux:

```bash
export PYTHONPATH=src
```

After the required processed data has been prepared, run the official test evaluation:

```bash
python -m predictive_maintenance.evaluate_official_test
```

**Optional:** Run this evaluation only if you need to generate or intentionally retrain the saved models. If you already have the required `.joblib` files, you can skip this step and proceed to the Streamlit Dashboard.

This command retrains both models, overwrites the saved model artifacts and metadata, and writes the official test evaluation reports.

**Warning:** Run this command only when you intentionally want to retrain the models. It can replace existing files in the `models/` directory.

The following option prevents evaluation report files from being saved. However, the function still retrains both models and overwrites their saved model artifacts and metadata:

```python
from predictive_maintenance.evaluate_official_test import evaluate_official_test

evaluate_official_test(save_reports=False)
```

**Important:** `save_reports=False` prevents report files from being saved, but it does not prevent model artifacts and metadata from being overwritten.

Other experiment modules can be run using the same module pattern, replacing the module name with the desired experiment.

## Streamlit Dashboard

The project includes a Streamlit dashboard for interactive Remaining Useful Life (RUL) predictions using the saved Random Forest and Extra Trees models.

### Run the Dashboard

Make sure the following saved model files and feature configuration are available in the `models/` directory:

- `random_forest_fd001.joblib`
- `extra_trees_fd001.joblib`
- `feature_columns.json`

Start the dashboard from the project root:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

### Dashboard Features

- Select an engine from the official FD001 test dataset.
- View RUL predictions from both saved models.
- Compare predictions with actual RUL values for retrospective evaluation.
- View prediction errors and official test performance metrics.
- Explore the project methodology and model architecture.
- Download a CSV prediction report.
- Download a PDF prediction report.

**Note:** The dashboard loads existing saved models. It does not train or retrain models when launched. The model files are not included in the GitHub repository and must be available locally before running the dashboard.

## Validation

Compile the project modules:

```bash
python -m compileall -q src/predictive_maintenance
```

Run the test suite:

```bash
python -m unittest discover -s tests
```

The current test suite contains basic configuration checks; it does not provide full end-to-end validation of the machine-learning pipeline.

## Experiment Reports

The `reports/` directory contains CSV files generated by the project's model evaluations and experiments.

### Model Evaluation

- `reports/baseline_results.csv` — baseline model results, including MAE and RMSE.
- `reports/model_comparison.csv` — model comparison results using MAE and RMSE.
- `reports/tree_model_comparison_runs.csv` — individual Random Forest and Extra Trees comparison runs across different training-engine counts.
- `reports/tree_model_comparison_summary.csv` — aggregated Random Forest and Extra Trees comparison metrics.
- `reports/official_test_results.csv` — official FD001 test metrics, including MAE, RMSE, and NASA score.
- `reports/official_test_predictions.csv` — actual RUL values and Random Forest and Extra Trees predictions for the official test engines.

### Limited-Data Experiments

- `reports/limited_data_results.csv` — results for different training-engine counts, including training rows, MAE, and RMSE.
- `reports/limited_data_repeated_runs.csv` — individual repeated limited-data experiment results.
- `reports/limited_data_summary.csv` — aggregated MAE and RMSE results for the limited-data experiments.
- `reports/limited_data_multisplit_runs.csv` — individual runs across different split seeds, training-engine counts, and validation-engine groups.
- `reports/limited_data_multisplit_summary.csv` — aggregated MAE and RMSE results for the multi-split experiments.
- `reports/multi_split_results.csv` — multi-split evaluation results comparing Random Forest and median-based metrics.
- `reports/per_engine_evaluation_runs.csv` — individual evaluation results using row-weighted and equal-engine MAE and RMSE.
- `reports/per_engine_evaluation_summary.csv` — aggregated row-weighted and equal-engine evaluation metrics.

### Feature Selection and RUL Capping

- `reports/feature_selection_runs.csv` — detailed results comparing three feature sets across different training-engine counts and split repeats.
- `reports/feature_selection_summary.csv` — aggregated MAE and RMSE results for the feature-selection experiments.
- `reports/rul_capping_runs.csv` — individual RUL-capping experiment results, including metrics on capped and original RUL targets.
- `reports/rul_capping_summary.csv` — aggregated results comparing capped-target and original-target metrics.

### Semi-Supervised Experiments

- `reports/semi_supervised_results.csv` — results comparing baseline and semi-supervised MAE and RMSE across split seeds and labeled-engine counts.
- `reports/semi_supervised_summary.csv` — summary of baseline and semi-supervised MAE and RMSE results.
- `reports/semi_supervised_randomized_runs.csv` — individual randomized semi-supervised experiment results.
- `reports/semi_supervised_randomized_summary.csv` — aggregated baseline and semi-supervised metrics across randomized runs.

### Bias-Correction Experiments

- `reports/independent_bias_calibration_results.csv` — evaluation metrics for the independent bias-calibration experiment.
- `reports/independent_bias_calibration_predictions.csv` — actual RUL values and original and corrected predictions from that experiment.
- `reports/repeated_bias_calibration_runs.csv` — results from each of the 10 repeated bias-calibration trials, including original and corrected metrics.
- `reports/repeated_bias_calibration_summary.csv` — average metrics and standard deviations for the repeated bias-calibration trials.
- `reports/validation_bias_correction.csv` — MAE, RMSE, and mean error for the validation bias-correction methods.


## Conclusion

This project investigated Remaining Useful Life (RUL) prediction using the NASA C-MAPSS FD001 dataset, with a focus on the effect of limited labeled engine trajectories on prediction performance.

Random Forest and Extra Trees were evaluated using the official FD001 test dataset. In this evaluation, Extra Trees achieved lower MAE (23.54 cycles), RMSE (31.98 cycles), and NASA score (28,214.11) than Random Forest. These results indicate better performance by Extra Trees in this particular evaluation.

The limited-data experiments showed that increasing the number of labeled training engines from 20 to 80 reduced prediction errors, with smaller improvements at higher training sizes. Feature-selection experiments did not show consistent benefits from removing sensor features. Pseudo-labeling improvements were small and inconsistent, while repeated bias-correction experiments showed promising but preliminary results under simulated partial-observation conditions.

Overall, the experiments provide an initial assessment of RUL prediction under limited failure data. Further evaluation using more independent engine splits, additional validation procedures, and realistic operating conditions is needed before drawing broader conclusions or considering real-world deployment.


## Project Status and Future Work

The project currently includes baseline model comparisons, official test evaluation, limited-data experiments, and exploratory bias-correction studies.

Feature-selection experiments did not show consistent performance improvements from removing constant sensors or `sensor_6` in the evaluated experiments.

Bias-correction experiments remain exploratory and require further validation across independent engine splits.

Potential next steps include:

- Repeating experiments with more robust validation procedures.
- Investigating methods designed for limited labeled failure trajectories.
- Evaluating calibration strategies under more realistic deployment conditions.
- Improving reproducibility and documenting experimental configurations.

## Disclaimer

This is an academic machine-learning research project using simulated engine data. Its predictions are experimental and are not validated for real-world aircraft maintenance or safety-critical decisions.
