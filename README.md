# AI-Based Predictive Maintenance with Limited Failure Data

A machine-learning research project investigating **Remaining Useful Life (RUL) prediction** for simulated aircraft engines, with a focus on model performance when fewer labeled training-engine trajectories are available.

## Research Objectives

- Build machine-learning baselines for RUL prediction.
- Investigate how the amount of labeled training data affects prediction accuracy.
- Compare Random Forest and Extra Trees regressors.
- Explore pseudo-labeling as a semi-supervised learning baseline.
- Evaluate feature selection and RUL target capping.
- Study calibration-based bias correction under simulated partial observation.
- Document experiments, limitations, and reproducible results.

## Dataset

**NASA C-MAPSS FD001**

- **Dataset source:** [NASA C-MAPSS FD001 on Zenodo](https://zenodo.org/records/15346912)
- **DOI:** [10.5281/zenodo.15346912](https://doi.org/10.5281/zenodo.15346912)
- **License:** CC BY 4.0

The dataset contains simulated aircraft-engine sensor measurements collected over operating cycles. Training trajectories run to failure, while test trajectories end before failure.

| Dataset characteristic | Description |
|---|---|
| Training engines | 100 |
| Test engines | 100 |
| Operating conditions | 1 |
| Simulated fault modes | 1 |
| Sensor measurements | 21 |
| Operating settings | 3 |

Official test RUL labels are supplied separately.

**Important:** These are simulated engine data, not real aircraft maintenance records.

## Technologies Used

- Python
- NumPy
- pandas
- scikit-learn
- Git and GitHub

## Methodology

1. Load and prepare the FD001 training data.
2. Calculate training RUL as the number of cycles remaining until the end of each training trajectory.
3. Split validation data by engine ID to prevent the same engine from appearing in both training and validation sets.
4. Compare baseline regressors and investigate limited-label training scenarios.
5. Train models on all 100 training engines.
6. Predict RUL for the last observed cycle of each of the 100 official test engines.
7. Evaluate predictions using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the asymmetric NASA scoring function.

For limited-data experiments, the number of labeled training engines is varied. This simulates having fewer labeled run-to-failure trajectories; it does not mean that individual cycles within a selected trajectory are unlabeled.

## Preliminary Official Test Results

The following results were obtained in the current experiment using the official FD001 test evaluation.

| Model | MAE (cycles) | RMSE (cycles) | NASA score |
|---|---:|---:|---:|
| Random Forest | 25.05 | 33.96 | 48,627.96 |
| **Extra Trees** | **23.54** | **31.98** | **28,214.11** |

Lower values are better for all three metrics.

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

## Project Structure

```text
Predictive-Maintenance-AI/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── notebooks/
├── reports/
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
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

The `data/`, `models/`, and `reports/` directories store datasets, model artifacts, and experiment outputs. Python source code is organized under `src/predictive_maintenance/`.

The source directory includes baseline models, official test evaluation, limited-data experiments, semi-supervised learning, feature selection, RUL capping, per-engine analysis, and bias-correction experiments. Backup scripts are retained for reference.

## Installation and Setup

### Prerequisites

- Python compatible with the pinned package versions in `requirements.txt`
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

Download the FD001 dataset from the [Zenodo dataset record](https://zenodo.org/records/15346912).

Place the required raw dataset files in the project's `data/raw/` directory. Follow the filenames and paths expected by the data-preparation scripts before running the experiments.

### 5. Run the Python Modules

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

Other experiment modules can be run using the same pattern, replacing the module name with the desired experiment.

## Validation

Compile the project modules:

```bash
python -m compileall -q src/predictive_maintenance
```

Run the test suite:

```bash
python -m unittest discover -s tests
```

## Experiment Reports

The repository includes experiment reports such as:

- `reports/official_test_results.csv` — official FD001 test metrics, including MAE, RMSE, and NASA score.
- `reports/official_test_predictions.csv` — actual RUL values and model predictions for the official test engines.
- `reports/repeated_bias_calibration_runs.csv` — results for each of the 10 repeated trials.
- `reports/repeated_bias_calibration_summary.csv` — average metrics and standard deviations.
- `reports/independent_bias_calibration_results.csv` — results from the initial independent calibration experiment.
- `reports/independent_bias_calibration_predictions.csv` — predictions from that experiment.
- `reports/validation_bias_correction.csv` — preliminary validation experiment.

## Current Status and Future Work

The project currently includes baseline model comparisons, official test evaluation, limited-data experiments, and exploratory bias-correction studies.

Potential next steps include:

- Repeating experiments with more robust validation procedures.
- Investigating methods designed for limited labeled failure trajectories.
- Evaluating calibration strategies under more realistic deployment conditions.
- Improving reproducibility and documenting experimental configurations.

## Disclaimer

This is an academic machine-learning research project using simulated engine data. Its predictions are experimental and are not validated for real-world aircraft maintenance or safety-critical decisions.
