@'
# AI-Based Predictive Maintenance with Limited Failure Data

A machine-learning research project that investigates Remaining Useful Life (RUL) prediction for simulated aircraft engines, including model performance when fewer labeled training-engine trajectories are available.

## Research Objectives

- Build machine-learning baselines for RUL prediction.
- Investigate how the amount of labeled training data affects prediction accuracy.
- Compare Random Forest and Extra Trees regressors.
- Explore simple pseudo-labeling as a semi-supervised learning baseline.
- Evaluate feature selection and RUL target capping.
- Document experiments, limitations, and reproducible results.

## Dataset

**NASA C-MAPSS FD001**

The dataset contains simulated aircraft-engine sensor measurements collected over operating cycles. Training trajectories run to failure, while test trajectories end before failure.

- 100 training-engine trajectories.
- 100 test-engine trajectories.
- One operating condition and one simulated fault mode.
- 21 sensor measurements and 3 operating settings.
- Official test RUL labels supplied separately.

The data is simulated, not real aircraft maintenance records.

## Methodology

1. Load and prepare the FD001 training data.
2. Calculate training RUL as the number of cycles remaining until the end of each training trajectory.
3. Split validation data by engine ID to prevent the same engine from appearing in both training and validation sets.
4. Compare baseline regressors and investigate limited-label training scenarios.
5. Train models on all 100 training engines.
6. Predict RUL for the last observed cycle of each of the 100 official test engines.
7. Evaluate predictions using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the asymmetric NASA scoring function.

For the limited-data experiments, the number of labeled training engines is varied. This simulates having fewer labeled run-to-failure trajectories; it does not mean that individual cycles within a selected trajectory are unlabeled.

## Preliminary Official Test Results

Results from the current experiment:

| Model | MAE (cycles) | RMSE (cycles) | NASA score |
|---|---:|---:|---:|
| Random Forest | 25.05 | 33.96 | 48,627.96 |
| Extra Trees | 23.54 | 31.98 | 28,214.11 |

Lower values are better for all three metrics.

Extra Trees performs better than Random Forest on these metrics in this experiment. These are preliminary results on FD001, not a claim of superiority over published research.

## Limited-Data Experiments

Experiments have investigated:

- Training with 20, 40, and 60 labeled engines.
- Random Forest versus Extra Trees.
- Feature removal for constant and near-constant sensors.
- Capped versus uncapped RUL targets.
- A simple pseudo-labeling approach.

These experiments are exploratory. Improvements from pseudo-labeling were small and inconsistent, so they do not yet establish a reliable benefit.

## Project Structure

```text
Predictive-Maintenance-AI/
|-- data/
|   |-- raw/
|   |-- processed/
|-- models/
|-- reports/
|-- src/
|   |-- predictive_maintenance/
|-- tests/
```

## Repeated Bias-Correction Experiment
A calibration-based bias-correction experiment was evaluated over 10 trials using separate training, calibration, and evaluation engine groups. Simulated cutoff cycles between 50% and 90% of each engine's lifetime were used to approximate partially observed engines.
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
The results suggest that calibration-based bias correction may improve RUL predictions under this simulated evaluation setup. However, the improvement was not consistent for MAE across all trials.
The evaluation uses simulated cutoff points from training trajectories that run to failure. This does not perfectly reproduce the official FD001 test distribution. The results are preliminary and must not be interpreted as an improvement in official test performance.
The official test results reported above remain unchanged. The bias-correction experiment has not been adopted as the final model.
### Experiment Reports
- `reports/repeated_bias_calibration_runs.csv` - results for each of the 10 trials.
- `reports/repeated_bias_calibration_summary.csv` - average metrics and standard deviations.
- `reports/independent_bias_calibration_results.csv` - results from the initial independent calibration experiment.
- `reports/independent_bias_calibration_predictions.csv` - predictions for that experiment.
- `reports/validation_bias_correction.csv` - preliminary validation experiment.
