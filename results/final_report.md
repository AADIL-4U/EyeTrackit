# Reading Difficulty Prediction - Final Report

## Project Overview
This project predicts the log-transformed total dwell time per word as a proxy for reading difficulty, using eye-movement features from the Provo Corpus dataset.

## Methods Used
- **Data:** Provo Corpus
- **Feature Engineering:** Log transformations, word-level linguistic features, and context features. Imputed and scaled using a preprocessing pipeline.
- **Models:** 
  1. Ridge Regression (Baseline)
  2. LSTM (Deep Learning sequence model)

## Metrics Summary
| Model   |   RMSE_log |   RMSE_ms |   Pearson_R |   R2_Score |
|:--------|-----------:|----------:|------------:|-----------:|
| Ridge   |     0.1650 |  673.3503 |      0.9489 |     0.9005 |
| LSTM    |     0.0880 |   52.1611 |      0.9858 |     0.9717 |

## Key Findings
- The preferred model based on Pearson Correlation is **LSTM**.
- The Ridge baseline achieved an R-squared of **0.9005**.
- Visualizations show how models match the actual reading speeds and highlight important features.

## Output Files
- Metrics Table: `results/metrics.csv`
- Actual vs Predicted: `results/actual_vs_predicted.png`
- LSTM Training Curves: `results/lstm_training_curves.png`
- Ridge Feature Importance: `results/ridge_feature_importance.png`
- SHAP Summary (Ridge): `results/shap_summary.png`
