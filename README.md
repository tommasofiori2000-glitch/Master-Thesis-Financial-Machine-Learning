# Master-Thesis-Financial-Machine-Learning

# Market Return Prediction with ML

Master Thesis Project – Data Science

## Overview
This project develops a machine learning pipeline for predicting forward market returns
under strict data-leakage constraints, inspired by Kaggle-style financial competitions.

## Models
- XGBoost regressor (base model)
- ElasticNet residual correction
- Walk-forward validation (Kaggle-like)

## Features
- Lagged returns
- Lagged risk-free rate
- Lagged market excess returns
- Macroeconomic and sector indicators (D*, E*, I*, M*, P*, S*, V*)

## Evaluation
- Walk-forward Sharpe Ratio
- Directional accuracy
- Decile monotonicity
- Correlation diagnostics

## Notes
Original dataset is not publicly available due to competition constraints.
