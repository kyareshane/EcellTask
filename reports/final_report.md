# 10-K SEC Filing Classification System

## Problem Statement

Develop an NLP-based classification system using SEC 10-K filings.

## Dataset

Source:
https://huggingface.co/datasets/winterForestStump/10-K_sec_filings

Sample Size:
5000 filings

## Preprocessing

- Combined Business and MD&A sections
- Removed missing values
- Text normalization
- Length feature generation

## Label Generation

Labels derived from document length quantiles:

- Class 0 = Low Complexity
- Class 1 = Medium Complexity
- Class 2 = High Complexity

## Feature Engineering

TF-IDF Vectorization

Parameters:

- max_features = 5000
- stop_words = english

## Models Trained

1. AdaBoost
2. XGBoost
3. CatBoost

## Results

| Model | Accuracy | F1 |
|---------|---------|---------|
| AdaBoost | 0.922 | 0.922 |
| XGBoost | 0.978 | 0.978 |
| CatBoost | 0.978 | 0.978 |

## Best Model

CatBoost

Reason:
Highest F1 score and robust performance.

## Deployment

FastAPI endpoint:

POST /predict

Input:

{
  "text":"This company operates several manufacturing facilities and reports annual revenue growth."
}

Output:

{
  "label":"Low Complexity",
  "confidence":0.9986596476217755
}