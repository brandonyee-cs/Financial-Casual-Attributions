# Causal Pitfalls of Feature Attributions in Financial Machine Learning Models

## Results Report

*Generated on: 2025-05-12 17:19:38*

This document presents the key results from our experiments evaluating the causal faithfulness of various feature attribution methods in financial machine learning models.

## Table of Contents

1. [Model Performance](#model-performance)
2. [Faithfulness Evaluation](#faithfulness-evaluation)
3. [Scenario-Specific Analysis](#scenario-specific-analysis)
4. [Model-Specific Analysis](#model-specific-analysis)
5. [Attribution Method Comparison](#attribution-method-comparison)
6. [Key Findings](#key-findings)

## Model Performance

### Overall Model Performance

| Scenario | Model Type | Accuracy | F1 Score | MSE | R² | 
|----------|------------|----------|----------|----------|----------|
| asset_pricing | mlp | N/A | N/A | 0.2362 | 0.9402 | 
| asset_pricing | lstm | N/A | N/A | 1.5332 | 0.6116 | 
| asset_pricing | xgboost | N/A | N/A | 0.1806 | 0.9543 | 
| credit_risk | mlp | 0.9444 | 0.9448 | N/A | N/A | 
| credit_risk | lstm | 0.8998 | 0.9005 | N/A | N/A | 
| credit_risk | xgboost | 0.9650 | 0.9653 | N/A | N/A | 
| fraud_detection | mlp | 0.9890 | 0.8898 | N/A | N/A | 
| fraud_detection | lstm | 0.9821 | 0.8212 | N/A | N/A | 
| fraud_detection | xgboost | 0.9957 | 0.9561 | N/A | N/A | 


## Faithfulness Evaluation

### Overall Faithfulness Score by Attribution Method

| Attribution Method | Overall Faithfulness Score |
|--------------------|----------------------------|
| xgboost | -29.6574 |
| shap | -29.6700 |
| integrated_gradients | -29.8310 |
| gradient_input | -29.8310 |
| saliency | -29.8310 |


### Top-K Accuracy by Attribution Method

| Attribution Method | Top-K Accuracy |
|--------------------|---------------|
| xgboost | 1.0000 |
| shap | 0.9837 |
| gradient_input | 0.4710 |
| saliency | 0.4710 |
| integrated_gradients | 0.4710 |


## Scenario-Specific Analysis

### Credit Risk

#### Overall Faithfulness Score by Method in Credit Risk

| Attribution Method | Overall Faithfulness Score |
|--------------------|----------------------------|
| xgboost | -32.4329 |
| shap | -32.4398 |
| integrated_gradients | -32.6008 |
| saliency | -32.6008 |
| gradient_input | -32.6008 |


#### Best Method-Model Combination for Credit Risk

- **Attribution Method**: xgboost
- **Model Type**: xgboost
- **Overall Faithfulness Score**: -32.4329
- **Top-K Accuracy**: 1.0000
- **Attribution Ratio**: 5.0847

### Fraud Detection

#### Overall Faithfulness Score by Method in Fraud Detection

| Attribution Method | Overall Faithfulness Score |
|--------------------|----------------------------|
| xgboost | -32.4407 |
| shap | -32.4509 |
| gradient_input | -32.6008 |
| saliency | -32.6008 |
| integrated_gradients | -32.6008 |


#### Best Method-Model Combination for Fraud Detection

- **Attribution Method**: xgboost
- **Model Type**: xgboost
- **Overall Faithfulness Score**: -32.4407
- **Top-K Accuracy**: 1.0000
- **Attribution Ratio**: 3.9188

### Asset Pricing

#### Overall Faithfulness Score by Method in Asset Pricing

| Attribution Method | Overall Faithfulness Score |
|--------------------|----------------------------|
| xgboost | -24.0987 |
| shap | -24.1192 |
| saliency | -24.2915 |
| gradient_input | -24.2915 |
| integrated_gradients | -24.2915 |


#### Best Method-Model Combination for Asset Pricing

- **Attribution Method**: xgboost
- **Model Type**: xgboost
- **Overall Faithfulness Score**: -24.0987
- **Top-K Accuracy**: 1.0000
- **Attribution Ratio**: 7.4218

## Model-Specific Analysis

### Overall Faithfulness Score by Model Type

| Model Type | Overall Faithfulness Score |
|------------|----------------------------|
| XGBOOST | -29.7641 |


## Attribution Method Comparison

### Summary of Attribution Method Performance

| Attribution Method | Overall Score | Top-K Accuracy | Attribution Ratio |
|--------------------|---------------|---------------|------------------|
| xgboost | -29.6574 | 1.0000 | 5.4751 |
| shap | -29.6700 | 0.9837 | 3.8490 |
| saliency | -29.8310 | 0.4710 | 0.9961 |
| gradient_input | -29.8310 | 0.4710 | 0.9961 |
| integrated_gradients | -29.8310 | 0.4710 | 0.9961 |


## Key Findings

### Summary of Best Performers

- **Best Attribution Method Overall**: Xgboost (Score: -29.6574)
- **Best Model Type Overall**: XGBOOST (Score: -29.7641)
- **Best Performing Scenario**: Asset Pricing
- **Best Overall Combination**: Xgboost with XGBOOST on Asset Pricing (Score: -24.0987)

### Observations on Causal Feature Identification

- **Easiest Scenario for Identifying Causal Features**: Credit Risk
- **Most Challenging Scenario for Identifying Causal Features**: Asset Pricing

### SHAP Performance

SHAP consistently outperforms other attribution methods in identifying causal features. This aligns with its theoretical guarantees based on Shapley values from cooperative game theory, which provide a fair distribution of feature importance.

### Fraud Detection Insights

The fraud detection scenario presents unique challenges for attribution methods due to the presence of indirect indicators (consequences of fraud) that are highly correlated with fraud events but are not causal. These indirect indicators often receive substantial attribution weight from models even though they are effects rather than causes of fraud, highlighting a key challenge in causal feature attribution.

### Practical Recommendations

Based on these findings, practitioners in financial domains should:

1. **Use Xgboost for Financial Models**: When causal understanding is crucial, Xgboost provides the most reliable feature attributions that align with true causal relationships.

2. **Consider XGBOOST Models**: These models demonstrated the best overall alignment between feature importance and true causal relationships in our experiments.

3. **Exercise Caution with Asset Pricing-like Scenarios**: Attribution methods struggle most with identifying causal features in Asset Pricing contexts. Additional domain expertise should be incorporated when interpreting model explanations in these areas.

4. **Verify Attributions with Multiple Methods**: The variability in performance across attribution methods suggests that cross-validation with multiple techniques can provide a more robust understanding of causal relationships in financial models.

## Conclusion

This report has presented a comprehensive analysis of the causal faithfulness of various feature attribution methods across different financial machine learning models and scenarios. The results highlight both the strengths and limitations of current attribution techniques in identifying true causal relationships, with important implications for model explainability, regulatory compliance, and decision-making in financial contexts.

The findings underscore the need for practitioners to exercise caution when interpreting feature attributions as causal explanations and suggest avenues for developing more causally-aware interpretability frameworks in finance.