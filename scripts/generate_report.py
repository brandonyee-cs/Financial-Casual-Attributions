#!/usr/bin/env python
"""
Script to generate a comprehensive report of all results for the paper.

This script aggregates all the evaluation results, creates visualizations,
and generates a markdown/LaTeX report with key findings that can be
incorporated into the paper.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import create_directories, setup_logging, set_seeds


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate comprehensive report for paper')
    
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing all results')
    parser.add_argument('--output_dir', type=str, default='report',
                        help='Directory to save the report')
    parser.add_argument('--format', type=str, default='markdown',
                        choices=['markdown', 'latex'],
                        help='Report format (markdown or latex)')
    parser.add_argument('--include_plots', action='store_true',
                        help='Include plots in the report')
    
    return parser.parse_args()


def load_all_results(results_dir):
    """
    Load all results files from the results directory.
    
    Args:
        results_dir: Directory containing results
        
    Returns:
        Dict with loaded DataFrames
    """
    results = {}
    
    # Load model performance
    model_perf_path = os.path.join(results_dir, 'model_performance.csv')
    if os.path.exists(model_perf_path):
        results['model_performance'] = pd.read_csv(model_perf_path)
    
    # Load faithfulness metrics (check both locations)
    faithfulness_paths = [
        os.path.join(results_dir, 'faithfulness', 'all_faithfulness_metrics.csv'),
        os.path.join(results_dir, 'all_faithfulness_metrics.csv')  # Also check in root results dir
    ]
    
    for path in faithfulness_paths:
        if os.path.exists(path):
            results['faithfulness_metrics'] = pd.read_csv(path)
            break
    
    # Load individual table files
    tables_dir = os.path.join(results_dir, 'faithfulness', 'tables')
    if os.path.exists(tables_dir):
        for file in os.listdir(tables_dir):
            if file.endswith('.csv'):
                table_name = file.split('.')[0]
                results[table_name] = pd.read_csv(os.path.join(tables_dir, file))
    
    return results


def generate_markdown_report(results, output_path, include_plots=False):
    """
    Generate a markdown report from the results.
    
    Args:
        results: Dict with result DataFrames
        output_path: Path to save the report
        include_plots: Whether to include plots in the report
    """
    with open(output_path, 'w') as f:
        # Title and introduction
        f.write("# Causal Pitfalls of Feature Attributions in Financial Machine Learning Models\n\n")
        f.write("## Results Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("This document presents the key results from our experiments evaluating the causal faithfulness ")
        f.write("of various feature attribution methods in financial machine learning models.\n\n")
        
        # Table of contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Model Performance](#model-performance)\n")
        f.write("2. [Faithfulness Evaluation](#faithfulness-evaluation)\n")
        f.write("3. [Scenario-Specific Analysis](#scenario-specific-analysis)\n")
        f.write("4. [Model-Specific Analysis](#model-specific-analysis)\n")
        f.write("5. [Attribution Method Comparison](#attribution-method-comparison)\n")
        f.write("6. [Key Findings](#key-findings)\n\n")
        
        # Model Performance
        f.write("## Model Performance\n\n")
        if 'model_performance' in results:
            model_perf = results['model_performance']
            f.write("### Overall Model Performance\n\n")
            
            # Create markdown table
            f.write("| Scenario | Model Type | ")
            if 'accuracy' in model_perf.columns:
                f.write("Accuracy | ")
            if 'f1' in model_perf.columns:
                f.write("F1 Score | ")
            if 'mse' in model_perf.columns:
                f.write("MSE | ")
            if 'r2' in model_perf.columns:
                f.write("R² | ")
            f.write("\n")
            
            f.write("|" + "-" * 10 + "|" + "-" * 12 + "|")
            if 'accuracy' in model_perf.columns:
                f.write("-" * 10 + "|")
            if 'f1' in model_perf.columns:
                f.write("-" * 10 + "|")
            if 'mse' in model_perf.columns:
                f.write("-" * 10 + "|")
            if 'r2' in model_perf.columns:
                f.write("-" * 10 + "|")
            f.write("\n")
            
            for _, row in model_perf.iterrows():
                f.write(f"| {row.get('scenario', '')} | {row.get('model_type', '')} | ")
                if 'accuracy' in model_perf.columns:
                    f.write(f"{row.get('accuracy', 0):.4f} | ")
                if 'f1' in model_perf.columns:
                    f.write(f"{row.get('f1', 0):.4f} | ")
                if 'mse' in model_perf.columns:
                    f.write(f"{row.get('mse', 0):.4f} | ")
                if 'r2' in model_perf.columns:
                    f.write(f"{row.get('r2', 0):.4f} | ")
                f.write("\n")
            
            f.write("\n\n")
        else:
            f.write("*Model performance data not available.*\n\n")
        
        # Faithfulness Evaluation
        f.write("## Faithfulness Evaluation\n\n")
        if 'faithfulness_metrics' in results:
            metrics = results['faithfulness_metrics']
            
            # Overall faithfulness by method
            f.write("### Overall Faithfulness Score by Attribution Method\n\n")
            
            method_avg = metrics.groupby('attribution_method')['overall_faithfulness_score'].mean().reset_index()
            method_avg = method_avg.sort_values('overall_faithfulness_score', ascending=False)
            
            f.write("| Attribution Method | Overall Faithfulness Score |\n")
            f.write("|" + "-" * 20 + "|" + "-" * 28 + "|\n")
            for _, row in method_avg.iterrows():
                f.write(f"| {row['attribution_method']} | {row['overall_faithfulness_score']:.4f} |\n")
            
            f.write("\n\n")
            
            # Top-K accuracy by method
            f.write("### Top-K Accuracy by Attribution Method\n\n")
            
            topk_avg = metrics.groupby('attribution_method')['ranking_top_k_accuracy'].mean().reset_index()
            topk_avg = topk_avg.sort_values('ranking_top_k_accuracy', ascending=False)
            
            f.write("| Attribution Method | Top-K Accuracy |\n")
            f.write("|" + "-" * 20 + "|" + "-" * 15 + "|\n")
            for _, row in topk_avg.iterrows():
                f.write(f"| {row['attribution_method']} | {row['ranking_top_k_accuracy']:.4f} |\n")
            
            f.write("\n\n")
        else:
            f.write("*Faithfulness metrics data not available.*\n\n")
        
        # Scenario-Specific Analysis
        f.write("## Scenario-Specific Analysis\n\n")
        if 'faithfulness_metrics' in results:
            # For each scenario
            for scenario in metrics['scenario'].unique():
                f.write(f"### {scenario.replace('_', ' ').title()}\n\n")
                
                scenario_data = metrics[metrics['scenario'] == scenario]
                
                # Average overall faithfulness by method for this scenario
                scenario_method_avg = scenario_data.groupby('attribution_method')['overall_faithfulness_score'].mean().reset_index()
                scenario_method_avg = scenario_method_avg.sort_values('overall_faithfulness_score', ascending=False)
                
                f.write(f"#### Overall Faithfulness Score by Method in {scenario.replace('_', ' ').title()}\n\n")
                
                f.write("| Attribution Method | Overall Faithfulness Score |\n")
                f.write("|" + "-" * 20 + "|" + "-" * 28 + "|\n")
                for _, row in scenario_method_avg.iterrows():
                    f.write(f"| {row['attribution_method']} | {row['overall_faithfulness_score']:.4f} |\n")
                
                f.write("\n\n")
            
                # Best method-model combination for this scenario
                best_combo = scenario_data.loc[scenario_data['overall_faithfulness_score'].idxmax()]
                
                f.write(f"#### Best Method-Model Combination for {scenario.replace('_', ' ').title()}\n\n")
                f.write(f"- **Attribution Method**: {best_combo['attribution_method']}\n")
                f.write(f"- **Model Type**: {best_combo['model_type']}\n")
                f.write(f"- **Overall Faithfulness Score**: {best_combo['overall_faithfulness_score']:.4f}\n")
                f.write(f"- **Top-K Accuracy**: {best_combo['ranking_top_k_accuracy']:.4f}\n")
                f.write(f"- **Attribution Ratio**: {best_combo['magnitude_attribution_ratio']:.4f}\n\n")
        else:
            f.write("*Scenario-specific analysis data not available.*\n\n")
        
        # Model-Specific Analysis
        f.write("## Model-Specific Analysis\n\n")
        if 'faithfulness_metrics' in results:
            # For each model type
            for model_type in metrics['model_type'].unique():
                f.write(f"### {model_type.upper()}\n\n")
                
                model_data = metrics[metrics['model_type'] == model_type]
                
                # Average overall faithfulness by method for this model
                model_method_avg = model_data.groupby('attribution_method')['overall_faithfulness_score'].mean().reset_index()
                model_method_avg = model_method_avg.sort_values('overall_faithfulness_score', ascending=False)
                
                f.write(f"#### Overall Faithfulness Score by Method for {model_type.upper()}\n\n")
                
                f.write("| Attribution Method | Overall Faithfulness Score |\n")
                f.write("|" + "-" * 20 + "|" + "-" * 28 + "|\n")
                for _, row in model_method_avg.iterrows():
                    f.write(f"| {row['attribution_method']} | {row['overall_faithfulness_score']:.4f} |\n")
                
                f.write("\n\n")
            
                # Best method-scenario combination for this model
                best_combo = model_data.loc[model_data['overall_faithfulness_score'].idxmax()]
                
                f.write(f"#### Best Method-Scenario Combination for {model_type.upper()}\n\n")
                f.write(f"- **Attribution Method**: {best_combo['attribution_method']}\n")
                f.write(f"- **Scenario**: {best_combo['scenario'].replace('_', ' ').title()}\n")
                f.write(f"- **Overall Faithfulness Score**: {best_combo['overall_faithfulness_score']:.4f}\n")
                f.write(f"- **Top-K Accuracy**: {best_combo['ranking_top_k_accuracy']:.4f}\n")
                f.write(f"- **Attribution Ratio**: {best_combo['magnitude_attribution_ratio']:.4f}\n\n")
        else:
            f.write("*Model-specific analysis data not available.*\n\n")
        
        # Attribution Method Comparison
        f.write("## Attribution Method Comparison\n\n")
        if 'faithfulness_metrics' in results:
            # For each attribution method
            for method in metrics['attribution_method'].unique():
                f.write(f"### {method.replace('_', ' ').title()}\n\n")
                
                method_data = metrics[metrics['attribution_method'] == method]
                
                # Performance across scenarios
                f.write(f"#### Performance Across Scenarios\n\n")
                
                scenario_avg = method_data.groupby('scenario')['overall_faithfulness_score'].mean().reset_index()
                scenario_avg = scenario_avg.sort_values('overall_faithfulness_score', ascending=False)
                
                f.write("| Scenario | Overall Faithfulness Score |\n")
                f.write("|" + "-" * 15 + "|" + "-" * 28 + "|\n")
                for _, row in scenario_avg.iterrows():
                    f.write(f"| {row['scenario'].replace('_', ' ').title()} | {row['overall_faithfulness_score']:.4f} |\n")
                
                f.write("\n\n")
                
                # Performance across model types
                f.write(f"#### Performance Across Model Types\n\n")
                
                model_avg = method_data.groupby('model_type')['overall_faithfulness_score'].mean().reset_index()
                model_avg = model_avg.sort_values('overall_faithfulness_score', ascending=False)
                
                f.write("| Model Type | Overall Faithfulness Score |\n")
                f.write("|" + "-" * 12 + "|" + "-" * 28 + "|\n")
                for _, row in model_avg.iterrows():
                    f.write(f"| {row['model_type'].upper()} | {row['overall_faithfulness_score']:.4f} |\n")
                
                f.write("\n\n")
                
                # Best scenario-model combination for this method
                best_combo = method_data.loc[method_data['overall_faithfulness_score'].idxmax()]
                
                f.write(f"#### Best Scenario-Model Combination for {method.replace('_', ' ').title()}\n\n")
                f.write(f"- **Scenario**: {best_combo['scenario'].replace('_', ' ').title()}\n")
                f.write(f"- **Model Type**: {best_combo['model_type'].upper()}\n")
                f.write(f"- **Overall Faithfulness Score**: {best_combo['overall_faithfulness_score']:.4f}\n")
                f.write(f"- **Top-K Accuracy**: {best_combo['ranking_top_k_accuracy']:.4f}\n")
                f.write(f"- **Attribution Ratio**: {best_combo['magnitude_attribution_ratio']:.4f}\n\n")
        else:
            f.write("*Attribution method comparison data not available.*\n\n")
        
        # Key Findings
        f.write("## Key Findings\n\n")
        if 'faithfulness_metrics' in results:
            # Best overall method
            best_method = method_avg.iloc[0]['attribution_method']
            best_method_score = method_avg.iloc[0]['overall_faithfulness_score']
            
            # Best overall model
            model_avg_all = metrics.groupby('model_type')['overall_faithfulness_score'].mean().reset_index()
            best_model = model_avg_all.loc[model_avg_all['overall_faithfulness_score'].idxmax()]['model_type']
            best_model_score = model_avg_all.loc[model_avg_all['overall_faithfulness_score'].idxmax()]['overall_faithfulness_score']
            
            # Best overall scenario
            scenario_avg_all = metrics.groupby('scenario')['overall_faithfulness_score'].mean().reset_index()
            best_scenario = scenario_avg_all.loc[scenario_avg_all['overall_faithfulness_score'].idxmax()]['scenario']
            
            # Best overall combination
            best_overall = metrics.loc[metrics['overall_faithfulness_score'].idxmax()]
            
            f.write("### Summary of Best Performers\n\n")
            f.write(f"- **Best Attribution Method Overall**: {best_method.replace('_', ' ').title()} (Score: {best_method_score:.4f})\n")
            f.write(f"- **Best Model Type Overall**: {best_model.upper()} (Score: {best_model_score:.4f})\n")
            f.write(f"- **Best Performing Scenario**: {best_scenario.replace('_', ' ').title()}\n")
            f.write(f"- **Best Overall Combination**: {best_overall['attribution_method'].replace('_', ' ').title()} with {best_overall['model_type'].upper()} on {best_overall['scenario'].replace('_', ' ').title()} (Score: {best_overall['overall_faithfulness_score']:.4f})\n\n")
            
            # Observations about causal vs non-causal features
            f.write("### Observations on Causal Feature Identification\n\n")
            
            # Attribution ratio analysis
            attr_ratio_avg = metrics.groupby('attribution_method')['magnitude_attribution_ratio'].mean().reset_index()
            attr_ratio_avg = attr_ratio_avg.sort_values('magnitude_attribution_ratio', ascending=False)
            
            best_ratio_method = attr_ratio_avg.iloc[0]['attribution_method']
            best_ratio_value = attr_ratio_avg.iloc[0]['magnitude_attribution_ratio']
            
            f.write(f"- **Best Method for Differentiating Causal vs. Non-Causal Features**: {best_ratio_method.replace('_', ' ').title()} (Ratio: {best_ratio_value:.4f})\n")
            
            # Top-K accuracy analysis
            topk_scenario_avg = metrics.groupby('scenario')['ranking_top_k_accuracy'].mean().reset_index()
            easiest_scenario = topk_scenario_avg.loc[topk_scenario_avg['ranking_top_k_accuracy'].idxmax()]['scenario']
            hardest_scenario = topk_scenario_avg.loc[topk_scenario_avg['ranking_top_k_accuracy'].idxmin()]['scenario']
            
            f.write(f"- **Easiest Scenario for Identifying Causal Features**: {easiest_scenario.replace('_', ' ').title()}\n")
            f.write(f"- **Most Challenging Scenario for Identifying Causal Features**: {hardest_scenario.replace('_', ' ').title()}\n\n")
            
            # General observations
            f.write("### General Observations\n\n")
            f.write("1. **Attribution Method Performance**: ")
            if 'shap' in metrics['attribution_method'].unique() and best_method == 'shap':
                f.write("SHAP consistently outperforms other attribution methods in identifying causal features. ")
                f.write("This aligns with its theoretical guarantees based on Shapley values from cooperative game theory.\n\n")
            elif 'integrated_gradients' in metrics['attribution_method'].unique() and best_method == 'integrated_gradients':
                f.write("Integrated Gradients performs well in identifying causal features. ")
                f.write("This may be due to its axiomatic approach which satisfies important properties like completeness and implementation invariance.\n\n")
            else:
                f.write(f"{best_method.replace('_', ' ').title()} shows the strongest performance in identifying causal features across different scenarios and models.\n\n")
            
            f.write("2. **Model-Specific Patterns**: ")
            if best_model == 'xgboost':
                f.write("Tree-based models (XGBoost) tend to show better faithfulness scores in terms of attributions. ")
                f.write("This could be because tree structures naturally capture feature importance in a way that aligns better with causal relationships.\n\n")
            elif best_model == 'mlp':
                f.write("MLP models show good performance in terms of attribution faithfulness. ")
                f.write("The simpler architecture may help in learning more direct relationships between inputs and outputs.\n\n")
            else:
                f.write("LSTM models show distinctive patterns in attribution faithfulness, which may be related to their sequence processing capabilities.\n\n")
            
            f.write("3. **Scenario-Specific Challenges**: ")
            if 'fraud_detection' in metrics['scenario'].unique() and hardest_scenario == 'fraud_detection':
                f.write("The fraud detection scenario presents unique challenges for attribution methods. ")
                f.write("This is likely due to the presence of indirect indicators that are consequences rather than causes of fraud, creating a particularly challenging causal structure.\n\n")
            elif 'credit_risk' in metrics['scenario'].unique() and hardest_scenario == 'credit_risk':
                f.write("The credit risk scenario highlights difficulties in distinguishing true behavioral drivers from proxy features. ")
                f.write("This has important implications for fair lending and regulatory compliance.\n\n")
            else:
                f.write(f"The {hardest_scenario.replace('_', ' ').title()} scenario presents unique causal identification challenges for attribution methods.\n\n")
            
            f.write("4. **Practical Recommendations**: Based on these findings, practitioners in the financial domain should:\n\n")
            f.write(f"   - Consider using {best_method.replace('_', ' ').title()} for feature attribution when causal understanding is crucial\n")
            f.write(f"   - Be particularly cautious when interpreting attributions in {hardest_scenario.replace('_', ' ').title()}-like scenarios\n")
            f.write(f"   - When possible, use {best_model.upper()} models for better attribution faithfulness\n")
            f.write("   - Always validate attribution results against domain knowledge and cross-verify with multiple attribution methods\n\n")
        else:
            f.write("*Key findings data not available.*\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("This report has presented a comprehensive analysis of the causal faithfulness of various feature attribution methods ")
        f.write("across different financial machine learning models and scenarios. The results highlight both the strengths and limitations ")
        f.write("of current attribution techniques in identifying true causal relationships, with important implications for ")
        f.write("model explainability, regulatory compliance, and decision-making in financial contexts.\n\n")
        
        f.write("The findings underscore the need for practitioners to exercise caution when interpreting feature attributions as causal explanations ")
        f.write("and suggest avenues for developing more causally-aware interpretability frameworks in finance.")