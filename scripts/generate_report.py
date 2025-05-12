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
    
    # Load faithfulness metrics
    faithfulness_path = os.path.join(results_dir, 'faithfulness', 'all_faithfulness_metrics.csv')
    if os.path.exists(faithfulness_path):
        results['faithfulness_metrics'] = pd.read_csv(faithfulness_path)
    
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


def generate_latex_report(results, output_path, include_plots=False):
    """
    Generate a LaTeX report from the results.
    
    Args:
        results: Dict with result DataFrames
        output_path: Path to save the report
        include_plots: Whether to include plots in the report
    """
    with open(output_path, 'w') as f:
        # Document setup
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{graphicx}\n")
        f.write("\\usepackage{float}\n")
        f.write("\\usepackage{caption}\n")
        f.write("\\usepackage{subcaption}\n")
        f.write("\\usepackage{hyperref}\n")
        f.write("\\usepackage{geometry}\n")
        f.write("\\geometry{margin=1in}\n\n")
        
        f.write("\\title{Causal Pitfalls of Feature Attributions in Financial Machine Learning Models: Results Report}\n")
        f.write(f"\\date{{{datetime.now().strftime('%Y-%m-%d')}}}\n")
        f.write("\\author{}\n\n")
        
        f.write("\\begin{document}\n\n")
        f.write("\\maketitle\n\n")
        
        # Introduction
        f.write("\\section{Introduction}\n\n")
        f.write("This document presents the key results from our experiments evaluating the causal faithfulness ")
        f.write("of various feature attribution methods in financial machine learning models. ")
        f.write("We analyze how well different attribution techniques identify truly causal features across three ")
        f.write("financial scenarios: asset pricing, credit risk assessment, and fraud detection.\n\n")
        
        # Model Performance
        f.write("\\section{Model Performance}\n\n")
        if 'model_performance' in results:
            model_perf = results['model_performance']
            f.write("\\subsection{Overall Model Performance}\n\n")
            
            # Create LaTeX table
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Model Performance Metrics}\n")
            f.write("\\begin{tabular}{lll")
            
            if 'accuracy' in model_perf.columns:
                f.write("r")
            if 'f1' in model_perf.columns:
                f.write("r")
            if 'mse' in model_perf.columns:
                f.write("r")
            if 'r2' in model_perf.columns:
                f.write("r")
            f.write("}\n")
            f.write("\\toprule\n")
            
            f.write("Scenario & Model Type")
            if 'accuracy' in model_perf.columns:
                f.write(" & Accuracy")
            if 'f1' in model_perf.columns:
                f.write(" & F1 Score")
            if 'mse' in model_perf.columns:
                f.write(" & MSE")
            if 'r2' in model_perf.columns:
                f.write(" & R$^2$")
            f.write(" \\\\\n")
            
            f.write("\\midrule\n")
            
            for _, row in model_perf.iterrows():
                f.write(f"{row.get('scenario', '').replace('_', ' ').title()} & {row.get('model_type', '').upper()}")
                if 'accuracy' in model_perf.columns:
                    f.write(f" & {row.get('accuracy', 0):.4f}")
                if 'f1' in model_perf.columns:
                    f.write(f" & {row.get('f1', 0):.4f}")
                if 'mse' in model_perf.columns:
                    f.write(f" & {row.get('mse', 0):.4f}")
                if 'r2' in model_perf.columns:
                    f.write(f" & {row.get('r2', 0):.4f}")
                f.write(" \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
        else:
            f.write("Model performance data not available.\n\n")
        
        # Faithfulness Evaluation
        f.write("\\section{Faithfulness Evaluation}\n\n")
        if 'faithfulness_metrics' in results:
            metrics = results['faithfulness_metrics']
            
            # Overall faithfulness by method
            f.write("\\subsection{Overall Faithfulness Score by Attribution Method}\n\n")
            
            method_avg = metrics.groupby('attribution_method')['overall_faithfulness_score'].mean().reset_index()
            method_avg = method_avg.sort_values('overall_faithfulness_score', ascending=False)
            
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Average Overall Faithfulness Score by Attribution Method}\n")
            f.write("\\begin{tabular}{lr}\n")
            f.write("\\toprule\n")
            f.write("Attribution Method & Overall Faithfulness Score \\\\\n")
            f.write("\\midrule\n")
            
            for _, row in method_avg.iterrows():
                f.write(f"{row['attribution_method'].replace('_', ' ').title()} & {row['overall_faithfulness_score']:.4f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Top-K accuracy by method
            f.write("\\subsection{Top-K Accuracy by Attribution Method}\n\n")
            
            topk_avg = metrics.groupby('attribution_method')['ranking_top_k_accuracy'].mean().reset_index()
            topk_avg = topk_avg.sort_values('ranking_top_k_accuracy', ascending=False)
            
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Average Top-K Accuracy by Attribution Method}\n")
            f.write("\\begin{tabular}{lr}\n")
            f.write("\\toprule\n")
            f.write("Attribution Method & Top-K Accuracy \\\\\n")
            f.write("\\midrule\n")
            
            for _, row in topk_avg.iterrows():
                f.write(f"{row['attribution_method'].replace('_', ' ').title()} & {row['ranking_top_k_accuracy']:.4f} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Performance heatmap (using table5 if available)
            if 'table5_average_by_scenario' in results:
                table5 = results['table5_average_by_scenario']
                
                f.write("\\subsection{Performance Across Scenarios and Methods}\n\n")
                
                f.write("\\begin{table}[H]\n")
                f.write("\\centering\n")
                f.write("\\caption{Overall Faithfulness Score by Scenario and Attribution Method}\n")
                f.write("\\begin{tabular}{l")
                for _ in range(len(table5.columns) - 1):
                    f.write("r")
                f.write("}\n")
                f.write("\\toprule\n")
                
                # Write header
                f.write("Scenario")
                for col in table5.columns[1:]:  # Skip the index column
                    f.write(f" & {col.replace('_', ' ').title()}")
                f.write(" \\\\\n")
                f.write("\\midrule\n")
                
                # Write data
                for idx, row in table5.iterrows():
                    f.write(f"{idx.replace('_', ' ').title()}")
                    for col in table5.columns[1:]:
                        f.write(f" & {row[col]:.3f}")
                    f.write(" \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
        else:
            f.write("Faithfulness metrics data not available.\n\n")
        
        # Scenario-Specific Analysis
        f.write("\\section{Scenario-Specific Analysis}\n\n")
        if 'faithfulness_metrics' in results:
            # For each scenario
            for scenario in metrics['scenario'].unique():
                f.write(f"\\subsection{{{scenario.replace('_', ' ').title()}}}\n\n")
                
                scenario_data = metrics[metrics['scenario'] == scenario]
                
                # Average overall faithfulness by method for this scenario
                scenario_method_avg = scenario_data.groupby('attribution_method')['overall_faithfulness_score'].mean().reset_index()
                scenario_method_avg = scenario_method_avg.sort_values('overall_faithfulness_score', ascending=False)
                
                f.write(f"\\subsubsection{{Overall Faithfulness Score by Method in {scenario.replace('_', ' ').title()}}}\n\n")
                
                f.write("\\begin{table}[H]\n")
                f.write("\\centering\n")
                f.write(f"\\caption{{Faithfulness Scores for {scenario.replace('_', ' ').title()}}}\n")
                f.write("\\begin{tabular}{lr}\n")
                f.write("\\toprule\n")
                f.write("Attribution Method & Overall Faithfulness Score \\\\\n")
                f.write("\\midrule\n")
                
                for _, row in scenario_method_avg.iterrows():
                    f.write(f"{row['attribution_method'].replace('_', ' ').title()} & {row['overall_faithfulness_score']:.4f} \\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
            
                # Best method-model combination for this scenario
                best_combo = scenario_data.loc[scenario_data['overall_faithfulness_score'].idxmax()]
                
                f.write(f"\\subsubsection{{Best Method-Model Combination for {scenario.replace('_', ' ').title()}}}\n\n")
                f.write("\\begin{itemize}\n")
                f.write(f"\\item \\textbf{{Attribution Method}}: {best_combo['attribution_method'].replace('_', ' ').title()}\n")
                f.write(f"\\item \\textbf{{Model Type}}: {best_combo['model_type'].upper()}\n")
                f.write(f"\\item \\textbf{{Overall Faithfulness Score}}: {best_combo['overall_faithfulness_score']:.4f}\n")
                f.write(f"\\item \\textbf{{Top-K Accuracy}}: {best_combo['ranking_top_k_accuracy']:.4f}\n")
                f.write(f"\\item \\textbf{{Attribution Ratio}}: {best_combo['magnitude_attribution_ratio']:.4f}\n")
                f.write("\\end{itemize}\n\n")
        else:
            f.write("Scenario-specific analysis data not available.\n\n")
        
        # Key Findings
        f.write("\\section{Key Findings}\n\n")
        if 'faithfulness_metrics' in results:
            # Best overall method
            method_avg = metrics.groupby('attribution_method')['overall_faithfulness_score'].mean().reset_index()
            best_method = method_avg.loc[method_avg['overall_faithfulness_score'].idxmax()]['attribution_method']
            best_method_score = method_avg.loc[method_avg['overall_faithfulness_score'].idxmax()]['overall_faithfulness_score']
            
            # Best overall model
            model_avg_all = metrics.groupby('model_type')['overall_faithfulness_score'].mean().reset_index()
            best_model = model_avg_all.loc[model_avg_all['overall_faithfulness_score'].idxmax()]['model_type']
            best_model_score = model_avg_all.loc[model_avg_all['overall_faithfulness_score'].idxmax()]['overall_faithfulness_score']
            
            # Best overall scenario
            scenario_avg_all = metrics.groupby('scenario')['overall_faithfulness_score'].mean().reset_index()
            best_scenario = scenario_avg_all.loc[scenario_avg_all['overall_faithfulness_score'].idxmax()]['scenario']
            
            # Best overall combination
            best_overall = metrics.loc[metrics['overall_faithfulness_score'].idxmax()]
            
            f.write("\\subsection{Summary of Best Performers}\n\n")
            f.write("\\begin{itemize}\n")
            f.write(f"\\item \\textbf{{Best Attribution Method Overall}}: {best_method.replace('_', ' ').title()} (Score: {best_method_score:.4f})\n")
            f.write(f"\\item \\textbf{{Best Model Type Overall}}: {best_model.upper()} (Score: {best_model_score:.4f})\n")
            f.write(f"\\item \\textbf{{Best Performing Scenario}}: {best_scenario.replace('_', ' ').title()}\n")
            f.write(f"\\item \\textbf{{Best Overall Combination}}: {best_overall['attribution_method'].replace('_', ' ').title()} with {best_overall['model_type'].upper()} on {best_overall['scenario'].replace('_', ' ').title()} (Score: {best_overall['overall_faithfulness_score']:.4f})\n")
            f.write("\\end{itemize}\n\n")
            
            # Observations about causal vs non-causal features
            f.write("\\subsection{Observations on Causal Feature Identification}\n\n")
            
            # Attribution ratio analysis
            attr_ratio_avg = metrics.groupby('attribution_method')['magnitude_attribution_ratio'].mean().reset_index()
            attr_ratio_avg = attr_ratio_avg.sort_values('magnitude_attribution_ratio', ascending=False)
            
            best_ratio_method = attr_ratio_avg.iloc[0]['attribution_method']
            best_ratio_value = attr_ratio_avg.iloc[0]['magnitude_attribution_ratio']
            
            # Top-K accuracy analysis
            topk_scenario_avg = metrics.groupby('scenario')['ranking_top_k_accuracy'].mean().reset_index()
            easiest_scenario = topk_scenario_avg.loc[topk_scenario_avg['ranking_top_k_accuracy'].idxmax()]['scenario']
            hardest_scenario = topk_scenario_avg.loc[topk_scenario_avg['ranking_top_k_accuracy'].idxmin()]['scenario']
            
            f.write("\\begin{itemize}\n")
            f.write(f"\\item \\textbf{{Best Method for Differentiating Causal vs. Non-Causal Features}}: {best_ratio_method.replace('_', ' ').title()} (Ratio: {best_ratio_value:.4f})\n")
            f.write(f"\\item \\textbf{{Easiest Scenario for Identifying Causal Features}}: {easiest_scenario.replace('_', ' ').title()}\n")
            f.write(f"\\item \\textbf{{Most Challenging Scenario for Identifying Causal Features}}: {hardest_scenario.replace('_', ' ').title()}\n")
            f.write("\\end{itemize}\n\n")
            
            # General observations
            f.write("\\subsection{General Observations}\n\n")
            f.write("\\begin{enumerate}\n")
            f.write("\\item \\textbf{Attribution Method Performance}: ")
            if 'shap' in metrics['attribution_method'].unique() and best_method == 'shap':
                f.write("SHAP consistently outperforms other attribution methods in identifying causal features. ")
                f.write("This aligns with its theoretical guarantees based on Shapley values from cooperative game theory.\n\n")
            elif 'integrated_gradients' in metrics['attribution_method'].unique() and best_method == 'integrated_gradients':
                f.write("Integrated Gradients performs well in identifying causal features. ")
                f.write("This may be due to its axiomatic approach which satisfies important properties like completeness and implementation invariance.\n\n")
            else:
                f.write(f"{best_method.replace('_', ' ').title()} shows the strongest performance in identifying causal features across different scenarios and models.\n\n")
            
            f.write("\\item \\textbf{Model-Specific Patterns}: ")
            if best_model == 'xgboost':
                f.write("Tree-based models (XGBoost) tend to show better faithfulness scores in terms of attributions. ")
                f.write("This could be because tree structures naturally capture feature importance in a way that aligns better with causal relationships.\n\n")
            elif best_model == 'mlp':
                f.write("MLP models show good performance in terms of attribution faithfulness. ")
                f.write("The simpler architecture may help in learning more direct relationships between inputs and outputs.\n\n")
            else:
                f.write("LSTM models show distinctive patterns in attribution faithfulness, which may be related to their sequence processing capabilities.\n\n")
            
            f.write("\\item \\textbf{Scenario-Specific Challenges}: ")
            if 'fraud_detection' in metrics['scenario'].unique() and hardest_scenario == 'fraud_detection':
                f.write("The fraud detection scenario presents unique challenges for attribution methods. ")
                f.write("This is likely due to the presence of indirect indicators that are consequences rather than causes of fraud, creating a particularly challenging causal structure.\n\n")
            elif 'credit_risk' in metrics['scenario'].unique() and hardest_scenario == 'credit_risk':
                f.write("The credit risk scenario highlights difficulties in distinguishing true behavioral drivers from proxy features. ")
                f.write("This has important implications for fair lending and regulatory compliance.\n\n")
            else:
                f.write(f"The {hardest_scenario.replace('_', ' ').title()} scenario presents unique causal identification challenges for attribution methods.\n\n")
            
            f.write("\\item \\textbf{Practical Recommendations}: Based on these findings, practitioners in the financial domain should:\n")
            f.write("\\begin{itemize}\n")
            f.write(f"\\item Consider using {best_method.replace('_', ' ').title()} for feature attribution when causal understanding is crucial\n")
            f.write(f"\\item Be particularly cautious when interpreting attributions in {hardest_scenario.replace('_', ' ').title()}-like scenarios\n")
            f.write(f"\\item When possible, use {best_model.upper()} models for better attribution faithfulness\n")
            f.write("\\item Always validate attribution results against domain knowledge and cross-verify with multiple attribution methods\n")
            f.write("\\end{itemize}\n")
            f.write("\\end{enumerate}\n\n")
        else:
            f.write("Key findings data not available.\n\n")
        
        # Conclusion
        f.write("\\section{Conclusion}\n\n")
        f.write("This report has presented a comprehensive analysis of the causal faithfulness of various feature attribution methods ")
        f.write("across different financial machine learning models and scenarios. The results highlight both the strengths and limitations ")
        f.write("of current attribution techniques in identifying true causal relationships, with important implications for ")
        f.write("model explainability, regulatory compliance, and decision-making in financial contexts.\n\n")
        
        f.write("The findings underscore the need for practitioners to exercise caution when interpreting feature attributions as causal explanations ")
        f.write("and suggest avenues for developing more causally-aware interpretability frameworks in finance.\n\n")
        
        # End document
        f.write("\\end{document}")


def main():
    """Main function to generate the report."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create directories
    dirs = create_directories()
    results_dir = args.results_dir if args.results_dir else dirs['results']
    output_dir = args.output_dir if args.output_dir else os.path.join(dirs['results'], 'report')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    logger.info("Loading results from {results_dir}...")
    results = load_all_results(results_dir)
    
    # Check if we have metrics
    if 'faithfulness_metrics' not in results:
        logger.warning("No faithfulness metrics found. The report may be incomplete.")
    
    # Generate report
    logger.info(f"Generating {args.format} report...")
    
    if args.format == 'markdown':
        output_path = os.path.join(output_dir, 'report.md')
        generate_markdown_report(results, output_path, args.include_plots)
    elif args.format == 'latex':
        output_path = os.path.join(output_dir, 'report.tex')
        generate_latex_report(results, output_path, args.include_plots)
    
    logger.info(f"Report saved to {output_path}")
    
    # If LaTeX, try to compile it
    if args.format == 'latex':
        try:
            import subprocess
            logger.info("Attempting to compile LaTeX report...")
            subprocess.run(['pdflatex', '-output-directory', output_dir, output_path], check=True)
            logger.info(f"PDF report saved to {os.path.join(output_dir, 'report.pdf')}")
        except Exception as e:
            logger.warning(f"Could not compile LaTeX report: {str(e)}")
            logger.info("You can compile it manually using pdflatex.")


if __name__ == "__main__":
    main()