#!/usr/bin/env python
"""
Script to evaluate the causal faithfulness of feature attributions.

This script evaluates how well different attribution methods identify truly
causal features in models trained on the three financial scenarios. It computes
various faithfulness metrics and generates comparative visualizations.
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (
    CausalFaithfulnessEvaluator,
    plot_attribution_heatmap,
    plot_comparison_results
)
from src.utils import (
    create_directories,
    save_config,
    load_config,
    setup_logging,
    set_seeds,
    load_dataset,
    load_model,
    save_results,
    load_results
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate causal faithfulness of feature attributions')
    
    parser.add_argument('--config', type=str, default='config/evaluation_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing datasets')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing attribution results')
    parser.add_argument('--output_dir', type=str, default='results/faithfulness',
                        help='Directory to save faithfulness evaluation results')
    parser.add_argument('--plots_dir', type=str, default='plots',
                        help='Directory to save plots')
    parser.add_argument('--scenarios', type=str, nargs='+',
                        default=['asset_pricing', 'credit_risk', 'fraud_detection'],
                        help='Scenarios to evaluate')
    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['mlp', 'lstm', 'xgboost'],
                        help='Model types to evaluate')
    parser.add_argument('--attribution_methods', type=str, nargs='+',
                        default=['saliency', 'gradient_input', 'integrated_gradients', 'shap'],
                        help='Attribution methods to evaluate')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def evaluate_scenario_model_attribution(scenario: str, model_type: str, attribution_method: str,
                                     data_dir: str, results_dir: str, output_dir: str, 
                                     plots_dir: str) -> pd.DataFrame:
    """
    Evaluate faithfulness for a specific scenario, model, and attribution method.
    
    Args:
        scenario: Name of scenario
        model_type: Type of model
        attribution_method: Name of attribution method
        data_dir: Directory containing datasets
        results_dir: Directory containing attribution results
        output_dir: Directory to save evaluation results
        plots_dir: Directory to save plots
        
    Returns:
        DataFrame with faithfulness metrics
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating {attribution_method} for {model_type} on {scenario}...")
    
    # Load dataset and causal info
    try:
        X, y, causal_info = load_dataset(scenario, data_dir)
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None
    
    # Load attribution results
    attribution_file = os.path.join(results_dir, f'{scenario}_{model_type}_{attribution_method}_attributions.csv')
    if not os.path.exists(attribution_file):
        logger.warning(f"Attribution file not found: {attribution_file}")
        return None
    
    attributions_df = pd.read_csv(attribution_file)
    
    # Determine causal and non-causal features
    if scenario == 'fraud_detection':
        causal_features = causal_info['causal_features']
        non_causal_features = causal_info['indirect_features']
    else:
        causal_features = causal_info['causal_features']
        non_causal_features = causal_info['spurious_features']
    
    # Create evaluator
    evaluator = CausalFaithfulnessEvaluator(causal_features, non_causal_features)
    
    # Evaluate faithfulness
    metrics = evaluator.evaluate_all_metrics(attributions_df)
    
    # Add metadata
    metrics_df = pd.DataFrame([{
        'scenario': scenario,
        'model_type': model_type,
        'attribution_method': attribution_method,
        **metrics
    }])
    
    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(output_dir, f'{scenario}_{model_type}_{attribution_method}_metrics.csv'), index=False)
    
    return metrics_df


def generate_summary_plots(all_metrics: pd.DataFrame, plots_dir: str):
    """
    Generate summary plots comparing faithfulness across scenarios, models, and methods.
    
    Args:
        all_metrics: DataFrame with all faithfulness metrics
        plots_dir: Directory to save plots
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating summary plots...")
    
    # Create plots directory if it doesn't exist
    summary_plots_dir = os.path.join(plots_dir, 'faithfulness')
    os.makedirs(summary_plots_dir, exist_ok=True)
    
    # Plot overall faithfulness score by attribution method and scenario
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=all_metrics,
        x='attribution_method',
        y='overall_faithfulness_score',
        hue='scenario',
        palette='viridis'
    )
    plt.title('Overall Faithfulness Score by Attribution Method and Scenario')
    plt.xlabel('Attribution Method')
    plt.ylabel('Overall Faithfulness Score')
    plt.xticks(rotation=45)
    plt.legend(title='Scenario')
    plt.tight_layout()
    plt.savefig(os.path.join(summary_plots_dir, 'overall_score_by_method_scenario.png'))
    plt.close()
    
    # Plot overall faithfulness score by attribution method and model type
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=all_metrics,
        x='attribution_method',
        y='overall_faithfulness_score',
        hue='model_type',
        palette='Set2'
    )
    plt.title('Overall Faithfulness Score by Attribution Method and Model Type')
    plt.xlabel('Attribution Method')
    plt.ylabel('Overall Faithfulness Score')
    plt.xticks(rotation=45)
    plt.legend(title='Model Type')
    plt.tight_layout()
    plt.savefig(os.path.join(summary_plots_dir, 'overall_score_by_method_model.png'))
    plt.close()
    
    # Plot top-K accuracy by attribution method and scenario
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=all_metrics,
        x='attribution_method',
        y='ranking_top_k_accuracy',
        hue='scenario',
        palette='viridis'
    )
    plt.title('Top-K Accuracy by Attribution Method and Scenario')
    plt.xlabel('Attribution Method')
    plt.ylabel('Top-K Accuracy')
    plt.xticks(rotation=45)
    plt.legend(title='Scenario')
    plt.tight_layout()
    plt.savefig(os.path.join(summary_plots_dir, 'topk_accuracy_by_method_scenario.png'))
    plt.close()
    
    # Plot attribution ratio by attribution method and model type
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=all_metrics,
        x='attribution_method',
        y='magnitude_attribution_ratio',
        hue='model_type',
        palette='Set2'
    )
    plt.title('Attribution Ratio (Causal/Non-Causal) by Attribution Method and Model Type')
    plt.xlabel('Attribution Method')
    plt.ylabel('Attribution Ratio (log scale)')
    plt.yscale('log')  # Use log scale for better visualization
    plt.xticks(rotation=45)
    plt.legend(title='Model Type')
    plt.tight_layout()
    plt.savefig(os.path.join(summary_plots_dir, 'attribution_ratio_by_method_model.png'))
    plt.close()
    
    # Heatmap of overall faithfulness score for all combinations
    pivot_data = all_metrics.pivot_table(
        index=['scenario', 'model_type'],
        columns='attribution_method',
        values='overall_faithfulness_score'
    )
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        pivot_data,
        annot=True,
        cmap='YlGnBu',
        fmt='.2f',
        linewidths=0.5
    )
    plt.title('Overall Faithfulness Score Across All Combinations')
    plt.tight_layout()
    plt.savefig(os.path.join(summary_plots_dir, 'overall_score_heatmap.png'))
    plt.close()
    
    logger.info(f"Summary plots saved to {summary_plots_dir}")


def generate_paper_tables(all_metrics: pd.DataFrame, output_dir: str):
    """
    Generate tables of results for the paper.
    
    Args:
        all_metrics: DataFrame with all faithfulness metrics
        output_dir: Directory to save tables
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating paper tables...")
    
    # Create tables directory
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    # Table 1: Overall faithfulness score by scenario, model, and method
    table1 = all_metrics.pivot_table(
        index=['scenario', 'model_type'],
        columns='attribution_method',
        values='overall_faithfulness_score'
    ).round(3)
    
    table1.to_csv(os.path.join(tables_dir, 'table1_overall_score.csv'))
    
    # Table 2: Top-K accuracy by scenario, model, and method
    table2 = all_metrics.pivot_table(
        index=['scenario', 'model_type'],
        columns='attribution_method',
        values='ranking_top_k_accuracy'
    ).round(3)
    
    table2.to_csv(os.path.join(tables_dir, 'table2_topk_accuracy.csv'))
    
    # Table 3: Mean attribution ratio by scenario, model, and method
    table3 = all_metrics.pivot_table(
        index=['scenario', 'model_type'],
        columns='attribution_method',
        values='magnitude_attribution_ratio'
    ).round(3)
    
    table3.to_csv(os.path.join(tables_dir, 'table3_attribution_ratio.csv'))
    
    # Table 4: Average performance across scenarios by model and method
    table4 = all_metrics.pivot_table(
        index=['model_type'],
        columns='attribution_method',
        values='overall_faithfulness_score',
        aggfunc='mean'
    ).round(3)
    
    table4.to_csv(os.path.join(tables_dir, 'table4_average_by_model.csv'))
    
    # Table 5: Average performance across models by scenario and method
    table5 = all_metrics.pivot_table(
        index=['scenario'],
        columns='attribution_method',
        values='overall_faithfulness_score',
        aggfunc='mean'
    ).round(3)
    
    table5.to_csv(os.path.join(tables_dir, 'table5_average_by_scenario.csv'))
    
    # Create a summary table with key metrics for each scenario-model-method combination
    summary_table = all_metrics[['scenario', 'model_type', 'attribution_method', 
                              'overall_faithfulness_score', 'ranking_top_k_accuracy',
                              'magnitude_attribution_ratio', 'stability_stability_score']]
    
    summary_table.to_csv(os.path.join(tables_dir, 'summary_metrics.csv'), index=False)
    
    # Also save as LaTeX tables for paper inclusion
    try:
        table1.to_latex(os.path.join(tables_dir, 'table1_overall_score.tex'))
        table2.to_latex(os.path.join(tables_dir, 'table2_topk_accuracy.tex'))
        table3.to_latex(os.path.join(tables_dir, 'table3_attribution_ratio.tex'))
        table4.to_latex(os.path.join(tables_dir, 'table4_average_by_model.tex'))
        table5.to_latex(os.path.join(tables_dir, 'table5_average_by_scenario.tex'))
    except Exception as e:
        logger.warning(f"Error generating LaTeX tables: {str(e)}")
    
    logger.info(f"Paper tables saved to {tables_dir}")


def main():
    """Main function to evaluate faithfulness metrics."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create directories
    dirs = create_directories()
    data_dir = args.data_dir if args.data_dir else dirs['data']
    results_dir = args.results_dir if args.results_dir else dirs['results']
    output_dir = args.output_dir if args.output_dir else os.path.join(dirs['results'], 'faithfulness')
    plots_dir = args.plots_dir if args.plots_dir else dirs['plots']
    
    # Set random seed
    set_seeds(args.random_state)
    
    # Log parameters
    logger.info(f"Evaluating faithfulness for scenarios: {args.scenarios}")
    logger.info(f"Model types: {args.model_types}")
    logger.info(f"Attribution methods: {args.attribution_methods}")
    
    # Evaluate faithfulness for each combination
    all_metrics = []
    
    for scenario in args.scenarios:
        for model_type in args.model_types:
            for attribution_method in args.attribution_methods:
                # Skip SHAP for LSTM (very slow)
                if attribution_method == 'shap' and model_type == 'lstm':
                    logger.warning(f"Skipping SHAP for LSTM model (can be very slow)")
                    continue
                
                metrics_df = evaluate_scenario_model_attribution(
                    scenario=scenario,
                    model_type=model_type,
                    attribution_method=attribution_method,
                    data_dir=data_dir,
                    results_dir=results_dir,
                    output_dir=output_dir,
                    plots_dir=plots_dir
                )
                
                if metrics_df is not None:
                    all_metrics.append(metrics_df)
    
    # Combine all metrics
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        
        # Save combined metrics
        os.makedirs(output_dir, exist_ok=True)
        combined_metrics.to_csv(os.path.join(output_dir, 'all_faithfulness_metrics.csv'), index=False)
        
        # Also save a copy directly in the results directory for report generation
        combined_metrics.to_csv(os.path.join(results_dir, 'all_faithfulness_metrics.csv'), index=False)
        
        # Generate summary plots
        generate_summary_plots(combined_metrics, plots_dir)
        
        # Generate paper tables
        generate_paper_tables(combined_metrics, output_dir)
        
        logger.info(f"All faithfulness metrics saved to {os.path.join(output_dir, 'all_faithfulness_metrics.csv')}")
    else:
        logger.warning("No metrics were generated. Check if attribution files exist.")
    
    logger.info("Faithfulness evaluation complete!")


if __name__ == "__main__":
    main()