#!/usr/bin/env python
"""
Script to analyze combinations of attribution methods, model types, and scenarios.

This script performs a detailed analysis of how different attribution methods
perform across various combinations of model types and scenarios, identifying
the best-performing combinations and extracting key insights.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import create_directories, setup_logging, set_seeds


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze attribution method combinations')
    
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing all results')
    parser.add_argument('--output_dir', type=str, default='results/analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--plots_dir', type=str, default='plots/analysis',
                        help='Directory to save analysis plots')
    
    return parser.parse_args()


def load_faithfulness_metrics(results_dir):
    """
    Load faithfulness metrics from results directory.
    
    Args:
        results_dir: Directory containing results
        
    Returns:
        DataFrame with faithfulness metrics
    """
    faithfulness_path = os.path.join(results_dir, 'faithfulness', 'all_faithfulness_metrics.csv')
    
    if os.path.exists(faithfulness_path):
        return pd.read_csv(faithfulness_path)
    else:
        raise FileNotFoundError(f"Faithfulness metrics file not found: {faithfulness_path}")


def analyze_best_combinations(metrics_df, output_dir, plots_dir):
    """
    Analyze the best combinations of attribution methods, model types, and scenarios.
    
    Args:
        metrics_df: DataFrame with faithfulness metrics
        output_dir: Directory to save analysis results
        plots_dir: Directory to save analysis plots
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing best combinations...")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_key=True)
    
    # Best combination overall
    best_overall = metrics_df.loc[metrics_df['overall_faithfulness_score'].idxmax()]
    
    # Best combination for each scenario
    best_by_scenario = metrics_df.loc[metrics_df.groupby('scenario')['overall_faithfulness_score'].idxmax()]
    
    # Best combination for each model type
    best_by_model = metrics_df.loc[metrics_df.groupby('model_type')['overall_faithfulness_score'].idxmax()]
    
    # Best combination for each attribution method
    best_by_method = metrics_df.loc[metrics_df.groupby('attribution_method')['overall_faithfulness_score'].idxmax()]
    
    # Save best combinations
    best_combinations = pd.concat([
        pd.DataFrame([best_overall]),
        best_by_scenario,
        best_by_model,
        best_by_method
    ], keys=['overall', 'by_scenario', 'by_model', 'by_method'])
    
    best_combinations.to_csv(os.path.join(output_dir, 'best_combinations.csv'))
    
    # Create interactive heatmap for all combinations
    plt.figure(figsize=(16, 12))
    
    # Pivot data for scenario-model-method combinations
    pivot_data = metrics_df.pivot_table(
        index=['scenario', 'model_type'],
        columns='attribution_method',
        values='overall_faithfulness_score'
    )
    
    # Plot heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        cmap='viridis',
        fmt='.3f',
        linewidths=0.5
    )
    
    plt.title('Overall Faithfulness Score for All Combinations')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_combinations_heatmap.png'), dpi=300)
    plt.close()
    
    logger.info(f"Best combinations analysis saved to {output_dir}")


def analyze_causal_vs_noncausal_attribution(metrics_df, output_dir, plots_dir):
    """
    Analyze how well attribution methods distinguish causal vs. non-causal features.
    
    Args:
        metrics_df: DataFrame with faithfulness metrics
        output_dir: Directory to save analysis results
        plots_dir: Directory to save analysis plots
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing causal vs. non-causal attribution...")
    
    # Ratio of causal to non-causal attribution
    plt.figure(figsize=(12, 8))
    
    # Calculate average attribution ratio by method
    ratio_by_method = metrics_df.groupby('attribution_method')['magnitude_attribution_ratio'].mean().reset_index()
    ratio_by_method = ratio_by_method.sort_values('magnitude_attribution_ratio', ascending=False)
    
    # Plot attribution ratio by method
    ax = sns.barplot(
        data=ratio_by_method,
        x='attribution_method',
        y='magnitude_attribution_ratio',
        palette='viridis'
    )
    
    plt.title('Average Attribution Ratio (Causal/Non-Causal) by Method')
    plt.xlabel('Attribution Method')
    plt.ylabel('Mean Attribution Ratio')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f'{bar.get_height():.2f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'attribution_ratio_by_method.png'), dpi=300)
    plt.close()
    
    # Compare attribution ratio across different scenarios
    plt.figure(figsize=(14, 8))
    
    # Calculate average attribution ratio by method and scenario
    ratio_by_method_scenario = metrics_df.groupby(['attribution_method', 'scenario'])['magnitude_attribution_ratio'].mean().reset_index()
    
    # Plot attribution ratio by method and scenario
    ax = sns.barplot(
        data=ratio_by_method_scenario,
        x='attribution_method',
        y='magnitude_attribution_ratio',
        hue='scenario',
        palette='Set2'
    )
    
    plt.title('Attribution Ratio by Method and Scenario')
    plt.xlabel('Attribution Method')
    plt.ylabel('Mean Attribution Ratio')
    plt.xticks(rotation=45)
    plt.legend(title='Scenario')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'attribution_ratio_by_method_scenario.png'), dpi=300)
    plt.close()
    
    # Top-K accuracy analysis
    plt.figure(figsize=(12, 8))
    
    # Calculate average top-K accuracy by method
    topk_by_method = metrics_df.groupby('attribution_method')['ranking_top_k_accuracy'].mean().reset_index()
    topk_by_method = topk_by_method.sort_values('ranking_top_k_accuracy', ascending=False)
    
    # Plot top-K accuracy by method
    ax = sns.barplot(
        data=topk_by_method,
        x='attribution_method',
        y='ranking_top_k_accuracy',
        palette='viridis'
    )
    
    plt.title('Average Top-K Accuracy by Attribution Method')
    plt.xlabel('Attribution Method')
    plt.ylabel('Top-K Accuracy')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.2f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'topk_accuracy_by_method.png'), dpi=300)
    plt.close()
    
    # Save summary statistics to CSV
    ratio_by_method.to_csv(os.path.join(output_dir, 'attribution_ratio_by_method.csv'), index=False)
    ratio_by_method_scenario.to_csv(os.path.join(output_dir, 'attribution_ratio_by_method_scenario.csv'), index=False)
    topk_by_method.to_csv(os.path.join(output_dir, 'topk_accuracy_by_method.csv'), index=False)
    
    logger.info(f"Causal vs. non-causal analysis saved to {output_dir}")


def analyze_model_impact(metrics_df, output_dir, plots_dir):
    """
    Analyze how different model types impact attribution performance.
    
    Args:
        metrics_df: DataFrame with faithfulness metrics
        output_dir: Directory to save analysis results
        plots_dir: Directory to save analysis plots
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing model impact on attribution performance...")
    
    # Overall faithfulness by model type
    plt.figure(figsize=(12, 8))
    
    # Calculate average faithfulness by model type
    model_avg = metrics_df.groupby('model_type')['overall_faithfulness_score'].mean().reset_index()
    model_avg = model_avg.sort_values('overall_faithfulness_score', ascending=False)
    
    # Plot faithfulness by model type
    ax = sns.barplot(
        data=model_avg,
        x='model_type',
        y='overall_faithfulness_score',
        palette='viridis'
    )
    
    plt.title('Average Faithfulness Score by Model Type')
    plt.xlabel('Model Type')
    plt.ylabel('Overall Faithfulness Score')
    
    # Add value labels on bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.2f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'faithfulness_by_model.png'), dpi=300)
    plt.close()
    
    # Faithfulness by model type and attribution method
    plt.figure(figsize=(14, 8))
    
    # Calculate average faithfulness by model type and attribution method
    model_method_avg = metrics_df.groupby(['model_type', 'attribution_method'])['overall_faithfulness_score'].mean().reset_index()
    
    # Plot faithfulness by model type and attribution method
    ax = sns.barplot(
        data=model_method_avg,
        x='model_type',
        y='overall_faithfulness_score',
        hue='attribution_method',
        palette='Set2'
    )
    
    plt.title('Faithfulness Score by Model Type and Attribution Method')
    plt.xlabel('Model Type')
    plt.ylabel('Overall Faithfulness Score')
    plt.legend(title='Attribution Method')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'faithfulness_by_model_method.png'), dpi=300)
    plt.close()
    
    # Variation across model types for each method and scenario
    plt.figure(figsize=(16, 12))
    
    # Calculate standard deviation of faithfulness across model types
    model_variation = metrics_df.groupby(['scenario', 'attribution_method'])['overall_faithfulness_score'].std().reset_index()
    model_variation = model_variation.pivot(index='scenario', columns='attribution_method', values='overall_faithfulness_score')
    
    # Plot heatmap of standard deviation
    sns.heatmap(
        model_variation,
        annot=True,
        cmap='YlOrRd',
        fmt='.3f',
        linewidths=0.5
    )
    
    plt.title('Variation (Std Dev) in Faithfulness Across Model Types')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_variation_heatmap.png'), dpi=300)
    plt.close()
    
    # Save summary statistics to CSV
    model_avg.to_csv(os.path.join(output_dir, 'faithfulness_by_model.csv'), index=False)
    model_method_avg.to_csv(os.path.join(output_dir, 'faithfulness_by_model_method.csv'), index=False)
    model_variation.to_csv(os.path.join(output_dir, 'model_variation.csv'))
    
    logger.info(f"Model impact analysis saved to {output_dir}")


def analyze_scenario_impact(metrics_df, output_dir, plots_dir):
    """
    Analyze how different scenarios impact attribution performance.
    
    Args:
        metrics_df: DataFrame with faithfulness metrics
        output_dir: Directory to save analysis results
        plots_dir: Directory to save analysis plots
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing scenario impact on attribution performance...")
    
    # Overall faithfulness by scenario
    plt.figure(figsize=(12, 8))
    
    # Calculate average faithfulness by scenario
    scenario_avg = metrics_df.groupby('scenario')['overall_faithfulness_score'].mean().reset_index()
    scenario_avg = scenario_avg.sort_values('overall_faithfulness_score', ascending=False)
    
    # Plot faithfulness by scenario
    ax = sns.barplot(
        data=scenario_avg,
        x='scenario',
        y='overall_faithfulness_score',
        palette='viridis'
    )
    
    plt.title('Average Faithfulness Score by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Overall Faithfulness Score')
    
    # Add value labels on bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.2f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'faithfulness_by_scenario.png'), dpi=300)
    plt.close()
    
    # Scenario challenge score (how much each scenario challenges attribution methods)
    plt.figure(figsize=(12, 8))
    
    # For each attribution method, calculate normalized rank of scenarios
    scenario_challenge = pd.DataFrame()
    
    for method in metrics_df['attribution_method'].unique():
        method_data = metrics_df[metrics_df['attribution_method'] == method]
        method_avg = method_data.groupby('scenario')['overall_faithfulness_score'].mean().reset_index()
        method_avg['rank'] = method_avg['overall_faithfulness_score'].rank(ascending=False)
        method_avg['attribution_method'] = method
        scenario_challenge = pd.concat([scenario_challenge, method_avg])
    
    # Calculate average rank of each scenario
    scenario_rank_avg = scenario_challenge.groupby('scenario')['rank'].mean().reset_index()
    scenario_rank_avg = scenario_rank_avg.sort_values('rank')
    
    # Plot average rank by scenario
    ax = sns.barplot(
        data=scenario_rank_avg,
        x='scenario',
        y='rank',
        palette='viridis'
    )
    
    plt.title('Average Rank of Scenarios Across Attribution Methods\n(Lower = Better Performance)')
    plt.xlabel('Scenario')
    plt.ylabel('Average Rank')
    
    # Add value labels on bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f'{bar.get_height():.2f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'scenario_challenge_rank.png'), dpi=300)
    plt.close()
    
    # Detailed scenario analysis by attribution method
    plt.figure(figsize=(14, 10))
    
    # Calculate average top-K accuracy by scenario and attribution method
    scenario_method_topk = metrics_df.groupby(['scenario', 'attribution_method'])['ranking_top_k_accuracy'].mean().reset_index()
    scenario_method_topk_pivot = scenario_method_topk.pivot(index='scenario', columns='attribution_method', values='ranking_top_k_accuracy')
    
    # Plot heatmap of top-K accuracy
    sns.heatmap(
        scenario_method_topk_pivot,
        annot=True,
        cmap='YlGnBu',
        fmt='.3f',
        linewidths=0.5
    )
    
    plt.title('Top-K Accuracy by Scenario and Attribution Method')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'topk_by_scenario_method_heatmap.png'), dpi=300)
    plt.close()
    
    # Save summary statistics to CSV
    scenario_avg.to_csv(os.path.join(output_dir, 'faithfulness_by_scenario.csv'), index=False)
    scenario_rank_avg.to_csv(os.path.join(output_dir, 'scenario_challenge_rank.csv'), index=False)
    scenario_method_topk_pivot.to_csv(os.path.join(output_dir, 'topk_by_scenario_method.csv'))
    
    logger.info(f"Scenario impact analysis saved to {output_dir}")


def main():
    """Main function to analyze attribution combinations."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create directories
    dirs = create_directories()
    results_dir = args.results_dir if args.results_dir else dirs['results']
    output_dir = args.output_dir if args.output_dir else os.path.join(dirs['results'], 'analysis')
    plots_dir = args.plots_dir if args.plots_dir else os.path.join(dirs['plots'], 'analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set random seed
    set_seeds(42)
    
    # Load faithfulness metrics
    try:
        metrics_df = load_faithfulness_metrics(results_dir)
        logger.info(f"Loaded faithfulness metrics with {len(metrics_df)} rows")
    except Exception as e:
        logger.error(f"Error loading faithfulness metrics: {str(e)}")
        sys.exit(1)
    
    # Analyze best combinations
    try:
        analyze_best_combinations(metrics_df, output_dir, plots_dir)
    except Exception as e:
        logger.error(f"Error analyzing best combinations: {str(e)}")
    
    # Analyze causal vs. non-causal attribution
    try:
        analyze_causal_vs_noncausal_attribution(metrics_df, output_dir, plots_dir)
    except Exception as e:
        logger.error(f"Error analyzing causal vs. non-causal attribution: {str(e)}")
    
    # Analyze model impact
    try:
        analyze_model_impact(metrics_df, output_dir, plots_dir)
    except Exception as e:
        logger.error(f"Error analyzing model impact: {str(e)}")
    
    # Analyze scenario impact
    try:
        analyze_scenario_impact(metrics_df, output_dir, plots_dir)
    except Exception as e:
        logger.error(f"Error analyzing scenario impact: {str(e)}")
    
    logger.info("Attribution combination analysis complete!")


if __name__ == "__main__":
    main()