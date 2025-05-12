#!/usr/bin/env python
"""
Script to compute attributions for trained models.

This script computes feature attributions for models trained on the financial scenarios
using different attribution methods:
- Saliency Map: Simple gradient-based attribution
- Gradient * Input: Gradients multiplied by input values
- Integrated Gradients: Integrating gradients along a path from baseline to input
- SHAP: SHapley Additive exPlanations
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attributions import compute_attributions
from src.utils import (
    create_directories,
    save_config,
    load_config,
    setup_logging,
    set_seeds,
    load_dataset,
    load_model
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute attributions for trained models')
    
    parser.add_argument('--config', type=str, default='config/attribution_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing datasets')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='results/attributions',
                        help='Directory to save attributions')
    parser.add_argument('--scenarios', type=str, nargs='+',
                        default=['asset_pricing', 'credit_risk', 'fraud_detection'],
                        help='Scenarios to compute attributions for')
    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['mlp'],
                        help='Types of models to use (mlp, lstm, xgboost)')
    parser.add_argument('--attribution_methods', type=str, nargs='+',
                        default=['saliency', 'gradient_input', 'integrated_gradients', 'shap'],
                        help='Attribution methods to use')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples to compute attributions for')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--plot', action='store_true',
                        help='Generate attribution plots')
    
    return parser.parse_args()


def plot_attributions(attributions, scenario, model_type, method, causal_info, plots_dir):
    """
    Plot attributions for a model and method.
    
    Args:
        attributions: DataFrame with attribution scores
        scenario: Name of scenario
        model_type: Type of model
        method: Attribution method
        causal_info: Dictionary with causal feature information
        plots_dir: Directory to save plots
    """
    from src.evaluation import plot_attribution_heatmap
    
    # Extract causal and non-causal features
    if scenario == 'fraud_detection':
        causal_features = causal_info['causal_features']
        non_causal_features = causal_info['indirect_features']
    else:
        causal_features = causal_info['causal_features']
        non_causal_features = causal_info['spurious_features']
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    plot_attribution_heatmap(
        attributions=attributions,
        causal_features=causal_features,
        non_causal_features=non_causal_features,
        title=f'{method.replace("_", " ").title()} Attributions - {model_type.upper()} ({scenario.replace("_", " ").title()})'
    )
    
    # Save figure
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'{scenario}_{model_type}_{method}_attributions.png'))
    plt.close()
    
    # Also create a bar plot of mean absolute attributions by feature
    plt.figure(figsize=(12, 6))
    
    # Calculate mean absolute attributions
    mean_attrs = attributions.abs().mean().sort_values(ascending=False)
    
    # Color bars based on feature type (causal or non-causal)
    colors = ['royalblue' if feat in causal_features else 'lightcoral' for feat in mean_attrs.index]
    
    # Plot
    bars = plt.bar(mean_attrs.index, mean_attrs.values, color=colors)
    plt.title(f'Mean {method.replace("_", " ").title()} Attribution Magnitude - {model_type.upper()} ({scenario.replace("_", " ").title()})')
    plt.xlabel('Feature')
    plt.ylabel('Mean Absolute Attribution')
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='royalblue', label='Causal Features'),
        Patch(facecolor='lightcoral', label='Non-Causal Features')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{scenario}_{model_type}_{method}_attribution_bars.png'))
    plt.close()


def main():
    """Main function to compute attributions."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config) if os.path.exists(args.config) else {}
    
    # Create directories
    dirs = create_directories()
    data_dir = args.data_dir if args.data_dir else dirs['data']
    model_dir = args.model_dir if args.model_dir else dirs['models']
    output_dir = args.output_dir if args.output_dir else os.path.join(dirs['results'], 'attributions')
    plots_dir = os.path.join(dirs['plots'], 'attributions')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set random seed
    set_seeds(args.random_state)
    
    # Log parameters
    logger.info(f"Computing attributions for scenarios: {args.scenarios}")
    logger.info(f"Model types: {args.model_types}")
    logger.info(f"Attribution methods: {args.attribution_methods}")
    
    # Compute attributions for each scenario, model type, and attribution method
    for scenario in args.scenarios:
        # Load dataset
        try:
            X, y, causal_info = load_dataset(scenario, data_dir)
            logger.info(f"Loaded {scenario} dataset: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            logger.error(f"Error loading {scenario} dataset: {str(e)}")
            continue
        
        # Sample data for attribution
        if len(X) > args.n_samples:
            X_sample = X.sample(args.n_samples, random_state=args.random_state)
            y_sample = y[X_sample.index]
        else:
            X_sample = X
            y_sample = y
        
        for model_type in args.model_types:
            # Load model
            try:
                model = load_model(scenario, model_type, model_dir)
                logger.info(f"Loaded {model_type} model for {scenario}")
            except Exception as e:
                logger.error(f"Error loading {model_type} model for {scenario}: {str(e)}")
                continue
            
            for method in args.attribution_methods:
                logger.info(f"Computing {method} attributions for {scenario} using {model_type} model...")
                
                try:
                    # Skip SHAP for LSTM models (can be very slow)
                    if method == 'shap' and model_type == 'lstm':
                        logger.warning("Skipping SHAP for LSTM model (can be very slow)")
                        continue
                    
                    # Compute attributions
                    attributions = compute_attributions(
                        model=model,
                        X=X_sample.values,
                        method_name=method,
                        feature_names=X.columns.tolist()
                    )
                    
                    # Save attributions
                    attribution_path = os.path.join(
                        output_dir, f'{scenario}_{model_type}_{method}_attributions.csv'
                    )
                    attributions.to_csv(attribution_path, index=False)
                    logger.info(f"Saved attributions to {attribution_path}")
                    
                    # Generate plots
                    if args.plot:
                        plot_attributions(
                            attributions=attributions,
                            scenario=scenario,
                            model_type=model_type,
                            method=method,
                            causal_info=causal_info,
                            plots_dir=plots_dir
                        )
                        logger.info(f"Generated attribution plots for {method} ({scenario}, {model_type})")
                    
                except Exception as e:
                    logger.error(f"Error computing {method} attributions for {scenario} using {model_type}: {str(e)}")
    
    logger.info("Attribution computation complete!")


if __name__ == "__main__":
    main()