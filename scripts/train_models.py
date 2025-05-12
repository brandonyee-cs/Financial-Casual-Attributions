#!/usr/bin/env python
"""
Script to train models on the generated financial datasets.

This script trains models for the three financial scenarios:
1. Asset Pricing: Predicting stock returns (regression)
2. Credit Risk: Predicting loan default (classification)
3. Fraud Detection: Identifying fraudulent transactions (classification)

Different model types can be trained for each scenario:
- MLP: Simple neural network
- LSTM: For time series data
- XGBoost: Gradient boosting trees
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

from src.models import (
    SimpleMLPModel,
    TimeSeriesLSTMModel,
    GradientBoostingWrapper,
    train_model_for_scenario
)
from src.utils import (
    create_directories,
    save_config,
    load_config,
    setup_logging,
    set_seeds,
    load_dataset
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train models on financial datasets')
    
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing datasets')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--scenarios', type=str, nargs='+',
                        default=['asset_pricing', 'credit_risk', 'fraud_detection'],
                        help='Scenarios to train models for')
    parser.add_argument('--model_types', type=str, nargs='+',
                        default=['mlp'],
                        help='Types of models to train (mlp, lstm, xgboost)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                        help='Print training progress')
    
    return parser.parse_args()


def plot_learning_curves(history, model_type, scenario, plots_dir):
    """Plot learning curves for neural networks."""
    if 'train_loss' not in history:
        return  # XGBoost doesn't return history with losses
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Learning Curves - {model_type.upper()} ({scenario.replace("_", " ").title()})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'{scenario}_{model_type}_learning_curve.png'))
    plt.close()


def main():
    """Main function to train models."""
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
    plots_dir = dirs['plots']
    
    # Set random seed
    set_seeds(args.random_state)
    
    # Log parameters
    logger.info(f"Training models for scenarios: {args.scenarios}")
    logger.info(f"Model types: {args.model_types}")
    
    # Train models for each scenario and model type
    results = []
    
    for scenario in args.scenarios:
        for model_type in args.model_types:
            logger.info(f"Training {model_type} model for {scenario}...")
            
            try:
                # Train model
                result = train_model_for_scenario(
                    scenario_name=scenario,
                    model_type=model_type,
                    data_dir=data_dir,
                    model_dir=model_dir,
                    verbose=args.verbose
                )
                
                # Store results
                metrics = result['metrics']
                metrics.update({
                    'scenario': scenario,
                    'model_type': model_type
                })
                results.append(metrics)
                
                # Plot learning curves
                if 'history' in result:
                    plot_learning_curves(result['history'], model_type, scenario, plots_dir)
                
                logger.info(f"Model trained successfully. Metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Error training {model_type} for {scenario}: {str(e)}")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(dirs['results'], 'model_performance.csv'), index=False)
        logger.info(f"Results saved to {os.path.join(dirs['results'], 'model_performance.csv')}")
        
        # Print summary table
        if args.verbose:
            print("\nModel Performance Summary:")
            print(results_df.to_string())
    
    logger.info("Model training complete!")


if __name__ == "__main__":
    main()