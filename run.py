#!/usr/bin/env python3
"""
Main script to run the complete analysis pipeline for causal faithfulness of 
feature attributions in financial machine learning models.

This script:
1. Generates synthetic data for three financial scenarios
2. Trains models on this data
3. Computes feature attributions
4. Evaluates the causal faithfulness of these attributions
5. Generates visualizations and a comprehensive report
"""

import os
import sys
import logging
import polars as pl
import numpy as np
from datetime import datetime

#pl.Config.set_engine_affinity(engine="streaming")

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import data_generation, models, and analysis modules
from src.data_generation import (
    AssetPricingScenario,
    CreditRiskScenario,
    FraudDetectionScenario,
    generate_all_datasets
)
from src.models import (
    SimpleMLPModel,
    TimeSeriesLSTMModel,
    GradientBoostingWrapper,
    train_model_for_scenario
)
from src.analysis import (
    create_directories,
    set_seeds,
    run_attribution_analysis,
    evaluate_faithfulness_metrics,
    generate_report,
    logger
)

# Configuration (hardcoded parameters)
CONFIG = {
    'data': {
        'n_samples': 10000,
        'random_state': 42,
        'scenarios': ['asset_pricing', 'credit_risk', 'fraud_detection'],
        'asset_pricing': {
            'n_causal': 2,
            'n_spurious': 3,
            'n_confounders': 2, 
            'noise_level': 0.2
        },
        'credit_risk': {
            'n_causal': 2,
            'n_proxy': 2,
            'n_confounders': 2,
            'noise_level': 0.2
        },
        'fraud_detection': {
            'n_causal': 2,
            'n_indirect': 2,
            'n_confounders': 2,
            'fraud_rate': 0.05,
            'noise_level': 0.2
        }
    },
    'models': {
        'types': ['mlp', 'lstm', 'xgboost'],
        'mlp': {
            'hidden_dims': [64, 32],
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100
        },
        'lstm': {
            'hidden_dim': 64,
            'num_layers': 2,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,
            'sequence_length': 10
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    },
    'attributions': {
        'methods': {
            'mlp': ['saliency', 'gradient_input', 'integrated_gradients', 'shap'],
            'lstm': ['saliency', 'gradient_input', 'integrated_gradients'],
            'xgboost': ['shap']
        },
        'n_samples': 1000,
        'random_state': 42
    },
    'evaluation': {
        'output_format': 'markdown'
    }
}

def generate_data(data_dir: str, config: dict) -> None:
    """
    Generate synthetic data for all scenarios.
    
    Args:
        data_dir: Directory to save data
        config: Configuration dict
    """
    logger.info("Generating synthetic data...")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate data for each scenario
    for scenario in config['data']['scenarios']:
        logger.info(f"Generating data for {scenario} scenario...")
        
        if scenario == 'asset_pricing':
            scenario_obj = AssetPricingScenario(
                n_samples=config['data']['n_samples'],
                random_state=config['data']['random_state'],
                n_causal=config['data']['asset_pricing']['n_causal'],
                n_spurious=config['data']['asset_pricing']['n_spurious'],
                n_confounders=config['data']['asset_pricing']['n_confounders'],
                noise_level=config['data']['asset_pricing']['noise_level']
            )
        elif scenario == 'credit_risk':
            scenario_obj = CreditRiskScenario(
                n_samples=config['data']['n_samples'],
                random_state=config['data']['random_state'],
                n_causal=config['data']['credit_risk']['n_causal'],
                n_proxy=config['data']['credit_risk']['n_proxy'],
                n_confounders=config['data']['credit_risk']['n_confounders'],
                noise_level=config['data']['credit_risk']['noise_level']
            )
        elif scenario == 'fraud_detection':
            scenario_obj = FraudDetectionScenario(
                n_samples=config['data']['n_samples'],
                random_state=config['data']['random_state'],
                n_causal=config['data']['fraud_detection']['n_causal'],
                n_indirect=config['data']['fraud_detection']['n_indirect'],
                n_confounders=config['data']['fraud_detection']['n_confounders'],
                fraud_rate=config['data']['fraud_detection']['fraud_rate'],
                noise_level=config['data']['fraud_detection']['noise_level']
            )
        else:
            logger.warning(f"Unknown scenario: {scenario}")
            continue
        
        # Generate data
        X, y, causal_info = scenario_obj.generate_data()
        
        # Convert pandas to polars if needed
        if not isinstance(X, pl.DataFrame):
            X = pl.from_pandas(X)
        
        if not isinstance(y, pl.Series):
            y_df = pl.DataFrame({y.name: y.values})
        else:
            y_df = pl.DataFrame({y.name: y.to_list()})
        
        # Save data
        X.write_csv(os.path.join(data_dir, f'{scenario}_features.csv'))
        y_df.write_csv(os.path.join(data_dir, f'{scenario}_target.csv'))
        
        # Save causal info (excluding DAG which isn't easily serializable)
        causal_info_df = pl.DataFrame({
            'feature': X.columns,
            'is_causal': [feat in causal_info['causal_features'] for feat in X.columns]
        })
        causal_info_df.write_csv(os.path.join(data_dir, f'{scenario}_causal_info.csv'))
        
        logger.info(f"Data for {scenario} saved to {data_dir}")

def train_models(data_dir: str, model_dir: str, results_dir: str, config: dict) -> None:
    """
    Train models for all scenarios.
    
    Args:
        data_dir: Directory containing data
        model_dir: Directory to save models
        results_dir: Directory to save results
        config: Configuration dict
    """
    logger.info("Training models...")
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Track model performance
    all_metrics = []
    
    # Train models for each scenario and model type
    for scenario in config['data']['scenarios']:
        for model_type in config['models']['types']:
            logger.info(f"Training {model_type} model for {scenario}...")
            
            try:
                # Train model
                result = train_model_for_scenario(
                    scenario_name=scenario,
                    model_type=model_type,
                    data_dir=data_dir,
                    model_dir=model_dir,
                    verbose=True
                )
                
                # Store results
                metrics = result['metrics']
                metrics.update({
                    'scenario': scenario,
                    'model_type': model_type
                })
                all_metrics.append(metrics)
                
                logger.info(f"Model trained successfully. Metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Error training {model_type} for {scenario}: {str(e)}")
    
    # Save results
    if all_metrics:
        results_df = pl.DataFrame(all_metrics)
        results_df.write_csv(os.path.join(results_dir, 'model_performance.csv'))
        logger.info(f"Results saved to {os.path.join(results_dir, 'model_performance.csv')}")

def compute_attributions(data_dir: str, model_dir: str, results_dir: str, plots_dir: str, config: dict) -> None:
    """
    Compute attributions for all models and methods.
    
    Args:
        data_dir: Directory containing data
        model_dir: Directory containing models
        results_dir: Directory to save results
        plots_dir: Directory to save plots
        config: Configuration dict
    """
    logger.info("Computing attributions...")
    
    # Create directories
    attribution_plots_dir = os.path.join(plots_dir, 'attributions')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(attribution_plots_dir, exist_ok=True)
    
    # Compute attributions for each scenario, model type, and attribution method
    for scenario in config['data']['scenarios']:
        for model_type in config['models']['types']:
            # Get attribution methods for this model type
            if model_type in config['attributions']['methods']:
                attribution_methods = config['attributions']['methods'][model_type]
            else:
                logger.warning(f"No attribution methods specified for {model_type}")
                continue
            
            for method in attribution_methods:
                # Skip SHAP for LSTM models (very slow)
                if method == 'shap' and model_type == 'lstm':
                    logger.warning(f"Skipping SHAP for LSTM model (can be very slow)")
                    continue
                
                try:
                    # Run attribution analysis
                    result = run_attribution_analysis(
                        scenario=scenario,
                        model_type=model_type,
                        attribution_method=method,
                        data_dir=data_dir,
                        model_dir=model_dir,
                        output_dir=results_dir,
                        plots_dir=attribution_plots_dir,
                        n_samples=config['attributions']['n_samples']
                    )
                    
                    logger.info(f"Attribution analysis complete for {method} on {model_type} ({scenario})")
                    logger.info(f"Results saved to {result['paths']['attributions']}")
                    logger.info(f"Metrics saved to {result['paths']['metrics']}")
                    logger.info(f"Plots saved to {result['paths']['heatmap']} and {result['paths']['barplot']}")
                    
                except Exception as e:
                    logger.error(f"Error computing {method} for {scenario} using {model_type}: {str(e)}")

def main():
    """Main function to run the complete analysis pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"fin_causal_attributions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    
    # Create directories
    dirs = create_directories()
    data_dir = dirs['data']
    model_dir = dirs['models']
    results_dir = dirs['results']
    plots_dir = dirs['plots']
    
    # Set random seed
    set_seeds(CONFIG['data']['random_state'])
    
    # Step 1: Generate synthetic data
    generate_data(data_dir, CONFIG)
    
    # Step 2: Train models
    train_models(data_dir, model_dir, results_dir, CONFIG)
    
    # Step 3: Compute attributions
    compute_attributions(data_dir, model_dir, results_dir, plots_dir, CONFIG)
    
    # Step 4: Evaluate faithfulness
    faithfulness_dir = os.path.join(results_dir, 'faithfulness')
    faithfulness_plots_dir = os.path.join(plots_dir, 'faithfulness')
    metrics_df = evaluate_faithfulness_metrics(results_dir, faithfulness_dir, faithfulness_plots_dir)
    
    # Step 5: Generate report
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report')
    report_path = generate_report(results_dir, report_dir, CONFIG['evaluation']['output_format'])
    
    logger.info("Analysis pipeline completed!")
    logger.info(f"Results saved to {results_dir}")
    logger.info(f"Plots saved to {plots_dir}")
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()