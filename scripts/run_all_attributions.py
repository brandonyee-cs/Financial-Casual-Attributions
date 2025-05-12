#!/usr/bin/env python3

import argparse
import os
import logging
import torch
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(base_dir):
    """Create necessary directories if they don't exist."""
    dirs = {
        'data': os.path.join(base_dir, 'data'),
        'models': os.path.join(base_dir, 'models'),
        'results': os.path.join(base_dir, 'results'),
        'plots': os.path.join(base_dir, 'plots'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def run_attributions(data_dir, model_dir, output_dir, scenarios, model_types, 
                    attribution_methods, n_samples, random_state, plot=True):
    """Run attribution computation for all combinations of scenarios and model types."""
    
    # Define attribution methods for each model type
    model_attribution_methods = {
        'mlp': ['saliency', 'gradient_input', 'integrated_gradients', 'shap'],
        'lstm': ['saliency', 'gradient_input', 'integrated_gradients'],
        'xgboost': ['shap']  # Changed from 'shapley' to 'shap'
    }
    
    # Run attributions for each combination
    for scenario in scenarios:
        for model_type in model_types:
            logger.info(f"Computing attributions for {model_type} model in {scenario} scenario...")
            
            # Get attribution methods for this model type
            methods = model_attribution_methods[model_type]
            
            # Construct command
            cmd = [
                'python3', 'scripts/run_attributions.py',
                '--data_dir', data_dir,
                '--model_dir', model_dir,
                '--output_dir', output_dir,
                '--scenarios', scenario,
                '--model_types', model_type,
                '--n_samples', str(n_samples),
                '--random_state', str(random_state)
            ]
            
            # Add attribution methods
            for method in methods:
                cmd.extend(['--attribution_methods', method])
            
            # Add plot flag if requested
            if plot:
                cmd.append('--plot')
            
            # Run the command
            try:
                import subprocess
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"Successfully computed attributions for {model_type} in {scenario}")
                if result.stdout:
                    logger.info(result.stdout)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to compute attributions for {model_type} in {scenario}")
                logger.error(f"Error: {e.stderr}")
                raise

def main():
    parser = argparse.ArgumentParser(description='Run attributions for all scenarios and model types')
    parser.add_argument('--base_dir', type=str, default='.',
                      help='Base directory for the project')
    parser.add_argument('--n_samples', type=int, default=1000,
                      help='Number of samples to use for attribution')
    parser.add_argument('--random_state', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--no_plot', action='store_true',
                      help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Setup directories
    dirs = setup_directories(args.base_dir)
    
    # Define scenarios and model types
    scenarios = ['asset_pricing', 'credit_risk', 'fraud_detection']
    model_types = ['mlp', 'lstm', 'xgboost']
    
    # Run attributions
    run_attributions(
        data_dir=dirs['data'],
        model_dir=dirs['models'],
        output_dir=dirs['results'],  # Save directly to results directory
        scenarios=scenarios,
        model_types=model_types,
        attribution_methods=None,  # Will be set per model type
        n_samples=args.n_samples,
        random_state=args.random_state,
        plot=not args.no_plot
    )
    
    logger.info("Completed all attribution computations!")

if __name__ == '__main__':
    main()