#!/usr/bin/env python
"""
Script to generate synthetic financial data for all scenarios.

This script creates datasets for the three financial scenarios:
1. Asset Pricing: Predicting stock returns
2. Credit Risk: Predicting loan default
3. Fraud Detection: Identifying fraudulent transactions

Each dataset includes causal and non-causal (spurious or indirect) features
along with target variables.
"""

import os
import sys
import argparse
import logging
import yaml

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation import (
    AssetPricingScenario,
    CreditRiskScenario,
    FraudDetectionScenario,
    generate_all_datasets
)
from src.utils import create_directories, save_config, setup_logging, set_seeds


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate synthetic financial data')
    
    parser.add_argument('--config', type=str, default='config/data_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save generated datasets')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples per dataset')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--scenarios', type=str, nargs='+',
                        default=['asset_pricing', 'credit_risk', 'fraud_detection'],
                        help='Scenarios to generate data for')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize DAGs for each scenario')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logging.warning(f"Config file {config_path} not found, using default parameters")
        return {}


def main():
    """Main function to generate datasets."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    data_config = config.get('data', {})
    data_config['n_samples'] = args.n_samples
    data_config['random_state'] = args.random_state
    
    # Create directories
    dirs = create_directories()
    output_dir = os.path.join(dirs['data'])
    
    # Set random seed
    set_seeds(data_config['random_state'])
    
    # Log parameters
    logger.info(f"Generating datasets with {data_config['n_samples']} samples "
               f"(random_state={data_config['random_state']})")
    
    # Generate all datasets
    if 'all' in args.scenarios:
        generate_all_datasets(
            output_dir=output_dir,
            n_samples=data_config['n_samples'],
            random_state=data_config['random_state']
        )
        logger.info(f"All datasets generated and saved to {output_dir}")
    else:
        # Generate specific scenarios
        os.makedirs(output_dir, exist_ok=True)
        
        for scenario in args.scenarios:
            logger.info(f"Generating {scenario} dataset...")
            
            if scenario == 'asset_pricing':
                scenario_obj = AssetPricingScenario(
                    n_samples=data_config['n_samples'],
                    random_state=data_config['random_state'],
                    n_causal=data_config.get('asset_pricing', {}).get('n_causal', 2),
                    n_spurious=data_config.get('asset_pricing', {}).get('n_spurious', 3),
                    n_confounders=data_config.get('asset_pricing', {}).get('n_confounders', 2),
                    noise_level=data_config.get('asset_pricing', {}).get('noise_level', 0.2)
                )
            elif scenario == 'credit_risk':
                scenario_obj = CreditRiskScenario(
                    n_samples=data_config['n_samples'],
                    random_state=data_config['random_state'],
                    n_causal=data_config.get('credit_risk', {}).get('n_causal', 2),
                    n_proxy=data_config.get('credit_risk', {}).get('n_proxy', 2),
                    n_confounders=data_config.get('credit_risk', {}).get('n_confounders', 2),
                    noise_level=data_config.get('credit_risk', {}).get('noise_level', 0.2)
                )
            elif scenario == 'fraud_detection':
                scenario_obj = FraudDetectionScenario(
                    n_samples=data_config['n_samples'],
                    random_state=data_config['random_state'],
                    n_causal=data_config.get('fraud_detection', {}).get('n_causal', 2),
                    n_indirect=data_config.get('fraud_detection', {}).get('n_indirect', 2),
                    n_confounders=data_config.get('fraud_detection', {}).get('n_confounders', 2),
                    fraud_rate=data_config.get('fraud_detection', {}).get('fraud_rate', 0.05),
                    noise_level=data_config.get('fraud_detection', {}).get('noise_level', 0.2)
                )
            else:
                logger.error(f"Unknown scenario: {scenario}")
                continue
            
            # Generate data
            X, y, causal_info = scenario_obj.generate_data()
            
            # Save data
            X.to_csv(os.path.join(output_dir, f'{scenario}_features.csv'), index=False)
            y.to_csv(os.path.join(output_dir, f'{scenario}_target.csv'), index=False)
            
            # Save causal info (excluding DAG which isn't easily serializable)
            import pandas as pd
            pd.DataFrame({
                'feature': X.columns,
                'is_causal': [feat in causal_info['causal_features'] for feat in X.columns]
            }).to_csv(os.path.join(output_dir, f'{scenario}_causal_info.csv'), index=False)
            
            # Visualize DAG if requested
            if args.visualize:
                import matplotlib.pyplot as plt
                scenario_obj.plot_dag()
                plt.savefig(os.path.join(dirs['plots'], f'{scenario}_dag.png'))
                logger.info(f"DAG visualization saved to {os.path.join(dirs['plots'], f'{scenario}_dag.png')}")
            
            logger.info(f"{scenario.capitalize()} dataset generated and saved to {output_dir}")
    
    # Save configuration
    config['data'] = data_config
    save_config(config, os.path.join(dirs['logs'], 'data_config.yaml'))
    
    logger.info("Data generation complete!")


if __name__ == "__main__":
    main()