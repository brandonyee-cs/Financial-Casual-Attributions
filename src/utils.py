import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
import json
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy.core.multiarray


def create_directories(base_dir: str = '.') -> Dict[str, str]:
    """
    Create directory structure for the project.
    
    Args:
        base_dir: Base directory
        
    Returns:
        Dict with directory paths
    """
    # Define directory structure
    directories = {
        'data': os.path.join(base_dir, 'data'),
        'models': os.path.join(base_dir, 'models'),
        'results': os.path.join(base_dir, 'results'),
        'plots': os.path.join(base_dir, 'plots'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    # Create directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return directories


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_results(results: pd.DataFrame, results_path: str) -> None:
    """
    Save results to CSV file.
    
    Args:
        results: Results DataFrame
        results_path: Path to save results
    """
    results.to_csv(results_path, index=False)


def load_results(results_path: str) -> pd.DataFrame:
    """
    Load results from CSV file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Results DataFrame
    """
    return pd.read_csv(results_path)


def setup_logging(log_dir: str = './logs', log_name: str = 'experiment.log') -> None:
    """
    Setup logging for the project.
    
    Args:
        log_dir: Directory for log files
        log_name: Name of log file
    """
    import logging
    
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    log_path = os.path.join(log_dir, log_name)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(scenario_name: str, data_dir: str = './data') -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Load dataset for a specific scenario.
    
    Args:
        scenario_name: Name of scenario ('asset_pricing', 'credit_risk', 'fraud_detection')
        data_dir: Directory containing data
        
    Returns:
        X: Features DataFrame
        y: Target Series
        causal_info: Dictionary with causal information
    """
    # Load features and target
    X = pd.read_csv(os.path.join(data_dir, f'{scenario_name}_features.csv'))
    y = pd.read_csv(os.path.join(data_dir, f'{scenario_name}_target.csv')).iloc[:, 0]
    
    # Load causal info
    causal_info_df = pd.read_csv(os.path.join(data_dir, f'{scenario_name}_causal_info.csv'))
    
    # Convert causal info to dictionary
    causal_features = causal_info_df[causal_info_df['is_causal']]['feature'].tolist()
    non_causal_features = causal_info_df[~causal_info_df['is_causal']]['feature'].tolist()
    
    if scenario_name == 'fraud_detection':
        causal_info = {
            'causal_features': causal_features,
            'indirect_features': non_causal_features
        }
    else:
        causal_info = {
            'causal_features': causal_features,
            'spurious_features': non_causal_features
        }
    
    return X, y, causal_info


def load_model(scenario_name: str, model_type: str = 'mlp', model_dir: str = './models') -> Any:
    """
    Load trained model for a specific scenario.
    
    Args:
        scenario_name: Name of scenario ('asset_pricing', 'credit_risk', 'fraud_detection')
        model_type: Type of model ('mlp', 'lstm', 'xgboost')
        model_dir: Directory containing models
        
    Returns:
        Loaded model
    """
    from src.models import SimpleMLPModel, TimeSeriesLSTMModel, GradientBoostingWrapper
    from sklearn.preprocessing import StandardScaler
    import torch.serialization
    
    # Add required classes to safe globals for PyTorch 2.6+
    torch.serialization.add_safe_globals([
        StandardScaler,
        numpy.core.multiarray.scalar
    ])
    
    # Define model path
    model_path = os.path.join(model_dir, f"{scenario_name}_{model_type}_model.pkl")
    
    # Initialize empty model
    if model_type == 'mlp':
        model = SimpleMLPModel(input_dim=1)  # Dummy input_dim, will be overwritten
    elif model_type == 'lstm':
        model = TimeSeriesLSTMModel(input_dim=1)  # Dummy input_dim, will be overwritten
    elif model_type == 'xgboost':
        model = GradientBoostingWrapper()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load model
    model.load(model_path)
    
    return model


def preprocess_data(X: pd.DataFrame, y: pd.Series, scaler=None) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Preprocess data for model training or evaluation.
    
    Args:
        X: Features DataFrame
        y: Target Series
        scaler: Optional scaler for features
        
    Returns:
        X_scaled: Scaled features
        y_array: Target array
        scaler: Fitted scaler
    """
    from sklearn.preprocessing import StandardScaler
    
    # Convert to numpy arrays
    X_array = X.values
    y_array = y.values
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)
    else:
        X_scaled = scaler.transform(X_array)
    
    return X_scaled, y_array, scaler


if __name__ == "__main__":
    # Example usage
    
    # Create directory structure
    directories = create_directories()
    print("Created directories:", directories)
    
    # Setup logging
    setup_logging()
    
    # Set random seeds
    set_seeds()
    
    # Save and load configuration
    config = {
        'data': {
            'n_samples': 10000,
            'random_state': 42
        },
        'model': {
            'type': 'mlp',
            'hidden_dims': [64, 32],
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100
        },
        'evaluation': {
            'attribution_methods': ['saliency', 'gradient_input', 'integrated_gradients', 'shap'],
            'n_samples': 100
        }
    }
    
    config_path = os.path.join(directories['logs'], 'config.yaml')
    save_config(config, config_path)
    
    loaded_config = load_config(config_path)
    print("Loaded configuration:", loaded_config)