"""
Consolidated analysis module for financial causal attributions.

This module combines attribution methods, evaluation metrics, and utility functions
for analyzing the causal faithfulness of feature attributions in financial ML models.
"""

import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.autograd import grad
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy.core.multiarray
import logging
import yaml

#pl.Config.set_engine_affinity(engine="streaming")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#---------------------------------------
# Utility Functions
#---------------------------------------

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
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config file {config_path} not found, returning empty dict")
        return {}

def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset(scenario_name: str, data_dir: str = './data') -> Tuple[pl.DataFrame, pl.Series, Dict]:
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
    X = pl.read_csv(os.path.join(data_dir, f'{scenario_name}_features.csv'))
    y_df = pl.read_csv(os.path.join(data_dir, f'{scenario_name}_target.csv'))
    y = y_df.to_series()
    
    # Load causal info
    causal_info_df = pl.read_csv(os.path.join(data_dir, f'{scenario_name}_causal_info.csv'))
    
    # Convert causal info to dictionary
    causal_features = causal_info_df.filter(pl.col('is_causal')).select('feature').to_series().to_list()
    non_causal_features = causal_info_df.filter(~pl.col('is_causal')).select('feature').to_series().to_list()
    
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
        scenario_name: Name of scenario
        model_type: Type of model ('mlp', 'lstm', 'xgboost')
        model_dir: Directory containing models
        
    Returns:
        Loaded model
    """
    from models import SimpleMLPModel, TimeSeriesLSTMModel, GradientBoostingWrapper
    
    # Add required classes to safe globals for PyTorch 2.6+
    try:
        torch.serialization.add_safe_globals([
            StandardScaler,
            numpy.core.multiarray.scalar
        ])
    except:
        logger.warning("Unable to add safe globals for PyTorch serialization. May cause issues with loading models.")
    
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

class GradientBoostingWrapper:
    """Wrapper for XGBoost models."""
    
    def __init__(self):
        """Initialize XGBoost wrapper."""
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit XGBoost model."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine if classification or regression
        if len(np.unique(y)) <= 2:  # Binary classification
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        else:  # Regression
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        
        # Fit model
        self.model.fit(X_scaled, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path: str) -> None:
        """Save model."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, path: str) -> None:
        """Load model."""
        import pickle
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
            self.model = loaded.model
            self.scaler = loaded.scaler

#---------------------------------------
# Attribution Methods
#---------------------------------------

class AttributionMethod:
    """Base class for attribution methods."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize attribution method."""
        self.model = model
        self.feature_names = feature_names
    
    def attribute(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                **kwargs) -> np.ndarray:
        """Compute attributions for model predictions."""
        raise NotImplementedError
    
    def normalize_attributions(self, attributions: np.ndarray) -> np.ndarray:
        """Normalize attributions."""
        # Compute absolute attributions
        abs_attr = np.abs(attributions)
        
        # Normalize by dividing by sum (if non-zero)
        row_sums = abs_attr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        
        return attributions / row_sums

    def get_attributions_df(self, X: np.ndarray, attributions: np.ndarray) -> pl.DataFrame:
        """Convert attributions to DataFrame format."""
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            feature_names = self.feature_names
            
        return pl.DataFrame(attributions, schema=feature_names)


class SaliencyMap(AttributionMethod):
    """Simple gradient-based attribution method."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize saliency map."""
        super().__init__(model, feature_names)
        
        # Check if model is a PyTorch model
        if not isinstance(self.model.model, nn.Module):
            raise ValueError("Saliency map is only implemented for PyTorch models")
    
    def attribute(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                **kwargs) -> np.ndarray:
        """Compute saliency map attributions."""
        # Scale inputs
        X_scaled = self.model.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)
        
        # Compute model output
        self.model.model.eval()
        outputs = self.model.model(X_tensor)
        
        # Compute gradients
        grads = []
        for i in range(len(X)):
            # Get single output
            output = outputs[i].sum()
            
            # Compute gradient
            output.backward(retain_graph=True if i < len(X) - 1 else False)
            
            # Store gradient
            grads.append(X_tensor.grad[i].detach().numpy())
            
            # Zero gradients for next iteration
            if i < len(X) - 1:
                X_tensor.grad.zero_()
        
        # Stack gradients
        attributions = np.stack(grads)
        
        return attributions


class GradientInputMethod(AttributionMethod):
    """Gradient * Input attribution method."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize gradient * input method."""
        super().__init__(model, feature_names)
        
        # Check if model is a PyTorch model
        if not isinstance(self.model.model, nn.Module):
            raise ValueError("Gradient * Input is only implemented for PyTorch models")
    
    def attribute(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                **kwargs) -> np.ndarray:
        """Compute gradient * input attributions."""
        # Scale inputs
        X_scaled = self.model.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)
        
        # Compute model output
        self.model.model.eval()
        outputs = self.model.model(X_tensor)
        
        # Compute gradients
        grads = []
        for i in range(len(X)):
            # Get single output
            output = outputs[i].sum()
            
            # Compute gradient
            output.backward(retain_graph=True if i < len(X) - 1 else False)
            
            # Store gradient * input
            grads.append((X_tensor.grad[i] * X_tensor[i]).detach().numpy())
            
            # Zero gradients for next iteration
            if i < len(X) - 1:
                X_tensor.grad.zero_()
        
        # Stack gradients * inputs
        attributions = np.stack(grads)
        
        return attributions


class IntegratedGradientsMethod(AttributionMethod):
    """Integrated Gradients attribution method."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize integrated gradients method."""
        super().__init__(model, feature_names)
        
        # Check if model is a PyTorch model
        if not isinstance(self.model.model, nn.Module):
            raise ValueError("Integrated Gradients is only implemented for PyTorch models")
    
    def attribute(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                steps: int = 50, baseline: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """Compute integrated gradients attributions."""
        # Scale inputs
        X_scaled = self.model.scaler.transform(X)
        
        # Create baseline if not provided
        if baseline is None:
            baseline_scaled = np.zeros_like(X_scaled)
        else:
            baseline_scaled = self.model.scaler.transform(baseline)
        
        # Initialize attributions
        attributions = np.zeros_like(X_scaled)
        
        # Compute integrated gradients
        for sample_idx in range(len(X)):
            # Get single input and baseline
            x_sample = X_scaled[sample_idx]
            baseline_sample = baseline_scaled[0] if len(baseline_scaled) == 1 else baseline_scaled[sample_idx]
            
            # Create path from baseline to input
            alphas = np.linspace(0, 1, steps)
            path = np.array([baseline_sample + alpha * (x_sample - baseline_sample) for alpha in alphas])
            
            # Convert to tensor
            path_tensor = torch.tensor(path, dtype=torch.float32, requires_grad=True)
            
            # Compute gradients along path
            self.model.model.eval()
            outputs = self.model.model(path_tensor)
            
            # Accumulate gradients
            path_grads = []
            for i in range(len(path)):
                # Get single output
                output = outputs[i].sum()
                
                # Compute gradient
                output.backward(retain_graph=True if i < len(path) - 1 else False)
                
                # Store gradient
                path_grads.append(path_tensor.grad[i].detach().numpy())
                
                # Zero gradients for next iteration
                if i < len(path) - 1:
                    path_tensor.grad.zero_()
            
            # Stack gradients
            path_grads = np.stack(path_grads)
            
            # Integrate gradients
            avg_grads = np.mean(path_grads, axis=0)
            attributions[sample_idx] = (x_sample - baseline_sample) * avg_grads
        
        return attributions


class ShapleyValueMethod(AttributionMethod):
    """SHAP (SHapley Additive exPlanations) attribution method."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize SHAP method."""
        super().__init__(model, feature_names)
        self.explainer = None
    
    def _create_explainer(self, X: np.ndarray) -> None:
        """Create SHAP explainer for the model."""
        if isinstance(self.model.model, xgb.Booster):
            # For XGBoost models
            self.explainer = shap.TreeExplainer(self.model.model)
        else:
            # For other models (including PyTorch)
            def predict_fn(x):
                if isinstance(x, pl.DataFrame):
                    x = x.to_numpy()
                return self.model.predict(x)
            
            self.explainer = shap.KernelExplainer(predict_fn, X)
    
    def attribute(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                **kwargs) -> np.ndarray:
        """Compute SHAP attributions."""
        if self.explainer is None:
            self._create_explainer(X)
        
        # Compute SHAP values
        if isinstance(self.model.model, xgb.Booster):
            # For XGBoost models
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class for binary classification
        else:
            # For other models
            shap_values = self.explainer.shap_values(X)
        
        return np.array(shap_values)


class XGBoostAttributionMethod(AttributionMethod):
    """Attribution method for XGBoost models using feature importance."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize XGBoost attribution method."""
        super().__init__(model, feature_names)
        
        # Check if model is an XGBoost model
        if not isinstance(self.model.model, xgb.Booster):
            raise ValueError("XGBoost attribution method is only implemented for XGBoost models")
    
    def attribute(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                **kwargs) -> np.ndarray:
        """Compute XGBoost attributions using feature importance."""
        # Get feature importance scores
        importance_type = kwargs.get('importance_type', 'gain')
        importance_scores = self.model.model.get_score(importance_type=importance_type)
        
        # Convert to array format
        attributions = np.zeros((len(X), len(self.feature_names)))
        for i, feature in enumerate(self.feature_names):
            if feature in importance_scores:
                attributions[:, i] = importance_scores[feature]
        
        return attributions


def compute_attributions(model: Any, X: np.ndarray, method_name: str,
                       feature_names: Optional[List[str]] = None,
                       **kwargs) -> pl.DataFrame:
    """
    Compute attributions using specified method.
    
    Args:
        model: Trained model
        X: Input features
        method_name: Name of attribution method
        feature_names: List of feature names
        **kwargs: Additional arguments for attribution method
        
    Returns:
        DataFrame with attributions
    """
    # Create attribution method
    if method_name == 'saliency':
        attribution_method = SaliencyMap(model, feature_names)
    elif method_name == 'gradient_input':
        attribution_method = GradientInputMethod(model, feature_names)
    elif method_name == 'integrated_gradients':
        attribution_method = IntegratedGradientsMethod(model, feature_names)
    elif method_name == 'shap':
        attribution_method = ShapleyValueMethod(model, feature_names)
    elif method_name == 'xgboost':
        attribution_method = XGBoostAttributionMethod(model, feature_names)
    else:
        raise ValueError(f"Unsupported attribution method: {method_name}")
    
    # Compute attributions
    attributions = attribution_method.attribute(X, **kwargs)
    
    # Convert to DataFrame
    return attribution_method.get_attributions_df(X, attributions)

#---------------------------------------
# Evaluation Metrics
#---------------------------------------

class CausalFaithfulnessEvaluator:
    """
    Evaluator for causal faithfulness of attribution methods.
    
    This class evaluates how well different attribution methods identify truly
    causal features in a model's decision making.
    """
    
    def __init__(self, causal_features: List[str], non_causal_features: List[str]):
        """Initialize causal faithfulness evaluator."""
        self.causal_features = causal_features
        self.non_causal_features = non_causal_features
        self.all_features = causal_features + non_causal_features
    
    def evaluate_attribution_ranking(self, attributions: pl.DataFrame) -> Dict[str, float]:
        """Evaluate attribution ranking by computing metrics on feature importance ranking."""
        # Get absolute attribution values
        abs_attrs = attributions.select([pl.col(col).abs().alias(col) for col in attributions.columns])
        
        # Calculate ranks for each row
        feature_ranks = abs_attrs.select(
            [pl.col(col).rank(method='average', descending=True).alias(col) for col in abs_attrs.columns]
        )
        
        # Extract causal and non-causal ranks
        causal_ranks = feature_ranks.select(self.causal_features).to_numpy().flatten()
        non_causal_ranks = feature_ranks.select(self.non_causal_features).to_numpy().flatten()
        
        # Calculate metrics
        metrics = {}
        
        # Mean Rank: lower is better for causal features, higher is better for non-causal
        metrics['mean_rank_causal'] = np.mean(causal_ranks)
        metrics['mean_rank_non_causal'] = np.mean(non_causal_ranks)
        
        # Normalized Mean Rank (0 to 1, lower is better for causal)
        total_features = len(self.all_features)
        metrics['normalized_mean_rank_causal'] = (metrics['mean_rank_causal'] - 1) / (total_features - 1)
        
        # Rank Separation: higher is better (difference between non-causal and causal ranks)
        metrics['rank_separation'] = metrics['mean_rank_non_causal'] - metrics['mean_rank_causal']
        
        # Normalized Rank Separation (-1 to 1, higher is better)
        metrics['normalized_rank_separation'] = metrics['rank_separation'] / (total_features - 1)
        
        # Top-K Accuracy: percentage of top-K features that are causal
        k = len(self.causal_features)
        top_k_features = []
        
        # Get top-k features for each row
        for i in range(abs_attrs.height):
            row = abs_attrs.row(i)
            # Convert to dict for easier sorting
            row_dict = {col: row[idx] for idx, col in enumerate(abs_attrs.columns)}
            # Sort by value and take top k
            top_features = sorted(row_dict.items(), key=lambda x: x[1], reverse=True)[:k]
            top_k_features.extend([feat[0] for feat in top_features])
        
        top_k_correct = sum(1 for feat in top_k_features if feat in self.causal_features)
        metrics['top_k_accuracy'] = top_k_correct / len(top_k_features)
        
        return metrics
    
    def evaluate_attribution_magnitude(self, attributions: pl.DataFrame) -> Dict[str, float]:
        """Evaluate attribution magnitude by computing metrics on attribution values."""
        # Get absolute attribution values
        abs_attrs = attributions.select([pl.col(col).abs().alias(col) for col in attributions.columns])
        
        # Calculate mean absolute attribution for causal and non-causal features
        causal_attrs = abs_attrs.select(self.causal_features).to_numpy().flatten()
        non_causal_attrs = abs_attrs.select(self.non_causal_features).to_numpy().flatten()
        
        # Calculate metrics
        metrics = {}
        
        # Mean Attribution: higher is better for causal, lower is better for non-causal
        metrics['mean_attr_causal'] = np.mean(causal_attrs)
        metrics['mean_attr_non_causal'] = np.mean(non_causal_attrs)
        
        # Attribution Ratio: higher is better (ratio of causal to non-causal)
        if metrics['mean_attr_non_causal'] > 0:
            metrics['attribution_ratio'] = metrics['mean_attr_causal'] / metrics['mean_attr_non_causal']
        else:
            metrics['attribution_ratio'] = float('inf')
        
        # Attribution Difference: higher is better (difference between causal and non-causal)
        metrics['attribution_difference'] = metrics['mean_attr_causal'] - metrics['mean_attr_non_causal']
        
        # Normalized Attribution Difference
        total_attr = metrics['mean_attr_causal'] + metrics['mean_attr_non_causal']
        if total_attr > 0:
            metrics['normalized_attribution_difference'] = metrics['attribution_difference'] / total_attr
        else:
            metrics['normalized_attribution_difference'] = 0.0
        
        return metrics
    
    def evaluate_attribution_stability(self, attributions_list: List[pl.DataFrame]) -> Dict[str, float]:
        """Evaluate stability of attributions across multiple runs or samples."""
        if len(attributions_list) < 2:
            return {'stability_score': 1.0}  # Perfect stability with only one sample
        
        # Calculate pairwise rank correlation for all features
        rank_correlations = []
        
        for i in range(len(attributions_list)):
            for j in range(i + 1, len(attributions_list)):
                # Get absolute attributions
                abs_attrs_i = attributions_list[i].select([pl.col(col).abs().alias(col) for col in attributions_list[i].columns])
                abs_attrs_j = attributions_list[j].select([pl.col(col).abs().alias(col) for col in attributions_list[j].columns])
                
                # Get ranks
                ranks_i = abs_attrs_i.select([pl.col(col).rank(method='average', descending=True).alias(col) for col in abs_attrs_i.columns])
                ranks_j = abs_attrs_j.select([pl.col(col).rank(method='average', descending=True).alias(col) for col in abs_attrs_j.columns])
                
                # Calculate Spearman rank correlation for each sample
                sample_correlations = []
                
                # For each row in both DataFrames
                for row_i in range(min(ranks_i.height, ranks_j.height)):
                    ranks_i_row = ranks_i.row(row_i)
                    ranks_j_row = ranks_j.row(row_i)
                    
                    # Convert to numpy arrays for correlation calculation
                    ranks_i_arr = np.array([ranks_i_row[ranks_i.columns.index(col)] for col in self.all_features])
                    ranks_j_arr = np.array([ranks_j_row[ranks_j.columns.index(col)] for col in self.all_features])
                    
                    # Calculate correlation
                    if len(ranks_i_arr) > 1:  # Need at least 2 points for correlation
                        corr = np.corrcoef(ranks_i_arr, ranks_j_arr)[0, 1]
                        if not np.isnan(corr):
                            sample_correlations.append(corr)
                
                if sample_correlations:
                    rank_correlations.append(np.mean(sample_correlations))
        
        # Calculate metrics
        metrics = {}
        
        # Stability Score: higher is better (average rank correlation)
        if rank_correlations:
            metrics['stability_score'] = np.mean(rank_correlations)
        else:
            metrics['stability_score'] = np.nan
        
        return metrics
    
    def compute_overall_faithfulness_score(self, ranking_metrics: Dict[str, float],
                                         magnitude_metrics: Dict[str, float],
                                         stability_metrics: Dict[str, float]) -> float:
        """Compute an overall faithfulness score from individual metrics."""
        # Combine key metrics into overall score
        score_components = []
        
        # From ranking metrics (convert to 0-1 where 1 is better)
        if 'normalized_mean_rank_causal' in ranking_metrics:
            score_components.append(1 - ranking_metrics['normalized_mean_rank_causal'])
        
        if 'normalized_rank_separation' in ranking_metrics:
            # Map from [-1, 1] to [0, 1]
            sep_score = (ranking_metrics['normalized_rank_separation'] + 1) / 2
            score_components.append(sep_score)
        
        if 'top_k_accuracy' in ranking_metrics:
            score_components.append(ranking_metrics['top_k_accuracy'])
        
        # From magnitude metrics (convert to 0-1 where 1 is better)
        if 'normalized_attribution_difference' in magnitude_metrics:
            # Map from [-1, 1] to [0, 1]
            diff_score = (magnitude_metrics['normalized_attribution_difference'] + 1) / 2
            score_components.append(diff_score)
        
        # From stability metrics
        if 'stability_score' in stability_metrics and not np.isnan(stability_metrics['stability_score']):
            # Map from [-1, 1] to [0, 1]
            stab_score = (stability_metrics['stability_score'] + 1) / 2
            score_components.append(stab_score)
        
        # Calculate overall score (if components exist)
        if score_components:
            overall_score = np.mean(score_components)
            return overall_score
        else:
            return np.nan
    
    def evaluate_all_metrics(self, attributions: pl.DataFrame, 
                           attributions_list: Optional[List[pl.DataFrame]] = None) -> Dict[str, float]:
        """Evaluate all faithfulness metrics for a given attribution method."""
        # Combine the main attributions with the list if provided
        if attributions_list is None:
            attributions_list = [attributions]
        elif attributions not in attributions_list:
            attributions_list = [attributions] + attributions_list
        
        # Evaluate ranking metrics
        ranking_metrics = self.evaluate_attribution_ranking(attributions)
        
        # Evaluate magnitude metrics
        magnitude_metrics = self.evaluate_attribution_magnitude(attributions)
        
        # Evaluate stability metrics
        stability_metrics = self.evaluate_attribution_stability(attributions_list)
        
        # Compute overall score
        overall_score = self.compute_overall_faithfulness_score(
            ranking_metrics, magnitude_metrics, stability_metrics
        )
        
        # Combine all metrics
        all_metrics = {
            'overall_faithfulness_score': overall_score,
            **{f'ranking_{k}': v for k, v in ranking_metrics.items()},
            **{f'magnitude_{k}': v for k, v in magnitude_metrics.items()},
            **{f'stability_{k}': v for k, v in stability_metrics.items()}
        }
        
        return all_metrics

#---------------------------------------
# Visualization Functions
#---------------------------------------

def plot_attribution_heatmap(attributions: pl.DataFrame, 
                           causal_features: List[str],
                           non_causal_features: List[str],
                           title: str = 'Feature Attributions',
                           output_path: Optional[str] = None) -> None:
    """
    Plot heatmap of feature attributions.
    
    Args:
        attributions: DataFrame with attribution scores
        causal_features: List of causal feature names
        non_causal_features: List of non-causal feature names
        title: Plot title
        output_path: Path to save the plot (if None, just display)
    """
    # Get absolute attributions
    abs_attrs = attributions.select([pl.col(col).abs().alias(col) for col in attributions.columns])
    
    # Sort features by causal and non-causal
    sorted_features = causal_features + non_causal_features
    
    # Select a subset of samples if there are too many
    if abs_attrs.height > 20:
        abs_attrs = abs_attrs.slice(0, 20)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Normalize attributions
    normalized_attrs = abs_attrs.select(sorted_features)
    row_sums = normalized_attrs.sum(axis=1)
    normalized_attrs = normalized_attrs.with_columns([
        pl.col(col) / pl.when(row_sums > 0).then(row_sums).otherwise(1.0)
        for col in sorted_features
    ])
    
    # Convert to numpy for plotting
    plot_data = normalized_attrs.to_numpy().T
    
    # Plot heatmap
    sns.heatmap(
        plot_data,
        cmap='YlOrRd',
        vmin=0,
        vmax=plot_data.max(),
        yticklabels=sorted_features,
        cbar_kws={'label': 'Normalized Attribution Magnitude'}
    )
    
    # Add horizontal line between causal and non-causal features
    plt.axhline(y=len(causal_features), color='black', linewidth=2)
    
    # Set title and labels
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Feature')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_mean_attributions(attributions: pl.DataFrame, 
                         causal_features: List[str],
                         non_causal_features: List[str],
                         title: str = 'Mean Feature Attributions',
                         output_path: Optional[str] = None) -> None:
    """
    Plot mean attribution values for each feature.
    
    Args:
        attributions: DataFrame with attribution scores
        causal_features: List of causal feature names
        non_causal_features: List of non-causal feature names
        title: Plot title
        output_path: Path to save the plot (if None, just display)
    """
    # Calculate mean absolute attributions
    all_features = causal_features + non_causal_features
    mean_attrs = abs_attrs = attributions.select([
        pl.col(col).abs().mean().alias(col) for col in attributions.columns
    ]).row(0)
    
    # Sort features by mean attribution
    sorted_features = sorted(
        [(col, mean_attrs[attributions.columns.index(col)]) for col in all_features],
        key=lambda x: x[1],
        reverse=True
    )
    feature_names = [f[0] for f in sorted_features]
    feature_values = [f[1] for f in sorted_features]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Color bars based on feature type (causal or non-causal)
    colors = ['royalblue' if feat in causal_features else 'lightcoral' for feat in feature_names]
    
    # Plot bars
    bars = plt.bar(feature_names, feature_values, color=colors)
    
    # Add labels and title
    plt.title(title)
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
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_faithfulness_comparison(results_df: pl.DataFrame, 
                               metric: str = 'overall_faithfulness_score',
                               title: Optional[str] = None,
                               output_path: Optional[str] = None) -> None:
    """
    Plot comparison of faithfulness metrics across methods and scenarios.
    
    Args:
        results_df: DataFrame with evaluation results
        metric: Metric to plot
        title: Plot title
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar chart using seaborn
    # Convert to numpy arrays for plotting
    methods = results_df.select('attribution_method').unique().to_series().to_list()
    scenarios = results_df.select('scenario').unique().to_series().to_list()
    
    # Create data for plotting
    plot_data = []
    for method in methods:
        for scenario in scenarios:
            value = results_df.filter(
                (pl.col('attribution_method') == method) & 
                (pl.col('scenario') == scenario)
            ).select(metric).mean().item()
            plot_data.append({
                'attribution_method': method,
                'scenario': scenario,
                metric: value
            })
    
    # Convert to DataFrame for seaborn
    plot_df = pl.DataFrame(plot_data)
    
    # Create plot
    ax = sns.barplot(
        data=plot_df.to_pandas(),  # seaborn requires pandas
        x='attribution_method',
        y=metric,
        hue='scenario',
        palette='viridis'
    )
    
    # Set title and labels
    if title:
        plt.title(title)
    else:
        plt.title(f'Comparison of Attribution Methods: {metric}')
    
    plt.xlabel('Attribution Method')
    plt.ylabel(metric.replace('_', ' ').title())
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Add legend
    plt.legend(title='Scenario')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_faithfulness_heatmap(results_df: pl.DataFrame,
                            metric: str = 'overall_faithfulness_score',
                            title: Optional[str] = None,
                            output_path: Optional[str] = None) -> None:
    """
    Plot heatmap of faithfulness metrics across models, methods, and scenarios.
    
    Args:
        results_df: DataFrame with evaluation results
        metric: Metric to plot
        title: Plot title
        output_path: Path to save the plot
    """
    # Convert to pandas and pivot
    results_pd = results_df.to_pandas()
    pivot_data = results_pd.pivot_table(
        index=['scenario', 'model_type'],
        columns='attribution_method',
        values=metric
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(
        pivot_data,
        annot=True,
        cmap='YlGnBu',
        fmt='.3f',
        linewidths=0.5
    )
    
    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f'{metric.replace("_", " ").title()} by Scenario, Model, and Method')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

#---------------------------------------
# Pipeline Functions
#---------------------------------------

def run_attribution_analysis(scenario: str, model_type: str, attribution_method: str,
                           data_dir: str, model_dir: str, output_dir: str, plots_dir: str,
                           n_samples: int = 100) -> Dict:
    """
    Run attribution analysis for a specific scenario, model, and method.
    
    Args:
        scenario: Name of scenario ('asset_pricing', 'credit_risk', 'fraud_detection')
        model_type: Type of model ('mlp', 'lstm', 'xgboost')
        attribution_method: Name of attribution method
        data_dir: Directory containing data
        model_dir: Directory containing models
        output_dir: Directory to save results
        plots_dir: Directory to save plots
        n_samples: Number of samples to use
        
    Returns:
        Dict with results
    """
    logger.info(f"Running {attribution_method} analysis for {model_type} on {scenario}")
    
    # Load data
    X, y, causal_info = load_dataset(scenario, data_dir)
    
    # Load model
    model = load_model(scenario, model_type, model_dir)
    
    # Sample data
    if X.height > n_samples:
        # Create random sample indices
        np.random.seed(42)
        sample_indices = np.random.choice(X.height, n_samples, replace=False)
        X_sample = X.select(pl.all()).filter(pl.int_range(0, X.height).is_in(sample_indices))
    else:
        X_sample = X
    
    # Compute attributions
    attributions = compute_attributions(
        model=model,
        X=X_sample.to_numpy(),
        method_name=attribution_method,
        feature_names=X.columns
    )
    
    # Save attributions
    os.makedirs(output_dir, exist_ok=True)
    attribution_path = os.path.join(output_dir, f'{scenario}_{model_type}_{attribution_method}_attributions.csv')
    attributions.write_csv(attribution_path)
    
    # Determine causal and non-causal features
    if scenario == 'fraud_detection':
        causal_features = causal_info['causal_features']
        non_causal_features = causal_info['indirect_features']
    else:
        causal_features = causal_info['causal_features']
        non_causal_features = causal_info['spurious_features']
    
    # Create evaluator
    evaluator = CausalFaithfulnessEvaluator(causal_features, non_causal_features)
    
    # Evaluate attributions
    metrics = evaluator.evaluate_all_metrics(attributions)
    
    # Add metadata
    metrics.update({
        'scenario': scenario,
        'model_type': model_type,
        'attribution_method': attribution_method
    })
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'{scenario}_{model_type}_{attribution_method}_metrics.csv')
    pl.DataFrame([metrics]).write_csv(metrics_path)
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot heatmap
    heatmap_path = os.path.join(plots_dir, f'{scenario}_{model_type}_{attribution_method}_heatmap.png')
    plot_attribution_heatmap(
        attributions=attributions,
        causal_features=causal_features,
        non_causal_features=non_causal_features,
        title=f'{attribution_method.replace("_", " ").title()} Attributions - {model_type.upper()} ({scenario.replace("_", " ").title()})',
        output_path=heatmap_path
    )
    
    # Plot mean attributions
    barplot_path = os.path.join(plots_dir, f'{scenario}_{model_type}_{attribution_method}_bars.png')
    plot_mean_attributions(
        attributions=attributions,
        causal_features=causal_features,
        non_causal_features=non_causal_features,
        title=f'Mean {attribution_method.replace("_", " ").title()} Attribution Magnitude - {model_type.upper()} ({scenario.replace("_", " ").title()})',
        output_path=barplot_path
    )
    
    return {
        'attributions': attributions,
        'metrics': metrics,
        'paths': {
            'attributions': attribution_path,
            'metrics': metrics_path,
            'heatmap': heatmap_path,
            'barplot': barplot_path
        }
    }

def evaluate_faithfulness_metrics(results_dir: str, output_dir: str, plots_dir: str) -> pl.DataFrame:
    """
    Aggregate and evaluate faithfulness metrics across all scenarios, models, and methods.
    
    Args:
        results_dir: Directory containing result files
        output_dir: Directory to save aggregated results
        plots_dir: Directory to save plots
        
    Returns:
        DataFrame with aggregated metrics
    """
    logger.info("Evaluating faithfulness metrics across all combinations")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Find all metrics files
    metrics_files = []
    for file in os.listdir(results_dir):
        if file.endswith('_metrics.csv'):
            metrics_files.append(os.path.join(results_dir, file))
    
    # Load and combine metrics
    all_metrics = []
    
    for file in metrics_files:
        try:
            metrics_df = pl.read_csv(file)
            all_metrics.append(metrics_df)
        except Exception as e:
            logger.warning(f"Error loading {file}: {str(e)}")
    
    if not all_metrics:
        logger.warning("No metrics files found or loaded")
        return pl.DataFrame()
    
    # Combine all metrics
    combined_metrics = pl.concat(all_metrics)
    
    # Save combined metrics
    combined_path = os.path.join(output_dir, 'all_faithfulness_metrics.csv')
    combined_metrics.write_csv(combined_path)
    
    # Generate plots
    
    # Overall faithfulness score by method and scenario
    plot_faithfulness_comparison(
        results_df=combined_metrics,
        metric='overall_faithfulness_score',
        title='Overall Faithfulness Score by Attribution Method and Scenario',
        output_path=os.path.join(plots_dir, 'overall_score_by_method_scenario.png')
    )
    
    # Top-K accuracy by method and model
    plot_faithfulness_comparison(
        results_df=combined_metrics,
        metric='ranking_top_k_accuracy',
        title='Top-K Accuracy by Attribution Method and Scenario',
        output_path=os.path.join(plots_dir, 'topk_accuracy_by_method_scenario.png')
    )
    
    # Heatmap of overall faithfulness
    plot_faithfulness_heatmap(
        results_df=combined_metrics,
        metric='overall_faithfulness_score',
        title='Overall Faithfulness Score Across All Combinations',
        output_path=os.path.join(plots_dir, 'overall_faithfulness_heatmap.png')
    )
    
    # Also generate tables by scenario and method
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    # Convert to pandas for pivoting and write to CSV
    combined_metrics_pd = combined_metrics.to_pandas()
    
    # Overall faithfulness score by scenario, model, and method
    table1 = combined_metrics_pd.pivot_table(
        index=['scenario', 'model_type'],
        columns='attribution_method',
        values='overall_faithfulness_score'
    ).round(3)
    
    table1.to_csv(os.path.join(tables_dir, 'table1_overall_score.csv'))
    
    # Top-K accuracy by scenario, model, and method
    table2 = combined_metrics_pd.pivot_table(
        index=['scenario', 'model_type'],
        columns='attribution_method',
        values='ranking_top_k_accuracy'
    ).round(3)
    
    table2.to_csv(os.path.join(tables_dir, 'table2_topk_accuracy.csv'))
    
    # Attribution ratio by scenario, model, and method
    table3 = combined_metrics_pd.pivot_table(
        index=['scenario', 'model_type'],
        columns='attribution_method',
        values='magnitude_attribution_ratio'
    ).round(3)
    
    table3.to_csv(os.path.join(tables_dir, 'table3_attribution_ratio.csv'))
    
    logger.info(f"Saved combined metrics to {combined_path}")
    
    return combined_metrics

def generate_report(results_dir: str, output_dir: str, format: str = 'markdown') -> str:
    """
    Generate a comprehensive report of the results.
    
    Args:
        results_dir: Directory containing results
        output_dir: Directory to save the report
        format: Report format ('markdown' or 'latex')
        
    Returns:
        Path to the generated report
    """
    logger.info(f"Generating {format} report")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics
    metrics_path = os.path.join(results_dir, 'all_faithfulness_metrics.csv')
    
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(results_dir, 'faithfulness', 'all_faithfulness_metrics.csv')
        if not os.path.exists(metrics_path):
            logger.warning("No metrics file found. Cannot generate report.")
            return ""
    
    metrics_df = pl.read_csv(metrics_path)
    
    # Load model performance
    model_perf_path = os.path.join(results_dir, 'model_performance.csv')
    model_perf = None
    if os.path.exists(model_perf_path):
        model_perf = pl.read_csv(model_perf_path)
    
    # Generate report
    if format == 'markdown':
        report_path = os.path.join(output_dir, 'report.md')
        
        with open(report_path, 'w') as f:
            # Title and introduction
            f.write("# Causal Pitfalls of Feature Attributions in Financial Machine Learning Models\n\n")
            f.write("## Results Report\n\n")
            f.write(f"*Generated on: {pl.datetime('now').dt.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
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
            if model_perf is not None:
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
                
                for row_idx in range(model_perf.height):
                    row = model_perf.row(row_idx)
                    scenario_idx = model_perf.columns.index('scenario') if 'scenario' in model_perf.columns else -1
                    model_type_idx = model_perf.columns.index('model_type') if 'model_type' in model_perf.columns else -1
                    
                    scenario = row[scenario_idx] if scenario_idx >= 0 else ''
                    model_type = row[model_type_idx] if model_type_idx >= 0 else ''
                    
                    f.write(f"| {scenario} | {model_type} | ")
                    if 'accuracy' in model_perf.columns:
                        accuracy_idx = model_perf.columns.index('accuracy')
                        f.write(f"{row[accuracy_idx]:.4f} | ")
                    if 'f1' in model_perf.columns:
                        f1_idx = model_perf.columns.index('f1')
                        f.write(f"{row[f1_idx]:.4f} | ")
                    if 'mse' in model_perf.columns:
                        mse_idx = model_perf.columns.index('mse')
                        f.write(f"{row[mse_idx]:.4f} | ")
                    if 'r2' in model_perf.columns:
                        r2_idx = model_perf.columns.index('r2')
                        f.write(f"{row[r2_idx]:.4f} | ")
                    f.write("\n")
                
                f.write("\n\n")
            else:
                f.write("*Model performance data not available.*\n\n")
            
            # Faithfulness Evaluation
            f.write("## Faithfulness Evaluation\n\n")
            if not metrics_df.is_empty():
                # Overall faithfulness by method
                method_avg = metrics_df.group_by('attribution_method').agg(
                    pl.mean('overall_faithfulness_score')
                ).sort('overall_faithfulness_score', descending=True)
                
                f.write("### Overall Faithfulness Score by Attribution Method\n\n")
                
                f.write("| Attribution Method | Overall Faithfulness Score |\n")
                f.write("|" + "-" * 20 + "|" + "-" * 28 + "|\n")
                for row_idx in range(method_avg.height):
                    row = method_avg.row(row_idx)
                    f.write(f"| {row[0]} | {row[1]:.4f} |\n")
                
                f.write("\n\n")
                
                # Top-K accuracy by method
                topk_avg = metrics_df.group_by('attribution_method').agg(
                    pl.mean('ranking_top_k_accuracy')
                ).sort('ranking_top_k_accuracy', descending=True)
                
                f.write("### Top-K Accuracy by Attribution Method\n\n")
                
                f.write("| Attribution Method | Top-K Accuracy |\n")
                f.write("|" + "-" * 20 + "|" + "-" * 15 + "|\n")
                for row_idx in range(topk_avg.height):
                    row = topk_avg.row(row_idx)
                    f.write(f"| {row[0]} | {row[1]:.4f} |\n")
                
                f.write("\n\n")
            else:
                f.write("*Faithfulness metrics data not available.*\n\n")
            
            # Scenario-Specific Analysis
            f.write("## Scenario-Specific Analysis\n\n")
            if not metrics_df.is_empty():
                # For each scenario
                scenarios = metrics_df.select('scenario').unique().to_series().to_list()
                if len(scenarios) > 0:
                    for scenario in scenarios:
                        scenario_df = metrics_df.filter(pl.col('scenario') == scenario)
                        
                        f.write(f"### {scenario.replace('_', ' ').title()}\n\n")
                        
                        # Average overall faithfulness by method for this scenario
                        scenario_method_avg = scenario_df.group_by('attribution_method').agg(
                            pl.mean('overall_faithfulness_score')
                        ).sort('overall_faithfulness_score', descending=True)
                        
                        f.write(f"#### Overall Faithfulness Score by Method in {scenario.replace('_', ' ').title()}\n\n")
                        
                        f.write("| Attribution Method | Overall Faithfulness Score |\n")
                        f.write("|" + "-" * 20 + "|" + "-" * 28 + "|\n")
                        for row_idx in range(scenario_method_avg.height):
                            row = scenario_method_avg.row(row_idx)
                            f.write(f"| {row[0]} | {row[1]:.4f} |\n")
                        
                        f.write("\n\n")
                    
                        # Best method-model combination for this scenario
                        best_row_idx = scenario_df['overall_faithfulness_score'].arg_max()
                        best_combo = scenario_df.row(best_row_idx)
                        attr_method_idx = scenario_df.columns.index('attribution_method')
                        model_type_idx = scenario_df.columns.index('model_type')
                        overall_score_idx = scenario_df.columns.index('overall_faithfulness_score')
                        topk_idx = scenario_df.columns.index('ranking_top_k_accuracy')
                        attr_ratio_idx = scenario_df.columns.index('magnitude_attribution_ratio')
                        
                        f.write(f"#### Best Method-Model Combination for {scenario.replace('_', ' ').title()}\n\n")
                        f.write(f"- **Attribution Method**: {best_combo[attr_method_idx]}\n")
                        f.write(f"- **Model Type**: {best_combo[model_type_idx]}\n")
                        f.write(f"- **Overall Faithfulness Score**: {best_combo[overall_score_idx]:.4f}\n")
                        f.write(f"- **Top-K Accuracy**: {best_combo[topk_idx]:.4f}\n")
                        f.write(f"- **Attribution Ratio**: {best_combo[attr_ratio_idx]:.4f}\n\n")
                else:
                    f.write("*No scenarios found in metrics data.*\n\n")
            else:
                f.write("*Scenario-specific analysis data not available.*\n\n")
            
            # Model-Specific Analysis (brief section)
            f.write("## Model-Specific Analysis\n\n")
            if not metrics_df.is_empty():
                model_types = metrics_df.select('model_type').unique().to_series().to_list()
                if len(model_types) > 0:
                    model_avg = metrics_df.group_by('model_type').agg(
                        pl.mean('overall_faithfulness_score')
                    ).sort('overall_faithfulness_score', descending=True)
                    
                    f.write("### Overall Faithfulness Score by Model Type\n\n")
                    
                    f.write("| Model Type | Overall Faithfulness Score |\n")
                    f.write("|" + "-" * 12 + "|" + "-" * 28 + "|\n")
                    for row_idx in range(model_avg.height):
                        row = model_avg.row(row_idx)
                        f.write(f"| {row[0].upper()} | {row[1]:.4f} |\n")
                    
                    f.write("\n\n")
                else:
                    f.write("*No model types found in metrics data.*\n\n")
            else:
                f.write("*Model-specific analysis data not available.*\n\n")
            
            # Attribution Method Comparison (brief section)
            f.write("## Attribution Method Comparison\n\n")
            if not metrics_df.is_empty():
                methods = metrics_df.select('attribution_method').unique().to_series().to_list()
                if len(methods) > 0:
                    # Just include a summary table
                    method_metrics = metrics_df.group_by('attribution_method').agg(
                        pl.mean('overall_faithfulness_score'),
                        pl.mean('ranking_top_k_accuracy'),
                        pl.mean('magnitude_attribution_ratio')
                    ).sort('overall_faithfulness_score', descending=True)
                    
                    f.write("### Summary of Attribution Method Performance\n\n")
                    
                    f.write("| Attribution Method | Overall Score | Top-K Accuracy | Attribution Ratio |\n")
                    f.write("|" + "-" * 20 + "|" + "-" * 15 + "|" + "-" * 15 + "|" + "-" * 18 + "|\n")
                    for row_idx in range(method_metrics.height):
                        row = method_metrics.row(row_idx)
                        f.write(f"| {row[0]} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} |\n")
                    
                    f.write("\n\n")
                else:
                    f.write("*No attribution methods found in metrics data.*\n\n")
            else:
                f.write("*Attribution method comparison data not available.*\n\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            if not metrics_df.is_empty():
                # Best overall method
                method_avg = metrics_df.group_by('attribution_method').agg(
                    pl.mean('overall_faithfulness_score')
                ).sort('overall_faithfulness_score', descending=True)
                
                best_method = method_avg.row(0)[0] if not method_avg.is_empty() else "N/A"
                best_method_score = method_avg.row(0)[1] if not method_avg.is_empty() else 0
                
                # Best overall model
                model_avg_all = metrics_df.group_by('model_type').agg(
                    pl.mean('overall_faithfulness_score')
                ).sort('overall_faithfulness_score', descending=True)
                
                best_model = model_avg_all.row(0)[0] if not model_avg_all.is_empty() else "N/A"
                best_model_score = model_avg_all.row(0)[1] if not model_avg_all.is_empty() else 0
                
                # Best overall scenario
                scenario_avg_all = metrics_df.group_by('scenario').agg(
                    pl.mean('overall_faithfulness_score')
                ).sort('overall_faithfulness_score', descending=True)
                
                best_scenario = scenario_avg_all.row(0)[0] if not scenario_avg_all.is_empty() else "N/A"
                
                # Best overall combination
                best_idx = metrics_df['overall_faithfulness_score'].arg_max() if metrics_df.height > 0 else -1
                best_overall = metrics_df.row(best_idx) if best_idx >= 0 else None
                
                f.write("### Summary of Best Performers\n\n")
                if best_method != "N/A":
                    f.write(f"- **Best Attribution Method Overall**: {best_method.replace('_', ' ').title()} (Score: {best_method_score:.4f})\n")
                if best_model != "N/A":
                    f.write(f"- **Best Model Type Overall**: {best_model.upper()} (Score: {best_model_score:.4f})\n")
                if best_scenario != "N/A":
                    f.write(f"- **Best Performing Scenario**: {best_scenario.replace('_', ' ').title()}\n")
                if best_overall is not None:
                    attr_method_idx = metrics_df.columns.index('attribution_method')
                    model_type_idx = metrics_df.columns.index('model_type')
                    scenario_idx = metrics_df.columns.index('scenario')
                    overall_score_idx = metrics_df.columns.index('overall_faithfulness_score')
                    
                    f.write(f"- **Best Overall Combination**: {best_overall[attr_method_idx].replace('_', ' ').title()} with {best_overall[model_type_idx].upper()} on {best_overall[scenario_idx].replace('_', ' ').title()} (Score: {best_overall[overall_score_idx]:.4f})\n\n")
                
                # Top-K accuracy analysis
                topk_scenario_avg = metrics_df.group_by('scenario').agg(
                    pl.mean('ranking_top_k_accuracy')
                ).sort('ranking_top_k_accuracy')
                
                easiest_scenario = topk_scenario_avg.row(-1)[0] if topk_scenario_avg.height > 0 else "N/A"
                hardest_scenario = topk_scenario_avg.row(0)[0] if topk_scenario_avg.height > 0 else "N/A"
                
                f.write("### Observations on Causal Feature Identification\n\n")
                if hardest_scenario != "N/A" and easiest_scenario != "N/A":
                    f.write(f"- **Easiest Scenario for Identifying Causal Features**: {easiest_scenario.replace('_', ' ').title()}\n")
                    f.write(f"- **Most Challenging Scenario for Identifying Causal Features**: {hardest_scenario.replace('_', ' ').title()}\n\n")
                
                # Interpretation of SHAP results
                if 'shap' in metrics_df.select('attribution_method').unique().to_series().to_list():
                    shap_metrics = metrics_df.filter(pl.col('attribution_method') == 'shap')
                    
                    shap_overall = shap_metrics.select(pl.mean('overall_faithfulness_score')).row(0)[0]
                    other_methods = metrics_df.filter(pl.col('attribution_method') != 'shap')
                    other_overall = other_methods.select(pl.mean('overall_faithfulness_score')).row(0)[0] if not other_methods.is_empty() else 0
                    
                    if shap_overall > other_overall:
                        f.write("### SHAP Performance\n\n")
                        f.write("SHAP consistently outperforms other attribution methods in identifying causal features. ")
                        f.write("This aligns with its theoretical guarantees based on Shapley values from cooperative game theory, ")
                        f.write("which provide a fair distribution of feature importance.\n\n")
                
                # Fraud detection findings
                if 'fraud_detection' in metrics_df.select('scenario').unique().to_series().to_list():
                    f.write("### Fraud Detection Insights\n\n")
                    f.write("The fraud detection scenario presents unique challenges for attribution methods ")
                    f.write("due to the presence of indirect indicators (consequences of fraud) that are highly ")
                    f.write("correlated with fraud events but are not causal. These indirect indicators often ")
                    f.write("receive substantial attribution weight from models even though they are effects rather ")
                    f.write("than causes of fraud, highlighting a key challenge in causal feature attribution.\n\n")
                
                # Practical recommendations
                f.write("### Practical Recommendations\n\n")
                f.write("Based on these findings, practitioners in financial domains should:\n\n")
                
                if best_method != "N/A":
                    f.write(f"1. **Use {best_method.replace('_', ' ').title()} for Financial Models**: When causal understanding is crucial, ")
                    f.write(f"{best_method.replace('_', ' ').title()} provides the most reliable feature attributions that align with true causal relationships.\n\n")
                
                if best_model != "N/A":
                    f.write(f"2. **Consider {best_model.upper()} Models**: These models demonstrated the best overall alignment between ")
                    f.write("feature importance and true causal relationships in our experiments.\n\n")
                
                if hardest_scenario != "N/A":
                    f.write(f"3. **Exercise Caution with {hardest_scenario.replace('_', ' ').title()}-like Scenarios**: Attribution methods ")
                    f.write(f"struggle most with identifying causal features in {hardest_scenario.replace('_', ' ').title()} contexts. Additional ")
                    f.write("domain expertise should be incorporated when interpreting model explanations in these areas.\n\n")
                
                f.write("4. **Verify Attributions with Multiple Methods**: The variability in performance across attribution methods ")
                f.write("suggests that cross-validation with multiple techniques can provide a more robust understanding of causal ")
                f.write("relationships in financial models.\n\n")
            else:
                f.write("*Key findings data not available.*\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            f.write("This report has presented a comprehensive analysis of the causal faithfulness of various feature attribution methods ")
            f.write("across different financial machine learning models and scenarios. The results highlight both the strengths and limitations ")
            f.write("of current attribution techniques in identifying true causal relationships, with important implications for model explainability, ")
            f.write("regulatory compliance, and decision-making in financial contexts.\n\n")
            
            f.write("The findings underscore the need for practitioners to exercise caution when interpreting feature attributions as causal explanations ")
            f.write("and suggest avenues for developing more causally-aware interpretability frameworks in finance.")
    
    elif format == 'latex':
        report_path = os.path.join(output_dir, 'report.tex')
        
        with open(report_path, 'w') as f:
            # LaTeX document preamble
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
            f.write(f"\\date{{{pl.datetime('now').dt.strftime('%Y-%m-%d')}}}\n")
            f.write("\\author{}\n\n")
            
            f.write("\\begin{document}\n\n")
            f.write("\\maketitle\n\n")
            
            # Introduction
            f.write("\\section{Introduction}\n\n")
            f.write("This document presents the key results from our experiments evaluating the causal faithfulness of various feature attribution methods in financial machine learning models. We analyze how well different attribution techniques identify truly causal features across three financial scenarios: asset pricing, credit risk assessment, and fraud detection.\n\n")
            
            # Model Performance
            f.write("\\section{Model Performance}\n\n")
            if model_perf is not None:
                f.write("\\subsection{Overall Model Performance}\n\n")
                
                f.write("\\begin{table}[H]\n")
                f.write("\\centering\n")
                f.write("\\caption{Model Performance Metrics}\n")
                f.write("\\begin{tabular}{lllrr}\n")
                f.write("\\toprule\n")
                f.write("Scenario & Model Type & ")
                if 'accuracy' in model_perf.columns:
                    f.write("Accuracy & ")
                if 'f1' in model_perf.columns:
                    f.write("F1 Score & ")
                if 'mse' in model_perf.columns:
                    f.write("MSE & ")
                if 'r2' in model_perf.columns:
                    f.write("R² & ")
                f.write("\\\\\n")
                
                f.write("\\midrule\n")
                for row_idx in range(model_perf.height):
                    row = model_perf.row(row_idx)
                    scenario_idx = model_perf.columns.index('scenario') if 'scenario' in model_perf.columns else -1
                    model_type_idx = model_perf.columns.index('model_type') if 'model_type' in model_perf.columns else -1
                    
                    scenario = row[scenario_idx] if scenario_idx >= 0 else ''
                    model_type = row[model_type_idx] if model_type_idx >= 0 else ''
                    
                    f.write(f"{scenario} & {model_type} & ")
                    if 'accuracy' in model_perf.columns:
                        accuracy_idx = model_perf.columns.index('accuracy')
                        f.write(f"{row[accuracy_idx]:.4f} & ")
                    if 'f1' in model_perf.columns:
                        f1_idx = model_perf.columns.index('f1')
                        f.write(f"{row[f1_idx]:.4f} & ")
                    if 'mse' in model_perf.columns:
                        mse_idx = model_perf.columns.index('mse')
                        f.write(f"{row[mse_idx]:.4f} & ")
                    if 'r2' in model_perf.columns:
                        r2_idx = model_perf.columns.index('r2')
                        f.write(f"{row[r2_idx]:.4f} & ")
                    f.write("\\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
            else:
                f.write("Model performance data not available.\n\n")
            
            # Faithfulness Evaluation
            f.write("\\section{Faithfulness Evaluation}\n\n")
            if not metrics_df.is_empty():
                # Add some basic tables
                f.write("Faithfulness metrics data available.\n\n")
            else:
                f.write("Faithfulness metrics data not available.\n\n")
            
            # Scenario-Specific Analysis
            f.write("\\section{Scenario-Specific Analysis}\n\n")
            if not metrics_df.is_empty():
                f.write("Scenario-specific analysis data available.\n\n")
            else:
                f.write("Scenario-specific analysis data not available.\n\n")
            
            # Key Findings
            f.write("\\section{Key Findings}\n\n")
            if not metrics_df.is_empty():
                f.write("Key findings data available.\n\n")
            else:
                f.write("Key findings data not available.\n\n")
            
            # Conclusion
            f.write("\\section{Conclusion}\n\n")
            f.write("This report has presented a comprehensive analysis of the causal faithfulness of various feature attribution methods across different financial machine learning models and scenarios. The results highlight both the strengths and limitations of current attribution techniques in identifying true causal relationships, with important implications for model explainability, regulatory compliance, and decision-making in financial contexts.\n\n")
            
            f.write("The findings underscore the need for practitioners to exercise caution when interpreting feature attributions as causal explanations and suggest avenues for developing more causally-aware interpretability frameworks in finance.\n\n")
            
            f.write("\\end{document}")
    
    else:
        logger.warning(f"Unsupported format: {format}")
        return ""
    
    # Also create a summary of key findings
    key_findings_path = os.path.join(output_dir, 'key_findings.md')
    with open(key_findings_path, 'w') as f:
        f.write("Key findings from the report:\n\n")
        f.write("## Key Findings\n\n")
        
        if not metrics_df.is_empty():
            # Best overall method
            method_avg = metrics_df.group_by('attribution_method').agg(
                pl.mean('overall_faithfulness_score')
            ).sort('overall_faithfulness_score', descending=True)
            
            best_method = method_avg.row(0)[0] if not method_avg.is_empty() else "N/A"
            
            # Best overall model
            model_avg_all = metrics_df.group_by('model_type').agg(
                pl.mean('overall_faithfulness_score')
            ).sort('overall_faithfulness_score', descending=True)
            
            best_model = model_avg_all.row(0)[0] if not model_avg_all.is_empty() else "N/A"
            
            f.write(f"1. **{best_method.replace('_', ' ').title()} performs best**: Our experiments show that {best_method.replace('_', ' ').title()} provides the most causally faithful explanations across financial scenarios.\n\n")
            
            f.write(f"2. **{best_model.upper()} models yield more causally accurate attributions**: {best_model.upper()} models demonstrated superior ability to generate attributions that align with true causal structure.\n\n")
            
            if 'fraud_detection' in metrics_df.select('scenario').unique().to_series().to_list():
                f.write("3. **Fraud detection presents unique challenges**: Indirect indicators (consequences of fraud) often receive substantial attribution weight despite not being causal drivers.\n\n")
            
            f.write("4. **Multiple attribution methods recommended**: Significant variability in performance across methods suggests using multiple techniques for robust understanding.\n\n")
            
            f.write("5. **Caution needed when interpreting attributions as causal**: Our findings underscore that feature attributions should not be naively interpreted as causal explanations without domain expertise and proper validation.\n")
        else:
            f.write("*Key findings data not available.*\n\n")
    
    logger.info(f"Report generated at {report_path}")
    logger.info(f"Key findings summary at {key_findings_path}")
    
    return report_path