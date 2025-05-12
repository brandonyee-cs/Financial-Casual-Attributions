import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import grad
import xgboost as xgb
import shap
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

class AttributionMethod:
    """Base class for attribution methods."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """
        Initialize the attribution method.
        
        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
    
    def attribute(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                **kwargs) -> np.ndarray:
        """
        Compute attributions for model predictions.
        
        Args:
            X: Input features
            y: Target labels (optional)
            **kwargs: Additional arguments
            
        Returns:
            Attributions for each feature
        """
        raise NotImplementedError
    
    def normalize_attributions(self, attributions: np.ndarray) -> np.ndarray:
        """
        Normalize attributions.
        
        Args:
            attributions: Computed attributions
            
        Returns:
            Normalized attributions
        """
        # Compute absolute attributions
        abs_attr = np.abs(attributions)
        
        # Normalize by dividing by sum (if non-zero)
        row_sums = abs_attr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        
        return attributions / row_sums

    def get_attributions_df(self, X: np.ndarray, attributions: np.ndarray) -> pd.DataFrame:
        """
        Convert attributions to DataFrame format.
        
        Args:
            X: Input features
            attributions: Computed attributions
            
        Returns:
            DataFrame with attributions
        """
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            feature_names = self.feature_names
            
        return pd.DataFrame(attributions, columns=feature_names)


class SaliencyMap(AttributionMethod):
    """
    Simple gradient-based attribution method.
    
    Computes attributions as gradients of model output with respect to inputs.
    """
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize saliency map."""
        super().__init__(model, feature_names)
        
        # Check if model is a PyTorch model
        if not isinstance(self.model.model, nn.Module):
            raise ValueError("Saliency map is only implemented for PyTorch models")
    
    def attribute(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                **kwargs) -> np.ndarray:
        """
        Compute saliency map attributions.
        
        Args:
            X: Input features
            y: Target labels (not used for saliency)
            
        Returns:
            Attribution scores
        """
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
    """
    Gradient * Input attribution method.
    
    Computes attributions as gradients of model output with respect to inputs,
    multiplied by the input values.
    """
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize gradient * input method."""
        super().__init__(model, feature_names)
        
        # Check if model is a PyTorch model
        if not isinstance(self.model.model, nn.Module):
            raise ValueError("Gradient * Input is only implemented for PyTorch models")
    
    def attribute(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                **kwargs) -> np.ndarray:
        """
        Compute gradient * input attributions.
        
        Args:
            X: Input features
            y: Target labels (not used)
            
        Returns:
            Attribution scores
        """
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
    """
    Integrated Gradients attribution method.
    
    Approximates the integral of gradients along a straight line from a baseline
    to the input.
    """
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize integrated gradients method."""
        super().__init__(model, feature_names)
        
        # Check if model is a PyTorch model
        if not isinstance(self.model.model, nn.Module):
            raise ValueError("Integrated Gradients is only implemented for PyTorch models")
    
    def attribute(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                steps: int = 50, baseline: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Compute integrated gradients attributions.
        
        Args:
            X: Input features
            y: Target labels (not used)
            steps: Number of steps for integral approximation
            baseline: Baseline input (default: zero)
            
        Returns:
            Attribution scores
        """
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
    """
    SHAP (SHapley Additive exPlanations) attribution method.
    
    Computes attributions using Shapley values, which fairly distribute the model's
    output among the input features.
    """
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize SHAP method."""
        super().__init__(model, feature_names)
        self.explainer = None
    
    def _create_explainer(self, X: np.ndarray) -> None:
        """
        Create SHAP explainer based on model type.
        
        Args:
            X: Sample of input features
        """
        # Scale inputs
        X_scaled = self.model.scaler.transform(X)
        
        # Check model type and create appropriate explainer
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'get_booster', False):
            # For XGBoost models, use TreeExplainer
            self.explainer = shap.TreeExplainer(self.model.model)
        else:
            # For PyTorch models, create a prediction function
            def predict_fn(x):
                x_tensor = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    return self.model.model(x_tensor).numpy()
            
            # Use KernelExplainer for black-box models
            background = shap.kmeans(X_scaled, 10)
            self.explainer = shap.KernelExplainer(predict_fn, background)
    
    def attribute(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                **kwargs) -> np.ndarray:
        """
        Compute SHAP values.
        
        Args:
            X: Input features
            y: Target labels (not used)
            
        Returns:
            SHAP values
        """
        # Create explainer if not created yet
        if self.explainer is None:
            self._create_explainer(X)
        
        # Scale inputs
        X_scaled = self.model.scaler.transform(X)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_scaled)
        
        # Convert to numpy array if needed
        if isinstance(shap_values, list):
            if len(shap_values) == 1:  # For regression or binary classification (single output)
                shap_values = shap_values[0]
            else:  # For multi-class (not supported in this implementation)
                raise ValueError("Multi-class SHAP values not supported")
        
        return shap_values


def compute_attributions(model: Any, X: np.ndarray, method_name: str,
                       feature_names: Optional[List[str]] = None,
                       **kwargs) -> pd.DataFrame:
    """
    Compute attributions for a given model and method.
    
    Args:
        model: Trained model
        X: Input features
        method_name: Name of attribution method
        feature_names: List of feature names
        **kwargs: Additional arguments for the attribution method
        
    Returns:
        DataFrame with attribution scores
    """
    # Create attribution method
    if method_name == 'saliency':
        method = SaliencyMap(model, feature_names)
    elif method_name == 'gradient_input':
        method = GradientInputMethod(model, feature_names)
    elif method_name == 'integrated_gradients':
        method = IntegratedGradientsMethod(model, feature_names)
    elif method_name == 'shap':
        method = ShapleyValueMethod(model, feature_names)
    else:
        raise ValueError(f"Unsupported attribution method: {method_name}")
    
    # Compute attributions
    attributions = method.attribute(X, **kwargs)
    
    # Convert to DataFrame
    attributions_df = method.get_attributions_df(X, attributions)
    
    return attributions_df


if __name__ == "__main__":
    # Example usage
    from models import SimpleMLPModel
    
    # Create a simple model
    model = SimpleMLPModel(input_dim=5, hidden_dims=[10], output_dim=1)
    
    # Create random data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    # Train model
    model.fit(X, y, epochs=10, verbose=True)
    
    # Compute attributions
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    
    # Saliency map
    saliency_attrs = compute_attributions(
        model=model,
        X=X[:5],
        method_name='saliency',
        feature_names=feature_names
    )
    print("Saliency Map Attributions:")
    print(saliency_attrs)
    
    # Gradient * Input
    grad_input_attrs = compute_attributions(
        model=model,
        X=X[:5],
        method_name='gradient_input',
        feature_names=feature_names
    )
    print("\nGradient * Input Attributions:")
    print(grad_input_attrs)
    
    # Integrated Gradients
    ig_attrs = compute_attributions(
        model=model,
        X=X[:5],
        method_name='integrated_gradients',
        feature_names=feature_names,
        steps=10
    )
    print("\nIntegrated Gradients Attributions:")
    print(ig_attrs)
    
    # SHAP (may take longer)
    print("\nComputing SHAP values (this may take a moment)...")
    shap_attrs = compute_attributions(
        model=model,
        X=X[:5],
        method_name='shap',
        feature_names=feature_names
    )
    print("SHAP Attributions:")
    print(shap_attrs)