import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from typing import Tuple, Dict, List, Optional, Union, Any

class SimpleMLPModel:
    """Simple Multi-Layer Perceptron neural network."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                output_dim: int = 1, task_type: str = 'regression',
                random_state: int = 42):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for binary classification or regression)
            task_type: 'regression' or 'classification'
            random_state: Random seed
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.task_type = task_type
        self.random_state = random_state
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Build network
        self.model = self._build_network()
        
        # Initialize optimizer
        self.optimizer = None
        self.criterion = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        
    def _build_network(self) -> nn.Module:
        """Build MLP network architecture."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        
        # Add sigmoid for binary classification
        if self.task_type == 'classification':
            layers.append(nn.Sigmoid())
            
        return nn.Sequential(*layers)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
          batch_size: int = 32, epochs: int = 100, 
          lr: float = 0.001, val_split: float = 0.2,
          verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the MLP model.
        
        Args:
            X: Input features
            y: Target variable
            batch_size: Batch size for training
            epochs: Number of training epochs
            lr: Learning rate
            val_split: Validation split ratio
            verbose: Whether to print training progress
            
        Returns:
            Dict with training and validation losses
        """
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=val_split, random_state=self.random_state
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1))
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Set optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        if self.task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            # Training
            epoch_loss = 0.0
            num_batches = 0
            
            for inputs, targets in train_loader:
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                history['val_loss'].append(val_loss)
            self.model.train()
            
            # Print progress
            if verbose and (epoch + 1) % (epochs // 10 or 1) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        # Scale data
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        
        # For classification, convert probabilities to binary labels
        if self.task_type == 'classification':
            predictions = (predictions > 0.5).astype(int)
            
        return predictions.reshape(-1)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Dict with evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {}
        if self.task_type == 'regression':
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y, y_pred)
        else:
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['f1'] = f1_score(y, y_pred)
            
        return metrics
    
    def save(self, path: str) -> None:
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'task_type': self.task_type,
            'random_state': self.random_state
        }, path)
    
    def load(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(path)
        
        # Recreate model architecture
        self.input_dim = checkpoint['input_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.output_dim = checkpoint['output_dim']
        self.task_type = checkpoint['task_type']
        self.random_state = checkpoint['random_state']
        self.model = self._build_network()
        
        # Load model weights and other components
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        
        if checkpoint['optimizer_state_dict']:
            self.optimizer = optim.Adam(self.model.parameters())
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class TimeSeriesLSTMModel:
    """LSTM model for time series financial data."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                output_dim: int = 1, task_type: str = 'regression',
                random_state: int = 42):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Number of features per time step
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            output_dim: Output dimension
            task_type: 'regression' or 'classification'
            random_state: Random seed
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.task_type = task_type
        self.random_state = random_state
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Build network
        self.model = self._build_network()
        
        # Initialize optimizer
        self.optimizer = None
        self.criterion = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        
    def _build_network(self) -> nn.Module:
        """Build LSTM network."""
        class LSTMNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim, task_type):
                super(LSTMNetwork, self).__init__()
                
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.task_type = task_type
                
                # LSTM layers
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                
                # Output layer
                self.fc = nn.Linear(hidden_dim, output_dim)
                
                # Activation for classification
                if task_type == 'classification':
                    self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                # Initialize hidden state
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                
                # Forward propagate LSTM
                lstm_out, _ = self.lstm(x, (h0, c0))
                
                # Get the last time step output
                out = self.fc(lstm_out[:, -1, :])
                
                # Apply sigmoid for classification
                if self.task_type == 'classification':
                    out = self.sigmoid(out)
                    
                return out
            
        return LSTMNetwork(self.input_dim, self.hidden_dim, self.num_layers, 
                         self.output_dim, self.task_type)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
          batch_size: int = 32, epochs: int = 100,
          sequence_length: int = 10, 
          lr: float = 0.001, val_split: float = 0.2,
          verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the LSTM model.
        
        Args:
            X: Input features (samples, features)
            y: Target variable (samples,)
            batch_size: Batch size for training
            epochs: Number of training epochs
            sequence_length: Length of input sequences
            lr: Learning rate
            val_split: Validation split ratio
            verbose: Whether to print training progress
            
        Returns:
            Dict with training and validation losses
        """
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y, sequence_length)
        
        # Split data
        train_size = int((1 - val_split) * len(X_seq))
        X_train, X_val = X_seq[:train_size], X_seq[train_size:]
        y_train, y_val = y_seq[:train_size], y_seq[train_size:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1))
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Set optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        if self.task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            # Training
            epoch_loss = 0.0
            num_batches = 0
            
            for inputs, targets in train_loader:
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                history['val_loss'].append(val_loss)
            self.model.train()
            
            # Print progress
            if verbose and (epoch + 1) % (epochs // 10 or 1) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Store sequence length for prediction
        self.sequence_length = sequence_length
        
        return history
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            X: Input features
            y: Target variable
            sequence_length: Length of input sequences
            
        Returns:
            X_seq: Sequence inputs
            y_seq: Sequence targets
        """
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
            
        return np.array(X_seq), np.array(y_seq)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained LSTM model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        # Scale data
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:i + self.sequence_length])
        
        X_seq = np.array(X_seq)
        X_tensor = torch.FloatTensor(X_seq)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        
        # For classification, convert probabilities to binary labels
        if self.task_type == 'classification':
            predictions = (predictions > 0.5).astype(int)
            
        # Pad predictions with NaNs for the sequence_length points at the beginning
        full_preds = np.full(len(X), np.nan)
        full_preds[self.sequence_length:] = predictions.reshape(-1)
            
        return full_preds
    
    def save(self, path: str) -> None:
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'task_type': self.task_type,
            'sequence_length': getattr(self, 'sequence_length', 10),
            'random_state': self.random_state
        }, path)
    
    def load(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(path)
        
        # Recreate model architecture
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.output_dim = checkpoint['output_dim']
        self.task_type = checkpoint['task_type']
        self.random_state = checkpoint['random_state']
        self.sequence_length = checkpoint['sequence_length']
        self.model = self._build_network()
        
        # Load model weights and other components
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        
        if checkpoint['optimizer_state_dict']:
            self.optimizer = optim.Adam(self.model.parameters())
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Dict with evaluation metrics
        """
        y_pred = self.predict(X)
        
        # Remove NaN predictions (from sequence padding)
        mask = ~np.isnan(y_pred)
        y_pred = y_pred[mask]
        y_true = y[mask]
        
        metrics = {}
        if self.task_type == 'regression':
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
        else:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred)
            
        return metrics


class GradientBoostingWrapper:
    """Wrapper for XGBoost model with consistent API."""
    
    def __init__(self, task_type: str = 'regression', random_state: int = 42, **params):
        """
        Initialize XGBoost wrapper.
        
        Args:
            task_type: 'regression' or 'classification'
            random_state: Random seed
            **params: Additional parameters for XGBoost
        """
        self.task_type = task_type
        self.random_state = random_state
        self.params = params
        
        # Set default parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }
        
        # Update with user-specified parameters
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
        
        # Initialize model
        if task_type == 'regression':
            self.model = xgb.XGBRegressor(**self.params)
        else:
            self.model = xgb.XGBClassifier(**self.params)
        
        # Preprocessing
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
          val_split: float = 0.2, 
          verbose: bool = True) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X: Input features
            y: Target variable
            val_split: Validation split ratio
            verbose: Whether to print training progress
            
        Returns:
            Dict with training information
        """
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=val_split, random_state=self.random_state
        )
        
        # Train model
        eval_set = [(X_val, y_val)]
        
        if verbose:
            self.model.fit(X_train, y_train, eval_set=eval_set, eval_metric='rmse' if self.task_type == 'regression' else 'logloss',
                         verbose=True)
        else:
            self.model.fit(X_train, y_train)
        
        # Store feature names for later use with SHAP
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        return {'model': self.model}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained XGBoost model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        # Scale data
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if self.task_type == 'regression':
            predictions = self.model.predict(X_scaled)
        else:
            # Get probabilities for classification
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            # Convert to binary labels
            predictions = (probabilities > 0.5).astype(int)
            
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Dict with evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {}
        if self.task_type == 'regression':
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y, y_pred)
        else:
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['f1'] = f1_score(y, y_pred)
            
        return metrics
    
    def save(self, path: str) -> None:
        """Save the model."""
        import joblib
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'task_type': self.task_type,
            'random_state': self.random_state,
            'params': self.params,
            'feature_names': getattr(self, 'feature_names', None)
        }, path)
    
    def load(self, path: str) -> None:
        """Load the model."""
        import joblib
        checkpoint = joblib.load(path)
        
        self.model = checkpoint['model']
        self.scaler = checkpoint['scaler']
        self.task_type = checkpoint['task_type']
        self.random_state = checkpoint['random_state']
        self.params = checkpoint['params']
        self.feature_names = checkpoint['feature_names']


def train_model_for_scenario(scenario_name: str, model_type: str = 'mlp', 
                           data_dir: str = './data', 
                           model_dir: str = './models',
                           verbose: bool = True) -> Dict[str, Any]:
    """
    Train a model for a specific financial scenario.
    
    Args:
        scenario_name: Name of scenario ('asset_pricing', 'credit_risk', 'fraud_detection')
        model_type: Type of model ('mlp', 'lstm', 'xgboost')
        data_dir: Directory containing data
        model_dir: Directory to save models
        verbose: Whether to print training progress
        
    Returns:
        Dict with model and evaluation metrics
    """
    import os
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    X = pd.read_csv(os.path.join(data_dir, f'{scenario_name}_features.csv')).values
    y = pd.read_csv(os.path.join(data_dir, f'{scenario_name}_target.csv')).iloc[:, 0].values
    
    # Determine task type
    task_type = 'regression' if scenario_name == 'asset_pricing' else 'classification'
    
    # Initialize model
    if model_type == 'mlp':
        model = SimpleMLPModel(
            input_dim=X.shape[1],
            hidden_dims=[64, 32],
            output_dim=1,
            task_type=task_type
        )
        
        # Train model
        history = model.fit(
            X=X,
            y=y,
            batch_size=64,
            epochs=100,
            lr=0.001,
            val_split=0.2,
            verbose=verbose
        )
        
    elif model_type == 'lstm':
        model = TimeSeriesLSTMModel(
            input_dim=X.shape[1],
            hidden_dim=64,
            num_layers=2,
            output_dim=1,
            task_type=task_type
        )
        
        # Train model
        history = model.fit(
            X=X,
            y=y,
            batch_size=64,
            epochs=100,
            sequence_length=10,
            lr=0.001,
            val_split=0.2,
            verbose=verbose
        )
        
    elif model_type == 'xgboost':
        model = GradientBoostingWrapper(
            task_type=task_type,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        
        # Train model
        history = model.fit(
            X=X,
            y=y,
            val_split=0.2,
            verbose=verbose
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Evaluate model
    metrics = model.evaluate(X, y)
    if verbose:
        print(f"Model evaluation on {scenario_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save model
    model_path = os.path.join(model_dir, f"{scenario_name}_{model_type}_model.pkl")
    model.save(model_path)
    
    if verbose:
        print(f"Model saved to {model_path}")
    
    return {
        'model': model,
        'metrics': metrics,
        'history': history
    }


if __name__ == "__main__":
    # Example usage
    result = train_model_for_scenario(
        scenario_name='asset_pricing',
        model_type='mlp',
        verbose=True
    )
    
    print("Training complete!")