"""
Module for generating synthetic financial data with known causal ground truths.

This module implements scenarios for asset pricing, credit risk, and fraud detection,
each with a specific causal structure.
"""

import numpy as np
import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional

#pl.Config.set_engine_affinity(engine="streaming")

class CausalScenario:
    """Base class for causal scenarios in finance."""
    
    def __init__(self, n_samples: int = 10000, random_state: int = 42):
        """
        Initialize the causal scenario.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.feature_names = []
        self.dag = nx.DiGraph()
        
    def _generate_confounders(self) -> np.ndarray:
        """Generate confounding variables."""
        raise NotImplementedError
        
    def _generate_causal_features(self, confounders: np.ndarray) -> np.ndarray:
        """Generate causal features based on confounders."""
        raise NotImplementedError
        
    def _generate_spurious_features(self, confounders: np.ndarray) -> np.ndarray:
        """Generate spurious features based on confounders."""
        raise NotImplementedError
        
    def _generate_target(self, causal_features: np.ndarray, 
                         confounders: np.ndarray) -> np.ndarray:
        """Generate target variable based on causal features and confounders."""
        raise NotImplementedError
        
    def generate_data(self) -> Tuple[pl.DataFrame, pl.Series, Dict]:
        """
        Generate synthetic data for the causal scenario.
        
        Returns:
            X: Feature DataFrame
            y: Target Series
            causal_info: Dictionary with causal information
        """
        # Generate data according to the causal structure
        confounders = self._generate_confounders()
        causal_features = self._generate_causal_features(confounders)
        spurious_features = self._generate_spurious_features(confounders)
        
        # Combine features
        X = np.concatenate([causal_features, spurious_features], axis=1)
        
        # Generate target
        y = self._generate_target(causal_features, confounders)
        
        # Create DataFrame and Series
        X_df = pl.DataFrame(X, schema=self.feature_names)
        
        # Child classes might override target name
        target_name = 'target'
        y_series = pl.Series(name=target_name, values=y)
        
        # Create causal info dictionary
        # Note: Child classes might override y_series.name and the content of causal_info
        causal_info = {
            'causal_features': [name for i, name in enumerate(self.feature_names) 
                               if i < causal_features.shape[1]],
            'spurious_features': [name for i, name in enumerate(self.feature_names) 
                                 if i >= causal_features.shape[1]], # This will be 'indirect_features' for Fraud
            'dag': self.dag
        }
        
        return X_df, y_series, causal_info
    
    def plot_dag(self) -> None:
        """Plot the causal DAG."""
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.dag, seed=self.random_state)
        nx.draw(self.dag, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, arrowsize=20, font_size=12)
        plt.title(f"Causal DAG for {self.__class__.__name__}")
        plt.show()


class AssetPricingScenario(CausalScenario):
    """
    Asset Pricing Scenario:
    
    Y: Future stock return (target)
    X_fundamental: True causal factor (e.g., earnings surprise)
    Z_confounder: Common driver (e.g., market sentiment)
    X_spurious: Feature correlated with Y but no causal link
    """
    
    def __init__(self, n_samples: int = 10000, random_state: int = 42,
                n_causal: int = 2, n_spurious: int = 3, n_confounders: int = 2,
                noise_level: float = 0.2):
        """
        Initialize the asset pricing scenario.
        
        Args:
            n_samples: Number of samples
            random_state: Random seed
            n_causal: Number of causal features
            n_spurious: Number of spurious features
            n_confounders: Number of confounding variables
            noise_level: Level of noise in the data
        """
        super().__init__(n_samples, random_state)
        self.n_causal = n_causal
        self.n_spurious = n_spurious
        self.n_confounders = n_confounders
        self.noise_level = noise_level
        
        # Define feature names
        self.feature_names = (
            [f'fundamental_{i}' for i in range(n_causal)] +
            [f'spurious_{i}' for i in range(n_spurious)]
        )
        
        # Setup DAG
        self._setup_dag()
        
    def _setup_dag(self) -> None:
        """Setup the causal DAG."""
        # Add nodes
        for i in range(self.n_confounders):
            self.dag.add_node(f'Z{i}')
        
        for i in range(self.n_causal):
            self.dag.add_node(f'fundamental_{i}')
            
        for i in range(self.n_spurious):
            self.dag.add_node(f'spurious_{i}')
            
        self.dag.add_node('return')
        
        # Add edges
        for z_idx in range(self.n_confounders):
            # Confounders affect causal features
            for i in range(self.n_causal):
                self.dag.add_edge(f'Z{z_idx}', f'fundamental_{i}')
            
            # Confounders affect spurious features
            for i in range(self.n_spurious):
                self.dag.add_edge(f'Z{z_idx}', f'spurious_{i}')
            
            # Confounders affect return directly
            self.dag.add_edge(f'Z{z_idx}', 'return')
        
        # Causal features affect return
        for i in range(self.n_causal):
            self.dag.add_edge(f'fundamental_{i}', 'return')
    
    def _generate_confounders(self) -> np.ndarray:
        """Generate market confounders like sentiment, sector trends, etc."""
        return self.rng.normal(0, 1, size=(self.n_samples, self.n_confounders))
    
    def _generate_causal_features(self, confounders: np.ndarray) -> np.ndarray:
        """Generate fundamental features that cause returns."""
        causal_features = np.zeros((self.n_samples, self.n_causal))
        
        for i in range(self.n_causal):
            # Base fundamental value (earnings surprise, etc.)
            base = self.rng.normal(0, 1, size=self.n_samples)
            
            # Add confounder effect
            confounder_effect = 0
            if self.n_confounders > 0:
                confounder_weights = self.rng.uniform(0.3, 0.7, size=self.n_confounders)
                confounder_effect = np.sum(confounders * confounder_weights, axis=1)
            
            # Combine with some noise
            causal_features[:, i] = base + confounder_effect + self.rng.normal(0, self.noise_level, 
                                                                             size=self.n_samples)
        return causal_features
    
    def _generate_spurious_features(self, confounders: np.ndarray) -> np.ndarray:
        """Generate spurious features (correlated but not causal)."""
        spurious_features = np.zeros((self.n_samples, self.n_spurious))
        
        for i in range(self.n_spurious):
            # Base spurious value (technical indicators, etc.)
            base = self.rng.normal(0, 1, size=self.n_samples)
            
            # Strong effect from confounders (that's why they correlate with return)
            confounder_effect = 0
            if self.n_confounders > 0:
                confounder_weights = self.rng.uniform(0.5, 0.9, size=self.n_confounders)
                confounder_effect = np.sum(confounders * confounder_weights, axis=1)
            
            # Combine with some noise
            spurious_features[:, i] = base + confounder_effect + self.rng.normal(0, self.noise_level, 
                                                                              size=self.n_samples)
        return spurious_features
    
    def _generate_target(self, causal_features: np.ndarray, 
                        confounders: np.ndarray) -> np.ndarray:
        """Generate stock returns."""
        # Base return from causal features
        causal_effect = 0
        if self.n_causal > 0:
            causal_weights = self.rng.uniform(0.5, 1.0, size=self.n_causal)
            causal_effect = np.sum(causal_features * causal_weights, axis=1)
        
        # Direct confounder effect on returns
        confounder_effect_on_target = 0
        if self.n_confounders > 0:
            confounder_target_weights = self.rng.uniform(0.4, 0.8, size=self.n_confounders)
            confounder_effect_on_target = np.sum(confounders * confounder_target_weights, axis=1)
        
        # Combine with some noise to get final return
        returns = causal_effect + confounder_effect_on_target + self.rng.normal(0, self.noise_level, 
                                                                                size=self.n_samples)
        return returns

    def generate_data(self) -> Tuple[pl.DataFrame, pl.Series, Dict]:
        # Generate data with explicit confounders
        confounders = self._generate_confounders()
        causal_features_arr = self._generate_causal_features(confounders)
        spurious_features_arr = self._generate_spurious_features(confounders)
        X_arr = np.concatenate([causal_features_arr, spurious_features_arr], axis=1)
        y_arr = self._generate_target(causal_features_arr, confounders)

        X_df = pl.DataFrame(X_arr, schema=self.feature_names)
        y_series = pl.Series(name='return', values=y_arr)

        causal_info = {
            'causal_features': [name for i, name in enumerate(self.feature_names) 
                               if i < self.n_causal],
            'spurious_features': [name for i, name in enumerate(self.feature_names) 
                                 if i >= self.n_causal],
            'dag': self.dag
        }
        return X_df, y_series, causal_info


class CreditRiskScenario(CausalScenario):
    """
    Credit Risk Scenario:
    
    Y: Loan default probability (target)
    X_causal_behavioral: True causal factors (e.g., missed payments)
    Z_confounder_socioeconomic: Factors like unemployment rate
    X_proxy_bias: Features correlated with protected attributes (spurious features)
    """
    
    def __init__(self, n_samples: int = 10000, random_state: int = 42,
                n_causal: int = 2, n_proxy: int = 2, n_confounders: int = 2,
                noise_level: float = 0.2):
        """Initialize the credit risk scenario."""
        super().__init__(n_samples, random_state)
        self.n_causal = n_causal
        self.n_proxy = n_proxy # n_spurious in base class context
        self.n_confounders = n_confounders
        self.noise_level = noise_level
        
        # Define feature names
        self.feature_names = (
            [f'behavioral_{i}' for i in range(n_causal)] +
            [f'proxy_{i}' for i in range(n_proxy)]
        )
        
        # Setup DAG
        self._setup_dag()
        
    def _setup_dag(self) -> None:
        """Setup the causal DAG."""
        # Add nodes
        for i in range(self.n_confounders):
            self.dag.add_node(f'Z{i}')
        
        for i in range(self.n_causal):
            self.dag.add_node(f'behavioral_{i}')
            
        for i in range(self.n_proxy):
            self.dag.add_node(f'proxy_{i}')
            
        self.dag.add_node('default')
        
        # Add edges
        for z_idx in range(self.n_confounders):
            # Confounders affect behavioral features
            for i in range(self.n_causal):
                self.dag.add_edge(f'Z{z_idx}', f'behavioral_{i}')
            
            # Confounders affect proxy features
            for i in range(self.n_proxy):
                self.dag.add_edge(f'Z{z_idx}', f'proxy_{i}')
            
            # Confounders affect default directly
            self.dag.add_edge(f'Z{z_idx}', 'default')
        
        # Behavioral features affect default
        for i in range(self.n_causal):
            self.dag.add_edge(f'behavioral_{i}', 'default')
    
    def _generate_confounders(self) -> np.ndarray:
        """Generate socioeconomic confounders."""
        return self.rng.normal(0, 1, size=(self.n_samples, self.n_confounders))
    
    def _generate_causal_features(self, confounders: np.ndarray) -> np.ndarray:
        """Generate behavioral features that cause default."""
        causal_features = np.zeros((self.n_samples, self.n_causal))
        
        for i in range(self.n_causal):
            # Base behavioral value (e.g., debt-to-income ratio)
            base = self.rng.normal(0, 1, size=self.n_samples)
            
            # Add confounder effect (socioeconomic factors affect behavior)
            confounder_effect = 0
            if self.n_confounders > 0:
                confounder_weights = self.rng.uniform(0.3, 0.7, size=self.n_confounders)
                confounder_effect = np.sum(confounders * confounder_weights, axis=1)
            
            # Combine with some noise
            causal_features[:, i] = base + confounder_effect + self.rng.normal(0, self.noise_level, 
                                                                             size=self.n_samples)
        return causal_features
    
    def _generate_spurious_features(self, confounders: np.ndarray) -> np.ndarray:
        """Generate proxy features (e.g., zip code) influenced by confounders."""
        proxy_features = np.zeros((self.n_samples, self.n_proxy))
        
        for i in range(self.n_proxy):
            # Base proxy value
            base = self.rng.normal(0, 1, size=self.n_samples)
            
            # Strong effect from confounders
            confounder_effect = 0
            if self.n_confounders > 0:
                confounder_weights = self.rng.uniform(0.6, 0.9, size=self.n_confounders)
                confounder_effect = np.sum(confounders * confounder_weights, axis=1)
            
            # Combine with some noise
            proxy_features[:, i] = base + confounder_effect + self.rng.normal(0, self.noise_level, 
                                                                           size=self.n_samples)
        return proxy_features
    
    def _generate_target(self, causal_features: np.ndarray, 
                        confounders: np.ndarray) -> np.ndarray:
        """Generate loan default probability."""
        # Effect from behavioral features
        behavioral_effect = 0
        if self.n_causal > 0:
            behavioral_weights = self.rng.uniform(0.6, 1.2, size=self.n_causal)
            behavioral_effect = np.sum(causal_features * behavioral_weights, axis=1)
        
        # Direct confounder effect
        confounder_effect_on_target = 0
        if self.n_confounders > 0:
            confounder_target_weights = self.rng.uniform(0.4, 0.8, size=self.n_confounders)
            confounder_effect_on_target = np.sum(confounders * confounder_target_weights, axis=1)
        
        # Combine effects with noise
        logits = behavioral_effect + confounder_effect_on_target + self.rng.normal(0, self.noise_level, 
                                                                      size=self.n_samples)
        
        # Convert to probability using sigmoid
        prob_default = 1 / (1 + np.exp(-logits))
        
        # Binary outcome (default/no default)
        default = (prob_default > 0.5).astype(int) # Standard threshold
        
        return default

    def generate_data(self) -> Tuple[pl.DataFrame, pl.Series, Dict]:
        # Generate data with explicit confounders
        confounders = self._generate_confounders()
        causal_features_arr = self._generate_causal_features(confounders)
        # n_proxy corresponds to n_spurious here
        spurious_features_arr = self._generate_spurious_features(confounders)
        X_arr = np.concatenate([causal_features_arr, spurious_features_arr], axis=1)
        y_arr = self._generate_target(causal_features_arr, confounders)

        X_df = pl.DataFrame(X_arr, schema=self.feature_names)
        y_series = pl.Series(name='default', values=y_arr)

        causal_info = {
            'causal_features': [name for i, name in enumerate(self.feature_names) 
                               if i < self.n_causal],
            # These are proxy features, but match 'spurious_features' key for consistency from base
            'spurious_features': [name for i, name in enumerate(self.feature_names) 
                                 if i >= self.n_causal],
            'dag': self.dag
        }
        return X_df, y_series, causal_info


class FraudDetectionScenario(CausalScenario):
    """
    Fraud Detection Scenario:
    
    Y: Probability of fraudulent transaction (target)
    X_causal_anomaly: Features truly indicative of fraud
    Z_confounder_context: Contextual factors
    X_indirect_indicator: Features that are consequences of fraud
    """
    
    def __init__(self, n_samples: int = 10000, random_state: int = 42,
                n_causal: int = 2, n_indirect: int = 2, n_confounders: int = 2,
                fraud_rate: float = 0.05, noise_level: float = 0.2):
        """Initialize the fraud detection scenario."""
        super().__init__(n_samples, random_state)
        self.n_causal = n_causal
        self.n_indirect = n_indirect # These are the "spurious" features in base class context
        self.n_confounders = n_confounders
        self.fraud_rate = fraud_rate
        self.noise_level = noise_level
        
        # Define feature names
        self.feature_names = (
            [f'anomaly_{i}' for i in range(n_causal)] +
            [f'indirect_{i}' for i in range(n_indirect)]
        )
        
        # Setup DAG
        self._setup_dag()
        
    def _setup_dag(self) -> None:
        """Setup the causal DAG."""
        # Add nodes
        for i in range(self.n_confounders):
            self.dag.add_node(f'Z{i}')
        
        for i in range(self.n_causal):
            self.dag.add_node(f'anomaly_{i}')
            
        for i in range(self.n_indirect):
            self.dag.add_node(f'indirect_{i}')
            
        self.dag.add_node('fraud')
        
        # Add edges
        for z_idx in range(self.n_confounders):
            # Confounders affect anomaly features
            for i in range(self.n_causal):
                self.dag.add_edge(f'Z{z_idx}', f'anomaly_{i}')
            
            # Confounders affect fraud directly
            self.dag.add_edge(f'Z{z_idx}', 'fraud')
        
        # Anomaly features cause fraud
        for i in range(self.n_causal):
            self.dag.add_edge(f'anomaly_{i}', 'fraud')
            
        # Fraud causes indirect indicators
        for i in range(self.n_indirect):
            self.dag.add_edge('fraud', f'indirect_{i}')
    
    def _generate_confounders(self) -> np.ndarray:
        """Generate contextual confounders (time, merchant category, etc.)."""
        return self.rng.normal(0, 1, size=(self.n_samples, self.n_confounders))
    
    def _generate_causal_features(self, confounders: np.ndarray) -> np.ndarray:
        """Generate anomaly features that cause fraud."""
        anomaly_features = np.zeros((self.n_samples, self.n_causal))
        
        for i in range(self.n_causal):
            # Base anomaly (unusual amount, velocity, etc.)
            base = self.rng.normal(0, 1, size=self.n_samples)
            
            # Add confounder effect
            confounder_effect = 0
            if self.n_confounders > 0:
                confounder_weights = self.rng.uniform(0.2, 0.5, size=self.n_confounders)
                confounder_effect = np.sum(confounders * confounder_weights, axis=1)
            
            # Combine with noise
            anomaly_features[:, i] = base + confounder_effect + self.rng.normal(0, self.noise_level, 
                                                                             size=self.n_samples)
        return anomaly_features
    
    def _generate_target(self, causal_features: np.ndarray, 
                        confounders: np.ndarray) -> np.ndarray:
        """Generate fraud indicator (before generating indirect features)."""
        # Effect from anomaly features
        anomaly_effect = 0
        if self.n_causal > 0:
            anomaly_weights = self.rng.uniform(0.8, 1.5, size=self.n_causal)
            anomaly_effect = np.sum(causal_features * anomaly_weights, axis=1)
        
        # Direct confounder effect
        confounder_effect_on_target = 0
        if self.n_confounders > 0:
            confounder_target_weights = self.rng.uniform(0.3, 0.6, size=self.n_confounders)
            confounder_effect_on_target = np.sum(confounders * confounder_target_weights, axis=1)
        
        # Combine with noise
        logits = anomaly_effect + confounder_effect_on_target + self.rng.normal(0, self.noise_level, 
                                                                    size=self.n_samples)
        
        # Convert to probability
        prob_fraud = 1 / (1 + np.exp(-logits))
        
        # Ensure desired fraud rate approximately
        threshold = np.percentile(prob_fraud, 100 * (1 - self.fraud_rate))
        fraud = (prob_fraud >= threshold).astype(int)
        
        return fraud
    
    def _generate_spurious_features(self, confounders: np.ndarray) -> np.ndarray:
        """
        Generate indirect indicators (consequences of fraud).
        This method is called by the base class's generate_data if not overridden.
        For FraudDetectionScenario, generate_data is overridden to correctly sequence
        target generation before indirect feature generation.
        This implementation here is for completeness if one were to call it, but it's
        less efficient as it recalculates causal_features and target.
        The primary logic is in the overridden generate_data method.
        """
        # First generate fraud indicator (re-calculating for this method's scope)
        # This is redundant if called from the main overridden generate_data flow
        temp_causal_features = self._generate_causal_features(confounders)
        fraud = self._generate_target(temp_causal_features, confounders)
        
        # Then generate indirect features based on fraud
        indirect_features = np.zeros((self.n_samples, self.n_indirect))
        
        for i in range(self.n_indirect):
            # Base value
            base = self.rng.normal(0, 1, size=self.n_samples)
            
            # Strong effect from fraud (since these are consequences)
            fraud_effect_strength = self.rng.uniform(1.0, 2.0)
            fraud_effect = fraud * fraud_effect_strength # fraud is 0 or 1
            
            # Add some noise
            indirect_features[:, i] = base + fraud_effect + self.rng.normal(0, self.noise_level, 
                                                                         size=self.n_samples)
        return indirect_features
    
    def generate_data(self) -> Tuple[pl.DataFrame, pl.Series, Dict]:
        """Override to handle the special case where fraud affects indirect features."""
        # Generate data
        confounders = self._generate_confounders()
        causal_features = self._generate_causal_features(confounders)
        fraud_target_array = self._generate_target(causal_features, confounders)
        
        # Generate indirect features based on the generated fraud_target_array
        indirect_features = np.zeros((self.n_samples, self.n_indirect))
        
        for i in range(self.n_indirect):
            # Base value
            base = self.rng.normal(0, 1, size=self.n_samples)
            
            # Strong effect from fraud (since these are consequences)
            fraud_effect_strength = self.rng.uniform(1.0, 2.0) # Random strength for this feature
            fraud_effect = fraud_target_array * fraud_effect_strength
            
            # Add some noise
            indirect_features[:, i] = base + fraud_effect + self.rng.normal(0, self.noise_level, 
                                                                         size=self.n_samples)
        
        # Combine features
        X_arr = np.concatenate([causal_features, indirect_features], axis=1)
        
        # Create DataFrame and Series
        X_df = pl.DataFrame(X_arr, schema=self.feature_names)
        y_series = pl.Series(name='fraud', values=fraud_target_array)
        
        # Create causal info dictionary
        causal_info = {
            'causal_features': [name for i, name in enumerate(self.feature_names) 
                               if i < self.n_causal],
            'indirect_features': [name for i, name in enumerate(self.feature_names) 
                                 if i >= self.n_causal], # These are the "indirect" ones
            'dag': self.dag
        }
        
        return X_df, y_series, causal_info


def generate_all_datasets(output_dir: str = './data', 
                         n_samples: int = 10000, 
                         random_state: int = 42) -> None:
    """
    Generate all financial datasets and save them to disk.
    
    Args:
        output_dir: Directory to save datasets
        n_samples: Number of samples per dataset
        random_state: Random seed
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate asset pricing data
    asset_scenario = AssetPricingScenario(n_samples=n_samples, random_state=random_state)
    X_asset, y_asset, causal_info_asset = asset_scenario.generate_data()
    
    # Save asset pricing data
    X_asset.write_csv(os.path.join(output_dir, 'asset_pricing_features.csv'))
    y_df_asset = pl.DataFrame({y_asset.name: y_asset})
    y_df_asset.write_csv(os.path.join(output_dir, 'asset_pricing_target.csv'))
    
    # Save causal info (excluding DAG which isn't easily serializable)
    causal_info_df_asset = pl.DataFrame({
        'feature': X_asset.columns,
        'is_causal': [feat in causal_info_asset['causal_features'] for feat in X_asset.columns]
    })
    causal_info_df_asset.write_csv(os.path.join(output_dir, 'asset_pricing_causal_info.csv'))
    
    # Generate credit risk data
    credit_scenario = CreditRiskScenario(n_samples=n_samples, random_state=random_state)
    X_credit, y_credit, causal_info_credit = credit_scenario.generate_data()
    
    # Save credit risk data
    X_credit.write_csv(os.path.join(output_dir, 'credit_risk_features.csv'))
    y_df_credit = pl.DataFrame({y_credit.name: y_credit})
    y_df_credit.write_csv(os.path.join(output_dir, 'credit_risk_target.csv'))
    
    causal_info_df_credit = pl.DataFrame({
        'feature': X_credit.columns,
        'is_causal': [feat in causal_info_credit['causal_features'] for feat in X_credit.columns]
    })
    causal_info_df_credit.write_csv(os.path.join(output_dir, 'credit_risk_causal_info.csv'))
    
    # Generate fraud detection data
    fraud_scenario = FraudDetectionScenario(n_samples=n_samples, random_state=random_state)
    X_fraud, y_fraud, causal_info_fraud = fraud_scenario.generate_data()
    
    # Save fraud detection data
    X_fraud.write_csv(os.path.join(output_dir, 'fraud_detection_features.csv'))
    y_df_fraud = pl.DataFrame({y_fraud.name: y_fraud})
    y_df_fraud.write_csv(os.path.join(output_dir, 'fraud_detection_target.csv'))
    
    causal_info_df_fraud = pl.DataFrame({
        'feature': X_fraud.columns,
        'is_causal': [feat in causal_info_fraud['causal_features'] for feat in X_fraud.columns]
    })
    causal_info_df_fraud.write_csv(os.path.join(output_dir, 'fraud_detection_causal_info.csv'))
    
    print(f"All datasets generated and saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    generate_all_datasets(n_samples=1000) # Smaller sample for quick test
    
    # Example of generating and visualizing a specific scenario
    asset_scenario_test = AssetPricingScenario(n_samples=100, n_causal=1, n_spurious=1, n_confounders=1)
    X, y, causal_info = asset_scenario_test.generate_data()
    asset_scenario_test.plot_dag()
    
    print("\nAsset Pricing Dataset (Test Sample):")
    print(f"X shape: {X.shape}")
    print(f"y name: {y.name}, y shape: {len(y)}")
    print("Causal features:", causal_info['causal_features'])
    print("Spurious features:", causal_info['spurious_features'])
    print(X.head())
    print(y.head())

    credit_scenario_test = CreditRiskScenario(n_samples=100, n_causal=1, n_proxy=1, n_confounders=1)
    X_c, y_c, causal_info_c = credit_scenario_test.generate_data()
    credit_scenario_test.plot_dag()
    print("\nCredit Risk Dataset (Test Sample):")
    print(f"X shape: {X_c.shape}")
    print(f"y name: {y_c.name}, y shape: {len(y_c)}")
    print("Causal features:", causal_info_c['causal_features'])
    print("Proxy (Spurious) features:", causal_info_c['spurious_features'])
    print(X_c.head())
    print(y_c.head())

    fraud_scenario_test = FraudDetectionScenario(n_samples=100, n_causal=1, n_indirect=1, n_confounders=1)
    X_f, y_f, causal_info_f = fraud_scenario_test.generate_data()
    fraud_scenario_test.plot_dag()
    print("\nFraud Detection Dataset (Test Sample):")
    print(f"X shape: {X_f.shape}")
    print(f"y name: {y_f.name}, y shape: {len(y_f)}")
    print("Causal features:", causal_info_f['causal_features'])
    print("Indirect features:", causal_info_f['indirect_features'])
    print(X_f.head())
    print(y_f.head())