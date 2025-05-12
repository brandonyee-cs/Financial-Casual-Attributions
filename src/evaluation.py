import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from typing import Dict, List, Optional, Union, Any, Tuple

class CausalFaithfulnessEvaluator:
    """
    Evaluator for causal faithfulness of attribution methods.
    
    This class evaluates how well different attribution methods identify truly
    causal features in a model's decision making.
    """
    
    def __init__(self, causal_features: List[str], non_causal_features: List[str]):
        """
        Initialize causal faithfulness evaluator.
        
        Args:
            causal_features: List of truly causal feature names
            non_causal_features: List of non-causal feature names
        """
        self.causal_features = causal_features
        self.non_causal_features = non_causal_features
        self.all_features = causal_features + non_causal_features
    
    def evaluate_attribution_ranking(self, attributions: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate attribution ranking by computing metrics on feature importance ranking.
        
        Args:
            attributions: DataFrame with attribution scores
            
        Returns:
            Dict with evaluation metrics
        """
        # Get absolute attribution values
        abs_attrs = attributions.abs()
        
        # Calculate mean attribution rank for causal and non-causal features
        feature_ranks = abs_attrs.rank(axis=1, ascending=False, method='average')
        
        causal_ranks = feature_ranks[self.causal_features].values.flatten()
        non_causal_ranks = feature_ranks[self.non_causal_features].values.flatten()
        
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
        
        for idx, row in abs_attrs.iterrows():
            top_features = row.nlargest(k).index.tolist()
            top_k_features.extend(top_features)
        
        top_k_correct = sum(1 for feat in top_k_features if feat in self.causal_features)
        metrics['top_k_accuracy'] = top_k_correct / len(top_k_features)
        
        return metrics
    
    def evaluate_attribution_magnitude(self, attributions: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate attribution magnitude by computing metrics on attribution values.
        
        Args:
            attributions: DataFrame with attribution scores
            
        Returns:
            Dict with evaluation metrics
        """
        # Get absolute attribution values
        abs_attrs = attributions.abs()
        
        # Calculate mean absolute attribution for causal and non-causal features
        causal_attrs = abs_attrs[self.causal_features].values.flatten()
        non_causal_attrs = abs_attrs[self.non_causal_features].values.flatten()
        
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
    
    def evaluate_attribution_stability(self, attributions_list: List[pd.DataFrame]) -> Dict[str, float]:
        """
        Evaluate stability of attributions across multiple runs or samples.
        
        Args:
            attributions_list: List of attribution DataFrames
            
        Returns:
            Dict with stability metrics
        """
        if len(attributions_list) < 2:
            return {'stability_score': 1.0}  # Perfect stability with only one sample
        
        # Calculate pairwise rank correlation for all features
        rank_correlations = []
        
        for i in range(len(attributions_list)):
            for j in range(i + 1, len(attributions_list)):
                # Get absolute attributions
                abs_attrs_i = attributions_list[i].abs()
                abs_attrs_j = attributions_list[j].abs()
                
                # Get ranks
                ranks_i = abs_attrs_i.rank(axis=1, ascending=False)
                ranks_j = abs_attrs_j.rank(axis=1, ascending=False)
                
                # Calculate Spearman rank correlation for each sample
                sample_correlations = []
                for idx in abs_attrs_i.index:
                    if idx in ranks_j.index:
                        corr = ranks_i.loc[idx].corr(ranks_j.loc[idx], method='spearman')
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
        """
        Compute an overall faithfulness score from individual metrics.
        
        Args:
            ranking_metrics: Metrics from evaluate_attribution_ranking
            magnitude_metrics: Metrics from evaluate_attribution_magnitude
            stability_metrics: Metrics from evaluate_attribution_stability
            
        Returns:
            Overall faithfulness score (0 to 1, higher is better)
        """
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
    
    def evaluate_all_metrics(self, attributions: pd.DataFrame, 
                           attributions_list: Optional[List[pd.DataFrame]] = None) -> Dict[str, float]:
        """
        Evaluate all faithfulness metrics for a given attribution method.
        
        Args:
            attributions: DataFrame with attribution scores
            attributions_list: Optional list of additional attribution DataFrames for stability
            
        Returns:
            Dict with all evaluation metrics
        """
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


def compare_attribution_methods(scenario_name: str,
                              attribution_methods: List[str],
                              X: pd.DataFrame,
                              model: Any,
                              causal_info: Dict[str, List[str]],
                              n_samples: int = 100) -> pd.DataFrame:
    """
    Compare different attribution methods for a given scenario.
    
    Args:
        scenario_name: Name of scenario ('asset_pricing', 'credit_risk', 'fraud_detection')
        attribution_methods: List of attribution method names to compare
        X: Features DataFrame
        model: Trained model
        causal_info: Dictionary with causal feature information
        n_samples: Number of samples to use for evaluation
        
    Returns:
        DataFrame with comparison results
    """
    from attributions import compute_attributions
    
    # Extract causal and non-causal features
    if scenario_name == 'fraud_detection':
        causal_features = causal_info['causal_features']
        non_causal_features = causal_info['indirect_features']
    else:
        causal_features = causal_info['causal_features']
        non_causal_features = causal_info['spurious_features']
    
    # Create evaluator
    evaluator = CausalFaithfulnessEvaluator(causal_features, non_causal_features)
    
    # Sample data for evaluation
    if len(X) > n_samples:
        X_sample = X.sample(n_samples, random_state=42)
    else:
        X_sample = X
    
    # Compute attributions for each method
    all_metrics = []
    
    for method_name in attribution_methods:
        print(f"Computing {method_name} attributions...")
        
        # Compute attributions
        attributions_df = compute_attributions(
            model=model,
            X=X_sample.values,
            method_name=method_name,
            feature_names=X.columns.tolist()
        )
        
        # Evaluate metrics
        metrics = evaluator.evaluate_all_metrics(attributions_df)
        
        # Add method and scenario info
        metrics['method'] = method_name
        metrics['scenario'] = scenario_name
        
        all_metrics.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_metrics)
    
    return results_df


def plot_comparison_results(results_df: pd.DataFrame, 
                          metric: str = 'overall_faithfulness_score',
                          title: Optional[str] = None) -> None:
    """
    Plot comparison results for different attribution methods.
    
    Args:
        results_df: DataFrame with comparison results
        metric: Metric to plot
        title: Plot title (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar chart by scenario and method
    ax = sns.barplot(
        data=results_df,
        x='method',
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
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    
    # Add legend
    plt.legend(title='Scenario')
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()


def plot_attribution_heatmap(attributions: pd.DataFrame, 
                           causal_features: List[str],
                           non_causal_features: List[str],
                           title: str = 'Feature Attributions') -> None:
    """
    Plot heatmap of feature attributions.
    
    Args:
        attributions: DataFrame with attribution scores
        causal_features: List of causal feature names
        non_causal_features: List of non-causal feature names
        title: Plot title
    """
    # Get absolute attributions
    abs_attrs = attributions.abs()
    
    # Sort features by causal and non-causal
    sorted_features = causal_features + non_causal_features
    
    # Select a subset of samples if there are too many
    if len(abs_attrs) > 20:
        abs_attrs = abs_attrs.iloc[:20]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Normalize attributions
    normalized_attrs = abs_attrs.copy()
    for idx, row in normalized_attrs.iterrows():
        row_sum = row.sum()
        if row_sum > 0:
            normalized_attrs.loc[idx] = row / row_sum
    
    # Plot heatmap
    sns.heatmap(
        normalized_attrs[sorted_features].T,
        cmap='YlOrRd',
        vmin=0,
        vmax=normalized_attrs.max().max(),
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
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    # Example usage
    from models import SimpleMLPModel
    from attributions import compute_attributions
    
    # Create a simple model
    model = SimpleMLPModel(input_dim=5, hidden_dims=[10], output_dim=1)
    
    # Create random data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    # Train model
    model.fit(X, y, epochs=10, verbose=True)
    
    # Define causal and non-causal features
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    causal_features = ['feature_1', 'feature_2']
    non_causal_features = ['feature_3', 'feature_4', 'feature_5']
    
    # Create evaluator
    evaluator = CausalFaithfulnessEvaluator(causal_features, non_causal_features)
    
    # Compute attributions
    attribution_methods = ['saliency', 'gradient_input', 'integrated_gradients']
    all_metrics = []
    
    for method_name in attribution_methods:
        print(f"Computing {method_name} attributions...")
        
        # Compute attributions
        attributions_df = compute_attributions(
            model=model,
            X=X,
            method_name=method_name,
            feature_names=feature_names
        )
        
        # Evaluate metrics
        metrics = evaluator.evaluate_all_metrics(attributions_df)
        
        # Add method info
        metrics['method'] = method_name
        metrics['scenario'] = 'example'
        
        all_metrics.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_metrics)
    
    # Print results
    print("\nComparison Results:")
    print(results_df[['method', 'overall_faithfulness_score']])
    
    # Plot results
    plot_comparison_results(results_df)
    
    # Plot attribution heatmap for one method
    attributions_df = compute_attributions(
        model=model,
        X=X[:10],
        method_name='integrated_gradients',
        feature_names=feature_names
    )
    
    plot_attribution_heatmap(
        attributions=attributions_df,
        causal_features=causal_features,
        non_causal_features=non_causal_features,
        title='Integrated Gradients Attributions'
    )