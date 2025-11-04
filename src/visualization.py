"""Visualization and Analysis Tools.

Matplotlib and Seaborn visualizations for term structure evolution,
path trajectories, and sensitivity analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional


class RatePathVisualizer:
    """Visualize simulated interest rate paths."""

    @staticmethod
    def plot_paths(paths: np.ndarray, dt: float, figsize: Tuple = (12, 6),
                  num_sample: int = 100):
        """Plot sample paths.

        Args:
            paths: Simulated paths (steps, paths)
            dt: Time step
            figsize: Figure size
            num_sample: Number of paths to plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        times = np.arange(paths.shape[0]) * dt
        
        num_plot = min(num_sample, paths.shape[1])
        indices = np.linspace(0, paths.shape[1] - 1, num_plot, dtype=int)
        
        for idx in indices:
            ax.plot(times, paths[:, idx], alpha=0.3, linewidth=0.8)
        
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Short Rate (%)')
        ax.set_title('Monte Carlo Interest Rate Paths')
        ax.grid(True, alpha=0.3)
        
        return fig, ax

    @staticmethod
    def plot_term_structure(paths: np.ndarray, dt: float,
                           figsize: Tuple = (12, 6)):
        """Plot term structure evolution.

        Args:
            paths: Simulated paths
            dt: Time step
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        times = np.arange(paths.shape[0]) * dt
        
        mean_rate = np.mean(paths, axis=1)
        std_rate = np.std(paths, axis=1)
        q05 = np.percentile(paths, 5, axis=1)
        q95 = np.percentile(paths, 95, axis=1)
        
        ax.plot(times, mean_rate, label='Mean', linewidth=2, color='blue')
        ax.fill_between(times, q05, q95, alpha=0.2, color='blue',
                        label='5th-95th percentile')
        ax.fill_between(times, mean_rate - std_rate, mean_rate + std_rate,
                       alpha=0.15, color='blue', label='±1 Std Dev')
        
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Short Rate (%)')
        ax.set_title('Term Structure Evolution')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return fig, ax

    @staticmethod
    def plot_distribution(paths: np.ndarray, times_idx: list,
                         figsize: Tuple = (12, 8)):
        """Plot rate distributions at different times.

        Args:
            paths: Simulated paths
            times_idx: Time step indices
            figsize: Figure size
        """
        num_times = len(times_idx)
        fig, axes = plt.subplots(1, num_times, figsize=figsize)
        
        if num_times == 1:
            axes = [axes]
        
        for ax, idx in zip(axes, times_idx):
            ax.hist(paths[idx], bins=50, density=True, alpha=0.7,
                   color='steelblue', edgecolor='black')
            ax.set_xlabel('Rate')
            ax.set_ylabel('Density')
            ax.set_title(f't = {idx}')
            ax.grid(True, alpha=0.3, axis='y')
        
        return fig, axes


class SensitivityAnalyzer:
    """Sensitivity analysis visualizations."""

    @staticmethod
    def plot_parameter_sensitivity(param_names: list, sensitivities: np.ndarray,
                                  figsize: Tuple = (10, 6)):
        """Plot parameter sensitivities as bar chart.

        Args:
            param_names: Parameter names
            sensitivities: Sensitivity values
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['green' if s > 0 else 'red' for s in sensitivities]
        ax.bar(param_names, sensitivities, color=colors, alpha=0.7,
              edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Sensitivity')
        ax.set_title('Parameter Sensitivities')
        ax.grid(True, alpha=0.3, axis='y')
        
        return fig, ax

    @staticmethod
    def plot_heatmap(data: np.ndarray, x_labels: list, y_labels: list,
                    figsize: Tuple = (12, 8)):
        """Plot sensitivity heatmap.

        Args:
            data: 2D sensitivity matrix
            x_labels: Column labels
            y_labels: Row labels
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels,
                   cmap='RdYlGn', center=0, ax=ax, cbar_kws={'label': 'Sensitivity'})
        ax.set_title('Sensitivity Heatmap')
        
        return fig, ax


class CorrelationAnalyzer:
    """Correlation and autocorrelation visualization."""

    @staticmethod
    def plot_acf(acf_values: np.ndarray, lags: int,
                figsize: Tuple = (10, 6)):
        """Plot autocorrelation function.

        Args:
            acf_values: ACF values
            lags: Number of lags
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        lags_arr = np.arange(lags)
        
        ax.stem(lags_arr, acf_values, basefmt=' ')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title('Autocorrelation Function')
        ax.grid(True, alpha=0.3)
        
        return fig, ax

    @staticmethod
    def plot_correlation_matrix(corr_matrix: np.ndarray,
                               var_names: list,
                               figsize: Tuple = (10, 8)):
        """Plot correlation matrix as heatmap.

        Args:
            corr_matrix: Correlation matrix
            var_names: Variable names
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, xticklabels=var_names, yticklabels=var_names,
                   cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax,
                   square=True, cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix')
        
        return fig, ax
