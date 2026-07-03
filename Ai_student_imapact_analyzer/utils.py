# utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    return fig

def plot_feature_importance(feature_names, importances, top_n=10):
    """Plot top N feature importances."""
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
    ax.set_title(f'Top {top_n} Feature Importances')
    return fig

def plot_confusion_matrix(cm, labels=None):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

def plot_actual_vs_predicted(y_true, y_pred):
    """Scatter plot for regression (Actual vs Predicted)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    return fig