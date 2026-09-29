import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]


def evaluate_models(models, cnn_probs, cnn_preds, X_train_scaled, y_train_res, X_train_res, X_test_scaled, X_test, y_test):
    results = {}

    # 1. Evaluate Classical Models
    for name, clf in models.items():
        if name == "Logistic Regression":
            clf.fit(X_train_scaled, y_train_res)
            probs = clf.predict_proba(X_test_scaled)[:, 1]
            preds = clf.predict(X_test_scaled)
        else:
            clf.fit(X_train_res, y_train_res)
            probs = clf.predict_proba(X_test)[:, 1]
            preds = clf.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, probs)
        results[name] = {
            'probs': probs,
            'preds': preds,
            'auc': roc_auc_score(y_test, probs),
            'accuracy': accuracy_score(y_test, preds) * 100,
            'recall': recall_score(y_test, preds, zero_division=0),
            'precision': precision_score(y_test, preds, zero_division=0),
            'f1': f1_score(y_test, preds, zero_division=0),
            'fpr': fpr,
            'tpr': tpr,
            'model': clf
        }

    # 2. Evaluate PyTorch 1D-CNN
    fpr, tpr, _ = roc_curve(y_test, cnn_probs)
    results["1D-CNN (PyTorch)"] = {
        'probs': cnn_probs,
        'preds': cnn_preds,
        'auc': roc_auc_score(y_test, cnn_probs),
        'accuracy': accuracy_score(y_test, cnn_preds) * 100,
        'recall': recall_score(y_test, cnn_preds, zero_division=0),
        'precision': precision_score(y_test, cnn_preds, zero_division=0),
        'f1': f1_score(y_test, cnn_preds, zero_division=0),
        'fpr': fpr,
        'tpr': tpr,
        'model': None
    }

    metrics_df = pd.DataFrame({
        m: [results[m]['accuracy'], results[m]['auc'], results[m]['recall'], results[m]['precision'], results[m]['f1']]
        for m in results
    }, index=['Accuracy (%)', 'ROC-AUC', 'Recall (Sensitivity)', 'Precision', 'F1-Score']).T

    return results, metrics_df


def plot_performance_and_localization(results, output_path="artifacts/final_pipeline_results.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left Subplot: ROC Curves
    colors = {
        'Logistic Regression': 'gray',
        'Random Forest': 'forestgreen',
        'Gradient Boosting': 'darkorange',
        '1D-CNN (PyTorch)': 'purple'
    }

    for name, res in results.items():
        axes[0].plot(res['fpr'], res['tpr'], label=f"{name} (AUC = {res['auc']:.4f})", color=colors[name], lw=2)

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate (Sensitivity)')
    axes[0].set_title('Cross-Subject Model ROC Curve Comparison', fontsize=11, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Right Subplot: Seizure Localization Importance
    rf_model = results['Random Forest']['model']
    importances = rf_model.feature_importances_.reshape(18, 9).sum(axis=1)

    sorted_idx = np.argsort(importances)[::-1]
    sorted_channels = [STANDARD_CHANNELS[i] for i in sorted_idx]
    sorted_scores = importances[sorted_idx]

    axes[1].bar(sorted_channels, sorted_scores, color='teal')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_title('Channel-Wise Seizure Localization Importance', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Summed Feature Importance')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    top_channel = sorted_channels[0]
    print(f"\n--> Primary Seizure Focus Identified at Channel: '{top_channel}'")
    print(f"--> Saved evaluation plots to '{output_path}'.")