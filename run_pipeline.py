import os
import numpy as np
from imblearn.over_sampling import SMOTE

from src.feature_extraction import extract_subject_dataset
from src.machine_learning_models import initialize_classical_models, train_cnn
from src.evaluation_and_visualise import evaluate_models, plot_performance_and_localization

ARTIFACTS_DIR = "/Users/apple/Downloads/EEG_Processing/artifacts"


def main():
    print("="*60)
    print(" END-TO-END CHB-MIT SEIZURE DETECTION PIPELINE ")
    print("="*60)

    # 1. Feature Extraction (or Load from disk if pre-extracted)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    train_file = os.path.join(ARTIFACTS_DIR, "X_train.npy")
    if os.path.exists(train_file):
        print("\n--> Loading feature matrices from artifacts/...")
        X_train = np.load(os.path.join(ARTIFACTS_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(ARTIFACTS_DIR, "y_train.npy"))
        X_test = np.load(os.path.join(ARTIFACTS_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))
    else:
        X_train, y_train = extract_subject_dataset("chb01")
        X_test, y_test = extract_subject_dataset("chb02")
        np.save(os.path.join(ARTIFACTS_DIR, "X_train.npy"), X_train)
        np.save(os.path.join(ARTIFACTS_DIR, "y_train.npy"), y_train)
        np.save(os.path.join(ARTIFACTS_DIR, "X_test.npy"), X_test)
        np.save(os.path.join(ARTIFACTS_DIR, "y_test.npy"), y_test)

    print(f"\nTrain Set (chb01): {X_train.shape[0]} windows | Seizures: {np.sum(y_train)}")
    print(f"Test Set  (chb02): {X_test.shape[0]} windows | Seizures: {np.sum(y_test)}")

    # 2. Oversample & Scale Features
    print("\n--> Balancing Class Distribution with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    mean_X = np.mean(X_train_res, axis=0)
    std_X = np.std(X_train_res, axis=0) + 1e-8
    X_train_scaled = (X_train_res - mean_X) / std_X
    X_test_scaled = (X_test - mean_X) / std_X

    # 3. Model Training & Evaluation
    models = initialize_classical_models()
    cnn_probs, cnn_preds = train_cnn(X_train_scaled, y_train_res, X_test_scaled)

    results, metrics_df = evaluate_models(
        models, cnn_probs, cnn_preds,
        X_train_scaled, y_train_res, X_train_res,
        X_test_scaled, X_test, y_test
    )

    print("\n" + "="*65)
    print("        CROSS-SUBJECT BENCHMARK EVALUATION (TEST: CHB02)")
    print("="*65)
    print(metrics_df.round(4))

    # 4. Generate & Save Plot
    plot_performance_and_localization(results)


if __name__ == "__main__":
    main()