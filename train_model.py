#!/usr/bin/env python3
"""
train_model.py (corrected)
- mode: unsupervised (autoencoder) or supervised (RandomForest)
- Loads train/val/test from dataset dir created by create_dataset.py
- Trains model and saves it
- Prints evaluation metrics with proper formatting
- Handles edge cases and provides detailed progress
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
import joblib
import sys

# TensorFlow imports with proper error handling
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks

    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Unsupervised mode will not work.")
    TF_AVAILABLE = False


class Colors:
    """ANSI color codes for terminal output"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    ENDC = "\033[0m"


def load_npz(path):
    """Load NPZ file and extract X and y"""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    d = np.load(path, allow_pickle=True)
    X = d["X"]
    y = d["y"] if "y" in d else None
    return X, y


def build_autoencoder(input_dim, latent_dim=16):
    """Build autoencoder architecture"""
    inp = layers.Input(shape=(input_dim,), name="input")

    # Encoder
    x = layers.Dense(max(64, input_dim // 2), activation="relu", name="encoder_1")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(max(32, input_dim // 4), activation="relu", name="encoder_2")(x)
    x = layers.BatchNormalization()(x)

    # Bottleneck
    bottleneck = layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)

    # Decoder
    x = layers.Dense(max(32, input_dim // 4), activation="relu", name="decoder_1")(
        bottleneck
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dense(max(64, input_dim // 2), activation="relu", name="decoder_2")(x)
    x = layers.BatchNormalization()(x)

    # Output
    out = layers.Dense(input_dim, activation="linear", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="autoencoder")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model


def train_unsupervised(args, X_train, X_val, X_test, y_train, y_test, models_dir):
    """Train unsupervised autoencoder"""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for unsupervised mode")

    print("\n" + "=" * 60)
    print("UNSUPERVISED TRAINING - Autoencoder")
    print("=" * 60)

    # Filter to benign samples if requested and labels exist
    if args.train_on_benign and y_train is not None:
        benign_labels = ["benign", "normal", "0", "benign\r"]
        mask = np.array(
            [str(label).lower().strip() in benign_labels for label in y_train]
        )
        X_train_filtered = X_train[mask]
        print(
            f"\n{Colors.GREEN}Filtered:{Colors.ENDC} {len(X_train_filtered):,} benign samples (from {len(X_train):,})"
        )
        X_train = X_train_filtered

    input_dim = X_train.shape[1]
    print(f"\nModel configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")

    # Build model
    ae = build_autoencoder(input_dim, latent_dim=args.latent_dim)
    print(f"\n{ae.summary()}")

    # Callbacks
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]

    # Train
    print(f"\nTraining for up to {args.epochs} epochs...")
    history = ae.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb_list,
        verbose=1,
    )

    # Save model
    ae.save(args.model_out)
    print(f"\n{Colors.GREEN}Saved:{Colors.ENDC} autoencoder to {args.model_out}")

    # Evaluate on test set
    print("\n" + "-" * 60)
    print("EVALUATION ON TEST SET")
    print("-" * 60)

    recon = ae.predict(X_test, verbose=0)
    mse = np.mean(np.square(recon - X_test), axis=1)

    # Save reconstruction errors
    np.save(models_dir / "reconstruction_mse_test.npy", mse)
    print(
        f"{Colors.GREEN}Saved:{Colors.ENDC} reconstruction errors to reconstruction_mse_test.npy"
    )

    # Statistics
    print(f"\nReconstruction error statistics:")
    print(f"  Mean: {np.mean(mse):.6f}")
    print(f"  Std:  {np.std(mse):.6f}")
    print(f"  Min:  {np.min(mse):.6f}")
    print(f"  Max:  {np.max(mse):.6f}")

    # If labels exist, compute ROC AUC
    if y_test is not None:
        benign_labels = ["benign", "normal", "0", "benign\r"]
        y_true = np.array(
            [
                0 if str(label).lower().strip() in benign_labels else 1
                for label in y_test
            ]
        )

        try:
            auc = roc_auc_score(y_true, mse)
            print(
                f"\n{Colors.GREEN}ROC AUC:{Colors.ENDC} {auc:.4f} (reconstruction MSE as anomaly score)"
            )

            # Save evaluation metrics
            eval_metrics = {
                "mode": "unsupervised",
                "roc_auc": float(auc),
                "mse_mean": float(np.mean(mse)),
                "mse_std": float(np.std(mse)),
                "latent_dim": args.latent_dim,
                "epochs_trained": len(history.history["loss"]),
            }

            with open(models_dir / "evaluation_metrics.json", "w") as f:
                json.dump(eval_metrics, f, indent=2)

        except Exception as e:
            print(
                f"{Colors.YELLOW}Warning:{Colors.ENDC} Could not compute ROC AUC: {e}"
            )

    print("\n" + "=" * 60)
    print(f"{Colors.GREEN}Complete:{Colors.ENDC} Unsupervised training finished")
    print("=" * 60)


def train_supervised(args, X_train, X_val, X_test, y_train, y_val, y_test, models_dir):
    """Train supervised Random Forest classifier"""
    print("\n" + "=" * 60)
    print("SUPERVISED TRAINING - Random Forest")
    print("=" * 60)

    if y_train is None:
        raise ValueError("No labels available for supervised mode")

    # Encode labels - fit on ALL labels to handle rare classes in val/test
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_val, y_test])
    le.fit(all_labels)

    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val) if y_val is not None else None
    y_test_enc = le.transform(y_test) if y_test is not None else None

    print(f"\nDataset information:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(le.classes_)}")

    print(f"\nClass distribution in training set:")
    unique, counts = np.unique(y_train_enc, return_counts=True)
    for label_idx, count in zip(unique, counts):
        label_name = le.classes_[label_idx]
        # Use safe ASCII encoding for display
        safe_name = str(label_name).encode("ascii", "replace").decode("ascii")
        print(f"  {safe_name}: {count:,} ({count/len(y_train)*100:.1f}%)")

    # Train model
    print(f"\nTraining Random Forest ({args.n_estimators} trees)...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        max_depth=20,
        min_samples_split=10,
    )

    clf.fit(X_train, y_train_enc)

    # Save model
    model_data = {
        "model": clf,
        "label_encoder": le,
        "feature_names": [f"feature_{i}" for i in range(X_train.shape[1])],
    }
    joblib.dump(model_data, args.model_out)
    print(f"\n{Colors.GREEN}Saved:{Colors.ENDC} Random Forest to {args.model_out}")

    # Evaluate
    print("\n" + "-" * 60)
    print("EVALUATION ON TEST SET")
    print("-" * 60)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test_enc, y_pred)

    print(f"\n{Colors.GREEN}Test Accuracy:{Colors.ENDC} {accuracy:.4f}")

    # Get labels that actually appear in predictions and test set
    unique_test_labels = np.unique(np.concatenate([y_test_enc, y_pred]))
    # Map to class names (only for classes that appear)
    test_target_names = [le.classes_[i] for i in unique_test_labels]

    print("\nClassification Report:")
    try:
        report = classification_report(
            y_test_enc,
            y_pred,
            labels=unique_test_labels,
            target_names=test_target_names,
            zero_division=0,
        )
        print(report)
    except Exception as e:
        print(f"Warning: Could not generate full report: {e}")
        # Fallback without target names
        print(classification_report(y_test_enc, y_pred, zero_division=0))

    print("\nConfusion Matrix (showing only classes present in test set):")
    cm = confusion_matrix(y_test_enc, y_pred, labels=unique_test_labels)
    print(f"Shape: {cm.shape}")
    print(cm)

    # ROC AUC for binary or multiclass
    auc = None
    if hasattr(clf, "predict_proba"):
        try:
            probs = clf.predict_proba(X_test)

            # Get the classes that the model was trained on
            model_classes = clf.classes_

            if len(model_classes) == 2:
                auc = roc_auc_score(y_test_enc, probs[:, 1])
                print(f"\n{Colors.GREEN}ROC AUC (binary):{Colors.ENDC} {auc:.4f}")
            else:
                # For multiclass, we need to handle the case where test set
                # may not have all classes that the model was trained on
                auc = roc_auc_score(
                    y_test_enc,
                    probs,
                    multi_class="ovr",
                    average="weighted",
                    labels=model_classes,
                )
                print(
                    f"\n{Colors.GREEN}ROC AUC (multiclass, weighted):{Colors.ENDC} {auc:.4f}"
                )

        except Exception as e:
            print(
                f"{Colors.YELLOW}Warning:{Colors.ENDC} Could not compute ROC AUC: {e}"
            )

    # Save metrics
    eval_metrics = {
        "mode": "supervised",
        "accuracy": float(accuracy),
        "roc_auc": float(auc) if auc is not None else None,
        "n_classes": int(len(le.classes_)),
        "n_test_classes": int(len(unique_test_labels)),
        "n_estimators": args.n_estimators,
        "classes": le.classes_.tolist(),
    }

    with open(models_dir / "evaluation_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(
        f"{Colors.GREEN}Saved:{Colors.ENDC} evaluation metrics to evaluation_metrics.json"
    )

    # Feature importance
    importances = clf.feature_importances_
    top_k = 10
    top_indices = np.argsort(importances)[-top_k:][::-1]

    print(f"\nTop {top_k} Feature Importances:")
    for idx in top_indices:
        print(f"  Feature {idx}: {importances[idx]:.4f}")

    print("\n" + "=" * 60)
    print(f"{Colors.GREEN}Complete:{Colors.ENDC} Supervised training finished")
    print("=" * 60)


def main(args):
    dataset_dir = Path(args.dataset_dir)
    models_dir = Path(args.model_out).parent
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Model output: {args.model_out}")

    # Load datasets
    print("\nLoading datasets...")
    try:
        X_train, y_train = load_npz(dataset_dir / "train.npz")
        X_val, y_val = load_npz(dataset_dir / "val.npz")
        X_test, y_test = load_npz(dataset_dir / "test.npz")
    except FileNotFoundError as e:
        print(f"{Colors.RED}Error:{Colors.ENDC} {e}")
        sys.exit(1)

    print(f"{Colors.GREEN}Loaded:{Colors.ENDC} training set {X_train.shape}")
    print(f"{Colors.GREEN}Loaded:{Colors.ENDC} validation set {X_val.shape}")
    print(f"{Colors.GREEN}Loaded:{Colors.ENDC} test set {X_test.shape}")

    # Train based on mode
    if args.mode == "unsupervised":
        train_unsupervised(args, X_train, X_val, X_test, y_train, y_test, models_dir)
    else:
        train_supervised(
            args, X_train, X_val, X_test, y_train, y_val, y_test, models_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train anomaly detection model (unsupervised or supervised)"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing train/val/test NPZ files",
    )
    parser.add_argument(
        "--mode",
        choices=["unsupervised", "supervised"],
        default="unsupervised",
        help="Training mode",
    )
    parser.add_argument(
        "--model-out",
        required=True,
        help="Path to save model (.h5 for autoencoder, .joblib for RF)",
    )

    # Unsupervised parameters
    parser.add_argument(
        "--latent-dim", type=int, default=16, help="Autoencoder latent dimension"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs for autoencoder"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for autoencoder"
    )
    parser.add_argument(
        "--train-on-benign",
        action="store_true",
        help="Train autoencoder only on benign samples",
    )

    # Supervised parameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees for Random Forest",
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"\n{Colors.RED}Error:{Colors.ENDC} {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
