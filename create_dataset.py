#!/usr/bin/env python3
"""
create_dataset.py (corrected)
- Loads normalized features and combined raw (from preprocess_normalize.py output)
- Handles both Parquet and CSV formats (fallback)
- Builds train/val/test splits
- If label info exists, stratified splits are used for supervised tasks
- Saves datasets to out-dir as .npz (features + labels) and CSV copies
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys


class Colors:
    """ANSI color codes for terminal output"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    ENDC = "\033[0m"


def load_data_with_fallback(base_path, filename_base):
    """Try to load parquet, fallback to CSV"""
    parquet_path = base_path / f"{filename_base}.parquet"
    csv_path = base_path / f"{filename_base}.csv"

    if parquet_path.exists():
        print(f"Loading {parquet_path.name}...")
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print(f"Loading {csv_path.name} (CSV fallback)...")
        return pd.read_csv(csv_path, low_memory=False)
    else:
        raise FileNotFoundError(
            f"Neither {parquet_path.name} nor {csv_path.name} found in {base_path}"
        )


def main(args):
    processed = Path(args.processed_dir)

    # Load manifest
    manifest_path = processed / "feature_list.json"
    if not manifest_path.exists():
        print(f"Error: feature_list.json not found in {processed}")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print("=" * 60)
    print("DATASET CREATION")
    print("=" * 60)

    # Load normalized features and raw data (with fallback to CSV)
    try:
        normalized = load_data_with_fallback(processed, "combined_normalized")
        raw = load_data_with_fallback(processed, "combined_raw")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"{Colors.GREEN}Loaded:{Colors.ENDC} normalized features {normalized.shape}")
    print(f"{Colors.GREEN}Loaded:{Colors.ENDC} raw data {raw.shape}")

    # Check for labels
    labels = None
    if "label" in raw.columns and raw["label"].notna().any():
        # Clean labels - replace special characters to avoid encoding issues
        labels = raw["label"].fillna("BENIGN").astype(str).str.strip()
        # Replace problematic characters (en-dash, em-dash, etc.) with ASCII equivalents
        labels = labels.str.replace("\x96", "-", regex=False)
        labels = labels.str.replace("\x97", "-", regex=False)
        labels = labels.str.replace("\u2013", "-", regex=False)  # en-dash
        labels = labels.str.replace("\u2014", "-", regex=False)  # em-dash

        # Count labels
        label_counts = labels.value_counts()
        print("\nLabel distribution:")
        for label, count in label_counts.items():
            # Ensure label is printable ASCII
            safe_label = label.encode("ascii", "replace").decode("ascii")
            print(f"  {safe_label}: {count:,} ({count/len(labels)*100:.1f}%)")

        print(
            f"\n{Colors.GREEN}Labels detected:{Colors.ENDC} using supervised split with stratification"
        )
    else:
        print(
            f"\n{Colors.YELLOW}No labels detected:{Colors.ENDC} producing unsupervised splits (no y)"
        )

    X = normalized.values
    y = labels.values if labels is not None else None

    print(f"\nFeature matrix shape: {X.shape}")

    # Perform splitting
    if args.time_based:
        print("\nUsing TIME-BASED splitting (preserves temporal order)")

        # Sort by timestamp
        if "timestamp" in raw.columns:
            raw_sorted = raw.sort_values("timestamp").reset_index(drop=True)
            sorted_indices = raw_sorted.index.values
        else:
            print("Warning: No timestamp column found, using row order")
            sorted_indices = np.arange(len(raw))

        n = len(sorted_indices)
        test_n = int(n * args.test_size)
        val_n = int(n * args.val_size)

        train_idx = sorted_indices[: n - test_n - val_n]
        val_idx = sorted_indices[n - test_n - val_n : n - test_n]
        test_idx = sorted_indices[n - test_n :]

        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]

        if y is not None:
            y_train = y[train_idx]
            y_val = y[val_idx]
            y_test = y[test_idx]
        else:
            y_train = y_val = y_test = None

    else:
        print("\nUsing RANDOM splitting with stratification (if labels exist)")

        if y is not None:
            # Stratified split
            try:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=args.test_size, stratify=y, random_state=42
                )
                val_split = args.val_size / (1.0 - args.test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=val_split,
                    stratify=y_temp,
                    random_state=42,
                )
            except ValueError as e:
                # If stratification fails (e.g., too few samples in some classes)
                print(f"Warning: Stratification failed ({e}), using random split")
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=args.test_size, random_state=42
                )
                val_split = args.val_size / (1.0 - args.test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_split, random_state=42
                )
        else:
            # No labels, simple random split
            X_temp, X_test = train_test_split(
                X, test_size=args.test_size, random_state=42
            )
            val_split = args.val_size / (1.0 - args.test_size)
            X_train, X_val = train_test_split(
                X_temp, test_size=val_split, random_state=42
            )
            y_train = y_val = y_test = None

    # Print split statistics
    print("\nDataset splits:")
    print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Create output directory
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save as NPZ (compressed)
    def save_npz(path, Xarr, yarr=None):
        if yarr is None:
            np.savez_compressed(path, X=Xarr)
        else:
            np.savez_compressed(path, X=Xarr, y=yarr)

    print("\nSaving datasets...")
    save_npz(out / "train.npz", X_train, y_train)
    save_npz(out / "val.npz", X_val, y_val)
    save_npz(out / "test.npz", X_test, y_test)
    print(
        f"  {Colors.GREEN}Saved:{Colors.ENDC} NPZ files (train.npz, val.npz, test.npz)"
    )

    # Save CSV copies for inspection
    csv_cols = manifest["final_feature_columns"]

    pd.DataFrame(X_train, columns=csv_cols).to_csv(out / "train.csv", index=False)
    pd.DataFrame(X_val, columns=csv_cols).to_csv(out / "val.csv", index=False)
    pd.DataFrame(X_test, columns=csv_cols).to_csv(out / "test.csv", index=False)
    print(
        f"  {Colors.GREEN}Saved:{Colors.ENDC} CSV files (train.csv, val.csv, test.csv)"
    )

    # Save labels separately if they exist
    if y_train is not None:
        pd.DataFrame({"label": y_train}).to_csv(out / "train_labels.csv", index=False)
        pd.DataFrame({"label": y_val}).to_csv(out / "val_labels.csv", index=False)
        pd.DataFrame({"label": y_test}).to_csv(out / "test_labels.csv", index=False)
        print(
            f"  {Colors.GREEN}Saved:{Colors.ENDC} label files (train_labels.csv, val_labels.csv, test_labels.csv)"
        )

    # Save split metadata
    split_info = {
        "split_method": "time-based" if args.time_based else "random",
        "test_size": args.test_size,
        "val_size": args.val_size,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "features": len(csv_cols),
        "has_labels": y_train is not None,
    }

    if y_train is not None:
        split_info["label_distribution_train"] = {
            str(k): int(v)
            for k, v in pd.Series(y_train).value_counts().to_dict().items()
        }
        split_info["label_distribution_test"] = {
            str(k): int(v)
            for k, v in pd.Series(y_test).value_counts().to_dict().items()
        }

    with open(out / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  {Colors.GREEN}Saved:{Colors.ENDC} split metadata (split_info.json)")

    print(f"\n{'='*60}")
    print(
        f"{Colors.GREEN}Complete:{Colors.ENDC} Dataset creation finished successfully"
    )
    print(f"  Output directory: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from preprocessed data"
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Directory where preprocess_normalize.py wrote its outputs",
    )
    parser.add_argument(
        "--out-dir", required=True, help="Where to save train/val/test datasets"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (0-1)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Proportion of data for validation (0-1)",
    )
    parser.add_argument(
        "--time-based",
        action="store_true",
        help="Use time-ordered splitting (good for time series)",
    )

    args = parser.parse_args()

    # Validation
    if args.test_size + args.val_size >= 1.0:
        print("Error: test_size + val_size must be < 1.0")
        sys.exit(1)

    try:
        main(args)
    except Exception as e:
        print(f"\n{Colors.RED}Error:{Colors.ENDC} {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
