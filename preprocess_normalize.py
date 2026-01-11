#!/usr/bin/env python3
"""
preprocess_normalize.py (memory-efficient chunked version)
- Safely loads CSV variants with encoding fallback (UTF-8 -> Latin-1)
- Skips malformed lines gracefully
- Processes files in chunks to avoid memory exhaustion
- Normalizes column names into unified schema
- Handles duplicate column names
- Saves intermediate results to avoid losing progress
- Creates scaled feature matrix with protocol encoding limits
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from dateutil import parser as dateparser
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import json
import sys
import traceback
import gc
import psutil
import warnings

warnings.filterwarnings("ignore")


class Colors:
    """ANSI color codes for terminal output"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    ENDC = "\033[0m"


UNIFIED_COLS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "flow_duration",
    "packet_count",
    "byte_count",
    "label",
]

COL_MAPPING = {
    "timestamp": [
        "timestamp",
        "time",
        "date",
        "datetime",
        "flow start",
        "starttime",
        "flow_start",
        "ts",
        "TimeStamp",
        "StartTime",
        "timestamp_utc",
    ],
    "src_ip": [
        "srcip",
        "sourceip",
        "source_ip",
        "src_ip",
        "source",
        "src",
        "src address",
        "ip.src",
        "Source IP",
    ],
    "dst_ip": [
        "dstip",
        "destinationip",
        "destination_ip",
        "dst_ip",
        "dest",
        "dst",
        "dst address",
        "ip.dst",
        "Destination IP",
    ],
    "src_port": [
        "srcport",
        "sport",
        "sourceport",
        "source_port",
        "src_port",
        "Source Port",
    ],
    "dst_port": [
        "dstport",
        "dport",
        "destinationport",
        "destination_port",
        "dst_port",
        "Destination Port",
    ],
    "protocol": ["protocol", "proto", "protocol_name", "Protocol", "ip.proto"],
    "flow_duration": [
        "flow_duration",
        "duration",
        "Flow Duration",
        "FlowDuration",
        "flow_len",
        "flowlength",
        "Flow Duration",
    ],
    "packet_count": [
        "packets",
        "packet_count",
        "pkt_count",
        "total_packets",
        "total_fwd_packets",
        "total_bwd_packets",
        "Total Fwd Packets",
        "Total Backward Packets",
    ],
    "byte_count": [
        "bytes",
        "byte_count",
        "total_bytes",
        "total_len",
        "total_fwd_bytes",
        "total_bwd_bytes",
        "Total Length of Fwd Packets",
        "Flow Bytes/s",
    ],
    "label": [
        "label",
        "attack",
        "attack_type",
        "class",
        "Category",
        "Label",
        "Labelled",
        " Label",
    ],
}


def get_memory_usage():
    """Return current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def try_read_csv(path):
    """
    Try reading CSV with multiple fallback strategies for encoding and malformed data.
    Returns DataFrame or raises Exception.
    """
    # Strategy 1: UTF-8 with bad line skipping
    try:
        return pd.read_csv(
            path, encoding="utf-8", on_bad_lines="skip", low_memory=False
        )
    except Exception as e1:
        pass

    # Strategy 2: Latin-1 encoding (common for Windows-1252 data with 0x96, 0x92 bytes)
    try:
        return pd.read_csv(
            path, encoding="latin1", on_bad_lines="skip", low_memory=False
        )
    except Exception as e2:
        pass

    # Strategy 3: ISO-8859-1 encoding
    try:
        return pd.read_csv(
            path, encoding="iso-8859-1", on_bad_lines="skip", low_memory=False
        )
    except Exception as e3:
        pass

    # Strategy 4: Try with python engine (slower but more flexible)
    try:
        return pd.read_csv(
            path, encoding="latin1", engine="python", on_bad_lines="skip"
        )
    except Exception as e4:
        pass

    # Strategy 5: Read as strings then convert
    try:
        df = pd.read_csv(path, encoding="latin1", dtype=str, on_bad_lines="skip")
        return df
    except Exception as e5:
        # Last resort: try cp1252 (Windows encoding)
        try:
            return pd.read_csv(
                path, encoding="cp1252", on_bad_lines="skip", low_memory=False
            )
        except Exception as e6:
            raise Exception(f"All CSV reading strategies failed. Last error: {e6}")


def make_unique_cols(cols):
    """Return list of column names where duplicates get a suffix: name__dup1, name__dup2, ..."""
    seen = {}
    out = []
    for c in cols:
        c = str(c).strip()  # Strip whitespace from column names
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out


def detect_col_name(col_name):
    """Detect which unified column this maps to"""
    c_low = str(col_name).strip().lower()
    for target, variants in COL_MAPPING.items():
        for v in variants:
            v_low = v.lower().strip()
            # Exact match first
            if c_low == v_low:
                return target
            # Then substring matches
            if v_low in c_low or c_low in v_low:
                return target
    return None


def map_columns(df):
    """Create mapping from original column names to unified schema"""
    mapping = {}
    mapped_targets = set()

    for c in df.columns:
        detected = detect_col_name(c)
        if detected and detected not in mapped_targets:
            mapping[c] = detected
            mapped_targets.add(detected)

    return mapping


def parse_timestamp(val):
    """Parse various timestamp formats"""
    if pd.isna(val):
        return pd.NaT
    try:
        return dateparser.parse(str(val), fuzzy=True)
    except Exception:
        try:
            # Try numeric epoch
            return pd.to_datetime(float(val), unit="s", errors="coerce")
        except Exception:
            return pd.NaT


def load_and_map_csv(path, max_rows=None):
    """Load CSV and map to unified schema"""
    df = try_read_csv(path)

    # Limit rows if needed (for memory management)
    if max_rows and len(df) > max_rows:
        print(f"  Limiting {path.name} to {max_rows} rows (original: {len(df)})")
        df = df.head(max_rows)

    # Ensure unique column names
    df.columns = make_unique_cols(df.columns)

    # Map to unified schema
    colmap = map_columns(df)
    if colmap:
        df = df.rename(columns=colmap)

    # Ensure all unified columns exist
    for c in UNIFIED_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Keep only unified columns (drop unmapped extras to save memory)
    cols_to_keep = [
        c for c in df.columns if c in UNIFIED_COLS or c.startswith(tuple(UNIFIED_COLS))
    ]
    df = df[cols_to_keep]

    # Parse timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(parse_timestamp)

    # Convert numeric columns and handle infinity/extreme values
    for c in ["flow_duration", "packet_count", "byte_count", "src_port", "dst_port"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Replace inf with NaN, then fill with 0
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0)
            # Clip extreme values
            df[c] = df[c].clip(-1e15, 1e15)

    # Ensure IP columns are strings (fix mixed type issues with parquet)
    for ip_col in ["src_ip", "dst_ip"]:
        if ip_col in df.columns:
            df[ip_col] = df[ip_col].astype(str).str.strip()
            df[ip_col] = df[ip_col].replace(["nan", "None", ""], "0.0.0.0")

    # Normalize protocol
    if "protocol" in df.columns:
        df["protocol"] = df["protocol"].astype(str).str.strip().fillna("0")

    # Clean label column
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip()

    return df


def process_in_batches(csv_paths, out_dir, args):
    """Process CSVs in batches to manage memory"""
    batch_size = 5  # Process 5 files at a time
    all_protocols = set()
    total_rows = 0
    errors = []

    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PHASE 1: Processing {len(csv_paths)} files in batches of {batch_size}")
    print(f"{'='*60}\n")

    batch_num = 0
    for i in range(0, len(csv_paths), batch_size):
        batch = csv_paths[i : i + batch_size]
        batch_dfs = []

        print(f"\nBatch {batch_num + 1}/{(len(csv_paths)-1)//batch_size + 1}")
        print(f"Memory before batch: {get_memory_usage():.1f} MB")

        for p in tqdm(batch, desc=f"Loading batch {batch_num+1}"):
            try:
                df = load_and_map_csv(p, max_rows=args.max_rows_per_file)
                batch_dfs.append(df)

                # Collect unique protocols
                if "protocol" in df.columns:
                    all_protocols.update(df["protocol"].unique())

                total_rows += len(df)

            except Exception as e:
                error_msg = str(e)[:200]  # Truncate long errors
                print(f"  {Colors.YELLOW}Skipped:{Colors.ENDC} {p.name}: {error_msg}")
                errors.append({"path": str(p), "error": error_msg})

        # Save batch if any files loaded
        if batch_dfs:
            try:
                batch_combined = pd.concat(batch_dfs, ignore_index=True, sort=False)

                # Sanitize types before saving to parquet (fix mixed type errors)
                for col in batch_combined.columns:
                    if batch_combined[col].dtype == object:
                        # Convert object columns to string to avoid parquet type conflicts
                        batch_combined[col] = (
                            batch_combined[col].astype(str).replace("nan", "")
                        )

                # Save batch chunk
                chunk_path = chunks_dir / f"chunk_{batch_num:03d}.parquet"
                batch_combined.to_parquet(chunk_path, index=False)
                print(
                    f"  {Colors.GREEN}Saved:{Colors.ENDC} chunk {batch_num} ({len(batch_combined):,} rows) -> {chunk_path.name}"
                )

                # Clear memory
                del batch_combined
                del batch_dfs
                gc.collect()

            except Exception as e:
                print(
                    f"  {Colors.RED}Failed:{Colors.ENDC} to save batch {batch_num}: {e}"
                )

        batch_num += 1
        print(f"Memory after batch: {get_memory_usage():.1f} MB")

    print(
        f"\n{Colors.GREEN}Complete:{Colors.ENDC} Phase 1 - {total_rows:,} total rows processed"
    )
    print(f"{Colors.GREEN}Found:{Colors.ENDC} {len(all_protocols)} unique protocols")

    return chunks_dir, all_protocols, errors


def merge_chunks(chunks_dir, out_dir, all_protocols, args):
    """Merge all chunks into final dataset"""
    print(f"\n{'='*60}")
    print(f"PHASE 2: Merging chunks and creating normalized features")
    print(f"{'='*60}\n")

    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    print(f"Found {len(chunk_files)} chunks to merge")

    if not chunk_files:
        raise SystemExit("No chunks found to merge!")

    # Determine top-K protocols from all data
    top_k = args.top_k_protocols
    if len(all_protocols) > top_k:
        print(f"Limiting protocols to top {top_k} (found {len(all_protocols)} unique)")

    # Read all chunks and combine (they're already processed)
    combined_chunks = []
    for chunk_path in tqdm(chunk_files, desc="Merging chunks"):
        df = pd.read_parquet(chunk_path)
        combined_chunks.append(df)

    combined = pd.concat(combined_chunks, ignore_index=True, sort=False)
    del combined_chunks
    gc.collect()

    print(
        f"{Colors.GREEN}Merged:{Colors.ENDC} dataset with {len(combined):,} rows, {len(combined.columns)} columns"
    )

    # Fill missing numeric values
    for c in ["packet_count", "byte_count", "flow_duration"]:
        if c not in combined.columns:
            combined[c] = 0
        combined[c] = pd.to_numeric(combined[c], errors="coerce").fillna(0)

    # Timestamp fallback
    if "timestamp" not in combined.columns or combined["timestamp"].isna().all():
        combined["timestamp"] = pd.Timestamp.now()

    # Normalize IPs
    combined["src_ip"] = (
        combined.get("src_ip", pd.Series(["0.0.0.0"] * len(combined)))
        .astype(str)
        .fillna("0.0.0.0")
    )
    combined["dst_ip"] = (
        combined.get("dst_ip", pd.Series(["0.0.0.0"] * len(combined)))
        .astype(str)
        .fillna("0.0.0.0")
    )

    # Sort by timestamp
    try:
        combined = combined.sort_values(by="timestamp").reset_index(drop=True)
    except Exception:
        combined = combined.reset_index(drop=True)

    # Save raw combined data
    print("\nSaving combined raw data...")
    combined_raw_path = out_dir / "combined_raw.parquet"
    try:
        combined.to_parquet(combined_raw_path, index=False)
        print(f"{Colors.GREEN}Saved:{Colors.ENDC} {combined_raw_path}")
    except Exception:
        combined_raw_path = out_dir / "combined_raw.csv"
        combined.to_csv(combined_raw_path, index=False)
        print(f"{Colors.GREEN}Saved:{Colors.ENDC} {combined_raw_path} (CSV fallback)")

    # Create normalized features
    print("\nCreating normalized feature matrix...")

    numeric_cols = [
        "flow_duration",
        "packet_count",
        "byte_count",
        "src_port",
        "dst_port",
    ]
    for c in numeric_cols:
        if c not in combined.columns:
            combined[c] = 0
        combined[c] = pd.to_numeric(combined[c], errors="coerce").fillna(0)

    # Protocol encoding (top-K only)
    combined["protocol"] = (
        combined.get("protocol", pd.Series(["0"] * len(combined)))
        .astype(str)
        .fillna("0")
    )
    proto_counts = combined["protocol"].value_counts().nlargest(top_k)
    protos_to_keep = proto_counts.index.tolist()
    combined["protocol_mapped"] = combined["protocol"].where(
        combined["protocol"].isin(protos_to_keep), other="OTHER"
    )
    protocol_dummies = pd.get_dummies(combined["protocol_mapped"], prefix="proto")

    print(f"  Protocol features: {len(protocol_dummies.columns)} columns")

    # Combine numeric + protocol features
    scaled_df = pd.concat(
        [
            combined[numeric_cols].reset_index(drop=True),
            protocol_dummies.reset_index(drop=True),
        ],
        axis=1,
    ).fillna(0)

    print(
        f"  Feature matrix: {scaled_df.shape[0]:,} rows × {scaled_df.shape[1]} features"
    )

    # Handle infinity and extremely large values before scaling
    print("\nCleaning data (replacing inf/large values)...")
    # Replace infinity with NaN, then fill with 0
    scaled_df = scaled_df.replace([np.inf, -np.inf], np.nan)
    scaled_df = scaled_df.fillna(0)

    # Clip extremely large values to prevent overflow
    max_val = np.finfo(np.float64).max / 10  # Safe threshold
    for col in scaled_df.columns:
        scaled_df[col] = scaled_df[col].clip(-max_val, max_val)

    print(
        f"  Data range after cleaning: [{scaled_df.min().min():.2e}, {scaled_df.max().max():.2e}]"
    )

    # Fit scaler
    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(scaled_df)
    scaled_df_norm = pd.DataFrame(scaled_arr, columns=scaled_df.columns)

    # Save scaler
    scaler_path = out_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"{Colors.GREEN}Saved:{Colors.ENDC} scaler to {scaler_path}")

    # Save normalized features
    print("\nSaving normalized features...")
    normalized_path = out_dir / "combined_normalized.parquet"
    try:
        scaled_df_norm.to_parquet(normalized_path, index=False)
        print(f"{Colors.GREEN}Saved:{Colors.ENDC} {normalized_path}")
    except Exception:
        normalized_path = out_dir / "combined_normalized.csv"
        scaled_df_norm.to_csv(normalized_path, index=False)
        print(f"{Colors.GREEN}Saved:{Colors.ENDC} {normalized_path} (CSV fallback)")

    return {
        "numeric_cols": numeric_cols,
        "protocol_columns": list(protocol_dummies.columns),
        "final_feature_columns": list(scaled_df_norm.columns),
        "combined_raw_path": str(combined_raw_path),
        "normalized_path": str(normalized_path),
        "total_rows": len(combined),
        "total_features": len(scaled_df_norm.columns),
    }


def main(args):
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files
    csv_paths = list(input_dir.rglob("*.csv"))
    print(f"\n{'='*60}")
    print(f"PREPROCESSING PIPELINE - Memory-Efficient Mode")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Found {len(csv_paths)} CSV files")
    print(f"Processing limit: {args.max_files} files")
    print(
        f"Max rows per file: {args.max_rows_per_file if args.max_rows_per_file else 'unlimited'}"
    )
    print(f"Protocol limit: top-{args.top_k_protocols}")
    print(f"Current memory: {get_memory_usage():.1f} MB")

    # Limit files if requested
    csv_paths = csv_paths[: args.max_files]

    if not csv_paths:
        raise SystemExit("No CSV files found! Check your input directory.")

    # Process in batches
    chunks_dir, all_protocols, errors = process_in_batches(csv_paths, out_dir, args)

    # Merge chunks and create normalized features
    manifest = merge_chunks(chunks_dir, out_dir, all_protocols, args)

    # Add error log to manifest
    manifest["skipped_files"] = errors
    manifest["files_processed"] = len(csv_paths) - len(errors)
    manifest["files_skipped"] = len(errors)

    # Save manifest
    manifest_path = out_dir / "feature_list.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"{Colors.GREEN}Files processed:{Colors.ENDC} {manifest['files_processed']}")
    print(f"{Colors.YELLOW}Files skipped:{Colors.ENDC} {manifest['files_skipped']}")
    print(f"{Colors.GREEN}Total rows:{Colors.ENDC} {manifest['total_rows']:,}")
    print(f"{Colors.GREEN}Total features:{Colors.ENDC} {manifest['total_features']}")
    print(f"{Colors.GREEN}Manifest saved:{Colors.ENDC} {manifest_path}")
    print(f"Final memory: {get_memory_usage():.1f} MB")

    if errors:
        print(
            f"\n{Colors.YELLOW}Warning:{Colors.ENDC} {len(errors)} files were skipped due to errors"
        )
        print(f"  Check {manifest_path} for details")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Memory-efficient CSV preprocessing with chunked processing"
    )
    p.add_argument("--input-dir", required=True, help="Root directory with CSV files")
    p.add_argument(
        "--out-dir", required=True, help="Output directory for processed data"
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=500,
        help="Maximum number of CSV files to process",
    )
    p.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Limit rows per file (use for very large CSVs, e.g., 500000)",
    )
    p.add_argument(
        "--top-k-protocols",
        type=int,
        default=10,
        help="Limit protocol one-hot encoding to top-K protocols",
    )

    args = p.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted:{Colors.ENDC} by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.RED}Fatal error:{Colors.ENDC} {e}")
        traceback.print_exc()
        sys.exit(1)
