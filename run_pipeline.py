#!/usr/bin/env python3
"""
run_pipeline.py - Complete ML Pipeline Runner
Executes preprocessing, dataset creation, and model training in sequence
with comprehensive error handling and progress tracking.
"""
import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import json


class Colors:
    """ANSI color codes for terminal output"""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.ENDC}\n")


def print_step(step_num, total_steps, step_name):
    """Print step header"""
    print(
        f"\n{Colors.BOLD}{Colors.BLUE}Step {step_num}/{total_steps}: {step_name}{Colors.ENDC}"
    )
    print(f"{Colors.BLUE}{'-'*70}{Colors.ENDC}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}Success: {message}{Colors.ENDC}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}Error: {message}{Colors.ENDC}")


def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}Warning: {message}{Colors.ENDC}")


def print_info(message):
    """Print info message"""
    print(f"{Colors.CYAN}Info: {message}{Colors.ENDC}")


def run_command(cmd, step_name, show_output=True):
    """
    Run a command and capture output
    Returns: (success: bool, stdout: str, stderr: str, return_code: int)
    """
    start_time = time.time()

    print_info(f"Executing: {' '.join(cmd)}")
    print_info(f"Started at: {datetime.now().strftime('%H:%M:%S')}")

    try:
        if show_output:
            # Run with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            stdout_lines = []
            for line in process.stdout:
                print(line, end="")  # Print in real-time
                stdout_lines.append(line)

            process.wait()
            stdout = "".join(stdout_lines)
            stderr = ""
            return_code = process.returncode
        else:
            # Run silently and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            stdout = result.stdout
            stderr = result.stderr
            return_code = result.returncode

        elapsed = time.time() - start_time

        if return_code == 0:
            print_success(f"{step_name} completed successfully in {elapsed:.1f}s")
            return True, stdout, stderr, return_code
        else:
            print_error(f"{step_name} failed with return code {return_code}")
            if stderr and not show_output:
                print(f"\n{Colors.RED}Error output:{Colors.ENDC}")
                print(stderr)
            return False, stdout, stderr, return_code

    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        return False, "", "Command not found", -1
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False, "", str(e), -1


def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size / (1024 * 1024)  # MB
        print_success(f"{description} exists ({size:.2f} MB)")
        return True
    else:
        print_error(f"{description} not found: {filepath}")
        return False


def save_pipeline_log(log_data, output_dir):
    """Save pipeline execution log"""
    log_path = Path(output_dir) / "pipeline_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print_success(f"Pipeline log saved: {log_path}")


def check_step_completed(step_name, output_dir):
    """
    Check if a pipeline step has already been completed by verifying output files.
    Returns: (completed: bool, message: str)
    """
    output_dir = Path(output_dir)
    processed_dir = output_dir / "processed"
    dataset_dir = output_dir / "dataset"
    models_dir = output_dir / "models"

    if step_name == "preprocessing":
        # Check for preprocessing outputs
        required = [
            processed_dir / "combined_normalized.parquet",
            processed_dir / "combined_raw.parquet",
            processed_dir / "scaler.joblib",
            processed_dir / "feature_list.json",
        ]
        # Also accept CSV fallbacks
        if not required[0].exists():
            required[0] = processed_dir / "combined_normalized.csv"
        if not required[1].exists():
            required[1] = processed_dir / "combined_raw.csv"

        if all(f.exists() for f in required):
            return True, "All preprocessing outputs found"
        missing = [str(f.name) for f in required if not f.exists()]
        return False, f"Missing: {', '.join(missing)}"

    elif step_name == "dataset_creation":
        required = [
            dataset_dir / "train.npz",
            dataset_dir / "val.npz",
            dataset_dir / "test.npz",
        ]
        if all(f.exists() for f in required):
            return True, "All dataset files found"
        missing = [str(f.name) for f in required if not f.exists()]
        return False, f"Missing: {', '.join(missing)}"

    elif step_name == "model_training":
        # Check for any model file
        model_files = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.h5"))
        if model_files:
            return True, f"Model found: {model_files[0].name}"
        return False, "No model file found"

    return False, "Unknown step"


def print_skip_message(step_num, step_name, reason):
    """Print a formatted skip message"""
    print(f"\n{Colors.YELLOW}[SKIP]{Colors.ENDC} Step {step_num}: {step_name}")
    print(f"  {Colors.CYAN}Reason:{Colors.ENDC} {reason}")
    print(f"  {Colors.CYAN}Tip:{Colors.ENDC} Use --force to re-run this step")


def main(args):
    """Main pipeline execution"""
    start_time = time.time()

    # Pipeline configuration
    steps_total = 3
    log_data = {"start_time": datetime.now(), "steps": [], "status": "running"}

    print_header("ML ANOMALY DETECTION PIPELINE")
    print_info(f"Input directory: {args.input_dir}")
    print_info(f"Output directory: {args.output_dir}")
    print_info(f"Mode: {args.mode}")
    print_info(f"Python: {sys.executable}")

    # Show checkpoint options
    if args.force:
        print_warning("Force mode: All steps will be re-run")
    if args.skip_to:
        print_warning(f"Skipping to step {args.skip_to}")
    if args.force_step:
        print_warning(f"Will force re-run step {args.force_step}")

    # Create output directories
    output_dir = Path(args.output_dir)
    processed_dir = output_dir / "processed"
    dataset_dir = output_dir / "dataset"
    models_dir = output_dir / "models"

    for d in [output_dir, processed_dir, dataset_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # STEP 1: PREPROCESSING
    # ===================================================================
    step1_skipped = False
    should_run_step1 = True

    # Check if we should skip this step
    if args.skip_to and args.skip_to > 1:
        should_run_step1 = False
        print_skip_message(1, "Preprocessing", f"--skip-to {args.skip_to} specified")
        step1_skipped = True
    elif not args.force and args.force_step != 1:
        completed, msg = check_step_completed("preprocessing", output_dir)
        if completed:
            should_run_step1 = False
            print_skip_message(1, "Preprocessing", msg)
            step1_skipped = True

    if should_run_step1:
        print_step(1, steps_total, "Data Preprocessing & Normalization")

        preprocess_cmd = [
            sys.executable,
            "preprocess_normalize.py",
            "--input-dir",
            args.input_dir,
            "--out-dir",
            str(processed_dir),
            "--max-files",
            str(args.max_files),
            "--top-k-protocols",
            str(args.top_k_protocols),
        ]

        if args.max_rows_per_file:
            preprocess_cmd.extend(["--max-rows-per-file", str(args.max_rows_per_file)])

        success, stdout, stderr, code = run_command(
            preprocess_cmd, "Preprocessing", show_output=True
        )

        log_data["steps"].append(
            {
                "step": 1,
                "name": "preprocessing",
                "command": " ".join(preprocess_cmd),
                "success": success,
                "return_code": code,
            }
        )

        if not success:
            print_error("Pipeline failed at preprocessing step")
            log_data["status"] = "failed"
            log_data["failed_at"] = "preprocessing"
            save_pipeline_log(log_data, output_dir)
            return 1

        # Verify preprocessing outputs
        print("\nVerifying preprocessing outputs...")
        required_files = [
            (processed_dir / "combined_normalized.parquet", "Normalized features"),
            (processed_dir / "combined_raw.parquet", "Raw combined data"),
            (processed_dir / "scaler.joblib", "Scaler model"),
            (processed_dir / "feature_list.json", "Feature manifest"),
        ]

        # Check for CSV fallback
        if not (processed_dir / "combined_normalized.parquet").exists():
            required_files[0] = (
                processed_dir / "combined_normalized.csv",
                "Normalized features (CSV)",
            )
        if not (processed_dir / "combined_raw.parquet").exists():
            required_files[1] = (
                processed_dir / "combined_raw.csv",
                "Raw combined data (CSV)",
            )

        all_exist = all(check_file_exists(f, desc) for f, desc in required_files)

        if not all_exist:
            print_error("Preprocessing outputs incomplete")
            log_data["status"] = "failed"
            log_data["failed_at"] = "preprocessing_verification"
            save_pipeline_log(log_data, output_dir)
            return 1
    else:
        log_data["steps"].append(
            {
                "step": 1,
                "name": "preprocessing",
                "skipped": True,
                "reason": "Already completed",
            }
        )

    # ===================================================================
    # STEP 2: DATASET CREATION
    # ===================================================================
    step2_skipped = False
    should_run_step2 = True

    # Check if we should skip this step
    if args.skip_to and args.skip_to > 2:
        should_run_step2 = False
        print_skip_message(2, "Dataset Creation", f"--skip-to {args.skip_to} specified")
        step2_skipped = True
    elif not args.force and args.force_step != 2:
        completed, msg = check_step_completed("dataset_creation", output_dir)
        if completed:
            should_run_step2 = False
            print_skip_message(2, "Dataset Creation", msg)
            step2_skipped = True

    if should_run_step2:
        print_step(2, steps_total, "Dataset Creation (Train/Val/Test Split)")

        dataset_cmd = [
            sys.executable,
            "create_dataset.py",
            "--processed-dir",
            str(processed_dir),
            "--out-dir",
            str(dataset_dir),
            "--test-size",
            str(args.test_size),
            "--val-size",
            str(args.val_size),
        ]

        if args.time_based:
            dataset_cmd.append("--time-based")

        success, stdout, stderr, code = run_command(
            dataset_cmd, "Dataset creation", show_output=True
        )

        log_data["steps"].append(
            {
                "step": 2,
                "name": "dataset_creation",
                "command": " ".join(dataset_cmd),
                "success": success,
                "return_code": code,
            }
        )

        if not success:
            print_error("Pipeline failed at dataset creation step")
            log_data["status"] = "failed"
            log_data["failed_at"] = "dataset_creation"
            save_pipeline_log(log_data, output_dir)
            return 1

        # Verify dataset outputs
        print("\nVerifying dataset outputs...")
        dataset_files = [
            (dataset_dir / "train.npz", "Training set (NPZ)"),
            (dataset_dir / "val.npz", "Validation set (NPZ)"),
            (dataset_dir / "test.npz", "Test set (NPZ)"),
            (dataset_dir / "train.csv", "Training set (CSV)"),
            (dataset_dir / "val.csv", "Validation set (CSV)"),
            (dataset_dir / "test.csv", "Test set (CSV)"),
        ]

        all_exist = all(check_file_exists(f, desc) for f, desc in dataset_files)

        if not all_exist:
            print_error("Dataset outputs incomplete")
            log_data["status"] = "failed"
            log_data["failed_at"] = "dataset_verification"
            save_pipeline_log(log_data, output_dir)
            return 1
    else:
        log_data["steps"].append(
            {
                "step": 2,
                "name": "dataset_creation",
                "skipped": True,
                "reason": "Already completed",
            }
        )

    # ===================================================================
    # STEP 3: MODEL TRAINING
    # ===================================================================
    step3_skipped = False
    should_run_step3 = True

    # Check if we should skip this step
    if not args.force and args.force_step != 3:
        completed, msg = check_step_completed("model_training", output_dir)
        if completed:
            should_run_step3 = False
            print_skip_message(3, "Model Training", msg)
            step3_skipped = True

    if should_run_step3:
        print_step(3, steps_total, f"Model Training ({args.mode.upper()} mode)")

        model_filename = (
            "autoencoder.h5" if args.mode == "unsupervised" else "random_forest.joblib"
        )
        model_path = models_dir / model_filename

        train_cmd = [
            sys.executable,
            "train_model.py",
            "--dataset-dir",
            str(dataset_dir),
            "--mode",
            args.mode,
            "--model-out",
            str(model_path),
        ]

        # Add mode-specific parameters
        if args.mode == "unsupervised":
            train_cmd.extend(
                [
                    "--latent-dim",
                    str(args.latent_dim),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                ]
            )
            if args.train_on_benign:
                train_cmd.append("--train-on-benign")
        else:  # supervised
            train_cmd.extend(["--n-estimators", str(args.n_estimators)])

        success, stdout, stderr, code = run_command(
            train_cmd, "Model training", show_output=True
        )

        log_data["steps"].append(
            {
                "step": 3,
                "name": "model_training",
                "command": " ".join(train_cmd),
                "success": success,
                "return_code": code,
            }
        )

        if not success:
            print_error("Pipeline failed at model training step")
            log_data["status"] = "failed"
            log_data["failed_at"] = "model_training"
            save_pipeline_log(log_data, output_dir)
            return 1

        # Verify model outputs
        print("\nVerifying model outputs...")
        if check_file_exists(model_path, f"Trained model ({args.mode})"):
            print_success("Model training completed successfully")
        else:
            print_error("Model file not found")
            log_data["status"] = "failed"
            log_data["failed_at"] = "model_verification"
            save_pipeline_log(log_data, output_dir)
            return 1
    else:
        log_data["steps"].append(
            {
                "step": 3,
                "name": "model_training",
                "skipped": True,
                "reason": "Already completed",
            }
        )

    # ===================================================================
    # PIPELINE COMPLETE
    # ===================================================================
    elapsed_total = time.time() - start_time
    log_data["end_time"] = datetime.now()
    log_data["total_time_seconds"] = elapsed_total
    log_data["status"] = "completed"

    # Count skipped vs executed steps
    skipped_count = sum(1 for s in log_data["steps"] if s.get("skipped", False))
    executed_count = len(log_data["steps"]) - skipped_count

    save_pipeline_log(log_data, output_dir)

    print_header("PIPELINE COMPLETED SUCCESSFULLY")
    print_success(f"Total execution time: {elapsed_total/60:.1f} minutes")
    if skipped_count > 0:
        print_info(f"Steps executed: {executed_count}, Steps skipped: {skipped_count}")
    print_success(f"All outputs saved to: {output_dir}")

    print("\n" + Colors.BOLD + "Generated Files:" + Colors.ENDC)
    print(f"  Processed data: {processed_dir}")
    print(f"  Datasets: {dataset_dir}")
    print(f"  Models: {models_dir}")
    print(f"  Pipeline log: {output_dir / 'pipeline_log.json'}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete ML Anomaly Detection Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with default settings
  python run_pipeline.py --input-dir "original datasets" --output-dir "ml_output"
  
  # Run with memory-limited preprocessing
  python run_pipeline.py --input-dir "original datasets" --output-dir "ml_output" --max-rows-per-file 500000
  
  # Run supervised training
  python run_pipeline.py --input-dir "original datasets" --output-dir "ml_output" --mode supervised
  
  # Run with time-based splitting for time series
  python run_pipeline.py --input-dir "original datasets" --output-dir "ml_output" --time-based
        """,
    )

    # Input/Output
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing raw CSV files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Root directory for all outputs"
    )

    # Checkpoint/Skip options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run all steps even if outputs exist",
    )
    parser.add_argument(
        "--skip-to",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Skip to a specific step (1=preprocess, 2=dataset, 3=train)",
    )
    parser.add_argument(
        "--force-step",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Force re-run a specific step only",
    )

    # Preprocessing parameters
    parser.add_argument(
        "--max-files", type=int, default=200, help="Maximum CSV files to process"
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Limit rows per CSV (e.g., 500000 for memory management)",
    )
    parser.add_argument(
        "--top-k-protocols",
        type=int,
        default=10,
        help="Limit protocol one-hot encoding to top K",
    )

    # Dataset split parameters
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set proportion (0-1)"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.1, help="Validation set proportion (0-1)"
    )
    parser.add_argument(
        "--time-based",
        action="store_true",
        help="Use time-ordered splitting instead of random",
    )

    # Training mode
    parser.add_argument(
        "--mode",
        choices=["unsupervised", "supervised"],
        default="unsupervised",
        help="Training mode: unsupervised (autoencoder) or supervised (random forest)",
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
        exit_code = main(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_error("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nFatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
