# ML-Based Anomaly Detection for SIEM/SOAR Integration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An ML-based network traffic anomaly detection system designed for integration with FortiSIEM/FortiSOAR environments. The system uses Random Forest classification to detect 36 different attack categories with **90.65% accuracy** and **0.9718 ROC-AUC**.

## 🎯 Project Overview

This project implements a complete ML pipeline for network intrusion detection:

- **Data Processing**: Handles 7.6M+ records from CIC-IDS-2017 and UNSW-NB15 datasets
- **Feature Engineering**: Extracts 16 features with protocol one-hot encoding
- **Model Training**: Random Forest classifier with optimized hyperparameters
- **SIEM Integration**: Ready for FortiSIEM/FortiSOAR deployment

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ENVIRONMENT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │   Endpoints  │   │   Network    │   │  FortiGate   │         │
│  │   + Sysmon   │   │   Devices    │   │   Firewall   │         │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘         │
│         │                  │                  │                  │
│         ▼                  ▼                  ▼                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                     FortiSIEM                         │       │
│  │            (Log Collection & Correlation)             │       │
│  └────────────────────────┬─────────────────────────────┘       │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────┐       │
│  │          ★ ML ANOMALY DETECTION ENGINE ★             │       │
│  │                                                       │       │
│  │   Preprocess → Normalize → Predict (RF Model)        │       │
│  │                                                       │       │
│  │   Accuracy: 90.65% | ROC-AUC: 0.9718                 │       │
│  └────────────────────────┬─────────────────────────────┘       │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                     FortiSOAR                         │       │
│  │    (Security Orchestration, Automation & Response)    │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Performance Metrics

| Metric             | Value     |
| ------------------ | --------- |
| Test Accuracy      | 90.65%    |
| ROC-AUC (Weighted) | 0.9718    |
| Training Samples   | 5,333,678 |
| Test Samples       | 1,523,909 |
| Attack Categories  | 36        |
| Features           | 16        |

### Feature Importance

| Rank | Feature        | Importance |
| ---- | -------------- | ---------- |
| 1    | flow_duration  | 47.89%     |
| 2    | byte_count     | 28.46%     |
| 3    | src_port       | 11.85%     |
| 4    | proto_tcp      | 4.30%      |
| 5    | proto_17 (UDP) | 3.92%      |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM recommended
- ~10GB disk space for datasets

### Installation

```bash
# Clone the repository
git clone https://github.com/Bendaoud-Bilal/ML-Based-Anomaly-Detection-project-for-SIEM-SOAR-Integration.git
cd ML-Based-Anomaly-Detection-project-for-SIEM-SOAR-Integration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Datasets

The original datasets are not included due to size. Download them from:

1. **CIC-IDS-2017**: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
2. **UNSW-NB15**: [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

Place the datasets in the `original datasets/` folder following this structure:

```
original datasets/
├── CIC-IDS-2017-CSVs/
│   └── MachineLearningCSV/
│       └── MachineLearningCVE/
│           ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│           ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
│           └── ... (other CSV files)
└── UNSW-NB15-CSVs/
    └── UNSW-NB15-CSVs/
        └── CSV Files/
            ├── UNSW-NB15_1.csv
            ├── UNSW-NB15_2.csv
            └── ... (other CSV files)
```

### Run the Pipeline

```bash
# Run the complete ML pipeline
python run_pipeline.py
```

Or run individual steps:

```bash
# Step 1: Preprocess and normalize data
python preprocess_normalize.py

# Step 2: Create train/val/test datasets
python create_dataset.py

# Step 3: Train the model
python train_model.py
```

## 📁 Project Structure

```
ML-Based-Anomaly-Detection/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── overview.md                  # Detailed project documentation
├── run_pipeline.py              # Main pipeline orchestrator
├── preprocess_normalize.py      # Data preprocessing & normalization
├── create_dataset.py            # Dataset splitting & preparation
├── train_model.py               # Model training & evaluation
├── ml_output/                   # Output directory (created by pipeline)
│   ├── processed/               # Processed data files
│   │   └── feature_list.json    # Feature metadata
│   ├── dataset/                 # Train/val/test splits
│   │   └── split_info.json      # Split metadata
│   ├── models/                  # Trained models
│   │   └── evaluation_metrics.json
│   └── pipeline_log.json        # Execution log
└── original datasets/           # Raw datasets (download separately)
```

## 📈 Datasets Used

### CIC-IDS-2017

- **Source**: Canadian Institute for Cybersecurity
- **Records**: ~2.8M flows
- **Attack Types**: DDoS, DoS, PortScan, Brute Force, Web Attacks, Infiltration, Bot

### UNSW-NB15

- **Source**: University of NSW, Australia
- **Records**: ~2.5M records
- **Attack Types**: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms

## 🔧 Model Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42
)
```

## 📋 Attack Categories Detected

| Category                   | Dataset      | Description                   |
| -------------------------- | ------------ | ----------------------------- |
| BENIGN                     | Both         | Normal traffic                |
| DDoS                       | CIC-IDS-2017 | Distributed Denial of Service |
| DoS Hulk                   | CIC-IDS-2017 | DoS using Hulk tool           |
| DoS slowloris              | CIC-IDS-2017 | Slowloris attack              |
| PortScan                   | CIC-IDS-2017 | Port scanning                 |
| FTP-Patator                | CIC-IDS-2017 | FTP brute force               |
| SSH-Patator                | CIC-IDS-2017 | SSH brute force               |
| Web Attack - XSS           | CIC-IDS-2017 | Cross-site scripting          |
| Web Attack - SQL Injection | CIC-IDS-2017 | SQL injection                 |
| Bot                        | CIC-IDS-2017 | Botnet traffic                |
| Exploits                   | UNSW-NB15    | Exploit attempts              |
| Fuzzers                    | UNSW-NB15    | Fuzzing attacks               |
| Reconnaissance             | UNSW-NB15    | Information gathering         |
| Backdoors                  | UNSW-NB15    | Backdoor access               |
| ...                        | ...          | (36 total categories)         |

## 🔌 SIEM Integration

### FortiSIEM Column Mapping

| ML Feature    | FortiSIEM Column     |
| ------------- | -------------------- |
| src_ip        | src_ip               |
| dst_ip        | dst_ip               |
| src_port      | (derived)            |
| dst_port      | (derived)            |
| protocol      | protocol             |
| flow_duration | (calculated)         |
| prediction    | siem_alerts.category |

### Example Integration Code

```python
import joblib

class AnomalyDetector:
    def __init__(self, model_path, scaler_path):
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = joblib.load(scaler_path)

    def predict(self, network_logs):
        X = self.preprocess(network_logs)
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
```

## 📝 Known Issues & Future Work

1. **Empty Labels**: ~29% of UNSW-NB15 data has empty labels
2. **Class Imbalance**: BENIGN traffic dominates (54%)
3. **Rare Attack Detection**: Some attack types have low recall

### Planned Improvements

- [ ] Implement SMOTE for class balancing
- [ ] Add deep learning models (LSTM, Transformer)
- [ ] Real-time streaming prediction API
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 References

- [CIC-IDS-2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [FortiSIEM Documentation](https://docs.fortinet.com/product/fortisiem)
- [FortiSOAR Documentation](https://docs.fortinet.com/product/fortisoar)

## 👤 Author

**Bendaoud Bilal**

---

_This project was developed as part of an internship focusing on AI-based cybersecurity tools and SIEM/SOAR integration._
