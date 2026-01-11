# Comprehensive Project Analysis: ML-Based Anomaly Detection for ENAFOR SIEM/SOAR Integration

## 1. Project Context

## Internship project

### 1.1 Project Positioning

| Theme                                        | Project Alignment                     | Implementation Evidence                                                                              |
| -------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **AI Monitoring, Control & Red-Teaming**     | Anomaly detection in deployed systems | ML model monitors network traffic for attack patterns; integrates with CALDERA for attack simulation |
| **AI-Assisted Cybersecurity Tools**          | AI-based threat detection             | Random Forest classifier detecting 36 attack categories with 90.65% accuracy                         |
| **Secure AI Deployment & Defense Hardening** | ML pipeline security                  | Memory-efficient processing, encoding fallbacks, robust error handling                               |

### 1.2 Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ENAFOR PRODUCTION ENVIRONMENT                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐                │
│  │ Windows Endpoints│   │ Network Devices │   │ FortiGate FW    │                │
│  │ + Sysmon Agents  │   │ Routers/Switches│   │                 │                │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘                │
│           │                     │                     │                          │
│           ▼                     ▼                     ▼                          │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │                         FortiSIEM                                     │       │
│  │              (Log Collection & Correlation)                           │       │
│  │  Tables: windows_event_logs, endpoint_security_logs, network_logs     │       │
│  └────────────────────────────────┬─────────────────────────────────────┘       │
│                                   │                                              │
│                                   ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │              ★ ML ANOMALY DETECTION ENGINE ★                         │       │
│  │                    (THIS SUB-PROJECT)                                 │       │
│  │                                                                       │       │
│  │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                │       │
│  │   │ Preprocess  │ → │  Normalize  │ → │   Predict   │                │       │
│  │   │ (Schema Map)│   │ (Scaling)   │   │ (RF Model)  │                │       │
│  │   └─────────────┘   └─────────────┘   └─────────────┘                │       │
│  │                                                                       │       │
│  │   Model: random_forest.joblib (535 MB)                               │       │
│  │   Accuracy: 90.65% | ROC-AUC: 0.9718                                 │       │
│  └────────────────────────────────┬─────────────────────────────────────┘       │
│                                   │                                              │
│                                   ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │                         FortiSOAR                                     │       │
│  │         (Security Orchestration, Automation & Response)               │       │
│  │              Receives: siem_alerts with ML predictions                │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset Source & Characteristics

### 2.1 Original Datasets Overview

| Dataset          | Type            | Source                               | Records       | Purpose                              |
| ---------------- | --------------- | ------------------------------------ | ------------- | ------------------------------------ |
| **CIC-IDS-2017** | Network Flow    | Canadian Institute for Cybersecurity | ~2.8M flows   | Modern intrusion detection benchmark |
| **UNSW-NB15**    | Network Traffic | University of NSW, Australia         | ~2.5M records | Diverse attack patterns              |

### 2.2 CIC-IDS-2017 Dataset Details

**Files Processed:**
| File | Day | Attack Type | Estimated Rows |
|------|-----|-------------|----------------|
| Monday-WorkingHours.pcap_ISCX.csv | Monday | BENIGN baseline | 529,918 |
| Tuesday-WorkingHours.pcap_ISCX.csv | Tuesday | FTP-Patator, SSH-Patator | ~450,000 |
| Wednesday-workingHours.pcap_ISCX.csv | Wednesday | DoS slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed | 692,703 |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | Thursday AM | Web Attack - Brute Force, XSS, SQL Injection | ~170,000 |
| Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv | Thursday PM | Infiltration | ~290,000 |
| Friday-WorkingHours-Morning.pcap_ISCX.csv | Friday AM | Bot | ~290,000 |
| Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv | Friday PM | PortScan | ~290,000 |
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv | Friday PM | DDoS | ~230,000 |

**Original CIC-IDS-2017 Schema (80+ features):**

```
Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol,
Timestamp, Flow Duration, Total Fwd Packets, Total Backward Packets,
Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max,
Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std,
Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean,
Bwd Packet Length Std, Flow Bytes/s, Flow Packets/s, Flow IAT Mean,
Flow IAT Std, Flow IAT Max, Flow IAT Min, Fwd IAT Total, Fwd IAT Mean,
... (80+ columns) ..., Label
```

### 2.3 UNSW-NB15 Dataset Details

**Files Processed:**
| File | Records | Content |
|------|---------|---------|
| UNSW-NB15_1.csv | 700,000 | Network flows batch 1 |
| UNSW-NB15_2.csv | 700,000 | Network flows batch 2 |
| UNSW-NB15_3.csv | 700,000 | Network flows batch 3 |
| UNSW-NB15_4.csv | 700,000 | Network flows batch 4 |
| UNSW_NB15_training-set.csv | 175,341 | **Skipped** (schema mismatch) |
| UNSW_NB15_testing-set.csv | 82,332 | **Skipped** (schema mismatch) |

**UNSW-NB15 Attack Categories:**
| Category | Description |
|----------|-------------|
| Fuzzers | Feeding random data to discover vulnerabilities |
| Analysis | Port scans, spam, HTML file penetrations |
| Backdoors | Bypassing security mechanisms |
| DoS | Denial of Service |
| Exploits | Exploiting known vulnerabilities |
| Generic | Hash function collision attacks |
| Reconnaissance | Information gathering before attacks |
| Shellcode | Code injection |
| Worms | Self-replicating malware |

---

## 3. Data Processing Pipeline

### 3.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DATA PROCESSING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║                    PHASE 1: CHUNKED DATA LOADING                          ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
│  ┌─────────────────┐                                                            │
│  │ original datasets│──┐                                                         │
│  │   26 CSV files   │  │                                                         │
│  └─────────────────┘  │                                                         │
│                       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        BATCH PROCESSING (5 files/batch)                  │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │ For each CSV file:                                               │    │    │
│  │  │  1. Try encoding: UTF-8 → Latin-1 → ISO-8859-1 → CP1252          │    │    │
│  │  │  2. Skip malformed lines (on_bad_lines='skip')                   │    │    │
│  │  │  3. Limit to max_rows_per_file (500,000)                         │    │    │
│  │  │  4. Make column names unique (handle duplicates)                 │    │    │
│  │  │  5. Map columns to UNIFIED_COLS schema                           │    │    │
│  │  │  6. Parse timestamps, convert numeric types                      │    │    │
│  │  │  7. Clean IP addresses (convert to strings)                      │    │    │
│  │  │  8. Normalize protocol names                                     │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                              │                                           │    │
│  │                              ▼                                           │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │ Save batch as Parquet chunk → chunks/chunk_XXX.parquet           │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║                    PHASE 2: MERGE & NORMALIZE                             ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │ chunk_000.parquet│    │ chunk_001.parquet│    │ chunk_004.parquet│              │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘              │
│           └──────────────────────┼──────────────────────┘                       │
│                                  ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         pd.concat() - Merge All Chunks                   │    │
│  │                              7,619,541 total rows                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                  │                                               │
│                                  ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    DATA CLEANING & VALIDATION                            │    │
│  │  • Fill missing numeric values with 0                                    │    │
│  │  • Replace infinity values with NaN → 0                                  │    │
│  │  • Clip extreme values to ±1e15                                          │    │
│  │  • Normalize IP addresses (string format)                                │    │
│  │  • Sort by timestamp                                                     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                  │                                               │
│                                  ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      FEATURE ENGINEERING                                 │    │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │    │
│  │  │ Numeric Features (5):                                              │  │    │
│  │  │   flow_duration, packet_count, byte_count, src_port, dst_port     │  │    │
│  │  └───────────────────────────────────────────────────────────────────┘  │    │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │    │
│  │  │ Protocol One-Hot Encoding (Top-10 → 11 columns):                   │  │    │
│  │  │   proto_tcp, proto_udp, proto_6, proto_17, proto_ospf,            │  │    │
│  │  │   proto_sctp, proto_unas, proto_6.0, proto_17.0, proto_, OTHER    │  │    │
│  │  └───────────────────────────────────────────────────────────────────┘  │    │
│  │                         TOTAL: 16 features                              │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                  │                                               │
│                                  ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      STANDARD SCALING                                    │    │
│  │  scaler = StandardScaler()                                               │    │
│  │  X_normalized = scaler.fit_transform(X)                                  │    │
│  │  Mean = 0, Std = 1 for all features                                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                  │                                               │
│                                  ▼                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ combined_raw    │  │combined_normalized│  │ scaler.joblib  │                  │
│  │  .parquet       │  │    .parquet      │  │                 │                  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Schema Normalization (Unified Schema)

**Column Mapping Strategy:**

| Unified Column  | CIC-IDS-2017 Mapping                       | UNSW-NB15 Mapping   |
| --------------- | ------------------------------------------ | ------------------- |
| `timestamp`     | Timestamp, Flow Start                      | timestamp_utc, time |
| `src_ip`        | Source IP                                  | srcip               |
| `dst_ip`        | Destination IP                             | dstip               |
| `src_port`      | Source Port                                | sport               |
| `dst_port`      | Destination Port                           | dsport              |
| `protocol`      | Protocol                                   | proto               |
| `flow_duration` | Flow Duration                              | dur                 |
| `packet_count`  | Total Fwd Packets + Total Backward Packets | Spkts + Dpkts       |
| `byte_count`    | Total Length of Fwd Packets                | sbytes + dbytes     |
| `label`         | Label                                      | attack_cat          |

### 3.3 Data Transformation Steps

```python
# Step 1: Encoding Fallback Chain
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

# Step 2: Numeric Conversion with Infinity Handling
df[col] = pd.to_numeric(df[col], errors='coerce')
df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
df[col] = df[col].clip(-1e15, 1e15)

# Step 3: Protocol One-Hot Encoding (Top-K)
top_k = 10
proto_counts = combined["protocol"].value_counts().nlargest(top_k)
protocol_dummies = pd.get_dummies(combined["protocol_mapped"], prefix="proto")

# Step 4: Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 4. Detailed Dataset Statistics

### 4.1 Processing Summary

| Metric                              | Value          |
| ----------------------------------- | -------------- |
| **Input CSV Files**                 | 26             |
| **Files Successfully Processed**    | 24             |
| **Files Skipped (Schema Mismatch)** | 2              |
| **Total Rows Processed**            | 7,619,541      |
| **Final Feature Count**             | 16             |
| **Parquet Chunks Created**          | 5              |
| **Processing Time**                 | ~15-20 minutes |
| **Memory Peak Usage**               | ~840 MB        |

### 4.2 Data Split Statistics

| Split          | Samples   | Percentage | Purpose               |
| -------------- | --------- | ---------- | --------------------- |
| **Training**   | 5,333,678 | 70%        | Model training        |
| **Validation** | 761,954   | 10%        | Hyperparameter tuning |
| **Test**       | 1,523,909 | 20%        | Final evaluation      |
| **Total**      | 7,619,541 | 100%       | -                     |

### 4.3 Label Distribution Analysis

#### 4.3.1 Attack vs Benign Distribution

| Category               | Training Count | Training % | Test Count | Test % |
| ---------------------- | -------------- | ---------- | ---------- | ------ |
| **BENIGN**             | 2,884,808      | 54.10%     | 822,829    | 54.00% |
| **ATTACK (All Types)** | 1,888,867      | 35.42%     | 571,131    | 37.48% |
| **UNLABELED**          | 1,558,713      | 29.22%     | 446,729    | 29.32% |
| **Noise/Invalid**      | 47             | <0.01%     | 12         | <0.01% |

#### 4.3.2 Detailed Attack Category Distribution (Training Set)

| Rank | Attack Category            | Count     | Percentage | Dataset Source |
| ---- | -------------------------- | --------- | ---------- | -------------- |
| 1    | BENIGN                     | 2,884,808 | 54.10%     | Both           |
| 2    | (Empty - UNSW unlabeled)   | 1,558,713 | 29.22%     | UNSW-NB15      |
| 3    | DoS Hulk                   | 323,609   | 6.07%      | CIC-IDS-2017   |
| 4    | PortScan                   | 222,558   | 4.17%      | CIC-IDS-2017   |
| 5    | DDoS                       | 179,238   | 3.36%      | CIC-IDS-2017   |
| 6    | Exploits                   | 47,629    | 0.89%      | UNSW-NB15      |
| 7    | Fuzzers                    | 23,540    | 0.44%      | UNSW-NB15      |
| 8    | DoS                        | 17,205    | 0.32%      | UNSW-NB15      |
| 9    | Reconnaissance             | 14,187    | 0.27%      | UNSW-NB15      |
| 10   | Generic                    | 13,883    | 0.26%      | UNSW-NB15      |
| 11   | FTP-Patator                | 11,172    | 0.21%      | CIC-IDS-2017   |
| 12   | SSH-Patator                | 8,316     | 0.16%      | CIC-IDS-2017   |
| 13   | DoS slowloris              | 8,111     | 0.15%      | CIC-IDS-2017   |
| 14   | DoS Slowhttptest           | 7,691     | 0.14%      | CIC-IDS-2017   |
| 15   | Backdoor                   | 2,875     | 0.05%      | UNSW-NB15      |
| 16   | Bot                        | 2,723     | 0.05%      | CIC-IDS-2017   |
| 17   | DoS GoldenEye              | 1,582     | 0.03%      | CIC-IDS-2017   |
| 18   | Analysis                   | 1,307     | 0.02%      | UNSW-NB15      |
| 19   | Shellcode                  | 1,102     | 0.02%      | UNSW-NB15      |
| 20   | Web Attack - Brute Force   | 2,111     | 0.04%      | CIC-IDS-2017   |
| 21   | Web Attack - XSS           | 900       | 0.02%      | CIC-IDS-2017   |
| 22   | Backdoors                  | 180       | <0.01%     | UNSW-NB15      |
| 23   | Worms                      | 118       | <0.01%     | UNSW-NB15      |
| 24   | Infiltration               | 56        | <0.01%     | CIC-IDS-2017   |
| 25   | Web Attack - SQL Injection | 29        | <0.01%     | CIC-IDS-2017   |

#### 4.3.3 Class Imbalance Visualization

```
                            LABEL DISTRIBUTION (Training Set)

BENIGN          ████████████████████████████████████████████████████████ 54.1%
(Empty)         ██████████████████████████████                           29.2%
DoS Hulk        ██████                                                    6.1%
PortScan        ████                                                      4.2%
DDoS            ███                                                       3.4%
Exploits        █                                                         0.9%
Fuzzers         ▌                                                         0.4%
DoS             ▌                                                         0.3%
Reconnaissance  ▌                                                         0.3%
Generic         ▌                                                         0.3%
FTP-Patator     ▌                                                         0.2%
Others (<0.2%)  ▌                                                         0.4%
                ├────────────────────────────────────────────────────────┤
                0%                                                      60%
```

### 4.4 Feature Statistics

#### 4.4.1 Feature Inventory

| Feature Name    | Type    | Description                              |
| --------------- | ------- | ---------------------------------------- |
| `flow_duration` | Numeric | Duration of network flow in microseconds |
| `packet_count`  | Numeric | Total packets in flow                    |
| `byte_count`    | Numeric | Total bytes transferred                  |
| `src_port`      | Numeric | Source port number (0-65535)             |
| `dst_port`      | Numeric | Destination port number (0-65535)        |
| `proto_`        | Binary  | Empty protocol indicator                 |
| `proto_17`      | Binary  | UDP protocol (IP protocol 17)            |
| `proto_17.0`    | Binary  | UDP protocol variant                     |
| `proto_6`       | Binary  | TCP protocol (IP protocol 6)             |
| `proto_6.0`     | Binary  | TCP protocol variant                     |
| `proto_OTHER`   | Binary  | Other/rare protocols                     |
| `proto_ospf`    | Binary  | OSPF routing protocol                    |
| `proto_sctp`    | Binary  | SCTP protocol                            |
| `proto_tcp`     | Binary  | TCP protocol (named)                     |
| `proto_udp`     | Binary  | UDP protocol (named)                     |
| `proto_unas`    | Binary  | Unassigned protocol                      |

#### 4.4.2 Protocol Distribution

| Protocol     | Count      | Percentage |
| ------------ | ---------- | ---------- |
| TCP (6/tcp)  | ~5,200,000 | ~68.2%     |
| UDP (17/udp) | ~2,100,000 | ~27.6%     |
| OSPF         | ~150,000   | ~2.0%      |
| SCTP         | ~80,000    | ~1.0%      |
| Other        | ~90,000    | ~1.2%      |

### 4.5 Output Files Manifest

| File Path                                         | Size    | Format  | Content                                |
| ------------------------------------------------- | ------- | ------- | -------------------------------------- |
| `ml_output/processed/combined_raw.parquet`        | ~500 MB | Parquet | Raw merged data with all columns       |
| `ml_output/processed/combined_normalized.parquet` | ~300 MB | Parquet | Scaled feature matrix (16 features)    |
| `ml_output/processed/scaler.joblib`               | ~2 KB   | Joblib  | Fitted StandardScaler object           |
| `ml_output/processed/feature_list.json`           | ~2 KB   | JSON    | Feature metadata and column mapping    |
| `ml_output/processed/chunks/chunk_000.parquet`    | ~100 MB | Parquet | Batch 1 intermediate                   |
| `ml_output/processed/chunks/chunk_001.parquet`    | ~100 MB | Parquet | Batch 2 intermediate                   |
| `ml_output/processed/chunks/chunk_002.parquet`    | ~100 MB | Parquet | Batch 3 intermediate                   |
| `ml_output/processed/chunks/chunk_003.parquet`    | ~100 MB | Parquet | Batch 4 intermediate                   |
| `ml_output/processed/chunks/chunk_004.parquet`    | ~100 MB | Parquet | Batch 5 intermediate                   |
| `ml_output/dataset/train.npz`                     | ~400 MB | NPZ     | Training features + labels             |
| `ml_output/dataset/val.npz`                       | ~60 MB  | NPZ     | Validation features + labels           |
| `ml_output/dataset/test.npz`                      | ~120 MB | NPZ     | Test features + labels                 |
| `ml_output/dataset/train.csv`                     | ~800 MB | CSV     | Training features (human-readable)     |
| `ml_output/dataset/val.csv`                       | ~120 MB | CSV     | Validation features                    |
| `ml_output/dataset/test.csv`                      | ~240 MB | CSV     | Test features                          |
| `ml_output/dataset/train_labels.csv`              | ~50 MB  | CSV     | Training labels                        |
| `ml_output/dataset/val_labels.csv`                | ~7 MB   | CSV     | Validation labels                      |
| `ml_output/dataset/test_labels.csv`               | ~15 MB  | CSV     | Test labels                            |
| `ml_output/dataset/split_info.json`               | ~3 KB   | JSON    | Split metadata and label distributions |
| `ml_output/models/random_forest.joblib`           | 535 MB  | Joblib  | Trained Random Forest model            |
| `ml_output/models/evaluation_metrics.json`        | ~2 KB   | JSON    | Model performance metrics              |
| `ml_output/pipeline_log.json`                     | ~1 KB   | JSON    | Pipeline execution log                 |

---

## 5. Model Training Results

### 5.1 Model Configuration

| Parameter                | Value                    |
| ------------------------ | ------------------------ |
| **Algorithm**            | Random Forest Classifier |
| **Number of Estimators** | 200 trees                |
| **Max Depth**            | 20                       |
| **Min Samples Split**    | 10                       |
| **Parallelization**      | 4 workers (n_jobs=-1)    |
| **Random State**         | 42                       |

### 5.2 Performance Metrics

| Metric                 | Value        | Interpretation                          |
| ---------------------- | ------------ | --------------------------------------- |
| **Test Accuracy**      | 90.65%       | Correct predictions / Total predictions |
| **ROC-AUC (Weighted)** | 0.9718       | Excellent discrimination ability        |
| **Training Time**      | 21.3 minutes | On 5.3M samples                         |
| **Inference Time**     | 48.6 seconds | On 1.5M test samples                    |
| **Model Size**         | 535 MB       | Stored as joblib                        |

### 5.3 Feature Importance Analysis

| Rank | Feature                     | Importance | Description                                   |
| ---- | --------------------------- | ---------- | --------------------------------------------- |
| 1    | `flow_duration` (Feature 0) | 0.4789     | **Most important** - Attack duration patterns |
| 2    | `byte_count` (Feature 2)    | 0.2846     | Data volume transferred                       |
| 3    | `src_port` (Feature 5)      | 0.1185     | Source port behavior                          |
| 4    | `proto_tcp` (Feature 8)     | 0.0430     | TCP protocol indicator                        |
| 5    | `proto_17` (Feature 6)      | 0.0392     | UDP protocol indicator                        |
| 6    | `proto_OTHER` (Feature 13)  | 0.0100     | Rare protocols                                |
| 7    | `proto_6` (Feature 10)      | 0.0080     | TCP variant                                   |
| 8    | `proto_udp` (Feature 14)    | 0.0052     | UDP variant                                   |
| 9    | `proto_ospf` (Feature 9)    | 0.0050     | OSPF routing                                  |
| 10   | `proto_unas` (Feature 15)   | 0.0034     | Unassigned                                    |

**Key Insight:** Flow duration and byte count together account for **76.35%** of model decisions, indicating that temporal and volumetric features are most discriminative for attack detection.

### 5.4 Per-Class Performance (Test Set)

| Class ID | Label    | Precision | Recall | F1-Score | Support |
| -------- | -------- | --------- | ------ | -------- | ------- |
| 0        | (Empty)  | 0.90      | 1.00   | 0.95     | 446,729 |
| 2        | BENIGN   | 0.93      | 0.95   | 0.94     | 822,829 |
| 7        | DDoS     | 0.87      | 0.51   | 0.64     | 51,199  |
| 10       | DoS Hulk | 0.85      | 0.65   | 0.74     | 92,669  |
| 13       | Exploits | 0.46      | 0.99   | 0.63     | 13,754  |
| 17       | Generic  | 0.53      | 0.71   | 0.61     | 4,033   |
| 20       | PortScan | 0.98      | 0.84   | 0.90     | 63,396  |

**Weighted Average:** Precision=0.90, Recall=0.91, F1=0.90

---

## 6. Data Quality Issues Identified

### 6.1 Issues and Mitigation

| Issue                  | Severity | Count/Impact           | Mitigation Applied                |
| ---------------------- | -------- | ---------------------- | --------------------------------- |
| **Empty Labels**       | High     | 1,558,713 rows (20.5%) | Treated as separate class         |
| **Encoding Errors**    | Medium   | 2 files skipped        | Multi-encoding fallback chain     |
| **Duplicate Labels**   | Medium   | 6 label variants       | Character normalization (TODO)    |
| **Garbage Labels**     | Low      | 47 rows                | Left as separate classes          |
| **Mixed Type Columns** | Medium   | IP columns             | Forced string conversion          |
| **Infinity Values**    | Medium   | Unknown count          | Replaced with 0, clipped to ±1e15 |
| **Malformed CSV Rows** | Low      | Skipped automatically  | `on_bad_lines='skip'`             |

### 6.2 Skipped Files

| File                         | Reason                                |
| ---------------------------- | ------------------------------------- |
| `UNSW_NB15_testing-set.csv`  | Schema mismatch - DataFrame.str error |
| `UNSW_NB15_training-set.csv` | Schema mismatch - DataFrame.str error |

---

## 7. FortiSIEM/FortiSOAR Integration Mapping

### 7.1 Feature Mapping to ENAFOR Tables

| ML Feature           | FortiSIEM Table | FortiSIEM Column | Notes                     |
| -------------------- | --------------- | ---------------- | ------------------------- |
| `src_ip`             | `network_logs`  | `src_ip`         | Direct mapping            |
| `dst_ip`             | `network_logs`  | `dst_ip`         | Direct mapping            |
| `src_port`           | `network_logs`  | (derived)        | Need to add to schema     |
| `dst_port`           | `network_logs`  | (derived)        | Need to add to schema     |
| `protocol`           | `network_logs`  | `protocol`       | Direct mapping            |
| `flow_duration`      | (derived)       | -                | Calculate from timestamps |
| `packet_count`       | (derived)       | -                | Aggregate from raw logs   |
| `byte_count`         | (derived)       | -                | Aggregate from raw logs   |
| `label` (prediction) | `siem_alerts`   | `category`       | ML output                 |

### 7.2 Production Integration Code Template

```python
# integration_bridge.py - FortiSIEM to ML Model Bridge

import joblib
import pandas as pd
import numpy as np

class AnomalyDetector:
    def __init__(self, model_path, scaler_path):
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = joblib.load(scaler_path)

    def preprocess(self, fortisiem_logs):
        """Transform FortiSIEM network_logs to ML features"""
        df = pd.DataFrame(fortisiem_logs)

        # Map FortiSIEM columns to unified schema
        df['flow_duration'] = self._calculate_duration(df)
        df['packet_count'] = df.get('packets', 0)
        df['byte_count'] = df.get('bytes', 0)

        # Protocol one-hot encoding (must match training)
        protocol_dummies = pd.get_dummies(df['protocol'], prefix='proto')

        # Combine features
        features = pd.concat([
            df[['flow_duration', 'packet_count', 'byte_count', 'src_port', 'dst_port']],
            protocol_dummies
        ], axis=1).fillna(0)

        # Scale
        return self.scaler.transform(features)

    def predict(self, fortisiem_logs):
        """Predict attack categories"""
        X = self.preprocess(fortisiem_logs)
        predictions = self.model.predict(X)
        labels = self.label_encoder.inverse_transform(predictions)

        # Confidence scores
        probas = self.model.predict_proba(X)
        confidences = np.max(probas, axis=1)

        return [
            {'category': label, 'confidence': float(conf)}
            for label, conf in zip(labels, confidences)
        ]
```

---

## 8. Summary Statistics Table (For Report)

### Table 8.1: Dataset Overview

| Statistic                | Value                       |
| ------------------------ | --------------------------- |
| Total records processed  | 7,619,541                   |
| Source datasets          | 2 (CIC-IDS-2017, UNSW-NB15) |
| Input CSV files          | 26                          |
| Successfully processed   | 24 (92.3%)                  |
| Skipped files            | 2 (7.7%)                    |
| Unique attack categories | 36                          |
| Final feature dimensions | 16                          |

### Table 8.2: Class Distribution Summary

| Class Type     | Count     | Percentage |
| -------------- | --------- | ---------- |
| Benign traffic | 3,707,637 | 48.7%      |
| Attack traffic | 1,906,461 | 25.0%      |
| Unlabeled      | 2,005,443 | 26.3%      |

### Table 8.3: Model Performance Summary

| Metric   | Training  | Validation | Test      |
| -------- | --------- | ---------- | --------- |
| Samples  | 5,333,678 | 761,954    | 1,523,909 |
| Accuracy | -         | -          | 90.65%    |
| ROC-AUC  | -         | -          | 0.9718    |

### Table 8.4: Processing Performance

| Metric              | Value        |
| ------------------- | ------------ |
| Total pipeline time | 24.1 minutes |
| Preprocessing time  | ~15 minutes  |
| Model training time | 21.3 minutes |
| Peak memory usage   | 840 MB       |
| Model file size     | 535 MB       |

---

## 9. Critical Issues to Address

### 9.1 Empty Label Problem (29.2% of data!)

```
"": 1,558,713 (29.2%)
```

Almost **1.6 million rows** have empty/missing labels. This is a significant data quality issue from the UNSW-NB15 dataset. The model learned to classify these as a separate class (class 0 with 100% recall).

**Impact:**

- Model treats empty labels as a valid class
- Inflates accuracy metrics artificially
- Reduces meaningful attack detection capability

**Recommended Fix:**

- Option A: Remove rows with empty labels before training
- Option B: Map empty labels to "UNKNOWN" or "BENIGN" based on domain knowledge
- Option C: Use only CIC-IDS-2017 dataset (fully labeled)

### 9.2 Duplicate Label Variants (Encoding Issues)

The `�` character is a replacement character from encoding issues. These should be merged:

| Current Duplicate Labels                                    | Should Be Merged To          |
| ----------------------------------------------------------- | ---------------------------- |
| `Web Attack - Brute Force` + `Web Attack � Brute Force`     | `Web Attack - Brute Force`   |
| `Web Attack - XSS` + `Web Attack � XSS`                     | `Web Attack - XSS`           |
| `Web Attack - Sql Injection` + `Web Attack � Sql Injection` | `Web Attack - Sql Injection` |

**Impact:**

- Same attack type counted as different classes
- Model learns redundant patterns
- Reduces sample size per class, hurting minority class detection

**Recommended Fix:**

```python
# Add to label cleaning in create_dataset.py
labels = labels.str.replace('�', '-', regex=False)
labels = labels.str.replace('\ufffd', '-', regex=False)
```

### 9.3 Garbage Labels (Data Type Leakage from UNSW)

```
Float: 8
Integer: 6
integer: 14
nominal: 4
Binary: 1
Timestamp: 1
normal: 1
```

These are **column type names**, not attack labels! They leaked from the UNSW dataset headers or metadata rows.

**Impact:**

- Noise in the label space
- Model wastes capacity on meaningless classes
- Confuses evaluation metrics

### 9.4 Poor Recall on Many Attack Types

| Attack Type      | Recall | Issue             |
| ---------------- | ------ | ----------------- |
| DoS Slowhttptest | 1%     | Almost all missed |
| DoS slowloris    | 1%     | Almost all missed |
| SSH-Patator      | 0%     | Completely missed |
| Reconnaissance   | 0%     | Completely missed |
| Fuzzers          | 0%     | Completely missed |
| Analysis         | 0%     | Completely missed |
| Shellcode        | 0%     | Completely missed |
| Backdoors        | 0%     | Completely missed |
| Worms            | 0%     | Completely missed |

**Impact:**

- Critical security threats go undetected
- High false negative rate for rare attacks
- Model biased toward majority classes

**Root Causes:**

1. Severe class imbalance (BENIGN dominates)
2. Insufficient samples for rare attack types
3. Random Forest default settings favor majority class

---

## 10. Recommended Fixes Summary

| Issue            | Priority | Fix                | Effort |
| ---------------- | -------- | ------------------ | ------ |
| Empty labels     | High     | Filter or remap    | Low    |
| Duplicate labels | Medium   | String replacement | Low    |
| Garbage labels   | Low      | Filter rows        | Low    |
| Poor recall      | High     | Class balancing    | Medium |
| Skipped files    | Medium   | Fix schema mapping | Medium |

---
