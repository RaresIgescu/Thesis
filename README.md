# Intrusion Detection System using Deep Reinforcement Learning

A Deep Q-Network (DQN) based Intrusion Detection System trained and evaluated on the CICIDS2017 dataset. The agent is modeled as a Markov Decision Process operating on normalized network flow features, with an asymmetric reward function designed to penalize missed attacks more heavily than false alarms.

**Author:** Igescu Rareș-Andrei  
**Institution:** University of Bucharest, Faculty of Mathematics and Computer Science  
**Supervisor:** Conf. univ. dr. Ciprian Ionuț Păduraru

---

## Results

| Metric | Value |
| --- | --- |
| Overall Accuracy | 98.2% |
| Attack Recall | 96.1% |
| False Positive Rate | 1.3% |
| Training Timesteps | 500,000 |
| Test Set Size | 20% (held-out, never seen during training) |

---

## Project Structure

```text
IDS using RL/
│
├── code/
│   ├── data/
│   │   └── cicids2017_cleaned.csv       # CICIDS2017 dataset (place here)
│   │
│   ├── training/
│   │   └── main.py                      # Training script
│   │
│   ├── realtime/
│   │   ├── realTime.py                  # Real-time detection engine
│   │   ├── config.py                    # Configuration (loaded from .env)
│   │   ├── dashboard.py                 # Flask web dashboard
│   │   ├── captures/                    # CICFlowMeter CSV output folder
│   │   │   └── flows.csv
│   │   └── logs/
│   │       └── detections.csv           # Detection history log
│   │
│   └── models/
│       └── generated/
│           ├── ids_dqn_agent.zip        # Trained DQN model
│           └── scaler_ids.pkl           # Fitted MinMaxScaler
│
└── README.md
```

---

## Prerequisites

### 1. Python dependencies

```bash
pip install pandas numpy scikit-learn gymnasium stable-baselines3 matplotlib joblib colorama flask cicflowmeter python-dotenv
```

### 2. Npcap (required for live packet capture on Windows)

Download and install from [npcap.com](https://npcap.com) with default settings, then restart your terminal.

### 3. Dataset

Download the CICIDS2017 dataset and place the cleaned CSV at:

```text
code/data/cicids2017_cleaned.csv
```

The file must contain a column named `Attack Type` with `Normal Traffic` for benign flows and attack type names for malicious ones.

---

## Environment Configuration

The real-time detector reads configuration from a `.env` file. Create `code/realtime/.env` with the following content:

```env
IDS_WATCH_FOLDER=code/realtime/captures
IDS_MODEL_PATH=code/models/generated/ids_dqn_agent
IDS_SCALER_PATH=code/models/generated/scaler_ids.pkl
IDS_LOG_PATH=code/realtime/logs/detections.csv
IDS_DASHBOARD_HOST=127.0.0.1
IDS_DASHBOARD_PORT=5000
IDS_MAX_DASHBOARD_ROWS=200
IDS_POLL_INTERVAL_SEC=3

IDS_EMAIL_ADDRESS=your_email@gmail.com
IDS_EMAIL_PASSWORD=your_gmail_app_password
IDS_EMAIL_RECIPIENT=your_email@gmail.com
IDS_ALERT_COOLDOWN_SECONDS=60
```

**Gmail App Password setup:**
Go to [myaccount.google.com](https://myaccount.google.com) → Security → 2-Step Verification → App Passwords. Generate one for "Mail" and paste it as `IDS_EMAIL_PASSWORD`.

---

## Step 1 — Train the Model

Run from the project root:

```bash
cd "IDS using RL"
python code/training/main.py
```

This will:

- Load and preprocess the CICIDS2017 dataset
- Split data 80/20 into train/test sets (stratified by class)
- Fit MinMaxScaler on training data only, then transform test data separately
- Train the DQN agent for 500,000 timesteps
- Save the trained model to `code/models/generated/ids_dqn_agent.zip`
- Save the scaler to `code/models/generated/scaler_ids.pkl`
- Evaluate on the held-out test set and generate two report figures:
  - `ids_dqn_report.png` — training accuracy curve, reward curve, confusion matrix
  - `ids_dqn_metrics.png` — per-class precision, recall, F1

Training takes approximately 15–30 minutes depending on hardware.

### Agent configuration

| Parameter | Value | Reason |
| --- | --- | --- |
| Algorithm | DQN | Efficient for discrete action spaces |
| Network architecture | [256, 256, 128] | High-dimensional observation space |
| Learning rate | 0.001 | Standard for DQN |
| Discount factor (γ) | 0.95 | Stable medium-term convergence |
| Exploration fraction | 0.30 | Extensive early exploration |
| ε final | 0.05 | Maintains minimum exploration |
| Replay buffer size | 1,000,000 | High experience diversity |
| Target network sync | every 10,000 steps | Stable training targets |
| Steps per episode | 1,000 | Maximum instance variety |

### Reward function

| Outcome | Reward |
| --- | --- |
| Correct classification | +1.0 |
| Missed attack (False Negative) | -2.0 |
| False alarm (False Positive) | -1.3 |

The asymmetric penalties ensure the agent prioritizes attack detection while still penalizing false alarms meaningfully.

---

## Step 2 — Real-Time Detection

The real-time pipeline requires two processes running simultaneously in separate terminals.

### Terminal 1 — Start CICFlowMeter

CICFlowMeter captures live network traffic and writes flow features to a CSV file that the detector polls every 3 seconds.

```bash
cicflowmeter -i Ethernet -c "code/realtime/captures/flows.csv"
```

Replace `Ethernet` with your actual interface name. To find it:

```powershell
Get-NetAdapter | Select-Object Name, Status
```

CICFlowMeter writes flows incrementally as TCP connections close (via FIN/RST flags) or after a 120-second timeout per flow.

### Terminal 2 — Start the IDS detector

```bash
cd "IDS using RL"
python code/realtime/realTime.py
```

This starts three things simultaneously:

- **Terminal output** — colour-coded verdict for each flow (GREEN = NORMAL, RED = ATTACK)
- **Web dashboard** — live statistics and flow table at `http://127.0.0.1:5000`
- **Email alerts** — sends an email on first attack detection, then at most once every 60 seconds

---

## Step 3 — Testing the System

### Option A — Inject test flows via dashboard (recommended)

The dashboard exposes an `/inject` endpoint that writes 10 real attack flows from the CICIDS2017 test set directly into the monitored CSV, triggering the full detection pipeline without external tools.

Click **SIMULATE ATTACK** in the dashboard, or run from the terminal:

```bash
curl http://127.0.0.1:5000/inject
```

From another device on the same local network:

```bash
curl http://192.168.x.x:5000/inject
```

All 10 injected flows will be classified as ATTACK, the dashboard will update in real time, and an email alert will be sent automatically.

### Option B — Generate real traffic with nmap

```bash
nmap -sS -p 80 127.0.0.1 --max-rate 1000
```

> **Note on domain gap:** Traffic generated by tools on a home or lab network will have different statistical distributions from the CICIDS2017 dataset, which was captured in a controlled environment in 2017. The agent may classify real-world generated attacks as normal — this is a known limitation of IDS systems trained on public datasets and is discussed in detail in the thesis.

### Option C — Generate traffic with Python Scapy

```python
from scapy.all import *
send(IP(dst="127.0.0.1")/TCP(dport=80, flags="S"), count=1000, verbose=0)
```

---

## Dashboard

Access the web interface at `http://127.0.0.1:5000` while the detector is running.

The dashboard displays:
- Total flows analysed
- Normal flows count
- Attacks detected count
- Live attack rate percentage
- Table of the 50 most recent flows, colour-coded by verdict

The table auto-refreshes every 3 seconds via asynchronous polling.

---

## How It Works

```text
CICFlowMeter (live capture)
        |
        v
  captures/flows.csv   <-- polled every 3 seconds
        |
        v
  Feature alignment + MinMaxScaler normalization
        |
        v
  DQN agent prediction  (0 = NORMAL, 1 = ATTACK)
        |
   +----+--------+----------+
   v             v          v
Terminal       Log      Dashboard + Email alert
```

The detector tracks byte offsets in the CSV so it only processes new rows on each poll cycle. If CICFlowMeter restarts and recreates the file, the offset resets automatically and the file is re-read from the beginning.

---

## Key Design Decisions

**Biased episode reset** — 50% of training episodes start on attack samples, compensating for the 83% normal / 17% attack class imbalance in CICIDS2017.

**Train/test split before scaling** — The MinMaxScaler is fit exclusively on training data and applied to the test set without refitting, preventing leakage of test statistics into the training pipeline.

**Reproducibility** — All random sources are seeded with `SEED = 42`: NumPy global RNG, Gymnasium environment RNG via `self.np_random`, and the DQN agent via `model.set_random_seed()`.

**Email cooldown** — A 60-second cooldown between email alerts prevents inbox flooding during sustained attacks while ensuring the first detection is always reported immediately.

---

## Limitations

- **Binary classification only** — the agent distinguishes normal from attack but does not identify the attack type (DDoS, Brute Force, Web Attack, etc.)
- **Flow-level granularity** — flows are classified independently with no temporal context across consecutive flows, making slow distributed attacks harder to detect
- **CICFlowMeter latency** — flows are written only after they close, introducing a delay of several seconds to over a minute between an attack occurring and it being detected
- **Domain gap** — the agent is trained on CICIDS2017 (generated in a lab in 2017); real-world traffic from home or enterprise networks may produce different feature distributions for the same attack types

---

## Citation

If you use this work in your research, please cite it as:

```bibtex
@thesis{igescu2026ids,
  author      = {Igescu, Rareș-Andrei},
  title       = {Sistem de Detectare a Intruziunilor Utilizând Învățarea prin Recompensă},
  type        = {Bachelor's Thesis},
  institution = {University of Bucharest, Faculty of Mathematics and Computer Science},
  year        = {2026},
  supervisor  = {Conf. univ. dr. Ciprian Ionuț Păduraru},
}
```

---

## License

MIT License

Copyright (c) 2026 Igescu Rareș-Andrei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
