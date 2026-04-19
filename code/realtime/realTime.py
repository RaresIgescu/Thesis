"""
Real-time IDS detector — reads CICFlowMeter CSV output, runs the trained
DQN model, and fans the result out to terminal, log file, and web dashboard.

Pipeline:
    CICFlowMeter (running separately) --> CSV file in WATCH_FOLDER
                                                |
                                                v
    polling loop --> align features --> MinMaxScaler --> DQN.predict()
                                                |
                             +------------------+------------------+
                             v                  v                  v
                       Terminal print      logs/detections.csv   Flask @ :5000
"""

import os
import csv
import time
import threading
import collections
from io import StringIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from colorama import init as colorama_init, Fore, Style
from stable_baselines3 import DQN

colorama_init()

# ============================================================
# CONFIGURATION — edit these for your setup
# ============================================================

WATCH_FOLDER = Path(r"C:\Users\riges\Desktop\IDS using RL\code\realtime\captures")
MODEL_PATH   = "generated/ids_dqn_agent"
SCALER_PATH  = "generated/scaler_ids.pkl"
LOG_PATH     = "logs/detections.csv"
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5000
MAX_DASHBOARD_ROWS = 200
POLL_INTERVAL_SEC = 3

# ============================================================
# CANONICAL FEATURE ORDER (must exactly match the training CSV)
# ============================================================

CANONICAL_COLUMNS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
    'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size',
    'Subflow Fwd Bytes', 'Init_Win_bytes_forward',
    'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Max', 'Idle Min',
]

# Map snake_case names (Python CICFlowMeter port) to training-set names
COLUMN_ALIASES = {
    'dst_port': 'Destination Port',
    'flow_duration': 'Flow Duration',
    'tot_fwd_pkts': 'Total Fwd Packets',
    'totlen_fwd_pkts': 'Total Length of Fwd Packets',
    'fwd_pkt_len_max': 'Fwd Packet Length Max',
    'fwd_pkt_len_min': 'Fwd Packet Length Min',
    'fwd_pkt_len_mean': 'Fwd Packet Length Mean',
    'fwd_pkt_len_std': 'Fwd Packet Length Std',
    'bwd_pkt_len_max': 'Bwd Packet Length Max',
    'bwd_pkt_len_min': 'Bwd Packet Length Min',
    'bwd_pkt_len_mean': 'Bwd Packet Length Mean',
    'bwd_pkt_len_std': 'Bwd Packet Length Std',
    'flow_byts_s': 'Flow Bytes/s',
    'flow_pkts_s': 'Flow Packets/s',
    'flow_iat_mean': 'Flow IAT Mean',
    'flow_iat_std': 'Flow IAT Std',
    'flow_iat_max': 'Flow IAT Max',
    'flow_iat_min': 'Flow IAT Min',
    'fwd_iat_tot': 'Fwd IAT Total',
    'fwd_iat_mean': 'Fwd IAT Mean',
    'fwd_iat_std': 'Fwd IAT Std',
    'fwd_iat_max': 'Fwd IAT Max',
    'fwd_iat_min': 'Fwd IAT Min',
    'bwd_iat_tot': 'Bwd IAT Total',
    'bwd_iat_mean': 'Bwd IAT Mean',
    'bwd_iat_std': 'Bwd IAT Std',
    'bwd_iat_max': 'Bwd IAT Max',
    'bwd_iat_min': 'Bwd IAT Min',
    'fwd_header_len': 'Fwd Header Length',
    'bwd_header_len': 'Bwd Header Length',
    'fwd_pkts_s': 'Fwd Packets/s',
    'bwd_pkts_s': 'Bwd Packets/s',
    'pkt_len_min': 'Min Packet Length',
    'pkt_len_max': 'Max Packet Length',
    'pkt_len_mean': 'Packet Length Mean',
    'pkt_len_std': 'Packet Length Std',
    'pkt_len_var': 'Packet Length Variance',
    'fin_flag_cnt': 'FIN Flag Count',
    'psh_flag_cnt': 'PSH Flag Count',
    'ack_flag_cnt': 'ACK Flag Count',
    'pkt_size_avg': 'Average Packet Size',
    'subflow_fwd_byts': 'Subflow Fwd Bytes',
    'init_fwd_win_byts': 'Init_Win_bytes_forward',
    'init_bwd_win_byts': 'Init_Win_bytes_backward',
    'fwd_act_data_pkts': 'act_data_pkt_fwd',
    'fwd_seg_size_min': 'min_seg_size_forward',
    'active_mean': 'Active Mean',
    'active_max': 'Active Max',
    'active_min': 'Active Min',
    'idle_mean': 'Idle Mean',
    'idle_max': 'Idle Max',
    'idle_min': 'Idle Min',
}

# ============================================================
# SHARED STATE
# ============================================================

detection_buffer = collections.deque(maxlen=MAX_DASHBOARD_ROWS)
stats = {
    'total': 0, 'attacks': 0, 'normal': 0,
    'start_time': datetime.now().isoformat(timespec='seconds'),
}
stats_lock = threading.Lock()

# ============================================================
# FEATURE ALIGNMENT
# ============================================================

def align_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with exactly CANONICAL_COLUMNS in exact order, cleaned."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.rename(columns=COLUMN_ALIASES, inplace=True)

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    aligned = df[CANONICAL_COLUMNS].copy()
    aligned = aligned.apply(pd.to_numeric, errors='coerce')
    aligned.replace([np.inf, -np.inf], np.nan, inplace=True)
    aligned.fillna(0, inplace=True)
    return aligned


def extract_metadata(df: pd.DataFrame) -> list:
    """Pull per-flow metadata for display/logging only — not used for prediction."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    out = []
    for _, row in df.iterrows():
        out.append({
            'timestamp': str(row.get('timestamp', row.get('Timestamp', ''))),
            'src_ip':    str(row.get('src_ip', row.get('Src IP', 'N/A'))),
            'dst_ip':    str(row.get('dst_ip', row.get('Dst IP', 'N/A'))),
            'dst_port':  str(row.get('dst_port', row.get('Destination Port', 'N/A'))),
            'protocol':  str(row.get('protocol', row.get('Protocol', 'N/A'))),
        })
    return out


# ============================================================
# DISPATCH — terminal + log + dashboard
# ============================================================

def dispatch(meta: dict, pred: int, log_writer, log_file):
    label = 'ATTACK' if pred == 1 else 'NORMAL'
    color = Fore.RED if pred == 1 else Fore.GREEN
    ts = datetime.now().strftime('%H:%M:%S')

    print(f"[{ts}] {color}{Style.BRIGHT}{label:<6}{Style.RESET_ALL}  "
          f"{meta['src_ip']:>15} -> {meta['dst_ip']:<15}:{meta['dst_port']:<5}  "
          f"proto={meta['protocol']}")

    log_writer.writerow([
        ts, meta['timestamp'], meta['src_ip'], meta['dst_ip'],
        meta['dst_port'], meta['protocol'], label,
    ])
    log_file.flush()  # make sure writes reach disk immediately

    detection_buffer.append({
        'local_time': ts,
        'flow_time':  meta['timestamp'],
        'src_ip':     meta['src_ip'],
        'dst_ip':     meta['dst_ip'],
        'dst_port':   meta['dst_port'],
        'protocol':   meta['protocol'],
        'prediction': label,
    })

    with stats_lock:
        stats['total']  += 1
        stats['attacks' if pred == 1 else 'normal'] += 1


# ============================================================
# DASHBOARD
# ============================================================

DASHBOARD_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>IDS Dashboard</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 24px; background: #F5F4EF; color: #2C2C2A; }
  h1 { margin: 0 0 4px; font-weight: 500; font-size: 22px; }
  .sub { color: #888780; font-size: 13px; margin-bottom: 20px; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
  .card { background: white; padding: 14px 16px; border-radius: 10px;
          border-left: 3px solid #888780; }
  .card.attack { border-color: #D85A30; }
  .card.normal { border-color: #1D9E75; }
  .card.rate   { border-color: #185FA5; }
  .val { font-size: 26px; font-weight: 500; margin: 2px 0; }
  .lbl { font-size: 11px; text-transform: uppercase; color: #5F5E5A; letter-spacing: 0.5px; }
  table { width: 100%; background: white; border-collapse: collapse;
          border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  th, td { padding: 9px 14px; text-align: left; font-size: 13px; border-bottom: 1px solid #EFEEE8; }
  th { background: #FAFAF6; font-weight: 500; color: #5F5E5A; font-size: 11px;
       text-transform: uppercase; letter-spacing: 0.5px; }
  tbody tr:last-child td { border-bottom: none; }
  tr.attack td { color: #A32D2D; }
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
  tr.attack .badge { background: #FCEBEB; color: #A32D2D; }
  tr.normal .badge { background: #EAF3DE; color: #3B6D11; }
  .note { color: #888780; font-size: 12px; margin-top: 14px; }
  .empty { padding: 40px; text-align: center; color: #888780; }
</style></head><body>
<h1>IDS Real-Time Dashboard</h1>
<div class="sub">Started <span id="start"></span> · auto-refreshes every 2s</div>
<div class="stats">
  <div class="card"><div class="lbl">Total flows</div><div class="val" id="total">0</div></div>
  <div class="card normal"><div class="lbl">Normal</div><div class="val" id="normal">0</div></div>
  <div class="card attack"><div class="lbl">Attacks</div><div class="val" id="attacks">0</div></div>
  <div class="card rate"><div class="lbl">Attack rate</div><div class="val" id="rate">0%</div></div>
</div>
<table>
  <thead><tr><th>Time</th><th>Source</th><th>Destination</th><th>Port</th><th>Proto</th><th>Verdict</th></tr></thead>
  <tbody id="tbody"><tr><td class="empty" colspan="6">Waiting for flows…</td></tr></tbody>
</table>
<div class="note">Showing up to 50 most recent flows. Full history in logs/detections.csv.</div>
<script>
async function update() {
  try {
    const r = await fetch('/recent');
    const d = await r.json();
    document.getElementById('start').textContent = d.stats.start_time;
    document.getElementById('total').textContent = d.stats.total;
    document.getElementById('normal').textContent = d.stats.normal;
    document.getElementById('attacks').textContent = d.stats.attacks;
    const rate = d.stats.total ? (d.stats.attacks / d.stats.total * 100).toFixed(1) : '0.0';
    document.getElementById('rate').textContent = rate + '%';
    const tb = document.getElementById('tbody');
    if (!d.detections.length) return;
    tb.innerHTML = d.detections.map(x => `
      <tr class="${x.prediction === 'ATTACK' ? 'attack' : 'normal'}">
        <td>${x.local_time}</td><td>${x.src_ip}</td><td>${x.dst_ip}</td>
        <td>${x.dst_port}</td><td>${x.protocol}</td>
        <td><span class="badge">${x.prediction}</span></td>
      </tr>`).join('');
  } catch (e) { console.error(e); }
}
update(); setInterval(update, 2000);
</script></body></html>
"""


def start_dashboard():
    try:
        from flask import Flask, jsonify
    except ImportError:
        print(f"{Fore.YELLOW}[!] Flask not installed, dashboard disabled.{Style.RESET_ALL}")
        return

    app = Flask(__name__)
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    @app.route('/')
    def index():
        return DASHBOARD_HTML

    @app.route('/recent')
    def recent():
        with stats_lock:
            s = dict(stats)
        return jsonify({
            'detections': list(detection_buffer)[-50:][::-1],
            'stats': s,
        })

    thread = threading.Thread(
        target=lambda: app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT,
                               debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    print(f"{Fore.GREEN}[+] Dashboard at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}{Style.RESET_ALL}")


# ============================================================
# POLLING LOOP — reads any new bytes from every CSV in the watch folder
# ============================================================

def process_csv(path: str, offsets: dict, headers: dict,
                model, scaler, log_writer, log_file) -> int:
    """Read any new complete rows from `path`, run inference, dispatch.
    Returns the number of new rows processed."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return 0

    offset = offsets.get(path, 0)
    if size <= offset:
        return 0

    # Open in binary mode so byte offsets are exact, regardless of BOM/encoding
    with open(path, 'rb') as f:
        f.seek(offset)
        new_bytes = f.read()
        new_offset = f.tell()

    # Decode, stripping BOM on the first read only
    if offset == 0:
        text = new_bytes.decode('utf-8-sig', errors='replace')
    else:
        text = new_bytes.decode('utf-8', errors='replace')

    if offset == 0:
        # First read — parse the header line
        header_end = text.find('\n')
        if header_end == -1:
            return 0  # header not fully written yet
        headers[path] = [c.strip() for c in text[:header_end].strip().split(',')]
        text = text[header_end + 1:]

    # Only advance past the last COMPLETE line — avoids half-written rows
    last_newline = text.rfind('\n')
    if last_newline == -1:
        return 0
    complete = text[:last_newline + 1]

    # If there's an incomplete trailing line, back the offset off by its byte length
    trailing = text[last_newline + 1:]
    offsets[path] = new_offset - len(trailing.encode('utf-8'))

    try:
        df = pd.read_csv(
            StringIO(complete), header=None,
            names=headers[path], on_bad_lines='skip',
        )
    except Exception as e:
        print(f"{Fore.YELLOW}[!] CSV parse error on {path}: {e}{Style.RESET_ALL}")
        return 0

    if df.empty:
        return 0

    aligned  = align_features(df)
    scaled   = scaler.transform(aligned.values.astype(np.float32))
    preds, _ = model.predict(scaled, deterministic=True)

    for meta, pred in zip(extract_metadata(df), preds):
        dispatch(meta, int(pred), log_writer, log_file)

    return len(df)


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"{Fore.CYAN}{'=' * 68}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  DQN Intrusion Detection System — Real-Time Mode{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 68}{Style.RESET_ALL}")

    os.makedirs(os.path.dirname(LOG_PATH) or '.', exist_ok=True)
    log_exists = os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0
    log_file = open(LOG_PATH, 'a', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['local_time', 'flow_timestamp', 'src_ip',
                             'dst_ip', 'dst_port', 'protocol', 'prediction'])
        log_file.flush()

    print(f"{Fore.CYAN}[*] Loading model from {MODEL_PATH}{Style.RESET_ALL}")
    model = DQN.load(MODEL_PATH)
    print(f"{Fore.CYAN}[*] Loading scaler from {SCALER_PATH}{Style.RESET_ALL}")
    scaler = joblib.load(SCALER_PATH)
    print(f"{Fore.GREEN}[+] Model and scaler loaded.{Style.RESET_ALL}")

    start_dashboard()

    if not WATCH_FOLDER.exists():
        print(f"{Fore.YELLOW}[!] Watch folder does not exist, creating: {WATCH_FOLDER}{Style.RESET_ALL}")
        WATCH_FOLDER.mkdir(parents=True, exist_ok=True)

    print(f"{Fore.GREEN}[+] Polling {WATCH_FOLDER} every {POLL_INTERVAL_SEC}s{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-' * 68}{Style.RESET_ALL}")
    print(f"{'TIME':<10} {'VERDICT':<7}  {'SOURCE':>15} -> {'DEST':<15}:{'PORT':<5}  proto")
    print(f"{Fore.CYAN}{'-' * 68}{Style.RESET_ALL}")

    offsets = {}   # path -> last byte offset processed
    headers = {}   # path -> list of column names

    try:
        while True:
            for csv_path in WATCH_FOLDER.glob("*.csv"):
                path = str(csv_path)
                try:
                    process_csv(path, offsets, headers,
                                model, scaler, log_writer, log_file)
                except Exception as e:
                    print(f"{Fore.YELLOW}[!] Error on {path}: "
                          f"{type(e).__name__}: {e}{Style.RESET_ALL}")
            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[!] Shutdown requested…{Style.RESET_ALL}")
    finally:
        log_file.close()
        with stats_lock:
            s = dict(stats)
        print(f"{Fore.CYAN}{'-' * 68}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Summary:  total={s['total']}  "
              f"normal={s['normal']}  attacks={s['attacks']}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Log written to: {LOG_PATH}{Style.RESET_ALL}")


if __name__ == '__main__':
    main()