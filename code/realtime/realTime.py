import os
import csv
import time
import threading
import collections
import smtplib
from io import StringIO
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import joblib
from colorama import init as colorama_init, Fore, Style
from stable_baselines3 import DQN
from dashboard import start_dashboard
from config import (
    WATCH_FOLDER, MODEL_PATH, SCALER_PATH, LOG_PATH,
    DASHBOARD_HOST, DASHBOARD_PORT, MAX_DASHBOARD_ROWS, POLL_INTERVAL_SEC,
    EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_RECIPIENT, ALERT_COOLDOWN_SECONDS,
    CANONICAL_COLUMNS, COLUMN_ALIASES,
)

colorama_init()

# ============================================================
# SHARED STATE
# ============================================================

detection_buffer = collections.deque(maxlen=MAX_DASHBOARD_ROWS)
stats = {
    'total': 0, 'attacks': 0, 'normal': 0,
    'start_time': datetime.now().isoformat(timespec='seconds'),
}
stats_lock = threading.Lock()
last_alert_time = 0.0

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
# EMAIL ALERT
# ============================================================

def send_attack_alert(meta: dict) -> None:
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time < ALERT_COOLDOWN_SECONDS:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[IDS ALERT] Attack Detected — {meta['src_ip']} → {meta['dst_ip']}"
    body = (
        f"INTRUSION DETECTION SYSTEM — SECURITY ALERT\n"
        f"{'=' * 45}\n\n"
        f"Timestamp   : {timestamp}\n"
        f"Source IP   : {meta['src_ip']}\n"
        f"Destination : {meta['dst_ip']}:{meta['dst_port']}\n"
        f"Protocol    : {meta['protocol']}\n"
        f"Status      : ATTACK DETECTED\n\n"
        f"The DQN agent has classified this flow as malicious traffic.\n"
        f"Please review the network logs immediately.\n\n"
        f"-- IDS System (DQN Agent)"
    )
    msg = MIMEMultipart()
    msg["From"]    = EMAIL_ADDRESS
    msg["To"]      = EMAIL_RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, EMAIL_RECIPIENT, msg.as_string())
        print(f"{Fore.YELLOW}[EMAIL] Alert sent for {meta['src_ip']} → {meta['dst_ip']}{Style.RESET_ALL}")
        last_alert_time = current_time
    except Exception as e:
        print(f"{Fore.YELLOW}[EMAIL ERROR] {e}{Style.RESET_ALL}")


# ============================================================
# DISPATCH — terminal + log + dashboard + email
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

    if pred == 1:
        send_attack_alert(meta)


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
    if size < offset:
        # File was recreated or truncated — reset and re-read from the start
        offsets.pop(path, None)
        headers.pop(path, None)
        offset = 0
    if size == offset:
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

    start_dashboard(detection_buffer, stats, stats_lock,
                    WATCH_FOLDER, DASHBOARD_HOST, DASHBOARD_PORT)

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