import os
import csv
import time
import threading
import collections
import smtplib
import joblib

from io import StringIO
from datetime import datetime
import numpy as np
import pandas as pd

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from stable_baselines3 import DQN
from dashboard import start_dashboard
from config import (
    WATCH_FOLDER, MODEL_PATH, SCALER_PATH, LOG_PATH,
    DASHBOARD_HOST, DASHBOARD_PORT, MAX_DASHBOARD_ROWS, POLL_INTERVAL_SEC,
    EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_RECIPIENT, ALERT_COOLDOWN_SECONDS,
    CANONICAL_COLUMNS, COLUMN_ALIASES,
)

# The block below is a shared-state area for the main and dashboard threads.
# `detection_buffer` holds the most recent detections for display in the dashboard.
# A "deque" is like a list with a maximum length: when new items are added beyond the limit, 
# the oldest items are automatically removed.

detection_buffer = collections.deque(maxlen=MAX_DASHBOARD_ROWS)
stats = {
    'total': 0, 'attacks': 0, 'normal': 0,
    'start_time': datetime.now().isoformat(timespec='seconds'),
}

# A lock is used to synchronize access to the `stats` dictionary 
# between the main thread (which updates stats on each detection) 
# and the dashboard thread (which reads stats to display).
stats_lock = threading.Lock()

# This tracks when the last email alert was sent, to enforce a cooldown period between alerts and avoid spamming.
last_alert_time = 0.0

# === FEATURE ALIGNMENT AND CLEANING ===

# CICFlowMeter may produce CSVs with varying column orders, missing columns, or extra columns.
# Thus, the function below does three main things:
# 1. Strips whitespace from column names and renames any known aliases to a canonical set of column names expected by the model
# 2. Renames snake_case columns to match the canonical format expected by the model
# 3. Ensures all canonical columns are present, filling missing ones with zeros

def align_features(df: pd.DataFrame) -> pd.DataFrame:
    # We use the standard pandas dataframe operations to clean and align the features. 
    df = df.copy()
    df.columns = df.columns.str.strip()

    # After stripping whitespaces, we rename any columns that match known aliases to their canonical names.
    df.rename(columns=COLUMN_ALIASES, inplace=True)

    # If a canonical column is missing from the input CSV, we add it with a default value of 0
    # so the script does not crash or produce errors when it encounters a CSV with missing columns
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = 0

    # After the columns have been reordered and any missing columns have been filled with zeros 
    # we convert all feature values to numeric types.
    alignedColumns = df[CANONICAL_COLUMNS].copy()
    alignedColumns = alignedColumns.apply(pd.to_numeric, errors='coerce')

    # Sometimes, CICFlowMeter can produce infinite values, for example when dividing by zero during feature calculation. 
    # This step replaces both -infinity and +infinity with NaN, and then fills all NaN values with 0. 
    # This way we ensure that the input data is clean and does not contain any invalid values 
    # that could cause issues during scaling or prediction.
    alignedColumns.replace([np.inf, -np.inf], np.nan, inplace=True)
    alignedColumns.fillna(0, inplace=True)
    return alignedColumns

# This function is separate from the model.
# It extracts relevant metadata from the raw CSV rows to be used in the dashboard display and email alerts.
# It tries with both snake case and spaced column names to maximize compatibility with different CICFlowMeter versions
def extract_metadata(df: pd.DataFrame) -> list:
    # Again, we start by stripping whitespace from column names to ensure consistent access.
    df = df.copy()
    df.columns = df.columns.str.strip()

    # We iterate over each row in the DataFrame and construct a dictionary containing the relevant metadata fields.
    out = []
    for _, row in df.iterrows():
        # For each metadata field, we first try to access it using the snake_case column name (e.g., 'src_ip'),
        # and if that fails, we try the spaced version (e.g., 'Src IP'). If neither is found, we use a default value.
        out.append({
            'timestamp': str(row.get('timestamp', row.get('Timestamp', ''))),
            'src_ip':    str(row.get('src_ip', row.get('Src IP', 'N/A'))),
            'dst_ip':    str(row.get('dst_ip', row.get('Dst IP', 'N/A'))),
            'dst_port':  str(row.get('dst_port', row.get('Destination Port', 'N/A'))),
            'protocol':  str(row.get('protocol', row.get('Protocol', 'N/A'))),
        })
    return out

# === EMAIL ALERT AND COOLDOWN ===

# In this section, we define a function to send an email alert whenever an attack is detected.
# The function constructs a well-formatted email containing the details of the detected attack,
# and uses the smtplib library to send the email via an SMTP server (in this case, Gmail's SMTP server).
# To prevent spamming, the function checks the time since the last alert was sent 
# and enforces a cooldown period defined by ALERT_COOLDOWN_SECONDS.
# If an alert is sent successfully, it updates the last_alert_time to the current time.
# The email includes the source and destination IPs, the destination port, the protocol 
# and a timestamp of when the attack was detected.

def send_attack_alert(metadata: dict) -> None:
    # We use the global variable `last_alert_time` to track when the last email alert was sent
    global last_alert_time
    # We get the current time and compare it to the last alert time. 
    # If the difference is less than the defined cooldown period we return early and do not send an alert.
    current_time = time.time()
    if current_time - last_alert_time < ALERT_COOLDOWN_SECONDS:
        return

    # Take the exact date and time when the attack was detected
    # and format it as a human-readable string 
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Construct the subject of the email, includingg the source and destination IPs
    subject = f"[IDS ALERT] Attack Detected — {metadata['src_ip']} → {metadata['dst_ip']}"

    # Construct the body of the email with a clear and informative message about the detected attack,
    body = (
        f"INTRUSION DETECTION SYSTEM — SECURITY ALERT\n"
        f"{'=' * 45}\n\n"
        f"Timestamp   : {timestamp}\n"
        f"Source IP   : {metadata['src_ip']}\n"
        f"Destination : {metadata['dst_ip']}:{metadata['dst_port']}\n"
        f"Protocol    : {metadata['protocol']}\n"
        f"Status      : ATTACK DETECTED\n\n"
        f"The DQN agent has classified this flow as malicious traffic.\n"
        f"Please review the network logs immediately.\n\n"
        f"-- IDS System (DQN Agent)"
    )

    # We use the email.mime libraries to construct a multipart email message with the defined subject and body.
    # We set the recepient and the sender of the email using the configuration variables defined in config.py.
    msg = MIMEMultipart()
    msg["From"]    = EMAIL_ADDRESS
    msg["To"]      = EMAIL_RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # The code below opens a connection on Gmail's SMTP server
    # and attempts to send the email. If any errors are encountered during this process 
    # (e.g., network issues, authentication errors), they are caught and printed to the console.
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, EMAIL_RECIPIENT, msg.as_string())
        print(f"[EMAIL] Alert sent for {metadata['src_ip']} → {metadata['dst_ip']}")

        # Update the last_alert_time to the current time after successfully sending an alert
        last_alert_time = current_time
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")

# === DISPATCH FUNCTION ===

# The dispatch function is responsible for handling the output of the model's predictions.
# It takes the metadata extracted from the CSV, the prediction made by the model
# and the log writer and log file objects for recording detections.
# The function performs several tasks:
# 1. It formats and prints a log message to the console with the timestamp, verdict (ATTACK or NORMAL) 
# source and destination IPs, port, and protocol.
# 2. It writes a new row to the log CSV file with the detection details.
# 3. It appends the detection to the `detection_buffer` for display in the dashboard.
# 4. It updates the `stats` dictionary to keep track of the total number of detections, 
# as well as the counts of attacks and normal traffic.
# 5. If the prediction indicates an attack, it calls the `send_attack_alert` function to send an email alert.

def dispatch(metadata: dict, prediction: int, log_writer, log_file):
    # metadata is a dictionary containing the relevant metadata fields extracted from the CSV row,
    # prediction is the integer prediction made by the model (0 for normal traffic, 1 for attack),
    # log_writer is a CSV writer object used to write rows to the log file, and
    # log_file is the file object for the log CSV file, which we flush after writing to ensure data is saved immediately.

    # Based on the prediction (0 for normal traffic, 1 for attack), 
    # we determine the label to display in the logs and dashboard.
    label = 'ATTACK' if prediction == 1 else 'NORMAL'
    current_time = datetime.now().strftime('%H:%M:%S')

    # We print a formatted log message to the console with the timestamp, verdict, source destination IPs
    # port and protocol.
    print(f"[{current_time}] {label:<6}  "
          f"{metadata['src_ip']:>15} -> {metadata['dst_ip']:<15}:{metadata['dst_port']:<5}  "
          f"proto={metadata['protocol']}")

    # We write a new row to the log CSV file with the detection details, including the local time, flow timestamp
    # source destination IPs, port, protocol and the prediction label.
    log_writer.writerow([
        current_time, metadata['timestamp'], metadata['src_ip'], metadata['dst_ip'],
        metadata['dst_port'], metadata['protocol'], label,
    ])
    # After writing to the log file, we call `log_file.flush()` to ensure that the data is immediately written to disk.
    log_file.flush()  

    # We append the detection details to the `detection_buffer`
    # which is a shared data structure used by the dashboard thread to display recent detections.
    detection_buffer.append({
        'local_time': current_time,
        'flow_time':  metadata['timestamp'],
        'src_ip':     metadata['src_ip'],
        'dst_ip':     metadata['dst_ip'],
        'dst_port':   metadata['dst_port'],
        'protocol':   metadata['protocol'],
        'prediction': label,
    })

    # We update the `stats` dictionary to keep track of the total number of detections 
    # as well as the counts of attacks and normal traffic.
    with stats_lock:
        stats['total']  += 1
        stats['attacks' if prediction == 1 else 'normal'] += 1

    # If the prediction indicates an attack (prediction == 1), we call the `send_attack_alert` function 
    # to send an email alert with the metadata of the detected attack.
    if prediction == 1:
        send_attack_alert(metadata)

# === LOOP ===

# The `process_csv` function is responsible for processing a single CSV file that has been detected in the watch folder.
# It reads the new data from the CSV file, aligns and scales the features, makes predictions
# using the loaded DQN model, and dispatches the results for logging, dashboard display, and email alerts.
# The function also handles the tracking of byte offsets to ensure that only new data is processed each time 
# it is called, and it manages the parsing of the CSV file, including handling of headers and incomplete lines.

def process_csv(path: str, offsets: dict, headers: dict, model, scaler, log_writer, log_file) -> int:
    # The function starts by trying to get the size of the file at the given path. 
    # If the file does not exist or cannot be accessed, it returns 0, indicating that no new rows were processed.
    try:
        size = os.path.getsize(path)
    except OSError:
        return 0

    # The function then checks the last byte offset that was processed for this file (if any) 
    # from the `offsets` dictionary. If the file size is smaller than the last offset, 
    # it means that the file was recreated or truncated, so it resets the offset and header information for this file.
    offset = offsets.get(path, 0)
    if size < offset:
        # File was recreated or truncated, reset and re-read from the start
        offsets.pop(path, None)
        headers.pop(path, None)
        offset = 0
    # If the file size is equal to the offset, it means there are no new bytes to read, so it returns 0.
    if size == offset:
        return 0

    # Open in binary mode so byte offsets are exact, regardless of BOM/encoding
    # We read the new bytes from the file starting from the last offset, and then decode them as UTF-8 text.
    with open(path, 'rb') as f:
        f.seek(offset)
        new_bytes = f.read()
        new_offset = f.tell()

    # CICFlowMeter may produce CSV files with a UTF-8 Byte Order Mark (BOM) at the beginning of the file 
    # which can interfere with parsing. If the offset is 0, we decode with 'utf-8-sig' to automatically 
    # handle and remove the BOM if it is present. For subsequent reads (offset > 0), we decode with standard 
    # 'utf-8' since the BOM would only be at the start of the file.
    # The BOM is Byte Order Mark, a special marker at the beginning of a text file that indicates the encoding used.
    if offset == 0:
        text = new_bytes.decode('utf-8-sig', errors='replace')
    else:
        text = new_bytes.decode('utf-8', errors='replace')

    # If this is the first time we are reading from the file (offset == 0)
    # we need to parse the header line to get the column names.
    if offset == 0:
        # We look for the first newline character to separate the header from the rest of the data.
        header_end = text.find('\n')

        # If no newline is found, it means the header line is not fully written yet 
        # so we return 0 and wait for the next poll.
        if header_end == -1:
            return 0  

        # We split the header line by commas to get the list of column names 
        # and we strip any whitespace from them.
        headers[path] = [c.strip() for c in text[:header_end].strip().split(',')]
        text = text[header_end + 1:]

    # We look for the last newline character in the text to ensure we only process complete lines of data.
    # If there is no newline, it means we have an incomplete line (e.g., the file is still being written to)
    # so we return 0 and wait for the next poll to process it when it is complete.
    last_newline = text.rfind('\n')
    if last_newline == -1:
        return 0
    
    # We take the text up to the last newline as the complete data to process 
    # and we keep any trailing incomplete line for the next poll.
    complete = text[:last_newline + 1]

    # If there's an incomplete trailing line, back the offset off by its byte length
    trailing = text[last_newline + 1:]
    offsets[path] = new_offset - len(trailing.encode('utf-8'))

    # We use pandas to read the complete CSV data from the string, using the header we parsed earlier.
    # We also specify `on_bad_lines='skip'` to skip any lines that cannot be parsed correctly
    # which can happen if the file is being written to while we are reading it.
    try:
        df = pd.read_csv(
            StringIO(complete), header=None,
            names=headers[path], on_bad_lines='skip',
        )
    except Exception as e:
        print(f"[!] CSV parse error on {path}: {e}")
        return 0

    # If the resulting DataFrame is empty (e.g., if all lines were bad or if there was no new data), we return 0.   
    if df.empty:
        return 0

    # We then call the `align_features` function to clean and align the features in the DataFrame,
    # and we use the loaded scaler to scale the features before making predictions with the model.
    # The `align_features` function ensures that the input features are in the correct format and order 
    # expected by the model, while the scaler transforms the features to the same scale as the training data 
    # which is crucial for accurate predictions.
    aligned  = align_features(df)
    scaled   = scaler.transform(aligned.values.astype(np.float32))
    preds, _ = model.predict(scaled, deterministic=True)

    # Finally, we iterate over the metadata extracted from the DataFrame and the corresponding predictions,
    # and we call the `dispatch` function for each row to handle logging, dashboard display
    # and email alerts based on the prediction results.
    for metadata, pred in zip(extract_metadata(df), preds):
        dispatch(metadata, int(pred), log_writer, log_file)

    # The function returns the number of rows that were processed, which is the length of the DataFrame.
    return len(df)

# === MAIN FUNCTION ===

# The `main` function is the entry point of the script. It performs the following tasks:
# 1. It prints a header to the console indicating that the DQN Intrusion Detection System is running in real-time mode.
# 2. It sets up the logging mechanism by creating the log file if it does not exist and writing the header row.
# 3. It loads the trained DQN model and the scaler from the specified paths.
# 4. It starts the dashboard in a separate thread to display real-time detections and statistics.
# 5. It checks if the watch folder exists and creates it if it does not.
# 6. It enters an infinite loop where it polls the watch folder for new CSV files at regular intervals defined by POLL_INTERVAL_SEC.
# For each CSV file found, it calls the `process_csv` function to process the new data and make predictions.
# The loop continues until a KeyboardInterrupt (e.g., Ctrl+C) is received 
# at which point it gracefully shuts down, prints a summary of the detections, and closes the log file.

def main():
    print('=' * 68)
    print('  DQN Intrusion Detection System — Real-Time Mode')
    print('=' * 68)

    # We set up the logging mechanism by ensuring the log directory exists and opening the log file in append mode.
    os.makedirs(os.path.dirname(LOG_PATH) or '.', exist_ok=True)
    log_exists = os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0
    log_file = open(LOG_PATH, 'a', newline='', encoding='utf-8')

    # We create a CSV writer object to write rows to the log file
    # and if the log file did not already exist, we write the header row with the column names.
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['local_time', 'flow_timestamp', 'src_ip',
                             'dst_ip', 'dst_port', 'protocol', 'prediction'])
        log_file.flush()

    # We load the trained DQN model and the scaler from the specified paths defined in the configuration.
    print(f"[*] Loading model from {MODEL_PATH}")
    model = DQN.load(MODEL_PATH)

    # We load the scaler using joblib, which is a common way to save and load scikit-learn objects.
    print(f"[*] Loading scaler from {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    print("[+] Model and scaler loaded.")

    # We start the dashboard in a separate thread by calling the `start_dashboard` function 
    # and passing the shared `detection_buffer`, `stats`, and `stats_lock` 
    # along with the dashboard configuration parameters.
    start_dashboard(detection_buffer, stats, stats_lock,
                    WATCH_FOLDER, DASHBOARD_HOST, DASHBOARD_PORT)

    # We check if the watch folder exists and if it does not, we create it. 
    # This is where the CICFlowMeter will output the CSV files that we need to process.
    if not WATCH_FOLDER.exists():
        print(f"[!] Watch folder does not exist, creating: {WATCH_FOLDER}")
        WATCH_FOLDER.mkdir(parents=True, exist_ok=True)

    # We enter an infinite loop where we poll the watch folder for new CSV files every POLL_INTERVAL_SEC seconds.
    # For each CSV file found in the watch folder, we call the `process_csv`
    # function to process the new data, make predictions, and handle logging and alerts.
    print(f"[+] Polling {WATCH_FOLDER} every {POLL_INTERVAL_SEC}s")
    print('-' * 68)
    print(f"{'TIME':<10} {'VERDICT':<7}  {'SOURCE':>15} -> {'DEST':<15}:{'PORT':<5}  proto")
    print('-' * 68)

    # We maintain two dictionaries, `offsets` and `headers` 
    # to track the byte offsets for each file and the headers of each CSV file respectively.
    # The `offsets` dictionary allows us to keep track of how much of each file we have already processed,
    # so that we only read and process new data each time we poll the files.
    # The `headers` dictionary stores the column names for each file, which is necessary 
    # for parsing the CSV data correctly, especially since different files may have 
    # different column orders or names depending on the version of CICFlowMeter used.
    offsets = {}   
    headers = {}   

    # We use a try-except block to handle any exceptions that may occur during the main loop.
    try:
        while True:
            for csv_path in WATCH_FOLDER.glob("*.csv"):
                path = str(csv_path)
                # We call the `process_csv` function for each CSV file found in the watch folder.
                # The function will read the new data from the file, make predictions, and handle logging and alerts.
                try:
                    process_csv(path, offsets, headers,
                                model, scaler, log_writer, log_file)
                except Exception as e:
                    print(f"[!] Error on {path}: {type(e).__name__}: {e}")
            time.sleep(POLL_INTERVAL_SEC)
    # If a KeyboardInterrupt is received (e.g., the user presses Ctrl+C), we catch it and print a shutdown message.
    except KeyboardInterrupt:
        print("\n[!] Shutdown requested…")
    finally:
        log_file.close()
        with stats_lock:
            s = dict(stats)
        print('-' * 68)
        print(f"Summary:  total={s['total']}  normal={s['normal']}  attacks={s['attacks']}")
        print(f"Log written to: {LOG_PATH}")


if __name__ == '__main__':

    main()