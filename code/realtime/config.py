import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / '.env')

# ============================================================
# PATHS & RUNTIME
# ============================================================

WATCH_FOLDER       = Path(os.environ['IDS_WATCH_FOLDER'])
MODEL_PATH         = os.environ['IDS_MODEL_PATH']
SCALER_PATH        = os.environ['IDS_SCALER_PATH']
LOG_PATH           = os.environ['IDS_LOG_PATH']
DASHBOARD_HOST     = os.environ['IDS_DASHBOARD_HOST']
DASHBOARD_PORT     = int(os.environ['IDS_DASHBOARD_PORT'])
MAX_DASHBOARD_ROWS = int(os.environ['IDS_MAX_DASHBOARD_ROWS'])
POLL_INTERVAL_SEC  = int(os.environ['IDS_POLL_INTERVAL_SEC'])

# ============================================================
# EMAIL ALERTS
# ============================================================

EMAIL_ADDRESS          = os.environ['IDS_EMAIL_ADDRESS']
EMAIL_PASSWORD         = os.environ['IDS_EMAIL_PASSWORD']
EMAIL_RECIPIENT        = os.environ['IDS_EMAIL_RECIPIENT']
ALERT_COOLDOWN_SECONDS = int(os.environ['IDS_ALERT_COOLDOWN_SECONDS'])

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

# Maps snake_case names (Python CICFlowMeter port) to training-set names
COLUMN_ALIASES = {
    'dst_port':          'Destination Port',
    'flow_duration':     'Flow Duration',
    'tot_fwd_pkts':      'Total Fwd Packets',
    'totlen_fwd_pkts':   'Total Length of Fwd Packets',
    'fwd_pkt_len_max':   'Fwd Packet Length Max',
    'fwd_pkt_len_min':   'Fwd Packet Length Min',
    'fwd_pkt_len_mean':  'Fwd Packet Length Mean',
    'fwd_pkt_len_std':   'Fwd Packet Length Std',
    'bwd_pkt_len_max':   'Bwd Packet Length Max',
    'bwd_pkt_len_min':   'Bwd Packet Length Min',
    'bwd_pkt_len_mean':  'Bwd Packet Length Mean',
    'bwd_pkt_len_std':   'Bwd Packet Length Std',
    'flow_byts_s':       'Flow Bytes/s',
    'flow_pkts_s':       'Flow Packets/s',
    'flow_iat_mean':     'Flow IAT Mean',
    'flow_iat_std':      'Flow IAT Std',
    'flow_iat_max':      'Flow IAT Max',
    'flow_iat_min':      'Flow IAT Min',
    'fwd_iat_tot':       'Fwd IAT Total',
    'fwd_iat_mean':      'Fwd IAT Mean',
    'fwd_iat_std':       'Fwd IAT Std',
    'fwd_iat_max':       'Fwd IAT Max',
    'fwd_iat_min':       'Fwd IAT Min',
    'bwd_iat_tot':       'Bwd IAT Total',
    'bwd_iat_mean':      'Bwd IAT Mean',
    'bwd_iat_std':       'Bwd IAT Std',
    'bwd_iat_max':       'Bwd IAT Max',
    'bwd_iat_min':       'Bwd IAT Min',
    'fwd_header_len':    'Fwd Header Length',
    'bwd_header_len':    'Bwd Header Length',
    'fwd_pkts_s':        'Fwd Packets/s',
    'bwd_pkts_s':        'Bwd Packets/s',
    'pkt_len_min':       'Min Packet Length',
    'pkt_len_max':       'Max Packet Length',
    'pkt_len_mean':      'Packet Length Mean',
    'pkt_len_std':       'Packet Length Std',
    'pkt_len_var':       'Packet Length Variance',
    'fin_flag_cnt':      'FIN Flag Count',
    'psh_flag_cnt':      'PSH Flag Count',
    'ack_flag_cnt':      'ACK Flag Count',
    'pkt_size_avg':      'Average Packet Size',
    'subflow_fwd_byts':  'Subflow Fwd Bytes',
    'init_fwd_win_byts': 'Init_Win_bytes_forward',
    'init_bwd_win_byts': 'Init_Win_bytes_backward',
    'fwd_act_data_pkts': 'act_data_pkt_fwd',
    'fwd_seg_size_min':  'min_seg_size_forward',
    'active_mean':       'Active Mean',
    'active_max':        'Active Max',
    'active_min':        'Active Min',
    'idle_mean':         'Idle Mean',
    'idle_max':          'Idle Max',
    'idle_min':          'Idle Min',
}
