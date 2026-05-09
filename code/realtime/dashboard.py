import threading
import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.model_selection import train_test_split

DASHBOARD_HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>IDS Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #000000;
    --surface:   #0d0d0d;
    --border:    #1e1e1e;
    --muted:     #444;
    --text:      #c8c8c8;
    --dim:       #555;

    --total-c:   #8ab4cc;
    --normal-c:  #6aab7e;
    --attack-c:  #c0514a;
    --rate-c:    #b08a3e;

    --total-bg:  rgba(138,180,204,0.07);
    --normal-bg: rgba(106,171,126,0.07);
    --attack-bg: rgba(192,81,74,0.07);
    --rate-bg:   rgba(176,138,62,0.07);

    --attack-row: rgba(192,81,74,0.06);
    --normal-row: rgba(106,171,126,0.04);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    padding: 48px 24px 48px;
  }

  .container {
    width: 100%;
    max-width: 900px;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 6px;
    flex-wrap: wrap;
    gap: 8px;
  }

  h1 {
    font-family: 'Share Tech Mono', monospace;
    font-size: 18px;
    font-weight: 400;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #e0e0e0;
  }

  .pulse-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--normal-c);
    margin-right: 8px;
    vertical-align: middle;
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(106,171,126,0.6); }
    50%       { opacity: 0.6; box-shadow: 0 0 0 4px rgba(106,171,126,0); }
  }

  .sub {
    font-size: 12px;
    color: var(--dim);
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.04em;
  }

  /* ── Divider ── */
  .divider {
    height: 1px;
    background: var(--border);
    margin: 18px 0 22px;
  }

  /* ── Stat cards ── */
  .stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 22px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }

  .card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 8px;
    opacity: 1;
  }

  .card.total  { border-top: 2px solid var(--total-c);  }
  .card.total::before  { background: var(--total-bg); }

  .card.normal { border-top: 2px solid var(--normal-c); }
  .card.normal::before { background: var(--normal-bg); }

  .card.attack { border-top: 2px solid var(--attack-c); }
  .card.attack::before { background: var(--attack-bg); }

  .card.rate   { border-top: 2px solid var(--rate-c);   }
  .card.rate::before   { background: var(--rate-bg); }

  .lbl {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 8px;
    font-family: 'Share Tech Mono', monospace;
  }

  .val {
    font-family: 'Share Tech Mono', monospace;
    font-size: 28px;
    line-height: 1;
    letter-spacing: 0.03em;
  }

  .card.total  .val { color: var(--total-c); }
  .card.normal .val { color: var(--normal-c); }
  .card.attack .val { color: var(--attack-c); }
  .card.rate   .val { color: var(--rate-c); }

  /* ── Table ── */
  .table-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }

  thead th {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    padding: 11px 16px;
    text-align: left;
    background: #080808;
    border-bottom: 1px solid var(--border);
    font-weight: 400;
  }

  tbody td {
    padding: 9px 16px;
    border-bottom: 1px solid #141414;
    font-size: 12px;
    font-family: 'Share Tech Mono', monospace;
    color: var(--dim);
    transition: background 0.15s;
  }

  tbody tr:last-child td { border-bottom: none; }

  tbody tr:hover td { background: rgba(255,255,255,0.02); }

  tbody tr.attack td { color: #a0524e; }
  tbody tr.normal td { color: #537a5d; }

  tbody tr.attack { background: var(--attack-row); }
  tbody tr.normal { background: var(--normal-row); }

  .badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 3px;
    font-size: 10px;
    letter-spacing: 0.08em;
    font-weight: 400;
  }

  tr.attack .badge {
    background: rgba(192,81,74,0.15);
    color: #c0514a;
    border: 1px solid rgba(192,81,74,0.25);
  }

  tr.normal .badge {
    background: rgba(106,171,126,0.12);
    color: #6aab7e;
    border: 1px solid rgba(106,171,126,0.2);
  }

  .empty {
    text-align: center;
    padding: 48px;
    color: var(--muted);
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.08em;
  }

  /* ── Footer ── */
  .note {
    margin-top: 14px;
    font-size: 11px;
    color: #333;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.04em;
    text-align: center;
  }

  /* ── Responsive ── */
  @media (max-width: 600px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    h1 { font-size: 15px; }
    .val { font-size: 22px; }
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <h1><span class="pulse-dot"></span>IDS Real-Time Dashboard</h1>
    <div style="display:flex;align-items:center;gap:16px">
      <div class="sub">Started <span id="start">—</span> &nbsp;·&nbsp; refreshes every 3s</div>
      <a href="/inject" style="font-family:'Share Tech Mono',monospace;font-size:11px;padding:6px 14px;background:rgba(192,81,74,0.15);color:#c0514a;border:1px solid rgba(192,81,74,0.35);border-radius:4px;text-decoration:none;letter-spacing:0.08em;">SIMULATE ATTACK</a>
    </div>
  </header>

  <div class="divider"></div>

  <div class="stats">
    <div class="card total">
      <div class="lbl">Total Flows</div>
      <div class="val" id="total">0</div>
    </div>
    <div class="card normal">
      <div class="lbl">Normal</div>
      <div class="val" id="normal">0</div>
    </div>
    <div class="card attack">
      <div class="lbl">Attacks</div>
      <div class="val" id="attacks">0</div>
    </div>
    <div class="card rate">
      <div class="lbl">Attack Rate</div>
      <div class="val" id="rate">0.0%</div>
    </div>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Source</th>
          <th>Destination</th>
          <th>Port</th>
          <th>Proto</th>
          <th>Verdict</th>
        </tr>
      </thead>
      <tbody id="tbody">
        <tr><td class="empty" colspan="6">Waiting for flows…</td></tr>
      </tbody>
    </table>
  </div>

  <div class="note">Showing up to 50 most recent flows · Full history in logs/detections.csv</div>

</div>

<script>
async function update() {
  try {
    const r = await fetch('/recent');
    const d = await r.json();
    document.getElementById('start').textContent   = d.stats.start_time;
    document.getElementById('total').textContent   = d.stats.total;
    document.getElementById('normal').textContent  = d.stats.normal;
    document.getElementById('attacks').textContent = d.stats.attacks;
    const rate = d.stats.total
      ? (d.stats.attacks / d.stats.total * 100).toFixed(1)
      : '0.0';
    document.getElementById('rate').textContent = rate + '%';

    const tb = document.getElementById('tbody');
    if (!d.detections.length) return;
    tb.innerHTML = d.detections.map(x => `
      <tr class="${x.prediction === 'ATTACK' ? 'attack' : 'normal'}">
        <td>${x.local_time}</td>
        <td>${x.src_ip}</td>
        <td>${x.dst_ip}</td>
        <td>${x.dst_port}</td>
        <td>${x.protocol}</td>
        <td><span class="badge">${x.prediction}</span></td>
      </tr>`).join('');
  } catch (e) { console.error(e); }
}

update();
setInterval(update, 2000);
</script>
</body>
</html>
"""


def start_dashboard(detection_buffer, stats, stats_lock, watch_folder, host, port):
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

    @app.route('/inject')
    def inject():
        try:
            data = pd.read_csv(r'C:\Users\riges\Desktop\IDS using RL\code\data\cicids2017_cleaned.csv')
            data.columns = data.columns.str.strip()
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.dropna(inplace=True)

            labels = data['Attack Type'].apply(lambda x: 0 if x == 'Normal Traffic' else 1).values
            features = data.drop('Attack Type', axis=1)

            _, test_f, _, test_l = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )

            attack_rows = test_f[test_l == 1].head(10).copy()
            attack_rows['src_ip'] = '10.0.0.1'
            attack_rows['dst_ip'] = '192.168.100.119'
            attack_rows['src_port'] = 1234
            attack_rows['protocol'] = 6
            attack_rows['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

            attack_rows.to_csv(str(watch_folder / 'flows.csv'), index=False)
            return '<script>setTimeout(()=>location.href="/",3000)</script><p style="font-family:monospace;padding:20px">Injecting 10 attack flows — redirecting to dashboard...</p>'
        except Exception as e:
            return f'Error: {e}', 500

    thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    print(f"{Fore.GREEN}[+] Dashboard at http://{host}:{port}{Style.RESET_ALL}")
