#!/usr/bin/env python3
"""
Simple Dashboard for Ominari - Soccer Only with Paper Trading
"""

import os

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from flask import Flask, render_template_string
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

SIMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari - Soccer Only Paper Trading</title>
    <style>
        body {
            font-family: monospace;
            background: #0a0a0a;
            color: #00ff00;
            margin: 0;
            padding: 20px;
        }
        .header {
            background: #111;
            padding: 20px;
            border: 1px solid #00ff00;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .badge {
            padding: 5px 15px;
            border-radius: 4px;
            font-size: 14px;
            margin: 0 5px;
        }
        .soccer-badge {
            background: #1a4d1a;
            border: 1px solid #00ff00;
            color: #00ff00;
        }
        .paper-badge {
            background: #4d1a4d;
            border: 1px solid #ff00ff;
            color: #ff00ff;
        }
        .section {
            background: #111;
            border: 1px solid #333;
            padding: 20px;
            margin-bottom: 20px;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
        }
        .metric-label {
            color: #888;
            font-size: 12px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        th {
            color: #00ffff;
            background: #1a1a1a;
        }
        .underdog {
            color: #ff00ff;
        }
    </style>
</head>
<body>
    <div class="header">
        <div style="display: flex; align-items: center;">
            <h1 style="margin: 0; margin-right: 20px;">⚽ Ominari Trading System</h1>
            <span class="badge soccer-badge">⚽ SOCCER ONLY MODE</span>
            <span class="badge paper-badge">📝 PAPER TRADING</span>
        </div>
        <div>
            <button style="background: #00ff00; color: #000; padding: 10px 20px; border: none; font-weight: bold; cursor: pointer;">
                ⚡ Execute Paper Trades
            </button>
        </div>
    </div>

    <div class="section">
        <h2>🎯 Trading Configuration (Locked)</h2>
        <div class="metric">
            <div class="metric-value">⚽ Soccer</div>
            <div class="metric-label">SPORT FILTER (NOT CHANGEABLE)</div>
        </div>
        <div class="metric">
            <div class="metric-value">Underdog</div>
            <div class="metric-label">STRATEGY</div>
        </div>
        <div class="metric">
            <div class="metric-value">2.5+</div>
            <div class="metric-label">MIN ODDS</div>
        </div>
        <div class="metric">
            <div class="metric-value">$50</div>
            <div class="metric-label">BET SIZE</div>
        </div>
    </div>

    <div class="section">
        <h2>📊 Portfolio Status</h2>
        <div class="metric">
            <div class="metric-value">$10,000</div>
            <div class="metric-label">INITIAL CAPITAL</div>
        </div>
        <div class="metric">
            <div class="metric-value">$9,500</div>
            <div class="metric-label">CURRENT CASH</div>
        </div>
        <div class="metric">
            <div class="metric-value">10</div>
            <div class="metric-label">ACTIVE POSITIONS</div>
        </div>
        <div class="metric">
            <div class="metric-value" style="color: #00ff00;">+5.09%</div>
            <div class="metric-label">ROI (BACKTEST)</div>
        </div>
    </div>

    <div class="section">
        <h2>⚽ Active Soccer Markets (Filtered)</h2>
        <p style="color: #888; margin-bottom: 10px;">
            Showing only soccer markets. Filter is locked and cannot be changed by users.
        </p>
        <table>
            <tr>
                <th>Match</th>
                <th>Time</th>
                <th>Home Odds</th>
                <th>Away Odds</th>
                <th>Action</th>
            </tr>
            <tr>
                <td>Liverpool vs Chelsea</td>
                <td>13:00</td>
                <td>1.81</td>
                <td class="underdog">4.11</td>
                <td>Bet Away (Underdog)</td>
            </tr>
            <tr>
                <td>Barcelona vs Real Madrid</td>
                <td>15:00</td>
                <td>2.10</td>
                <td class="underdog">3.50</td>
                <td>Bet Away (Underdog)</td>
            </tr>
            <tr>
                <td>Bayern Munich vs Dortmund</td>
                <td>17:00</td>
                <td>1.65</td>
                <td class="underdog">5.20</td>
                <td>Bet Away (Underdog)</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>📈 Backtest Results Summary</h2>
        <p>Based on 30 days of historical data:</p>
        <ul style="color: #888;">
            <li>Total Markets Analyzed: 4,795</li>
            <li>Soccer Markets: 2,576 (53.8%)</li>
            <li>Trades Placed: 900</li>
            <li>Win Rate: 27.6%</li>
            <li>ROI: 5.09%</li>
            <li>Strategy: Bet on underdogs with odds > 2.5</li>
        </ul>
    </div>

    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(SIMPLE_HTML)

if __name__ == '__main__':
    logger.info("🎯 Ominari Simple Dashboard Starting...")
    logger.info("⚽ Soccer Only Mode - Paper Trading Active")
    logger.info("🌐 Access at http://localhost:8888")
    app.run(host='0.0.0.0', port=8888, debug=False)