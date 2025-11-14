#!/usr/bin/env python3
"""
Real-time Trading Performance Monitor
Displays portfolio performance, active positions, and trading metrics
"""

import json
import os
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify
import pandas as pd
from sqlalchemy import func, desc

from database_v2 import db_manager
from models import Bet, BettingSession, Market

app = Flask(__name__)

# HTML template for performance dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ominari Trading Performance</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0a0a0a;
            color: #e0e0e0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #00ff88;
            margin-bottom: 30px;
            text-align: center;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .metric-card {
            background: #1a1a1a;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #333;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .positive {
            color: #00ff88;
        }
        .negative {
            color: #ff4444;
        }
        .chart-container {
            background: #1a1a1a;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            height: 400px;
        }
        .positions-table {
            background: #1a1a1a;
            padding: 20px;
            border-radius: 12px;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        th {
            color: #00ff88;
            font-weight: 600;
        }
        .status-active {
            color: #00ff88;
        }
        .status-won {
            color: #00ff88;
            font-weight: bold;
        }
        .status-lost {
            color: #ff4444;
        }
        .refresh-info {
            text-align: center;
            color: #666;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Ominari Trading Performance</h1>
        
        <div class="metrics-grid" id="metrics">
            <!-- Metrics will be loaded here -->
        </div>
        
        <div class="chart-container">
            <canvas id="performanceChart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="exposureChart"></canvas>
        </div>
        
        <div class="positions-table">
            <h2>Active Positions</h2>
            <table id="positionsTable">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Market</th>
                        <th>Outcome</th>
                        <th>Stake</th>
                        <th>Odds</th>
                        <th>Edge</th>
                        <th>Status</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody id="positionsBody">
                    <!-- Positions will be loaded here -->
                </tbody>
            </table>
        </div>
        
        <div class="refresh-info">
            Auto-refreshing every 10 seconds | Last update: <span id="lastUpdate"></span>
        </div>
    </div>
    
    <script>
        let performanceChart;
        let exposureChart;
        
        // Initialize charts
        function initCharts() {
            // Performance chart
            const perfCtx = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(perfCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            grid: { color: '#333' },
                            ticks: { color: '#888' }
                        },
                        x: {
                            grid: { color: '#333' },
                            ticks: { color: '#888' }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
            
            // Exposure chart
            const expCtx = document.getElementById('exposureChart').getContext('2d');
            exposureChart = new Chart(expCtx, {
                type: 'bar',
                data: {
                    labels: ['Current Exposure', 'Available Capital'],
                    datasets: [{
                        data: [0, 100],
                        backgroundColor: ['#ff6b6b', '#00ff88']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: { color: '#333' },
                            ticks: { 
                                color: '#888',
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#888' }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
        
        // Update dashboard data
        async function updateDashboard() {
            try {
                const response = await fetch('/api/performance');
                const data = await response.json();
                
                // Update metrics
                const metricsHtml = `
                    <div class="metric-card">
                        <div class="metric-label">Current Bankroll</div>
                        <div class="metric-value">$${data.metrics.bankroll.toFixed(2)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total P&L</div>
                        <div class="metric-value ${data.metrics.pnl >= 0 ? 'positive' : 'negative'}">
                            ${data.metrics.pnl >= 0 ? '+' : ''}$${data.metrics.pnl.toFixed(2)}
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">ROI</div>
                        <div class="metric-value ${data.metrics.roi >= 0 ? 'positive' : 'negative'}">
                            ${data.metrics.roi >= 0 ? '+' : ''}${data.metrics.roi.toFixed(2)}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">${data.metrics.win_rate.toFixed(1)}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Active Positions</div>
                        <div class="metric-value">${data.metrics.active_positions}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Avg Edge</div>
                        <div class="metric-value positive">${data.metrics.avg_edge.toFixed(2)}%</div>
                    </div>
                `;
                document.getElementById('metrics').innerHTML = metricsHtml;
                
                // Update performance chart
                if (data.history.length > 0) {
                    performanceChart.data.labels = data.history.map(h => 
                        new Date(h.timestamp).toLocaleTimeString()
                    );
                    performanceChart.data.datasets[0].data = data.history.map(h => h.value);
                    performanceChart.update();
                }
                
                // Update exposure chart
                const exposurePct = data.metrics.exposure_pct;
                exposureChart.data.datasets[0].data = [exposurePct, 100 - exposurePct];
                exposureChart.update();
                
                // Update positions table
                let positionsHtml = '';
                data.positions.forEach(pos => {
                    const statusClass = pos.status === 'pending' ? 'status-active' : 
                                       pos.status === 'won' ? 'status-won' : 'status-lost';
                    
                    positionsHtml += `
                        <tr>
                            <td>${new Date(pos.placed_at).toLocaleString()}</td>
                            <td>${pos.home_team} vs ${pos.away_team}</td>
                            <td>${pos.outcome}</td>
                            <td>$${pos.stake.toFixed(2)}</td>
                            <td>${pos.odds.toFixed(2)}</td>
                            <td class="positive">${pos.edge?.toFixed(2) || 'N/A'}%</td>
                            <td class="${statusClass}">${pos.status.toUpperCase()}</td>
                            <td class="${pos.pnl >= 0 ? 'positive' : 'negative'}">
                                ${pos.pnl ? (pos.pnl >= 0 ? '+' : '') + '$' + pos.pnl.toFixed(2) : '-'}
                            </td>
                        </tr>
                    `;
                });
                document.getElementById('positionsBody').innerHTML = positionsHtml;
                
                // Update last refresh time
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
                
            } catch (error) {
                console.error('Error updating dashboard:', error);
            }
        }
        
        // Initialize and start updates
        initCharts();
        updateDashboard();
        setInterval(updateDashboard, 10000);  // Update every 10 seconds
    </script>
</body>
</html>
"""


@app.route('/')
def dashboard():
    """Render the performance dashboard"""
    return render_template_string(DASHBOARD_TEMPLATE)


@app.route('/api/performance')
def api_performance():
    """Get performance data API endpoint"""
    try:
        # Load latest performance data
        performance_data = load_performance_data()
        
        # Get active positions
        positions = get_recent_positions()
        
        # Get performance history
        history = get_performance_history()
        
        return jsonify({
            'metrics': performance_data,
            'positions': positions,
            'history': history
        })
        
    except Exception as e:
        app.logger.error(f"Error getting performance data: {e}")
        return jsonify({
            'metrics': get_default_metrics(),
            'positions': [],
            'history': []
        })


def load_performance_data():
    """Load latest performance metrics from file"""
    try:
        # Read the last line from performance log
        if os.path.exists('logs/trading_performance.jsonl'):
            with open('logs/trading_performance.jsonl', 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    data = json.loads(last_line)
                    return data.get('portfolio', get_default_metrics())
        
        return get_default_metrics()
        
    except Exception as e:
        app.logger.error(f"Error loading performance data: {e}")
        return get_default_metrics()


def get_default_metrics():
    """Get default metrics structure"""
    return {
        'bankroll': 10000.0,
        'initial_bankroll': 10000.0,
        'pnl': 0.0,
        'roi': 0.0,
        'active_positions': 0,
        'total_exposure': 0.0,
        'exposure_pct': 0.0,
        'win_rate': 0.0,
        'avg_edge': 0.0
    }


def get_recent_positions():
    """Get recent betting positions"""
    positions = []
    
    try:
        with db_manager.get_db_session() as db:
            # Get recent bets
            recent_bets = db.query(Bet).join(
                BettingSession
            ).filter(
                BettingSession.is_paper == True
            ).order_by(
                desc(Bet.placed_at)
            ).limit(50).all()
            
            for bet in recent_bets:
                # Calculate P&L
                if bet.status == 'won':
                    pnl = bet.payout - bet.stake
                elif bet.status == 'lost':
                    pnl = -bet.stake
                else:
                    pnl = None
                    
                positions.append({
                    'id': bet.id,
                    'placed_at': bet.placed_at.isoformat(),
                    'home_team': bet.home_team,
                    'away_team': bet.away_team,
                    'outcome': bet.outcome,
                    'stake': float(bet.stake),
                    'odds': float(bet.odds),
                    'status': bet.status,
                    'pnl': float(pnl) if pnl else None,
                    'edge': getattr(bet, 'edge', None)
                })
                
    except Exception as e:
        app.logger.error(f"Error getting positions: {e}")
        
    return positions


def get_performance_history():
    """Get performance history for chart"""
    history = []
    
    try:
        if os.path.exists('logs/trading_performance.jsonl'):
            with open('logs/trading_performance.jsonl', 'r') as f:
                lines = f.readlines()[-100:]  # Last 100 entries
                
                for line in lines:
                    data = json.loads(line)
                    history.append({
                        'timestamp': data['timestamp'],
                        'value': data['portfolio']['bankroll']
                    })
                    
    except Exception as e:
        app.logger.error(f"Error getting history: {e}")
        
    return history


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'performance_monitor'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8889, debug=False)