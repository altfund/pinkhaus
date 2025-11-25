/**
 * Ominari Trading Dashboard - Modern Interactive JavaScript
 * Professional financial interface with Chart.js integration
 */

class OminariDashboard {
    constructor() {
        this.portfolioChart = null;
        this.positionChart = null;
        this.websocket = null;
        this.lastUpdate = new Date();
        
        // Chart configuration with altfund2 color scheme
        this.chartColors = {
            primary: '#4B9CD3',
            success: '#28a745',
            danger: '#dc3545',
            warning: '#fd7e14',
            light: '#f8f9fa',
            dark: '#343a40'
        };
        
        this.init();
    }
    
    init() {
        this.initializeCharts();
        this.setupWebSocket();
        this.loadInitialData();
        this.setupEventListeners();
    }
    
    initializeCharts() {
        this.initPortfolioChart();
        this.initPositionChart();
    }
    
    initPortfolioChart() {
        const ctx = document.getElementById('portfolioChart');
        if (!ctx) return;
        
        this.portfolioChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
                    borderColor: this.chartColors.primary,
                    backgroundColor: `${this.chartColors.primary}20`,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        titleColor: '#333',
                        bodyColor: '#666',
                        borderColor: this.chartColors.primary,
                        borderWidth: 1,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                return `Portfolio: $${context.parsed.y.toLocaleString()}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'hour',
                            displayFormats: {
                                hour: 'MMM DD HH:mm'
                            }
                        },
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#666',
                            font: {
                                size: 11
                            }
                        }
                    },
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: '#f0f0f0'
                        },
                        ticks: {
                            color: '#666',
                            font: {
                                size: 11
                            },
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }
    
    initPositionChart() {
        const ctx = document.getElementById('positionChart');
        if (!ctx) return;
        
        this.positionChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Cash', 'Active Positions'],
                datasets: [{
                    data: [100, 0],
                    backgroundColor: [this.chartColors.light, this.chartColors.primary],
                    borderWidth: 0,
                    cutout: '60%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true,
                            pointStyle: 'circle',
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        titleColor: '#333',
                        bodyColor: '#666',
                        borderColor: this.chartColors.primary,
                        borderWidth: 1,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                const percentage = ((context.parsed / context.dataset.data.reduce((a, b) => a + b, 0)) * 100).toFixed(1);
                                return `${context.label}: ${percentage}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    setupWebSocket() {
        if (typeof io !== 'undefined') {
            this.websocket = io();
            
            this.websocket.on('dashboard_update', (data) => {
                this.updateDashboard(data);
                this.showToast('Dashboard updated with new data');
            });
            
            this.websocket.on('connect', () => {
                console.log('WebSocket connected');
                this.updateHeartbeat(true);
            });
            
            this.websocket.on('disconnect', () => {
                console.log('WebSocket disconnected');
                this.updateHeartbeat(false);
            });
        }
    }
    
    async loadInitialData() {
        try {
            await Promise.all([
                this.loadPortfolioData(),
                this.loadMarketData(),
                this.loadTradesData()
            ]);
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showToast('Error loading dashboard data', 'error');
        }
    }
    
    async loadPortfolioData() {
        try {
            const response = await fetch('/api/portfolio');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.updatePortfolioHeader(data.portfolio);
                this.updatePortfolioChart(data.historical || []);
                this.updatePositionChart(data.positions || {});
            } else {
                console.warn('Portfolio API returned error:', data.error || 'Unknown error');
                this.showToast('Portfolio data unavailable', 'warning');
            }
        } catch (error) {
            console.error('Error loading portfolio data:', error);
            this.showToast('Failed to load portfolio data', 'error');
        }
    }
    
    async loadMarketData() {
        try {
            const response = await fetch('/api/markets');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.updateMarketGrid(data.markets);
            }
        } catch (error) {
            console.error('Error loading market data:', error);
        }
    }
    
    async loadTradesData() {
        try {
            const response = await fetch('/api/trades');
            const data = await response.json();

            // API returns {success: true, trades: [...]}
            if (data.success && Array.isArray(data.trades)) {
                this.updateTradesTable(data.trades);
            }
        } catch (error) {
            console.error('Error loading trades data:', error);
        }
    }
    
    updatePortfolioHeader(portfolio) {
        const elements = {\n            portfolioValue: document.getElementById('portfolioValue'),\n            portfolioChange: document.getElementById('portfolioChange'),\n            openPositions: document.getElementById('openPositions'),\n            totalTrades: document.getElementById('totalTrades'),\n            activeStake: document.getElementById('activeStake')\n        };\n        \n        if (elements.portfolioValue) {\n            elements.portfolioValue.textContent = `$${portfolio.value?.toLocaleString() || '0.00'}`;\n        }\n        \n        if (elements.portfolioChange && portfolio.change !== undefined) {\n            const changeClass = portfolio.change >= 0 ? 'positive' : 'negative';\n            const changeSign = portfolio.change >= 0 ? '+' : '';\n            elements.portfolioChange.textContent = `${changeSign}$${portfolio.change?.toLocaleString() || '0.00'} (${changeSign}${portfolio.changePercent?.toFixed(1) || '0.0'}%)`;\n            elements.portfolioChange.className = `portfolio-change ${changeClass}`;\n        }\n        \n        if (elements.openPositions) {\n            elements.openPositions.textContent = portfolio.openPositions || '0';\n        }\n        \n        if (elements.totalTrades) {\n            elements.totalTrades.textContent = portfolio.totalTrades || '0';\n        }\n        \n        if (elements.activeStake) {\n            elements.activeStake.textContent = `$${portfolio.activeStake?.toLocaleString() || '0'}`;\n        }\n    }\n    \n    updatePortfolioChart(historical) {\n        if (!this.portfolioChart || !historical) return;\n        \n        const labels = historical.map(point => new Date(point.timestamp));\n        const data = historical.map(point => point.value);\n        \n        this.portfolioChart.data.labels = labels;\n        this.portfolioChart.data.datasets[0].data = data;\n        this.portfolioChart.update('none');\n    }\n    \n    updatePositionChart(positions) {\n        if (!this.positionChart || !positions) return;\n        \n        const cash = positions.cash || 0;\n        const activeStake = positions.activeStake || 0;\n        \n        this.positionChart.data.datasets[0].data = [cash, activeStake];\n        this.positionChart.update('none');\n    }\n    \n    updateMarketGrid(markets) {\n        const grid = document.getElementById('marketGrid');\n        const countElement = document.getElementById('marketCount');\n        \n        if (!grid) return;\n        \n        if (countElement) {\n            countElement.textContent = `${markets?.length || 0} opportunities found`;\n        }\n        \n        if (!markets || markets.length === 0) {\n            grid.innerHTML = '<div class=\"col-12 text-center p-4\"><p class=\"text-muted\">No market opportunities available</p></div>';\n            return;\n        }\n        \n        grid.innerHTML = markets.map(market => this.createMarketCard(market)).join('');\n    }\n    \n    createMarketCard(market) {\n        const edgeColor = market.edge > 10 ? 'success' : market.edge > 5 ? 'warning' : 'primary';\n        \n        return `\n            <div class=\"market-card\">\n                <div class=\"market-name\">${market.name}</div>\n                <div class=\"market-odds\">\n                    <div class=\"odds-item\">\n                        <div class=\"odds-value\">${market.homeOdds}</div>\n                        <div class=\"odds-label\">Home</div>\n                    </div>\n                    <div class=\"odds-item\">\n                        <div class=\"odds-value\">${market.awayOdds}</div>\n                        <div class=\"odds-label\">Away</div>\n                    </div>\n                </div>\n                <div class=\"edge-indicator\">\n                    <span class=\"text-${edgeColor}\">Edge: <strong class=\"edge-value\">+${market.edge}%</strong></span>\n                </div>\n                <div class=\"mt-2 small text-muted\">\n                    <i class=\"fas fa-clock me-1\"></i>${new Date(market.startTime).toLocaleString()}\n                </div>\n            </div>\n        `;\n    }\n    \n    updateTradesTable(trades) {\n        const tbody = document.getElementById('tradesTableBody');\n        if (!tbody) return;\n        \n        if (!trades || trades.length === 0) {\n            tbody.innerHTML = '<tr><td colspan=\"6\" class=\"text-center p-4\"><p class=\"text-muted\">No recent trades</p></td></tr>';\n            return;\n        }\n        \n        tbody.innerHTML = trades.slice(0, 10).map(trade => {\n            const statusClass = trade.status === 'closed' ? 'success' : 'warning';\n            const pnlClass = trade.pnl > 0 ? 'text-success' : trade.pnl < 0 ? 'text-danger' : 'text-muted';\n            \n            return `\n                <tr>\n                    <td>${trade.match || 'N/A'}</td>\n                    <td>${trade.outcome || 'N/A'}</td>\n                    <td>$${(trade.stake || 0).toLocaleString()}</td>\n                    <td>${(trade.odds || 0).toFixed(2)}</td>\n                    <td><span class=\"status-indicator status-${statusClass}\">${trade.status || 'pending'}</span></td>\n                    <td class=\"${pnlClass}\">$${(trade.pnl || 0).toFixed(2)}</td>\n                </tr>\n            `;\n        }).join('');\n    }\n    \n    updateHeartbeat(connected) {\n        const status = document.getElementById('heartbeatStatus');\n        if (!status) return;\n        \n        if (connected) {\n            status.innerHTML = '<span class=\"heartbeat-indicator\"></span>System Active';\n            status.style.borderLeftColor = this.chartColors.success;\n        } else {\n            status.innerHTML = '<span style=\"color: #dc3545;\">●</span> Disconnected';\n            status.style.borderLeftColor = this.chartColors.danger;\n        }\n    }\n    \n    showToast(message, type = 'info') {\n        const toast = document.getElementById('systemToast');\n        const messageEl = document.getElementById('toastMessage');\n        \n        if (toast && messageEl) {\n            messageEl.textContent = message;\n            const bsToast = new bootstrap.Toast(toast);\n            bsToast.show();\n        }\n    }\n    \n    setupEventListeners() {\n        // Update last update timestamp\n        setInterval(() => {\n            const lastUpdateEl = document.getElementById('lastUpdate');\n            if (lastUpdateEl) {\n                const timeAgo = Math.floor((Date.now() - this.lastUpdate.getTime()) / 1000);\n                let timeText;\n                \n                if (timeAgo < 60) {\n                    timeText = `${timeAgo}s ago`;\n                } else if (timeAgo < 3600) {\n                    timeText = `${Math.floor(timeAgo / 60)}m ago`;\n                } else {\n                    timeText = `${Math.floor(timeAgo / 3600)}h ago`;\n                }\n                \n                lastUpdateEl.textContent = `Last updated: ${timeText}`;\n            }\n        }, 1000);\n    }\n    \n    updateDashboard(data) {\n        if (data.portfolio) {\n            this.updatePortfolioHeader(data.portfolio);\n        }\n        \n        if (data.markets) {\n            this.updateMarketGrid(data.markets);\n        }\n        \n        if (data.trades) {\n            this.updateTradesTable(data.trades);\n        }\n        \n        this.lastUpdate = new Date();\n    }\n}\n\n// Global functions for template integration\nlet dashboardInstance;\n\nfunction initializeDashboard() {\n    dashboardInstance = new OminariDashboard();\n}\n\nfunction refreshData() {\n    if (dashboardInstance) {\n        dashboardInstance.loadInitialData();\n        dashboardInstance.showToast('Refreshing all data...');\n    }\n}\n\nfunction refreshMarkets() {\n    if (dashboardInstance) {\n        dashboardInstance.loadMarketData();\n        dashboardInstance.showToast('Market data refreshed');\n    }\n}\n\nfunction refreshTrades() {\n    if (dashboardInstance) {\n        dashboardInstance.loadTradesData();\n        dashboardInstance.showToast('Trades data refreshed');\n    }\n}\n\nfunction refreshDashboardData() {\n    if (dashboardInstance) {\n        dashboardInstance.loadInitialData();\n    }\n}\n\n// Export for module systems\nif (typeof module !== 'undefined' && module.exports) {\n    module.exports = { OminariDashboard, initializeDashboard, refreshData };\n}