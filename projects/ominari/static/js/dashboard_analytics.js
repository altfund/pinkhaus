/**
 * Enhanced Dashboard Analytics
 * Advanced portfolio analytics and visualization components
 */

class DashboardAnalytics {
    constructor(dashboardInstance) {
        this.dashboard = dashboardInstance;
        this.performanceData = {
            dailyReturns: [],
            monthlyReturns: [],
            drawdowns: [],
            volatility: null,
            sharpeRatio: null
        };
        this.init();
    }

    init() {
        this.setupPerformanceMetrics();
        this.setupAdvancedCharts();
        this.setupRiskMetrics();
    }

    async loadPerformanceData() {
        try {
            // Get extended portfolio data
            const response = await fetch('/api/portfolio/analytics');
            if (response.ok) {
                const data = await response.json();
                this.updatePerformanceData(data);
            }
        } catch (error) {
            console.error('Error loading analytics data:', error);
        }
    }

    setupPerformanceMetrics() {
        this.createPerformanceCards();
    }

    createPerformanceCards() {
        const metricsContainer = document.getElementById('performanceMetrics');
        if (!metricsContainer) return;

        const metricsHTML = `
            <div class="row">
                <div class="col-md-3 mb-3">
                    <div class="metric-card">
                        <div class="metric-icon">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <div class="metric-content">
                            <h6 class="metric-title">Sharpe Ratio</h6>
                            <div class="metric-value" id="sharpeRatio">-</div>
                            <div class="metric-change" id="sharpeChange">Risk-adjusted return</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-3">
                    <div class="metric-card">
                        <div class="metric-icon">
                            <i class="fas fa-shield-alt"></i>
                        </div>
                        <div class="metric-content">
                            <h6 class="metric-title">Max Drawdown</h6>
                            <div class="metric-value" id="maxDrawdown">-</div>
                            <div class="metric-change" id="drawdownPeriod">Worst decline</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-3">
                    <div class="metric-card">
                        <div class="metric-icon">
                            <i class="fas fa-bullseye"></i>
                        </div>
                        <div class="metric-content">
                            <h6 class="metric-title">Win Rate</h6>
                            <div class="metric-value" id="winRate">-</div>
                            <div class="metric-change" id="winCount">Profitable trades</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-3">
                    <div class="metric-card">
                        <div class="metric-icon">
                            <i class="fas fa-heartbeat"></i>
                        </div>
                        <div class="metric-content">
                            <h6 class="metric-title">Volatility</h6>
                            <div class="metric-value" id="volatility">-</div>
                            <div class="metric-change" id="volatilityPeriod">Daily std dev</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        metricsContainer.innerHTML = metricsHTML;
    }

    setupAdvancedCharts() {
        this.setupDrawdownChart();
        this.setupReturnDistribution();
    }

    setupDrawdownChart() {
        const ctx = document.getElementById('drawdownChart');
        if (!ctx) return;

        this.drawdownChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Drawdown %',
                    data: [],
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        max: 0,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Drawdown: ${context.parsed.y.toFixed(2)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    setupReturnDistribution() {
        const ctx = document.getElementById('returnDistribution');
        if (!ctx) return;

        this.returnChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Frequency',
                    data: [],
                    backgroundColor: '#4B9CD3',
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Daily Return Distribution'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Daily Return (%)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequency'
                        }
                    }
                }
            }
        });
    }

    setupRiskMetrics() {
        // Calculate and display risk metrics
        this.updateRiskMetrics();
    }

    updatePerformanceData(data) {
        this.performanceData = { ...this.performanceData, ...data };
        this.updateMetricCards();
        this.updateAdvancedCharts();
    }

    updateMetricCards() {
        const elements = {
            sharpeRatio: document.getElementById('sharpeRatio'),
            maxDrawdown: document.getElementById('maxDrawdown'),
            winRate: document.getElementById('winRate'),
            volatility: document.getElementById('volatility')
        };

        if (elements.sharpeRatio) {
            const sharpe = this.performanceData.sharpeRatio || 0;
            elements.sharpeRatio.textContent = sharpe.toFixed(2);
            elements.sharpeRatio.className = `metric-value ${sharpe > 1 ? 'positive' : sharpe > 0 ? 'neutral' : 'negative'}`;
        }

        if (elements.maxDrawdown) {
            const drawdown = this.performanceData.maxDrawdown || 0;
            elements.maxDrawdown.textContent = `${drawdown.toFixed(1)}%`;
            elements.maxDrawdown.className = 'metric-value negative';
        }

        if (elements.winRate) {
            const winRate = this.performanceData.winRate || 0;
            elements.winRate.textContent = `${winRate.toFixed(1)}%`;
            elements.winRate.className = `metric-value ${winRate > 60 ? 'positive' : winRate > 50 ? 'neutral' : 'negative'}`;
        }

        if (elements.volatility) {
            const vol = this.performanceData.volatility || 0;
            elements.volatility.textContent = `${(vol * 100).toFixed(1)}%`;
            elements.volatility.className = `metric-value ${vol < 0.02 ? 'positive' : vol < 0.04 ? 'neutral' : 'negative'}`;
        }
    }

    updateAdvancedCharts() {
        if (this.drawdownChart && this.performanceData.drawdowns) {
            this.updateDrawdownChart();
        }

        if (this.returnChart && this.performanceData.dailyReturns) {
            this.updateReturnDistribution();
        }
    }

    updateDrawdownChart() {
        const drawdowns = this.performanceData.drawdowns || [];
        const labels = drawdowns.map((_, i) => `Day ${i + 1}`);
        
        this.drawdownChart.data.labels = labels;
        this.drawdownChart.data.datasets[0].data = drawdowns;
        this.drawdownChart.update();
    }

    updateReturnDistribution() {
        const returns = this.performanceData.dailyReturns || [];
        if (returns.length === 0) return;

        // Create histogram bins
        const binCount = 20;
        const min = Math.min(...returns);
        const max = Math.max(...returns);
        const binWidth = (max - min) / binCount;

        const bins = Array(binCount).fill(0);
        const labels = [];

        for (let i = 0; i < binCount; i++) {
            const binStart = min + i * binWidth;
            const binEnd = min + (i + 1) * binWidth;
            labels.push(`${(binStart * 100).toFixed(1)}`);

            // Count returns in this bin
            returns.forEach(ret => {
                if (ret >= binStart && ret < binEnd) {
                    bins[i]++;
                }
            });
        }

        this.returnChart.data.labels = labels;
        this.returnChart.data.datasets[0].data = bins;
        this.returnChart.update();
    }

    updateRiskMetrics() {
        // Calculate Value at Risk (VaR)
        const returns = this.performanceData.dailyReturns || [];
        if (returns.length > 0) {
            const sortedReturns = [...returns].sort((a, b) => a - b);
            const var95 = sortedReturns[Math.floor(returns.length * 0.05)];
            const var99 = sortedReturns[Math.floor(returns.length * 0.01)];

            // Update VaR display if elements exist
            const var95El = document.getElementById('var95');
            const var99El = document.getElementById('var99');

            if (var95El) var95El.textContent = `${(var95 * 100).toFixed(2)}%`;
            if (var99El) var99El.textContent = `${(var99 * 100).toFixed(2)}%`;
        }
    }

    // Advanced portfolio analytics
    calculateSharpeRatio(returns, riskFreeRate = 0.02) {
        if (!returns || returns.length === 0) return 0;
        
        const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
        const annualizedReturn = avgReturn * 252; // 252 trading days
        
        const variance = returns.reduce((sum, ret) => {
            return sum + Math.pow(ret - avgReturn, 2);
        }, 0) / returns.length;
        
        const volatility = Math.sqrt(variance * 252);
        
        return volatility === 0 ? 0 : (annualizedReturn - riskFreeRate) / volatility;
    }

    calculateMaxDrawdown(portfolioValues) {
        if (!portfolioValues || portfolioValues.length === 0) return 0;
        
        let maxDrawdown = 0;
        let peak = portfolioValues[0];
        
        for (let value of portfolioValues) {
            if (value > peak) {
                peak = value;
            }
            
            const drawdown = (peak - value) / peak;
            if (drawdown > maxDrawdown) {
                maxDrawdown = drawdown;
            }
        }
        
        return maxDrawdown * 100; // Return as percentage
    }

    // Export analytics data
    exportAnalytics() {
        const data = {
            performanceMetrics: this.performanceData,
            timestamp: new Date().toISOString(),
            summary: {
                sharpeRatio: this.performanceData.sharpeRatio,
                maxDrawdown: this.performanceData.maxDrawdown,
                volatility: this.performanceData.volatility,
                winRate: this.performanceData.winRate
            }
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `portfolio_analytics_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Initialize analytics when dashboard is ready
if (typeof window !== 'undefined') {
    window.DashboardAnalytics = DashboardAnalytics;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardAnalytics;
}