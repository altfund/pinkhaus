# 🎯 Ominari Comprehensive Dashboard

## ✅ Dashboard is Live!

The comprehensive dashboard is now running and accessible at:

### 🌐 http://localhost:8889

## 📊 Features

The dashboard displays ALL relevant data from both the Overtime API and blockchain:

### 1. **Overview Statistics**
- Total Markets: 7,590 (5,774 API + 1,816 Blockchain)
- Active Markets with real-time updates
- Markets with Odds tracking
- Sports coverage (12+ different sports)

### 2. **Sport Distribution** 
Visual breakdown showing:
- Soccer: 1,744 markets
- Table Tennis: 1,078 markets  
- Unknown: 2,103 markets
- Football: 292 markets
- eSports: 259 markets
- Baseball: 130 markets
- Hockey: 80 markets
- And more...

### 3. **Data Sources**
Complete breakdown of all data sources:
- `api_live_real`: 5,774 markets (Overtime API)
- `blockchain_optimism_v2`: 1,119 markets
- `blockchain_arbitrum_v2`: 598 markets
- `blockchain_v2_optimism`: 97 markets

### 4. **Live Markets**
Shows upcoming markets in the next 24 hours with:
- Team names
- Sport classification
- Real-time odds (when available)
- Data source (API or Blockchain)

### 5. **Blockchain Integration**
- Recent blockchain markets from Arbitrum & Optimism
- Historical blockchain data
- Odds fetched from on-chain SportsAMMV2 contracts

### 6. **Sport Analysis Table**
Comprehensive breakdown by sport showing:
- Total markets per sport
- API markets count
- Blockchain markets count
- Markets with odds
- Average odds

### 7. **Best Odds Section**
Highlights markets with the best betting opportunities

## 🔄 Auto-Refresh
The dashboard automatically refreshes every 30 seconds to show the latest data.

## 📡 Data Sources

1. **Overtime API** (`https://api.overtime.io/overtime-v2/`)
   - Games info endpoint for live games
   - Sports endpoint for official sport definitions

2. **Blockchain Data**
   - Arbitrum RPC: `https://arb1.arbitrum.io/rpc`
   - Optimism RPC: `https://mainnet.optimism.io`
   - SportsAMMV2 contracts for odds

## 🎨 Features
- Clean, terminal-style design
- Real-time data updates
- Color-coded sources (Blue for API, Orange for Blockchain)
- Responsive grid layout
- Sport distribution visualization
- Comprehensive statistics

## 🚀 Access
Simply open your browser and navigate to:
**http://localhost:8889**

The dashboard is now displaying all the relevant Overtime data in one comprehensive view!