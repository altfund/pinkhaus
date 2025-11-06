import React, { useState, useEffect } from 'react';
import { useWeb3, useOminariContract, useChainName, useFormattedAddress } from '../hooks/useWeb3';
import './Web3Dashboard.css';

export function Web3Dashboard() {
  const { connected, loading, error, chainId, connect, disconnect, switchNetwork } = useWeb3();
  const { createSession, placeBet, optimizePortfolio, getSession, getSessionPositions } = useOminariContract();
  const chainName = useChainName();
  const formattedAddress = useFormattedAddress();
  
  const [activeSession, setActiveSession] = useState(null);
  const [positions, setPositions] = useState([]);
  const [markets, setMarkets] = useState([]);
  const [initialBankroll, setInitialBankroll] = useState('100');
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);

  // Load active session if connected
  useEffect(() => {
    if (connected && activeSession) {
      loadSessionData();
    }
  }, [connected, activeSession]);

  const loadSessionData = async () => {
    try {
      const sessionData = await getSession(activeSession);
      const positionsData = await getSessionPositions(activeSession);
      setPositions(positionsData);
      
      // Update session info
      document.getElementById('session-bankroll').textContent = `${sessionData.currentBankroll} ETH`;
      document.getElementById('session-profit').textContent = `${sessionData.totalProfit} ETH`;
      document.getElementById('session-bets').textContent = sessionData.totalBetsPlaced;
      document.getElementById('session-wins').textContent = sessionData.totalBetsWon;
    } catch (err) {
      console.error('Failed to load session data:', err);
    }
  };

  const handleCreateSession = async () => {
    if (!connected) {
      await connect();
      return;
    }
    
    setIsCreatingSession(true);
    try {
      const sessionId = await createSession(initialBankroll);
      setActiveSession(sessionId);
      alert(`Session created! ID: ${sessionId}`);
    } catch (err) {
      console.error('Failed to create session:', err);
      alert('Failed to create session: ' + err.message);
    } finally {
      setIsCreatingSession(false);
    }
  };

  const handleOptimizePortfolio = async () => {
    if (!activeSession) {
      alert('Please create a session first');
      return;
    }
    
    setIsOptimizing(true);
    try {
      // Get market IDs from current markets
      const marketIds = markets.map(m => m.marketId);
      const chunkDuration = 120; // 2 hours in minutes
      
      const { stakes, outcomes } = await optimizePortfolio(activeSession, marketIds, chunkDuration);
      
      // Display optimization results
      console.log('Optimization results:', { stakes, outcomes });
      alert(`Portfolio optimized! ${stakes.length} positions recommended.`);
      
      // Reload session data
      await loadSessionData();
    } catch (err) {
      console.error('Failed to optimize portfolio:', err);
      alert('Failed to optimize portfolio: ' + err.message);
    } finally {
      setIsOptimizing(false);
    }
  };

  const handlePlaceBet = async (marketId, stake, outcome) => {
    if (!activeSession) {
      alert('Please create a session first');
      return;
    }
    
    try {
      const positionId = await placeBet(activeSession, marketId, stake, outcome);
      alert(`Bet placed! Position ID: ${positionId}`);
      await loadSessionData();
    } catch (err) {
      console.error('Failed to place bet:', err);
      alert('Failed to place bet: ' + err.message);
    }
  };

  const renderWalletSection = () => {
    if (!connected) {
      return (
        <div className="wallet-section">
          <button className="connect-btn" onClick={connect} disabled={loading}>
            {loading ? 'Connecting...' : 'Connect Wallet'}
          </button>
          {error && <div className="error-msg">{error}</div>}
        </div>
      );
    }
    
    return (
      <div className="wallet-section connected">
        <div className="wallet-info">
          <span className="address">{formattedAddress}</span>
          <span className="network">{chainName}</span>
        </div>
        <button className="disconnect-btn" onClick={disconnect}>
          Disconnect
        </button>
        {chainId !== 137 && (
          <button className="switch-network-btn" onClick={() => switchNetwork(137)}>
            Switch to Polygon
          </button>
        )}
      </div>
    );
  };

  return (
    <div className="web3-dashboard">
      <header className="dashboard-header">
        <h1>🔗 Ominari DApp Trading</h1>
        {renderWalletSection()}
      </header>
      
      <div className="dashboard-content">
        {/* Session Management */}
        <section className="session-section">
          <h2>Trading Session</h2>
          {!activeSession ? (
            <div className="create-session">
              <input
                type="number"
                value={initialBankroll}
                onChange={(e) => setInitialBankroll(e.target.value)}
                placeholder="Initial bankroll (ETH)"
                min="0.01"
                step="0.01"
              />
              <button onClick={handleCreateSession} disabled={isCreatingSession}>
                {isCreatingSession ? 'Creating...' : 'Create Session'}
              </button>
            </div>
          ) : (
            <div className="session-info">
              <div className="info-grid">
                <div className="info-card">
                  <div className="label">Current Bankroll</div>
                  <div className="value" id="session-bankroll">-</div>
                </div>
                <div className="info-card">
                  <div className="label">Total Profit</div>
                  <div className="value" id="session-profit">-</div>
                </div>
                <div className="info-card">
                  <div className="label">Bets Placed</div>
                  <div className="value" id="session-bets">-</div>
                </div>
                <div className="info-card">
                  <div className="label">Bets Won</div>
                  <div className="value" id="session-wins">-</div>
                </div>
              </div>
            </div>
          )}
        </section>
        
        {/* Portfolio Optimization */}
        <section className="optimization-section">
          <h2>Kelly Optimization</h2>
          <button 
            className="optimize-btn" 
            onClick={handleOptimizePortfolio}
            disabled={!activeSession || isOptimizing}
          >
            {isOptimizing ? 'Optimizing...' : 'Optimize Portfolio'}
          </button>
        </section>
        
        {/* Active Positions */}
        <section className="positions-section">
          <h2>Active Positions</h2>
          <div className="positions-grid">
            {positions.length === 0 ? (
              <p className="no-positions">No active positions</p>
            ) : (
              positions.map((position) => (
                <div key={position.id} className="position-card">
                  <div className="position-header">
                    <span>Position #{position.id}</span>
                    <span className={position.isWon ? 'won' : position.isSettled ? 'lost' : 'pending'}>
                      {position.isSettled ? (position.isWon ? 'Won' : 'Lost') : 'Pending'}
                    </span>
                  </div>
                  <div className="position-details">
                    <div>Stake: {position.stake} ETH</div>
                    <div>Odds: {position.odds.toFixed(3)}</div>
                    <div>Outcome: {['Home', 'Draw', 'Away'][position.outcome]}</div>
                    {position.isSettled && position.isWon && (
                      <div>Payout: {position.payout} ETH</div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </section>
        
        {/* Market Browser */}
        <section className="markets-section">
          <h2>Available Markets</h2>
          <div className="markets-info">
            <p>Markets are fetched from on-chain data via TheGraph</p>
          </div>
        </section>
      </div>
    </div>
  );
}