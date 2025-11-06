// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IOminariTrading.sol";
import "../interfaces/IKellyOptimizer.sol";
import "../interfaces/IChunkManager.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

/**
 * @title OminariTradingEngine
 * @dev Main trading engine for the Ominari DApp
 */
contract OminariTradingEngine is IOminariTrading, Ownable, ReentrancyGuard, Pausable {
    // State variables
    IKellyOptimizer public kellyOptimizer;
    IChunkManager public chunkManager;
    address public oracleAddress;
    
    uint256 public nextSessionId = 1;
    uint256 public nextPositionId = 1;
    
    uint256 public constant MIN_BANKROLL = 0.01 ether;
    uint256 public constant MAX_POSITIONS_PER_SESSION = 100;
    uint256 public constant SESSION_TIMEOUT = 30 days;
    
    // Mappings
    mapping(uint256 => TradingSession) public sessions;
    mapping(uint256 => Position) public positions;
    mapping(uint256 => uint256[]) public sessionPositions;
    mapping(address => uint256[]) public traderSessions;
    mapping(bytes32 => MarketData) public markets;
    
    // Modifiers
    modifier onlyOracle() {
        require(msg.sender == oracleAddress, "Only oracle can call");
        _;
    }
    
    modifier onlySessionOwner(uint256 sessionId) {
        require(sessions[sessionId].trader == msg.sender, "Not session owner");
        _;
    }
    
    modifier sessionActive(uint256 sessionId) {
        require(sessions[sessionId].isActive, "Session not active");
        require(
            block.timestamp - sessions[sessionId].lastActivityTime < SESSION_TIMEOUT,
            "Session timed out"
        );
        _;
    }
    
    constructor(address _kellyOptimizer, address _chunkManager) {
        kellyOptimizer = IKellyOptimizer(_kellyOptimizer);
        chunkManager = IChunkManager(_chunkManager);
    }
    
    /**
     * @dev Create a new trading session
     */
    function createSession(uint256 initialBankroll) 
        external 
        override 
        nonReentrant 
        whenNotPaused 
        returns (uint256 sessionId) 
    {
        require(initialBankroll >= MIN_BANKROLL, "Insufficient initial bankroll");
        
        sessionId = nextSessionId++;
        
        sessions[sessionId] = TradingSession({
            id: sessionId,
            trader: msg.sender,
            initialBankroll: initialBankroll,
            currentBankroll: initialBankroll,
            startTime: block.timestamp,
            lastActivityTime: block.timestamp,
            isActive: true,
            totalBetsPlaced: 0,
            totalBetsWon: 0,
            totalProfit: 0
        });
        
        traderSessions[msg.sender].push(sessionId);
        
        emit SessionCreated(sessionId, msg.sender, initialBankroll);
    }
    
    /**
     * @dev Place a bet in a session
     */
    function placeBet(
        uint256 sessionId,
        bytes32 marketId,
        uint256 stake,
        uint8 outcome
    ) 
        external 
        override 
        nonReentrant 
        whenNotPaused
        onlySessionOwner(sessionId)
        sessionActive(sessionId)
        returns (uint256 positionId) 
    {
        TradingSession storage session = sessions[sessionId];
        MarketData memory market = markets[marketId];
        
        require(market.maturityDate > block.timestamp, "Market expired");
        require(!market.isResolved, "Market already resolved");
        require(outcome < 3, "Invalid outcome");
        require(stake > 0 && stake <= session.currentBankroll, "Invalid stake");
        
        positionId = nextPositionId++;
        
        positions[positionId] = Position({
            id: positionId,
            sessionId: sessionId,
            marketId: marketId,
            stake: stake,
            odds: market.odds[outcome],
            outcome: outcome,
            timestamp: block.timestamp,
            isSettled: false,
            isWon: false,
            payout: 0
        });
        
        session.currentBankroll -= stake;
        session.totalBetsPlaced++;
        session.lastActivityTime = block.timestamp;
        sessionPositions[sessionId].push(positionId);
        
        emit PositionPlaced(positionId, sessionId, marketId, stake, outcome);
    }
    
    /**
     * @dev Settle a position (called by oracle)
     */
    function settlePosition(uint256 positionId) 
        external 
        override 
        nonReentrant
        onlyOracle 
    {
        Position storage position = positions[positionId];
        require(!position.isSettled, "Already settled");
        
        MarketData memory market = markets[position.marketId];
        require(market.isResolved, "Market not resolved");
        
        position.isSettled = true;
        
        if (market.winningOutcome == position.outcome) {
            position.isWon = true;
            position.payout = position.stake * position.odds / 1e18;
            
            TradingSession storage session = sessions[position.sessionId];
            session.currentBankroll += position.payout;
            session.totalBetsWon++;
            session.totalProfit += int256(position.payout) - int256(position.stake);
        }
        
        emit PositionSettled(positionId, position.isWon, position.payout);
    }
    
    /**
     * @dev Close a trading session
     */
    function closeSession(uint256 sessionId) 
        external 
        override 
        nonReentrant
        onlySessionOwner(sessionId) 
    {
        TradingSession storage session = sessions[sessionId];
        require(session.isActive, "Session already closed");
        
        // Check all positions are settled
        uint256[] memory positionIds = sessionPositions[sessionId];
        for (uint256 i = 0; i < positionIds.length; i++) {
            require(positions[positionIds[i]].isSettled, "Unsettled positions exist");
        }
        
        session.isActive = false;
        session.totalProfit = int256(session.currentBankroll) - int256(session.initialBankroll);
        
        emit SessionClosed(sessionId, session.currentBankroll, uint256(session.totalProfit));
    }
    
    /**
     * @dev Optimize portfolio using Kelly criterion
     */
    function optimizePortfolio(
        uint256 sessionId,
        bytes32[] calldata marketIds,
        uint256 chunkDurationMinutes
    ) 
        external 
        override 
        whenNotPaused
        onlySessionOwner(sessionId)
        sessionActive(sessionId)
        returns (uint256[] memory stakes, uint8[] memory outcomes) 
    {
        TradingSession memory session = sessions[sessionId];
        
        // Prepare data for optimization
        uint256[][3] memory odds = new uint256[][3](marketIds.length);
        uint256[][3] memory probabilities = new uint256[][3](marketIds.length);
        
        for (uint256 i = 0; i < marketIds.length; i++) {
            MarketData memory market = markets[marketIds[i]];
            odds[i] = market.odds;
            // In production, probabilities would come from oracle
            // For now, using implied probabilities
            for (uint256 j = 0; j < 3; j++) {
                probabilities[i][j] = 1e18 / market.odds[j];
            }
        }
        
        // Call Kelly optimizer
        IKellyOptimizer.OptimizationParams memory params = IKellyOptimizer.OptimizationParams({
            bankroll: session.currentBankroll,
            maxStakePercentage: 500, // 5% max per bet
            confidenceThreshold: 100, // 1% minimum edge
            useHalfKelly: true
        });
        
        IKellyOptimizer.MarketOpportunity[] memory opportunities = kellyOptimizer.optimizePortfolio(
            params,
            marketIds,
            odds,
            probabilities
        );
        
        // Extract stakes and outcomes
        stakes = new uint256[](opportunities.length);
        outcomes = new uint8[](opportunities.length);
        
        for (uint256 i = 0; i < opportunities.length; i++) {
            stakes[i] = opportunities[i].recommendedStake;
            outcomes[i] = opportunities[i].bestOutcome;
        }
        
        emit PortfolioOptimized(sessionId, block.timestamp, marketIds.length);
    }
    
    // View functions
    function getSession(uint256 sessionId) external view override returns (TradingSession memory) {
        return sessions[sessionId];
    }
    
    function getPosition(uint256 positionId) external view override returns (Position memory) {
        return positions[positionId];
    }
    
    function getSessionPositions(uint256 sessionId) external view override returns (Position[] memory) {
        uint256[] memory positionIds = sessionPositions[sessionId];
        Position[] memory sessionPositionList = new Position[](positionIds.length);
        
        for (uint256 i = 0; i < positionIds.length; i++) {
            sessionPositionList[i] = positions[positionIds[i]];
        }
        
        return sessionPositionList;
    }
    
    function getActiveSessionsForTrader(address trader) external view override returns (uint256[] memory) {
        uint256[] memory allSessions = traderSessions[trader];
        uint256 activeCount = 0;
        
        // Count active sessions
        for (uint256 i = 0; i < allSessions.length; i++) {
            if (sessions[allSessions[i]].isActive) {
                activeCount++;
            }
        }
        
        // Create array of active sessions
        uint256[] memory activeSessions = new uint256[](activeCount);
        uint256 index = 0;
        
        for (uint256 i = 0; i < allSessions.length; i++) {
            if (sessions[allSessions[i]].isActive) {
                activeSessions[index++] = allSessions[i];
            }
        }
        
        return activeSessions;
    }
    
    function calculateKellyFraction(
        uint256 bankroll,
        uint256 odds,
        uint256 probability
    ) external pure override returns (uint256) {
        // Simple Kelly formula: f = (p(b+1) - 1) / b
        // where p = probability, b = odds - 1
        uint256 b = odds - 1e18;
        uint256 numerator = probability * odds - 1e18;
        uint256 fraction = numerator / b;
        
        // Apply half-Kelly for safety
        return fraction / 2;
    }
    
    // Admin functions
    function setOracle(address _oracle) external onlyOwner {
        oracleAddress = _oracle;
    }
    
    function updateMarket(
        bytes32 marketId,
        string memory homeTeam,
        string memory awayTeam,
        uint256[3] memory odds,
        uint256 maturityDate
    ) external onlyOracle {
        markets[marketId] = MarketData({
            marketId: marketId,
            homeTeam: homeTeam,
            awayTeam: awayTeam,
            odds: odds,
            maturityDate: maturityDate,
            isResolved: false,
            winningOutcome: 0
        });
    }
    
    function resolveMarket(bytes32 marketId, uint8 winningOutcome) external onlyOracle {
        markets[marketId].isResolved = true;
        markets[marketId].winningOutcome = winningOutcome;
    }
    
    function pause() external onlyOwner {
        _pause();
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
}