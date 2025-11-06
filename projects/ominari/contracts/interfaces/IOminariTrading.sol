// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IOminariTrading
 * @dev Interface for the main Ominari trading engine
 */
interface IOminariTrading {
    // Structs
    struct TradingSession {
        uint256 id;
        address trader;
        uint256 initialBankroll;
        uint256 currentBankroll;
        uint256 startTime;
        uint256 lastActivityTime;
        bool isActive;
        uint256 totalBetsPlaced;
        uint256 totalBetsWon;
        uint256 totalProfit;
    }

    struct Position {
        uint256 id;
        uint256 sessionId;
        bytes32 marketId;
        uint256 stake;
        uint256 odds;
        uint8 outcome; // 0: home, 1: draw, 2: away
        uint256 timestamp;
        bool isSettled;
        bool isWon;
        uint256 payout;
    }

    struct MarketData {
        bytes32 marketId;
        string homeTeam;
        string awayTeam;
        uint256[3] odds; // [home, draw, away]
        uint256 maturityDate;
        bool isResolved;
        uint8 winningOutcome;
    }

    // Events
    event SessionCreated(uint256 indexed sessionId, address indexed trader, uint256 bankroll);
    event PositionPlaced(uint256 indexed positionId, uint256 indexed sessionId, bytes32 marketId, uint256 stake, uint8 outcome);
    event PositionSettled(uint256 indexed positionId, bool won, uint256 payout);
    event SessionClosed(uint256 indexed sessionId, uint256 finalBankroll, uint256 totalProfit);
    event PortfolioOptimized(uint256 indexed sessionId, uint256 timestamp, uint256 marketCount);

    // Core Functions
    function createSession(uint256 initialBankroll) external returns (uint256 sessionId);
    
    function placeBet(
        uint256 sessionId,
        bytes32 marketId,
        uint256 stake,
        uint8 outcome
    ) external returns (uint256 positionId);
    
    function settlePosition(uint256 positionId) external;
    
    function closeSession(uint256 sessionId) external;
    
    function optimizePortfolio(
        uint256 sessionId,
        bytes32[] calldata marketIds,
        uint256 chunkDurationMinutes
    ) external returns (uint256[] memory stakes, uint8[] memory outcomes);
    
    // View Functions
    function getSession(uint256 sessionId) external view returns (TradingSession memory);
    
    function getPosition(uint256 positionId) external view returns (Position memory);
    
    function getSessionPositions(uint256 sessionId) external view returns (Position[] memory);
    
    function getActiveSessionsForTrader(address trader) external view returns (uint256[] memory);
    
    function calculateKellyFraction(
        uint256 bankroll,
        uint256 odds,
        uint256 probability
    ) external pure returns (uint256);
}