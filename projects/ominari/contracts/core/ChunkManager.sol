// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IChunkManager.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title ChunkManager
 * @dev Manages dynamic time-based market chunking
 */
contract ChunkManager is IChunkManager, Ownable {
    uint256 private nextChunkId = 1;
    
    mapping(uint256 => Chunk) public chunks;
    mapping(string => SportTimings) public sportTimings;
    mapping(uint256 => uint256) public marketToChunk;
    
    constructor() {
        // Initialize default sport timings
        sportTimings["soccer"] = SportTimings({
            sport: "soccer",
            averageDuration: 105,
            settlementTime: 15,
            minGapBetweenChunks: 15
        });
        
        sportTimings["basketball"] = SportTimings({
            sport: "basketball",
            averageDuration: 150,
            settlementTime: 10,
            minGapBetweenChunks: 10
        });
        
        sportTimings["tennis"] = SportTimings({
            sport: "tennis",
            averageDuration: 120,
            settlementTime: 5,
            minGapBetweenChunks: 10
        });
    }
    
    /**
     * @dev Create optimal chunks from a set of markets
     */
    function createChunks(
        bytes32[] calldata marketIds,
        uint256[] calldata maturityDates,
        string[] calldata sports,
        ChunkingParams calldata params
    ) external override returns (uint256[] memory chunkIds) {
        require(marketIds.length == maturityDates.length, "Array length mismatch");
        require(marketIds.length == sports.length, "Array length mismatch");
        
        // Sort markets by maturity date (would need external sorting in production)
        // For now, assume they're already sorted
        
        uint256[] memory createdChunks = new uint256[](marketIds.length);
        uint256 chunkCount = 0;
        uint256 currentChunkId = 0;
        uint256 currentChunkEndTime = 0;
        uint256 marketsInCurrentChunk = 0;
        
        for (uint256 i = 0; i < marketIds.length; i++) {
            SportTimings memory timing = sportTimings[sports[i]];
            uint256 marketStartTime = maturityDates[i];
            uint256 marketEndTime = marketStartTime + (timing.averageDuration * 60);
            uint256 settlementEndTime = marketEndTime + (timing.settlementTime * 60);
            
            // Check if we need a new chunk
            bool needNewChunk = false;
            
            if (currentChunkId == 0) {
                needNewChunk = true;
            } else if (marketStartTime > currentChunkEndTime + (params.minChunkDuration * 60)) {
                needNewChunk = true;
            } else if (marketsInCurrentChunk >= params.maxMarketsPerChunk) {
                needNewChunk = true;
            }
            
            // Create new chunk if needed
            if (needNewChunk) {
                currentChunkId = nextChunkId++;
                
                chunks[currentChunkId] = Chunk({
                    id: currentChunkId,
                    startTime: marketStartTime,
                    endTime: marketEndTime,
                    settlementTime: settlementEndTime,
                    marketIds: new bytes32[](0),
                    totalCapitalAllocated: 0,
                    expectedCapitalReturn: 0,
                    isActive: true,
                    isSettled: false
                });
                
                currentChunkEndTime = settlementEndTime;
                marketsInCurrentChunk = 0;
                createdChunks[chunkCount++] = currentChunkId;
                
                emit ChunkCreated(
                    currentChunkId,
                    marketStartTime,
                    settlementEndTime,
                    1
                );
            }
            
            // Add market to current chunk
            marketToChunk[uint256(marketIds[i])] = currentChunkId;
            marketsInCurrentChunk++;
            
            // Update chunk end time if needed
            if (settlementEndTime > currentChunkEndTime) {
                chunks[currentChunkId].endTime = marketEndTime;
                chunks[currentChunkId].settlementTime = settlementEndTime;
                currentChunkEndTime = settlementEndTime;
            }
        }
        
        // Resize array to actual chunks created
        chunkIds = new uint256[](chunkCount);
        for (uint256 i = 0; i < chunkCount; i++) {
            chunkIds[i] = createdChunks[i];
        }
        
        return chunkIds;
    }
    
    /**
     * @dev Get markets for a specific chunk
     */
    function getChunkMarkets(uint256 chunkId) 
        external view override returns (bytes32[] memory) 
    {
        return chunks[chunkId].marketIds;
    }
    
    /**
     * @dev Update empirical timing data for a sport
     */
    function updateTimingData(
        string calldata sport,
        uint256 actualDuration,
        uint256 actualSettlementTime
    ) external override onlyOwner {
        SportTimings storage timing = sportTimings[sport];
        
        // Update with exponential moving average
        uint256 alpha = 200; // 20% weight to new data
        
        timing.averageDuration = (timing.averageDuration * (1000 - alpha) + actualDuration * alpha) / 1000;
        timing.settlementTime = (timing.settlementTime * (1000 - alpha) + actualSettlementTime * alpha) / 1000;
        
        emit TimingsUpdated(sport, timing.averageDuration, timing.settlementTime);
    }
    
    /**
     * @dev Get timing parameters for a sport
     */
    function getSportTimings(string calldata sport) 
        external view override returns (SportTimings memory) 
    {
        return sportTimings[sport];
    }
    
    /**
     * @dev Settle a chunk and record actual timings
     */
    function settleChunk(uint256 chunkId, uint256 capitalReturned) 
        external override onlyOwner 
    {
        Chunk storage chunk = chunks[chunkId];
        require(!chunk.isSettled, "Already settled");
        
        chunk.isSettled = true;
        chunk.isActive = false;
        chunk.expectedCapitalReturn = capitalReturned;
        
        emit ChunkSettled(chunkId, block.timestamp, capitalReturned);
    }
    
    /**
     * @dev Calculate optimal chunk size based on historical data
     */
    function calculateOptimalChunkSize(
        string calldata sport,
        uint256 targetCapital
    ) external view override returns (uint256 optimalDuration, uint256 maxMarkets) {
        SportTimings memory timing = sportTimings[sport];
        
        // Base calculation on sport characteristics
        uint256 totalCycleTime = timing.averageDuration + timing.settlementTime + timing.minGapBetweenChunks;
        
        // Aim for chunks that complete within 3-6 hours
        uint256 targetDuration = 4 * 60; // 4 hours in minutes
        maxMarkets = targetDuration / totalCycleTime;
        
        if (maxMarkets < 3) maxMarkets = 3;
        if (maxMarkets > 10) maxMarkets = 10;
        
        optimalDuration = maxMarkets * totalCycleTime;
        
        return (optimalDuration, maxMarkets);
    }
}