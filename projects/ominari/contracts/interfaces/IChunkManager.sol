// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IChunkManager
 * @dev Interface for dynamic time-based market chunking
 */
interface IChunkManager {
    struct Chunk {
        uint256 id;
        uint256 startTime;
        uint256 endTime;
        uint256 settlementTime;
        bytes32[] marketIds;
        uint256 totalCapitalAllocated;
        uint256 expectedCapitalReturn;
        bool isActive;
        bool isSettled;
    }

    struct ChunkingParams {
        uint256 minChunkDuration; // Minimum minutes between chunks
        uint256 maxChunkDuration; // Maximum minutes for a chunk
        uint256 settlementBuffer; // Minutes to add for settlement
        uint256 maxMarketsPerChunk; // Max markets in one chunk
        uint256 targetCapitalPerChunk; // Target capital allocation
    }

    struct SportTimings {
        string sport;
        uint256 averageDuration; // Average match duration in minutes
        uint256 settlementTime; // Average settlement time after match
        uint256 minGapBetweenChunks; // Minimum gap required
    }

    event ChunkCreated(
        uint256 indexed chunkId,
        uint256 startTime,
        uint256 endTime,
        uint256 marketCount
    );

    event ChunkSettled(
        uint256 indexed chunkId,
        uint256 actualSettlementTime,
        uint256 capitalReturned
    );

    event TimingsUpdated(
        string sport,
        uint256 duration,
        uint256 settlementTime
    );

    /**
     * @dev Create optimal chunks from a set of markets
     * @param marketIds Array of market identifiers
     * @param maturityDates Array of market maturity timestamps
     * @param sports Array of sport types for each market
     * @param params Chunking parameters
     * @return chunkIds Array of created chunk IDs
     */
    function createChunks(
        bytes32[] calldata marketIds,
        uint256[] calldata maturityDates,
        string[] calldata sports,
        ChunkingParams calldata params
    ) external returns (uint256[] memory chunkIds);

    /**
     * @dev Get markets for a specific chunk
     * @param chunkId The chunk identifier
     * @return marketIds Array of market IDs in the chunk
     */
    function getChunkMarkets(
        uint256 chunkId
    ) external view returns (bytes32[] memory marketIds);

    /**
     * @dev Update empirical timing data for a sport
     * @param sport Sport identifier
     * @param actualDuration Observed match duration
     * @param actualSettlementTime Observed settlement time
     */
    function updateTimingData(
        string calldata sport,
        uint256 actualDuration,
        uint256 actualSettlementTime
    ) external;

    /**
     * @dev Get timing parameters for a sport
     * @param sport Sport identifier
     * @return timings The timing parameters
     */
    function getSportTimings(
        string calldata sport
    ) external view returns (SportTimings memory timings);

    /**
     * @dev Settle a chunk and record actual timings
     * @param chunkId The chunk to settle
     * @param capitalReturned Actual capital returned
     */
    function settleChunk(
        uint256 chunkId,
        uint256 capitalReturned
    ) external;

    /**
     * @dev Calculate optimal chunk size based on historical data
     * @param sport Sport type
     * @param targetCapital Target capital allocation
     * @return optimalDuration Recommended chunk duration
     * @return maxMarkets Recommended max markets
     */
    function calculateOptimalChunkSize(
        string calldata sport,
        uint256 targetCapital
    ) external view returns (uint256 optimalDuration, uint256 maxMarkets);
}