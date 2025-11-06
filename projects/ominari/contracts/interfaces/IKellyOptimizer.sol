// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IKellyOptimizer
 * @dev Interface for Kelly Criterion optimization calculations
 */
interface IKellyOptimizer {
    struct OptimizationParams {
        uint256 bankroll;
        uint256 maxStakePercentage; // Max percentage of bankroll per bet (basis points)
        uint256 confidenceThreshold; // Minimum edge required (basis points)
        bool useHalfKelly; // Conservative mode
    }

    struct MarketOpportunity {
        bytes32 marketId;
        uint256[3] odds; // [home, draw, away]
        uint256[3] probabilities; // Model probabilities
        uint256[3] edges; // Expected edge for each outcome
        uint8 bestOutcome;
        uint256 kellyFraction;
        uint256 recommendedStake;
    }

    event OptimizationCompleted(
        address indexed trader,
        uint256 marketsAnalyzed,
        uint256 positionsRecommended,
        uint256 totalStakeAllocated
    );

    /**
     * @dev Calculate Kelly fraction for a single bet
     * @param odds Decimal odds (scaled by 1e18)
     * @param probability Win probability (scaled by 1e18)
     * @param useHalfKelly Whether to use conservative half-Kelly
     * @return kellyFraction The optimal betting fraction (scaled by 1e18)
     */
    function calculateKellyFraction(
        uint256 odds,
        uint256 probability,
        bool useHalfKelly
    ) external pure returns (uint256 kellyFraction);

    /**
     * @dev Optimize portfolio across multiple markets
     * @param params Optimization parameters
     * @param marketIds Array of market identifiers
     * @param odds Array of odds for each market [home, draw, away]
     * @param probabilities Model probabilities for each market
     * @return opportunities Array of market opportunities with recommended stakes
     */
    function optimizePortfolio(
        OptimizationParams calldata params,
        bytes32[] calldata marketIds,
        uint256[][3] calldata odds,
        uint256[][3] calldata probabilities
    ) external view returns (MarketOpportunity[] memory opportunities);

    /**
     * @dev Calculate expected value for a bet
     * @param stake Bet amount
     * @param odds Decimal odds
     * @param probability Win probability
     * @return expectedValue The expected value of the bet
     */
    function calculateExpectedValue(
        uint256 stake,
        uint256 odds,
        uint256 probability
    ) external pure returns (int256 expectedValue);

    /**
     * @dev Validate optimization parameters
     * @param params Parameters to validate
     * @return isValid Whether parameters are valid
     * @return reason Error message if invalid
     */
    function validateParameters(
        OptimizationParams calldata params
    ) external pure returns (bool isValid, string memory reason);
}