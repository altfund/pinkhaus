// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IKellyOptimizer.sol";

/**
 * @title KellyOptimizer
 * @dev On-chain Kelly Criterion calculations for optimal bet sizing
 */
contract KellyOptimizer is IKellyOptimizer {
    uint256 private constant PRECISION = 1e18;
    uint256 private constant MAX_KELLY_FRACTION = 0.25e18; // 25% max
    
    /**
     * @dev Calculate Kelly fraction for a single bet
     */
    function calculateKellyFraction(
        uint256 odds,
        uint256 probability,
        bool useHalfKelly
    ) external pure override returns (uint256 kellyFraction) {
        require(odds >= PRECISION, "Odds must be >= 1");
        require(probability <= PRECISION, "Probability must be <= 1");
        
        // Kelly formula: f = (p(b+1) - 1) / b
        // where p = probability, b = odds - 1
        
        uint256 b = odds - PRECISION;
        if (b == 0) return 0;
        
        uint256 expectedReturn = (probability * odds) / PRECISION;
        if (expectedReturn <= PRECISION) return 0; // No edge
        
        uint256 numerator = expectedReturn - PRECISION;
        kellyFraction = (numerator * PRECISION) / b;
        
        // Apply half-Kelly if requested
        if (useHalfKelly) {
            kellyFraction = kellyFraction / 2;
        }
        
        // Cap at maximum fraction
        if (kellyFraction > MAX_KELLY_FRACTION) {
            kellyFraction = MAX_KELLY_FRACTION;
        }
        
        return kellyFraction;
    }
    
    /**
     * @dev Optimize portfolio across multiple markets
     */
    function optimizePortfolio(
        OptimizationParams calldata params,
        bytes32[] calldata marketIds,
        uint256[][3] calldata odds,
        uint256[][3] calldata probabilities
    ) external view override returns (MarketOpportunity[] memory opportunities) {
        require(marketIds.length == odds.length, "Array length mismatch");
        require(marketIds.length == probabilities.length, "Array length mismatch");
        
        opportunities = new MarketOpportunity[](marketIds.length);
        uint256 opportunityCount = 0;
        
        for (uint256 i = 0; i < marketIds.length; i++) {
            // Find best outcome for this market
            uint256 bestOutcome = 0;
            uint256 bestEdge = 0;
            uint256 bestKelly = 0;
            
            for (uint256 j = 0; j < 3; j++) {
                if (odds[i][j] == 0) continue; // Skip if no odds
                
                // Calculate edge
                uint256 expectedValue = (probabilities[i][j] * odds[i][j]) / PRECISION;
                
                if (expectedValue > PRECISION) {
                    uint256 edge = expectedValue - PRECISION;
                    
                    if (edge > bestEdge && edge >= params.confidenceThreshold) {
                        bestOutcome = j;
                        bestEdge = edge;
                        
                        // Calculate Kelly fraction
                        uint256 b = odds[i][j] - PRECISION;
                        if (b > 0) {
                            bestKelly = (edge * PRECISION) / b;
                            if (params.useHalfKelly) {
                                bestKelly = bestKelly / 2;
                            }
                        }
                    }
                }
            }
            
            // If we found a good opportunity
            if (bestEdge > 0) {
                // Calculate stake based on Kelly fraction and max stake limit
                uint256 kellyStake = (params.bankroll * bestKelly) / PRECISION;
                uint256 maxStake = (params.bankroll * params.maxStakePercentage) / 10000;
                uint256 recommendedStake = kellyStake < maxStake ? kellyStake : maxStake;
                
                opportunities[opportunityCount] = MarketOpportunity({
                    marketId: marketIds[i],
                    odds: odds[i],
                    probabilities: probabilities[i],
                    edges: [uint256(0), uint256(0), uint256(0)], // Would calculate all
                    bestOutcome: uint8(bestOutcome),
                    kellyFraction: bestKelly,
                    recommendedStake: recommendedStake
                });
                
                // Calculate edges for all outcomes
                for (uint256 k = 0; k < 3; k++) {
                    if (odds[i][k] > 0) {
                        uint256 ev = (probabilities[i][k] * odds[i][k]) / PRECISION;
                        opportunities[opportunityCount].edges[k] = ev > PRECISION ? ev - PRECISION : 0;
                    }
                }
                
                opportunityCount++;
            }
        }
        
        // Resize array to actual opportunities found
        assembly {
            mstore(opportunities, opportunityCount)
        }
        
        emit OptimizationCompleted(
            msg.sender,
            marketIds.length,
            opportunityCount,
            _calculateTotalStake(opportunities)
        );
        
        return opportunities;
    }
    
    /**
     * @dev Calculate expected value for a bet
     */
    function calculateExpectedValue(
        uint256 stake,
        uint256 odds,
        uint256 probability
    ) external pure override returns (int256 expectedValue) {
        uint256 expectedReturn = (stake * odds * probability) / PRECISION;
        expectedValue = int256(expectedReturn) - int256(stake);
    }
    
    /**
     * @dev Validate optimization parameters
     */
    function validateParameters(
        OptimizationParams calldata params
    ) external pure override returns (bool isValid, string memory reason) {
        if (params.bankroll == 0) {
            return (false, "Bankroll cannot be zero");
        }
        if (params.maxStakePercentage > 1000) { // 10%
            return (false, "Max stake percentage too high");
        }
        if (params.maxStakePercentage == 0) {
            return (false, "Max stake percentage cannot be zero");
        }
        return (true, "");
    }
    
    /**
     * @dev Internal function to calculate total stake
     */
    function _calculateTotalStake(MarketOpportunity[] memory opportunities) 
        private pure returns (uint256 total) 
    {
        for (uint256 i = 0; i < opportunities.length; i++) {
            total += opportunities[i].recommendedStake;
        }
    }
}