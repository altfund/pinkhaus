// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title GasOptimizedStorage
 * @dev Library for gas-efficient storage patterns
 */
library GasOptimizedStorage {
    
    /**
     * @dev Pack position data into a single storage slot
     * Layout (256 bits total):
     * - stake: 128 bits (enough for 340282366920938463463 wei)
     * - odds: 32 bits (max 4294.967295 with 6 decimal precision)
     * - timestamp: 40 bits (enough until year 36812)
     * - outcome: 2 bits (0-2 for home/draw/away)
     * - isSettled: 1 bit
     * - isWon: 1 bit
     * - reserved: 52 bits for future use
     */
    struct PackedPosition {
        uint256 data;
        uint256 payoutAndIds; // Separate slot for payout + IDs
    }
    
    function packPosition(
        uint128 stake,
        uint32 odds,
        uint40 timestamp,
        uint8 outcome,
        bool isSettled,
        bool isWon
    ) internal pure returns (uint256) {
        require(outcome < 3, "Invalid outcome");
        
        uint256 packed = uint256(stake);
        packed |= uint256(odds) << 128;
        packed |= uint256(timestamp) << 160;
        packed |= uint256(outcome) << 200;
        packed |= uint256(isSettled ? 1 : 0) << 202;
        packed |= uint256(isWon ? 1 : 0) << 203;
        
        return packed;
    }
    
    function unpackStake(uint256 packed) internal pure returns (uint128) {
        return uint128(packed);
    }
    
    function unpackOdds(uint256 packed) internal pure returns (uint32) {
        return uint32(packed >> 128);
    }
    
    function unpackTimestamp(uint256 packed) internal pure returns (uint40) {
        return uint40(packed >> 160);
    }
    
    function unpackOutcome(uint256 packed) internal pure returns (uint8) {
        return uint8(packed >> 200) & 0x03;
    }
    
    function unpackIsSettled(uint256 packed) internal pure returns (bool) {
        return ((packed >> 202) & 0x01) == 1;
    }
    
    function unpackIsWon(uint256 packed) internal pure returns (bool) {
        return ((packed >> 203) & 0x01) == 1;
    }
    
    /**
     * @dev Pack session data efficiently
     * Uses 2 storage slots instead of 10
     */
    struct PackedSession {
        uint256 slot1; // bankrolls + timestamps
        uint256 slot2; // counters + flags
        address trader;
    }
    
    function packSessionSlot1(
        uint128 initialBankroll,
        uint128 currentBankroll
    ) internal pure returns (uint256) {
        return uint256(initialBankroll) | (uint256(currentBankroll) << 128);
    }
    
    function packSessionSlot2(
        uint32 startTime,
        uint32 lastActivity,
        uint16 betsPlaced,
        uint16 betsWon,
        bool isActive
    ) internal pure returns (uint256) {
        uint256 packed = uint256(startTime);
        packed |= uint256(lastActivity) << 32;
        packed |= uint256(betsPlaced) << 64;
        packed |= uint256(betsWon) << 80;
        packed |= uint256(isActive ? 1 : 0) << 96;
        return packed;
    }
}