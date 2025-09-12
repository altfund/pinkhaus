#!/usr/bin/env python3
"""Overtime V2 Quote Service for getting execution quotes."""

import os
import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class OvertimeQuoteService:
    """Service for fetching quotes from Overtime V2 API."""
    
    def __init__(self):
        self.api_key = os.getenv("OVERTIME_API_KEY")
        self.network_id = os.getenv("OVERTIME_NETWORK_ID", "10")  # Default to Optimism
        self.base_url = "https://api.overtime.io/overtime-v2"
        self.quote_url = f"{self.base_url}/networks/{self.network_id}/quote"
        
        if not self.api_key:
            raise ValueError("OVERTIME_API_KEY environment variable not set")
    
    def get_quote(self, 
                  buy_in_amount: float,
                  trade_data: List[Dict[str, Any]],
                  collateral: str = "THALES") -> Optional[Dict[str, Any]]:
        """
        Get quote from Overtime for a set of trades.
        
        Args:
            buy_in_amount: Amount to bet
            trade_data: List of market data for trades
            collateral: Collateral type (THALES, ETH, WETH)
            
        Returns:
            Quote response from Overtime API
        """
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "buyInAmount": buy_in_amount,
            "tradeData": trade_data,
            "collateral": collateral
        }
        
        try:
            logger.info(f"Fetching quote for {len(trade_data)} positions with buy-in: ${buy_in_amount}")
            response = requests.post(self.quote_url, 
                                   headers=headers, 
                                   json=payload,
                                   timeout=10)
            response.raise_for_status()
            
            quote = response.json()
            logger.info(f"Quote received - Total odds: {quote['quoteData']['totalQuote']['decimal']:.3f}")
            return quote
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching quote: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            return None
    
    def prepare_trade_data(self, market: Dict[str, Any], position: int) -> Dict[str, Any]:
        """
        Prepare trade data for a single market position.
        
        Args:
            market: Market data from Overtime API
            position: Position to bet on (0=Home, 1=Away, 2=Draw for soccer)
            
        Returns:
            Trade data formatted for quote request
        """
        return {
            "gameId": market.get("gameId"),
            "sportId": market.get("sportId", 1),  # Soccer
            "typeId": market.get("typeId", 0),    # Winner market
            "maturity": market.get("maturity"),
            "status": market.get("status", 0),
            "line": market.get("line", 0),
            "playerId": market.get("playerId", 0),
            "odds": market.get("normalizedImpliedOdds", []),
            "merkleProof": market.get("proof", []),
            "position": position,
            "combinedPositions": market.get("combinedPositions", [[], [], []]),
            "live": False
        }
    
    def get_parlay_quote(self, trades: List[Dict[str, Any]], total_stake: float) -> Optional[Dict[str, Any]]:
        """
        Get quote for a parlay (multiple positions).
        
        Args:
            trades: List of trade data for parlay
            total_stake: Total amount to stake
            
        Returns:
            Quote response
        """
        return self.get_quote(total_stake, trades)
    
    def get_single_quote(self, market: Dict[str, Any], position: int, stake: float) -> Optional[Dict[str, Any]]:
        """
        Get quote for a single position.
        
        Args:
            market: Market data
            position: Position to bet on
            stake: Amount to stake
            
        Returns:
            Quote response
        """
        trade_data = [self.prepare_trade_data(market, position)]
        return self.get_quote(stake, trade_data)
    
    def extract_execution_price(self, quote: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract execution price details from quote.
        
        Args:
            quote: Quote response from API
            
        Returns:
            Dictionary with decimal odds, payout, profit
        """
        if not quote or 'quoteData' not in quote:
            return {}
        
        quote_data = quote['quoteData']
        total_quote = quote_data.get('totalQuote', {})
        payout = quote_data.get('payout', {})
        profit = quote_data.get('potentialProfit', {})
        
        return {
            'decimal_odds': total_quote.get('decimal', 0),
            'american_odds': total_quote.get('american', 0),
            'normalized_implied': total_quote.get('normalizedImplied', 0),
            'payout_amount': payout.get('THALES', 0),
            'payout_usd': payout.get('usd', 0),
            'profit_amount': profit.get('THALES', 0),
            'profit_usd': profit.get('usd', 0),
            'profit_percentage': profit.get('percentage', 0),
            'buy_in_usd': quote_data.get('buyInAmountInUsd', 0),
            'liquidity_usd': quote.get('liquidityData', {}).get('ticketLiquidityInUsd', 0)
        }


# Example usage
if __name__ == "__main__":
    # Test the quote service
    service = OvertimeQuoteService()
    
    # Example market data (would come from markets API)
    test_market = {
        "gameId": "0x3230323530393034444530453841303500000000000000000000000000000000",
        "sportId": 1,
        "typeId": 0,
        "maturity": 1719342000,
        "status": 0,
        "line": 0,
        "playerId": 0,
        "normalizedImpliedOdds": [0.740740740741, 0.102249488753, 0.208768267223],
        "proof": [
            "0xc4788d799bccce5adea24c9a3088da1072ba0a4e7405184cf164cd8bc8dc715e",
            "0x8f2f3a9252434ac2320a8b9833608ef0bd382a145880216e28fc16048bbdf8b2"
        ]
    }
    
    # Get quote for position 1 (Away) with $20 stake
    quote = service.get_single_quote(test_market, position=1, stake=20)
    if quote:
        details = service.extract_execution_price(quote)
        print(f"Execution quote: {details}")