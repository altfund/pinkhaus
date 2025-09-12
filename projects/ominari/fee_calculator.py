#!/usr/bin/env python3
"""Fee calculation utilities for paper trading, matching Overtime's fee structure."""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class FeeCalculator:
    """Calculate fees for Overtime Markets trades."""
    
    # Default fee structure based on evaluate_open_markets.py
    DEFAULT_SAFEBOX_FEE = 0.02  # 2% SafeBox fee
    DEFAULT_SKEW_FEE = 0.01     # 1% default skew impact
    
    def __init__(self):
        """Initialize fee calculator."""
        self.safebox_fee = self.DEFAULT_SAFEBOX_FEE
        self.default_skew_fee = self.DEFAULT_SKEW_FEE
        
    def calculate_entry_fees(self, 
                           stake: float, 
                           odds: float,
                           skew_fee: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate fees when entering a position.
        
        Args:
            stake: Base stake amount
            odds: Decimal odds for the position
            skew_fee: Optional skew fee (defaults to 1%)
            
        Returns:
            Dictionary with fee breakdown
        """
        # Use provided skew or default
        skew = skew_fee if skew_fee is not None else self.default_skew_fee
        
        # Total fee percentage
        total_fee_pct = self.safebox_fee + skew
        
        # Calculate amounts
        fee_amount = stake * total_fee_pct
        execution_stake = stake + fee_amount
        
        # Adjusted odds after fees
        adjusted_odds = odds / (1.0 + total_fee_pct)
        
        return {
            'stake': stake,
            'safebox_fee_pct': self.safebox_fee,
            'skew_fee_pct': skew,
            'total_fee_pct': total_fee_pct,
            'fee_amount': fee_amount,
            'execution_stake': execution_stake,
            'entry_odds': odds,
            'adjusted_odds': adjusted_odds,
            'implied_prob': 1.0 / adjusted_odds if adjusted_odds > 0 else 0
        }
    
    def calculate_exit_payout(self,
                            stake: float,
                            odds: float,
                            won: bool,
                            fee_info: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate payout when position settles.
        
        Args:
            stake: Original stake amount (before fees)
            odds: Entry odds
            won: Whether the bet won
            fee_info: Fee info from entry (if available)
            
        Returns:
            Dictionary with payout details
        """
        if not won:
            # Lost - lose entire execution stake
            execution_stake = fee_info.get('execution_stake', stake) if fee_info else stake
            return {
                'gross_payout': 0,
                'net_payout': 0,
                'pnl': -execution_stake,
                'roi_pct': -100.0
            }
        
        # Won - calculate payout
        # Gross payout is stake * odds
        gross_payout = stake * odds
        
        # No additional fees on exit for Overtime (fees paid on entry)
        net_payout = gross_payout
        
        # P&L calculation
        execution_stake = fee_info.get('execution_stake', stake) if fee_info else stake
        pnl = net_payout - execution_stake
        
        # ROI calculation
        roi_pct = (pnl / execution_stake) * 100 if execution_stake > 0 else 0
        
        return {
            'gross_payout': gross_payout,
            'net_payout': net_payout,
            'pnl': pnl,
            'roi_pct': roi_pct
        }
    
    def extract_quote_fees(self, quote: Dict) -> Optional[Dict[str, float]]:
        """
        Extract fee information from Overtime quote response.
        
        Args:
            quote: Quote response from Overtime API
            
        Returns:
            Dictionary with fee breakdown or None
        """
        try:
            if not quote or 'quoteData' not in quote:
                return None
                
            quote_data = quote['quoteData']
            
            # Try to extract fee info from quote
            # Note: Actual quote structure may vary - this is based on typical responses
            
            # Calculate implied fees from odds difference
            buy_in = quote_data.get('buyInAmountInUsd', 0)
            payout = quote_data.get('payout', {}).get('usd', 0)
            total_quote = quote_data.get('totalQuote', {})
            decimal_odds = total_quote.get('decimal', 0)
            
            if buy_in > 0 and decimal_odds > 0:
                # Implied fee from difference between theoretical and actual payout
                theoretical_payout = buy_in * decimal_odds
                if theoretical_payout > payout:
                    fee_amount = theoretical_payout - payout
                    fee_pct = fee_amount / buy_in
                    
                    # Estimate breakdown (SafeBox is fixed, rest is skew)
                    safebox_amount = buy_in * self.safebox_fee
                    skew_amount = max(0, fee_amount - safebox_amount)
                    skew_pct = skew_amount / buy_in if buy_in > 0 else 0
                    
                    return {
                        'safebox_fee_pct': self.safebox_fee,
                        'skew_fee_pct': skew_pct,
                        'total_fee_pct': fee_pct,
                        'fee_amount': fee_amount,
                        'execution_stake': buy_in,
                        'quoted_odds': decimal_odds
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting quote fees: {e}")
            return None
    
    def format_fee_display(self, fee_info: Dict[str, float]) -> str:
        """Format fee information for display."""
        total_pct = fee_info.get('total_fee_pct', 0) * 100
        safebox_pct = fee_info.get('safebox_fee_pct', 0) * 100
        skew_pct = fee_info.get('skew_fee_pct', 0) * 100
        
        return (f"Fees: {total_pct:.1f}% total "
                f"(SafeBox: {safebox_pct:.1f}%, Skew: {skew_pct:.1f}%)")


# Example usage
if __name__ == "__main__":
    calc = FeeCalculator()
    
    # Example: $100 bet at 2.5 odds
    stake = 100
    odds = 2.5
    
    # Calculate entry fees
    entry_fees = calc.calculate_entry_fees(stake, odds)
    print("Entry fees:", entry_fees)
    print(calc.format_fee_display(entry_fees))
    
    # Calculate payout for win
    win_payout = calc.calculate_exit_payout(stake, odds, won=True, fee_info=entry_fees)
    print("\nWin payout:", win_payout)
    
    # Calculate payout for loss
    loss_payout = calc.calculate_exit_payout(stake, odds, won=False, fee_info=entry_fees)
    print("\nLoss payout:", loss_payout)