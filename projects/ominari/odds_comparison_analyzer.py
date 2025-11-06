#!/usr/bin/env python3
"""Compare odds between blockchain and API sources to find opportunities"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import and_, or_
from unified_data_fetcher import UnifiedDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OddsComparisonAnalyzer:
    """Analyzes odds differences between blockchain and API sources"""
    
    def __init__(self):
        self.min_edge_threshold = 0.02  # 2% minimum edge to consider
        self.max_odds_difference = 0.5   # Maximum 50% difference to consider valid
        
        # Load blockchain connections
        self.blockchain_connections = {}
        try:
            with open('blockchain_connections.json', 'r') as f:
                self.blockchain_connections = json.load(f)
        except FileNotFoundError:
            logger.warning("No blockchain connections found")
    
    def fetch_odds_data(self) -> pd.DataFrame:
        """Fetch all odds data from database"""
        with db_manager.get_db_session() as db:
            # Get all markets with odds
            query = db.query(
                Market.source_id,
                Market.source,
                Market.home_team,
                Market.away_team,
                Market.sport,
                Market.maturity_date,
                Odd.outcome,
                Odd.decimal_odds,
                Odd.american_odds,
                Odd.source.label('odds_source'),
                Odd.updated_at
            ).join(
                Odd, Market.source_id == Odd.source_id
            ).filter(
                and_(
                    Market.sport.ilike('%soccer%'),
                    Market.is_finished == False,
                    Market.maturity_date > datetime.now(timezone.utc)
                )
            ).all()
            
            # Convert to DataFrame
            data = []
            for row in query:
                odds_value = row.decimal_odds if row.decimal_odds else self._american_to_decimal(row.american_odds)
                data.append({
                    'market_id': row.source_id,
                    'market_source': row.source,
                    'home_team': row.home_team,
                    'away_team': row.away_team,
                    'sport': row.sport,
                    'maturity_date': row.maturity_date,
                    'outcome': row.outcome.lower(),
                    'odds': odds_value,
                    'odds_source': row.odds_source,
                    'updated_at': row.updated_at
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Fetched {len(df)} odds records")
            return df
    
    def _american_to_decimal(self, american_odds: float) -> float:
        """Convert American odds to decimal"""
        if not american_odds:
            return 0.0
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def compare_odds(self, df: pd.DataFrame) -> List[Dict]:
        """Compare odds between different sources"""
        comparisons = []
        
        # Group by market and outcome
        grouped = df.groupby(['home_team', 'away_team', 'outcome'])
        
        for (home, away, outcome), group in grouped:
            if len(group) < 2:
                continue  # Need at least 2 sources to compare
            
            # Get blockchain and API odds
            blockchain_odds = group[group['market_source'].str.contains('blockchain')]['odds'].values
            api_odds = group[group['market_source'].str.contains('overtime|api')]['odds'].values
            
            if len(blockchain_odds) > 0 and len(api_odds) > 0:
                bc_odds = blockchain_odds[0]
                api_odds_val = api_odds[0]
                
                # Calculate difference
                odds_diff = abs(bc_odds - api_odds_val)
                odds_diff_pct = odds_diff / min(bc_odds, api_odds_val)
                
                # Check if significant difference
                if odds_diff_pct > self.min_edge_threshold and odds_diff_pct < self.max_odds_difference:
                    comparison = {
                        'home_team': home,
                        'away_team': away,
                        'outcome': outcome,
                        'blockchain_odds': bc_odds,
                        'api_odds': api_odds_val,
                        'difference': odds_diff,
                        'difference_pct': odds_diff_pct,
                        'better_source': 'blockchain' if bc_odds > api_odds_val else 'api',
                        'implied_prob_bc': 1/bc_odds if bc_odds > 0 else 0,
                        'implied_prob_api': 1/api_odds_val if api_odds_val > 0 else 0
                    }
                    comparisons.append(comparison)
        
        # Sort by difference percentage
        comparisons.sort(key=lambda x: x['difference_pct'], reverse=True)
        
        return comparisons
    
    def find_arbitrage_opportunities(self, df: pd.DataFrame) -> List[Dict]:
        """Find arbitrage opportunities between sources"""
        opportunities = []
        
        # Group by match
        match_groups = df.groupby(['home_team', 'away_team'])
        
        for (home, away), match_df in match_groups:
            # Get best odds for each outcome
            best_odds = {}
            
            for outcome in ['home', 'draw', 'away']:
                outcome_df = match_df[match_df['outcome'] == outcome]
                if not outcome_df.empty:
                    best_odds[outcome] = {
                        'odds': outcome_df['odds'].max(),
                        'source': outcome_df.loc[outcome_df['odds'].idxmax(), 'market_source']
                    }
            
            # Check for arbitrage
            if len(best_odds) >= 2:  # Need at least 2 outcomes
                # Calculate implied probabilities
                total_prob = sum(1/v['odds'] for v in best_odds.values() if v['odds'] > 0)
                
                # If total probability < 1, there's an arbitrage opportunity
                if total_prob < 0.98:  # 2% margin for fees
                    arb_margin = 1 - total_prob
                    
                    opportunity = {
                        'home_team': home,
                        'away_team': away,
                        'best_odds': best_odds,
                        'total_implied_prob': total_prob,
                        'arbitrage_margin': arb_margin,
                        'arbitrage_pct': arb_margin * 100,
                        'required_stakes': self._calculate_arbitrage_stakes(best_odds, 1000)  # $1000 total
                    }
                    opportunities.append(opportunity)
        
        # Sort by arbitrage margin
        opportunities.sort(key=lambda x: x['arbitrage_margin'], reverse=True)
        
        return opportunities
    
    def _calculate_arbitrage_stakes(self, best_odds: Dict, total_stake: float) -> Dict:
        """Calculate optimal stakes for arbitrage"""
        stakes = {}
        total_prob = sum(1/v['odds'] for v in best_odds.values() if v['odds'] > 0)
        
        for outcome, data in best_odds.items():
            if data['odds'] > 0:
                stake = (total_stake * (1/data['odds'])) / total_prob
                stakes[outcome] = {
                    'stake': round(stake, 2),
                    'potential_return': round(stake * data['odds'], 2),
                    'source': data['source']
                }
        
        return stakes
    
    def generate_report(self) -> Dict:
        """Generate comprehensive odds comparison report"""
        # Fetch data
        df = self.fetch_odds_data()
        
        # Run comparisons
        comparisons = self.compare_odds(df)
        arbitrage_opps = self.find_arbitrage_opportunities(df)
        
        # Generate summary statistics
        summary = {
            'total_markets': df['market_id'].nunique(),
            'total_odds_records': len(df),
            'significant_differences': len(comparisons),
            'arbitrage_opportunities': len(arbitrage_opps),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Top opportunities
        top_differences = comparisons[:10] if comparisons else []
        top_arbitrage = arbitrage_opps[:5] if arbitrage_opps else []
        
        report = {
            'summary': summary,
            'top_odds_differences': top_differences,
            'arbitrage_opportunities': top_arbitrage,
            'analysis': {
                'avg_difference': sum(c['difference_pct'] for c in comparisons) / len(comparisons) if comparisons else 0,
                'best_arbitrage_margin': arbitrage_opps[0]['arbitrage_margin'] if arbitrage_opps else 0,
                'markets_with_differences': len(comparisons),
                'profitable_arbitrage': len([a for a in arbitrage_opps if a['arbitrage_margin'] > 0.02])
            }
        }
        
        return report


def main():
    """Run odds comparison analysis"""
    analyzer = OddsComparisonAnalyzer()
    
    print("📊 Odds Comparison Analysis\n")
    
    # Generate report
    report = analyzer.generate_report()
    
    # Display summary
    print("📈 Summary:")
    print(f"  Total markets analyzed: {report['summary']['total_markets']}")
    print(f"  Significant differences found: {report['summary']['significant_differences']}")
    print(f"  Arbitrage opportunities: {report['summary']['arbitrage_opportunities']}")
    
    # Show top differences
    if report['top_odds_differences']:
        print("\n🔍 Top Odds Differences:")
        for i, diff in enumerate(report['top_odds_differences'][:5]):
            print(f"\n{i+1}. {diff['home_team']} vs {diff['away_team']} - {diff['outcome'].upper()}")
            print(f"   Blockchain: {diff['blockchain_odds']:.2f}")
            print(f"   API: {diff['api_odds']:.2f}")
            print(f"   Difference: {diff['difference_pct']*100:.1f}%")
            print(f"   Better odds on: {diff['better_source']}")
    
    # Show arbitrage opportunities
    if report['arbitrage_opportunities']:
        print("\n💰 Arbitrage Opportunities:")
        for i, arb in enumerate(report['arbitrage_opportunities'][:3]):
            print(f"\n{i+1}. {arb['home_team']} vs {arb['away_team']}")
            print(f"   Margin: {arb['arbitrage_pct']:.1f}%")
            print(f"   Required stakes:")
            for outcome, stake_data in arb['required_stakes'].items():
                print(f"     {outcome.upper()}: ${stake_data['stake']:.2f} @ {arb['best_odds'][outcome]['odds']:.2f} ({stake_data['source']})")
            print(f"   Guaranteed return: ${list(arb['required_stakes'].values())[0]['potential_return']:.2f}")
    
    # Save report
    with open('odds_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print("\n💾 Full report saved to odds_comparison_report.json")


if __name__ == "__main__":
    main()