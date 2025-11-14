#!/usr/bin/env python3
"""
Test edge calculation functionality
"""
import psycopg2
from collections import namedtuple

# Create a mock odd object
Odd = namedtuple('Odd', ['outcome', 'decimal_odds', 'normalized_implied'])

def calculate_edge(odds_by_outcome):
    """Calculate edge based on normalized implied probability vs fair odds"""
    edges = {}
    
    # Get implied probabilities
    total_prob = 0
    probs = {}
    for outcome, odd in odds_by_outcome.items():
        if odd and hasattr(odd, 'normalized_implied') and odd.normalized_implied:
            probs[outcome] = odd.normalized_implied
            total_prob += odd.normalized_implied
        else:
            probs[outcome] = 0
    
    # Calculate fair probabilities (removing margin)
    if total_prob <= 0:
        return {}
    
    fair_probs = {k: v / total_prob for k, v in probs.items()}
    
    # Calculate edge for each outcome
    for outcome, odd in odds_by_outcome.items():
        if odd and hasattr(odd, 'decimal_odds') and odd.decimal_odds and outcome in fair_probs and fair_probs[outcome] > 0:
            fair_odds = 1 / fair_probs[outcome]
            actual_odds = odd.decimal_odds
            edge = ((actual_odds / fair_odds) - 1) * 100
            edges[outcome] = round(edge, 2)
        else:
            edges[outcome] = 0
    
    return edges

def test_edge_calculation():
    """Test edge calculation with real data"""
    
    conn = psycopg2.connect(
        host='localhost', 
        port=5999, 
        user='ominari_user', 
        password='ominari_2025_secure', 
        database='ominari_production'
    )
    cur = conn.cursor()
    
    # Get a sample market
    cur.execute("""
        SELECT DISTINCT m.source_id, m.home_team, m.away_team 
        FROM market m
        JOIN odd o ON m.source_id = o.source_id
        WHERE m.sport = 'Soccer'
        AND o.normalized_implied > 0
        LIMIT 1
    """)
    
    market = cur.fetchone()
    if not market:
        print("No soccer markets found")
        return
    
    source_id, home_team, away_team = market
    print(f"\nTesting market: {home_team} vs {away_team}")
    
    # Get all odds for this market
    cur.execute("""
        SELECT outcome, decimal_odds, normalized_implied
        FROM odd
        WHERE source_id = %s
        AND decimal_odds NOT IN (2.5, 2.8, 3.0)
    """, (source_id,))
    
    odds_data = cur.fetchall()
    
    # Create odd objects and group by outcome
    odds_by_outcome = {}
    for outcome, decimal_odds, normalized_implied in odds_data:
        odd = Odd(outcome, decimal_odds, normalized_implied)
        odds_by_outcome[outcome] = odd
        print(f"  {outcome}: {decimal_odds} (implied: {normalized_implied})")
    
    # Calculate edges
    edges = calculate_edge(odds_by_outcome)
    
    print("\nCalculated edges:")
    for outcome, edge in edges.items():
        print(f"  {outcome}: {edge:+.2f}%")
    
    # Check if calculations make sense
    total_implied = sum(o.normalized_implied for o in odds_by_outcome.values() if o.normalized_implied)
    margin = total_implied - 1.0
    print(f"\nTotal implied probability: {total_implied:.4f}")
    print(f"Bookmaker margin: {margin*100:.2f}%")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    test_edge_calculation()