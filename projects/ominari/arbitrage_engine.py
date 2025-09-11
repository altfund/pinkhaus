#!/usr/bin/env python3
"""
Multi-Chain Arbitrage Engine

Identifies and executes arbitrage opportunities across:
1. Cross-chain price differences (same market, different chains)
2. Oracle lag arbitrage (Chainlink vs on-chain prices)
3. Liquidity imbalances between chains
4. Gas cost arbitrage (execution on cheaper chains)
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from decimal import Decimal
import numpy as np
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArbitrageType(Enum):
    """Types of arbitrage opportunities."""
    CROSS_CHAIN = "cross_chain"
    ORACLE_LAG = "oracle_lag"
    LIQUIDITY_IMBALANCE = "liquidity_imbalance"
    GAS_ARBITRAGE = "gas_arbitrage"
    SETTLEMENT_TIME = "settlement_time"


@dataclass
class MarketPrice:
    """Price data from a specific source."""
    chain: str
    market_id: str
    source: str  # 'amm', 'chainlink', 'api'
    home_odds: Decimal
    away_odds: Decimal
    draw_odds: Optional[Decimal]
    liquidity: Decimal
    last_update: datetime
    gas_cost: Decimal
    latency_ms: int
    
    @property
    def implied_probabilities(self) -> Dict[str, Decimal]:
        """Calculate implied probabilities."""
        home_prob = Decimal(1) / self.home_odds
        away_prob = Decimal(1) / self.away_odds
        draw_prob = Decimal(1) / self.draw_odds if self.draw_odds else Decimal(0)
        
        total = home_prob + away_prob + draw_prob
        
        return {
            'home': home_prob / total,
            'away': away_prob / total,
            'draw': draw_prob / total if self.draw_odds else None
        }


@dataclass
class ArbitrageOpportunity:
    """Identified arbitrage opportunity."""
    opportunity_id: str
    type: ArbitrageType
    profit_bps: int  # Basis points
    size_limit: Decimal
    chains: List[str]
    market_id: str
    legs: List[Dict[str, Any]]
    time_window_seconds: int
    risk_score: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def expected_profit(self) -> Decimal:
        """Calculate expected profit."""
        return self.size_limit * Decimal(self.profit_bps) / Decimal(10000)


class CrossChainArbitrageDetector:
    """Detects price differences across chains."""
    
    def __init__(self, min_profit_bps: int = 50):
        self.min_profit_bps = min_profit_bps
        self.chain_configs = {
            'optimism': {'block_time': 2, 'gas_mult': 1.0},
            'arbitrum': {'block_time': 0.25, 'gas_mult': 0.8},
            'base': {'block_time': 2, 'gas_mult': 0.9},
            'polygon': {'block_time': 2, 'gas_mult': 0.5}
        }
    
    def find_opportunities(self, market_prices: Dict[str, MarketPrice]) -> List[ArbitrageOpportunity]:
        """Find cross-chain arbitrage opportunities."""
        opportunities = []
        
        # Group by market
        markets = {}
        for chain, price in market_prices.items():
            if price.market_id not in markets:
                markets[price.market_id] = {}
            markets[price.market_id][chain] = price
        
        # Check each market
        for market_id, chain_prices in markets.items():
            if len(chain_prices) < 2:
                continue
                
            # Find best prices across chains
            best_home = min(chain_prices.items(), key=lambda x: x[1].home_odds)
            best_away = min(chain_prices.items(), key=lambda x: x[1].away_odds)
            
            worst_home = max(chain_prices.items(), key=lambda x: x[1].home_odds)
            worst_away = max(chain_prices.items(), key=lambda x: x[1].away_odds)
            
            # Calculate arbitrage for home
            home_arb = self._calculate_arbitrage(
                buy_chain=best_home[0],
                buy_price=best_home[1].home_odds,
                sell_chain=worst_home[0],
                sell_price=worst_home[1].home_odds,
                buy_gas=best_home[1].gas_cost,
                sell_gas=worst_home[1].gas_cost
            )
            
            if home_arb and home_arb['profit_bps'] > self.min_profit_bps:
                opportunities.append(ArbitrageOpportunity(
                    opportunity_id=f"cross_chain_{market_id}_home_{datetime.now().timestamp()}",
                    type=ArbitrageType.CROSS_CHAIN,
                    profit_bps=home_arb['profit_bps'],
                    size_limit=min(best_home[1].liquidity, worst_home[1].liquidity) * Decimal('0.1'),
                    chains=[best_home[0], worst_home[0]],
                    market_id=market_id,
                    legs=[
                        {'action': 'buy', 'chain': best_home[0], 'outcome': 'home', 'odds': float(best_home[1].home_odds)},
                        {'action': 'sell', 'chain': worst_home[0], 'outcome': 'home', 'odds': float(worst_home[1].home_odds)}
                    ],
                    time_window_seconds=30,
                    risk_score=self._calculate_risk_score(home_arb)
                ))
            
            # Similar for away
            away_arb = self._calculate_arbitrage(
                buy_chain=best_away[0],
                buy_price=best_away[1].away_odds,
                sell_chain=worst_away[0], 
                sell_price=worst_away[1].away_odds,
                buy_gas=best_away[1].gas_cost,
                sell_gas=worst_away[1].gas_cost
            )
            
            if away_arb and away_arb['profit_bps'] > self.min_profit_bps:
                opportunities.append(ArbitrageOpportunity(
                    opportunity_id=f"cross_chain_{market_id}_away_{datetime.now().timestamp()}",
                    type=ArbitrageType.CROSS_CHAIN,
                    profit_bps=away_arb['profit_bps'],
                    size_limit=min(best_away[1].liquidity, worst_away[1].liquidity) * Decimal('0.1'),
                    chains=[best_away[0], worst_away[0]],
                    market_id=market_id,
                    legs=[
                        {'action': 'buy', 'chain': best_away[0], 'outcome': 'away', 'odds': float(best_away[1].away_odds)},
                        {'action': 'sell', 'chain': worst_away[0], 'outcome': 'away', 'odds': float(worst_away[1].away_odds)}
                    ],
                    time_window_seconds=30,
                    risk_score=self._calculate_risk_score(away_arb)
                ))
        
        return opportunities
    
    def _calculate_arbitrage(self, buy_chain: str, buy_price: Decimal, 
                           sell_chain: str, sell_price: Decimal,
                           buy_gas: Decimal, sell_gas: Decimal) -> Optional[Dict]:
        """Calculate arbitrage profit including gas costs."""
        if sell_price <= buy_price:
            return None
        
        # Gross profit
        gross_profit = (sell_price - buy_price) / buy_price
        
        # Net of gas costs (simplified)
        total_gas = buy_gas + sell_gas
        gas_impact = total_gas / Decimal(1000)  # Assume $1000 position size
        
        net_profit = gross_profit - gas_impact
        profit_bps = int(net_profit * 10000)
        
        if profit_bps > 0:
            return {
                'profit_bps': profit_bps,
                'gross_profit': float(gross_profit),
                'gas_cost': float(total_gas),
                'net_profit': float(net_profit)
            }
        
        return None
    
    def _calculate_risk_score(self, arb: Dict) -> float:
        """Calculate risk score for arbitrage."""
        # Simple risk model
        risk = 0.0
        
        # Gas cost risk
        gas_ratio = arb['gas_cost'] / (arb['gross_profit'] * 1000)
        risk += min(gas_ratio * 0.5, 0.5)
        
        # Profit margin risk
        if arb['profit_bps'] < 100:
            risk += 0.3
        elif arb['profit_bps'] < 50:
            risk += 0.5
        
        return min(risk, 1.0)


class OracleArbitrageDetector:
    """Detects arbitrage between Chainlink oracles and AMM prices."""
    
    def __init__(self):
        self.oracle_configs = {
            'chainlink': {
                'update_frequency': 3600,  # 1 hour
                'deviation_threshold': 0.005,  # 0.5%
                'latency_ms': 1000
            },
            'uma': {
                'update_frequency': 300,  # 5 minutes
                'deviation_threshold': 0.01,
                'latency_ms': 2000
            }
        }
    
    def find_oracle_lag_opportunities(self, 
                                    amm_prices: Dict[str, MarketPrice],
                                    oracle_prices: Dict[str, MarketPrice]) -> List[ArbitrageOpportunity]:
        """Find opportunities from oracle update lag."""
        opportunities = []
        
        for market_id in amm_prices:
            if market_id not in oracle_prices:
                continue
                
            amm = amm_prices[market_id]
            oracle = oracle_prices[market_id]
            
            # Check if oracle is stale
            time_diff = (datetime.now(timezone.utc) - oracle.last_update).total_seconds()
            
            if time_diff > 300:  # Oracle older than 5 minutes
                # Calculate price divergence
                home_div = abs(float(amm.home_odds - oracle.home_odds)) / float(oracle.home_odds)
                away_div = abs(float(amm.away_odds - oracle.away_odds)) / float(oracle.away_odds)
                
                max_div = max(home_div, away_div)
                
                if max_div > 0.01:  # 1% divergence
                    profit_bps = int(max_div * 10000)
                    
                    opportunities.append(ArbitrageOpportunity(
                        opportunity_id=f"oracle_lag_{market_id}_{datetime.now().timestamp()}",
                        type=ArbitrageType.ORACLE_LAG,
                        profit_bps=profit_bps,
                        size_limit=amm.liquidity * Decimal('0.05'),  # 5% of liquidity
                        chains=[amm.chain],
                        market_id=market_id,
                        legs=[
                            {
                                'action': 'trade',
                                'source': 'amm',
                                'oracle_price': float(oracle.home_odds),
                                'amm_price': float(amm.home_odds),
                                'divergence': home_div
                            }
                        ],
                        time_window_seconds=int(self.oracle_configs['chainlink']['update_frequency'] - time_diff),
                        risk_score=min(time_diff / 3600, 1.0)  # Higher risk with older oracle
                    ))
        
        return opportunities


class LiquidityArbitrageDetector:
    """Detects arbitrage from liquidity imbalances."""
    
    def find_liquidity_arbitrage(self, market_prices: Dict[str, MarketPrice]) -> List[ArbitrageOpportunity]:
        """Find opportunities from liquidity imbalances."""
        opportunities = []
        
        # Group by market
        markets = {}
        for chain, price in market_prices.items():
            if price.market_id not in markets:
                markets[price.market_id] = []
            markets[price.market_id].append((chain, price))
        
        for market_id, chain_data in markets.items():
            if len(chain_data) < 2:
                continue
                
            # Sort by liquidity
            sorted_by_liq = sorted(chain_data, key=lambda x: x[1].liquidity)
            
            low_liq = sorted_by_liq[0]
            high_liq = sorted_by_liq[-1]
            
            liq_ratio = float(high_liq[1].liquidity / low_liq[1].liquidity)
            
            if liq_ratio > 5:  # 5x liquidity difference
                # Low liquidity chains often have worse prices
                # Check for price inefficiency
                price_diff = abs(float(low_liq[1].home_odds - high_liq[1].home_odds))
                
                if price_diff > 0.05:  # Significant price difference
                    profit_bps = int(price_diff * 100)
                    
                    opportunities.append(ArbitrageOpportunity(
                        opportunity_id=f"liquidity_{market_id}_{datetime.now().timestamp()}",
                        type=ArbitrageType.LIQUIDITY_IMBALANCE,
                        profit_bps=profit_bps,
                        size_limit=low_liq[1].liquidity * Decimal('0.2'),  # 20% of low liquidity
                        chains=[low_liq[0], high_liq[0]],
                        market_id=market_id,
                        legs=[
                            {
                                'action': 'provide_liquidity',
                                'chain': low_liq[0],
                                'current_liquidity': float(low_liq[1].liquidity)
                            },
                            {
                                'action': 'arb_trade', 
                                'chain': high_liq[0],
                                'current_liquidity': float(high_liq[1].liquidity)
                            }
                        ],
                        time_window_seconds=300,
                        risk_score=0.3  # Lower risk due to liquidity provision
                    ))
        
        return opportunities


class ArbitrageEngine:
    """Main arbitrage engine coordinating all strategies."""
    
    def __init__(self):
        self.cross_chain = CrossChainArbitrageDetector()
        self.oracle = OracleArbitrageDetector()
        self.liquidity = LiquidityArbitrageDetector()
        self.active_positions = {}
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load arbitrage configuration."""
        return {
            'max_position_size': Decimal('10000'),  # $10k max per arb
            'max_gas_percentage': 0.02,  # 2% max gas cost
            'min_profit_bps': 20,  # 0.2% minimum profit
            'max_concurrent_arbs': 10,
            'chains': {
                'optimism': {'enabled': True, 'gas_limit': Decimal('100')},
                'arbitrum': {'enabled': True, 'gas_limit': Decimal('80')},
                'base': {'enabled': True, 'gas_limit': Decimal('90')},
                'polygon': {'enabled': True, 'gas_limit': Decimal('50')}
            }
        }
    
    async def scan_for_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan all chains for arbitrage opportunities."""
        all_opportunities = []
        
        # Get current market prices from all sources
        market_prices = await self._fetch_all_market_prices()
        oracle_prices = await self._fetch_oracle_prices()
        
        # Cross-chain arbitrage
        cross_chain_opps = self.cross_chain.find_opportunities(market_prices)
        all_opportunities.extend(cross_chain_opps)
        
        # Oracle arbitrage
        oracle_opps = self.oracle.find_oracle_lag_opportunities(market_prices, oracle_prices)
        all_opportunities.extend(oracle_opps)
        
        # Liquidity arbitrage
        liquidity_opps = self.liquidity.find_liquidity_arbitrage(market_prices)
        all_opportunities.extend(liquidity_opps)
        
        # Filter and rank
        filtered = self._filter_opportunities(all_opportunities)
        ranked = self._rank_opportunities(filtered)
        
        return ranked[:self.config['max_concurrent_arbs']]
    
    async def _fetch_all_market_prices(self) -> Dict[str, MarketPrice]:
        """Fetch prices from all chains."""
        # This would connect to the blockchain readers
        prices = {}
        
        # Mock data for example
        for chain in ['optimism', 'arbitrum', 'base']:
            prices[chain] = MarketPrice(
                chain=chain,
                market_id='EPL_2024_Liverpool_Chelsea',
                source='amm',
                home_odds=Decimal('2.10') + Decimal(np.random.uniform(-0.05, 0.05)),
                away_odds=Decimal('3.50') + Decimal(np.random.uniform(-0.05, 0.05)),
                draw_odds=Decimal('3.20') + Decimal(np.random.uniform(-0.05, 0.05)),
                liquidity=Decimal(np.random.uniform(10000, 100000)),
                last_update=datetime.now(timezone.utc) - timedelta(seconds=np.random.randint(0, 60)),
                gas_cost=Decimal(np.random.uniform(5, 20)),
                latency_ms=int(np.random.uniform(10, 100))
            )
        
        return prices
    
    async def _fetch_oracle_prices(self) -> Dict[str, MarketPrice]:
        """Fetch prices from oracles."""
        # Mock Chainlink oracle data
        return {
            'EPL_2024_Liverpool_Chelsea': MarketPrice(
                chain='ethereum',  # Chainlink on Ethereum
                market_id='EPL_2024_Liverpool_Chelsea',
                source='chainlink',
                home_odds=Decimal('2.08'),
                away_odds=Decimal('3.45'),
                draw_odds=Decimal('3.25'),
                liquidity=Decimal('1000000'),  # Deep liquidity
                last_update=datetime.now(timezone.utc) - timedelta(minutes=45),  # Stale
                gas_cost=Decimal('50'),  # High gas on mainnet
                latency_ms=1000
            )
        }
    
    def _filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Filter opportunities based on criteria."""
        filtered = []
        
        for opp in opportunities:
            # Check minimum profit
            if opp.profit_bps < self.config['min_profit_bps']:
                continue
                
            # Check gas costs
            total_gas = sum(self.config['chains'][chain]['gas_limit'] for chain in opp.chains)
            gas_percentage = float(total_gas / opp.size_limit)
            
            if gas_percentage > self.config['max_gas_percentage']:
                continue
                
            # Check if chains are enabled
            if not all(self.config['chains'][chain]['enabled'] for chain in opp.chains):
                continue
                
            filtered.append(opp)
        
        return filtered
    
    def _rank_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Rank opportunities by profitability and risk."""
        
        def score_opportunity(opp: ArbitrageOpportunity) -> float:
            # Profit score (higher is better)
            profit_score = opp.profit_bps / 100
            
            # Risk penalty (lower is better)
            risk_penalty = opp.risk_score * 5
            
            # Time sensitivity bonus
            time_bonus = 1.0 if opp.time_window_seconds < 60 else 0.0
            
            # Type preference
            type_scores = {
                ArbitrageType.CROSS_CHAIN: 1.0,
                ArbitrageType.ORACLE_LAG: 1.5,  # Prefer oracle arbs
                ArbitrageType.LIQUIDITY_IMBALANCE: 0.8,
                ArbitrageType.GAS_ARBITRAGE: 0.5
            }
            type_score = type_scores.get(opp.type, 0.5)
            
            return (profit_score - risk_penalty + time_bonus) * type_score
        
        return sorted(opportunities, key=score_opportunity, reverse=True)
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> Dict[str, Any]:
        """Execute an arbitrage opportunity."""
        logger.info(f"Executing arbitrage: {opportunity.opportunity_id}")
        
        # Record start
        start_time = datetime.now(timezone.utc)
        
        try:
            # Pre-flight checks
            if not await self._pre_execution_checks(opportunity):
                return {'status': 'rejected', 'reason': 'pre-flight checks failed'}
            
            # Execute legs atomically
            results = []
            for leg in opportunity.legs:
                result = await self._execute_leg(leg, opportunity)
                results.append(result)
                
                if not result['success']:
                    # Rollback previous legs
                    await self._rollback_legs(results[:-1], opportunity)
                    return {'status': 'failed', 'reason': f"leg {len(results)} failed"}
            
            # Calculate actual profit
            actual_profit = await self._calculate_actual_profit(results, opportunity)
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return {
                'status': 'success',
                'opportunity_id': opportunity.opportunity_id,
                'expected_profit_bps': opportunity.profit_bps,
                'actual_profit_bps': actual_profit,
                'execution_time_seconds': execution_time,
                'legs_executed': len(results)
            }
            
        except Exception as e:
            logger.error(f"Arbitrage execution error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _pre_execution_checks(self, opportunity: ArbitrageOpportunity) -> bool:
        """Perform pre-execution validation."""
        # Check market still exists
        # Check liquidity still available
        # Check gas prices haven't spiked
        # Check we have sufficient balance
        return True
    
    async def _execute_leg(self, leg: Dict, opportunity: ArbitrageOpportunity) -> Dict:
        """Execute a single leg of the arbitrage."""
        # This would interact with smart contracts
        return {
            'success': True,
            'tx_hash': f"0x{opportunity.opportunity_id[:8]}",
            'gas_used': 100000,
            'execution_price': leg.get('odds', 0)
        }
    
    async def _rollback_legs(self, executed_legs: List[Dict], opportunity: ArbitrageOpportunity):
        """Rollback executed legs if arbitrage fails."""
        logger.warning(f"Rolling back arbitrage: {opportunity.opportunity_id}")
        # Implement rollback logic
    
    async def _calculate_actual_profit(self, results: List[Dict], opportunity: ArbitrageOpportunity) -> int:
        """Calculate actual profit from execution results."""
        # Compare expected vs actual execution prices
        # Account for actual gas costs
        # Return profit in basis points
        return opportunity.profit_bps - 10  # Mock: slightly less than expected


async def demo_arbitrage_system():
    """Demonstrate the arbitrage system."""
    logger.info("🏃 Arbitrage Engine Demo")
    logger.info("=" * 60)
    
    engine = ArbitrageEngine()
    
    # Scan for opportunities
    opportunities = await engine.scan_for_opportunities()
    
    logger.info(f"\n📊 Found {len(opportunities)} arbitrage opportunities:")
    
    for i, opp in enumerate(opportunities[:5], 1):
        logger.info(f"\n{i}. {opp.type.value.upper()} Arbitrage")
        logger.info(f"   Market: {opp.market_id}")
        logger.info(f"   Chains: {' ↔ '.join(opp.chains)}")
        logger.info(f"   Profit: {opp.profit_bps} bps ({opp.profit_bps/100:.2f}%)")
        logger.info(f"   Size Limit: ${opp.size_limit:,.2f}")
        logger.info(f"   Expected Profit: ${opp.expected_profit:,.2f}")
        logger.info(f"   Risk Score: {opp.risk_score:.2f}")
        logger.info(f"   Time Window: {opp.time_window_seconds}s")
    
    # Execute best opportunity
    if opportunities:
        best = opportunities[0]
        logger.info(f"\n🚀 Executing best opportunity...")
        result = await engine.execute_arbitrage(best)
        logger.info(f"   Result: {result}")
    
    # Show arbitrage strategies summary
    logger.info("\n📈 Arbitrage Strategies Summary:")
    logger.info("1. Cross-Chain: Exploit price differences between Optimism/Arbitrum/Base")
    logger.info("2. Oracle Lag: Trade when Chainlink updates lag behind market moves") 
    logger.info("3. Liquidity: Provide liquidity on thin chains, arb on deep chains")
    logger.info("4. Gas Arbitrage: Execute on cheapest chain when profitable")
    logger.info("5. Settlement: Exploit different settlement times between chains")


if __name__ == "__main__":
    asyncio.run(demo_arbitrage_system())