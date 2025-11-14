#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ominari Trading System - Main Application
FastAPI-based REST API and background task orchestration.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import asyncio
import logging
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import time

# Local imports
from database import get_db
from models import Market
from signals import SIGNAL_PROVIDERS
from paper_trading_engine import PaperTradingEngine, PaperOrder
from signal_registry import SignalRegistry, DynamicWeightManager, SignalAggregator
from alpha_research_pipeline import AlphaResearchPipeline, AlphaSignal
from carver_framework import create_carver_system
from deployment_config import DeploymentManager
from graphql_client import OvertimeGraphQLClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
signal_registry = SignalRegistry()
weight_manager = DynamicWeightManager(signal_registry)
signal_aggregator = SignalAggregator(signal_registry, weight_manager)
alpha_pipeline = AlphaResearchPipeline()
paper_engine = PaperTradingEngine()
deployment = DeploymentManager()
carver_system = create_carver_system()

# Background tasks
background_tasks = set()


# Pydantic models
class SignalRegistration(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class BacktestRequest(BaseModel):
    signal_names: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0


class PaperTradeRequest(BaseModel):
    source_id: str
    market_type: str
    bet_name: str
    side: str = "buy"
    size: float
    limit_price: Optional[float] = None
    signal_name: str
    expected_edge: float


class WeightUpdateRequest(BaseModel):
    method: str = "bayesian"
    lookback_days: int = 30


class AlphaSignalRequest(BaseModel):
    name: str
    description: str
    formula: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Ominari Trading System")
    
    # Initialize signal registry with default signals
    for provider_instance in SIGNAL_PROVIDERS:
        try:
            # SIGNAL_PROVIDERS already contains instances, not classes
            signal_registry.register(provider_instance)
        except Exception as e:
            logger.error(f"Failed to register {provider_instance.name}: {e}")
    
    # Start background tasks
    if deployment.is_feature_enabled('paper_trading'):
        task = asyncio.create_task(paper_trading_loop())
        background_tasks.add(task)
        
    if deployment.config.monitoring['enabled']:
        task = asyncio.create_task(metrics_collection_loop())
        background_tasks.add(task)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ominari Trading System")
    
    # Cancel background tasks
    for task in background_tasks:
        task.cancel()
        
    await asyncio.gather(*background_tasks, return_exceptions=True)


# Create FastAPI app
app = FastAPI(
    title="Ominari Trading System",
    description="Systematic sports betting trading platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": deployment.environment
    }


@app.get("/health/detailed")
async def detailed_health(db: Session = Depends(get_db)):
    """Detailed health check with component status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": deployment.environment,
        "components": {}
    }
    
    # Check database
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db.commit()
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        health_status["components"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
        db.rollback()
    
    # Check signal registry
    health_status["components"]["signals"] = {
        "count": len(signal_registry.signals),
        "active": signal_registry.list_signals(active_only=True)
    }
    
    # Check blockchain connection
    if deployment.is_feature_enabled('live_trading'):
        try:
            w3 = deployment.get_rpc_provider()
            if w3.is_connected():
                health_status["components"]["blockchain"] = {
                    "connected": True,
                    "block_number": w3.eth.block_number
                }
            else:
                health_status["components"]["blockchain"] = "disconnected"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["components"]["blockchain"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
    
    return health_status


# Signal management endpoints
@app.get("/signals")
async def list_signals(active_only: bool = True):
    """List all registered signals."""
    return {
        "signals": signal_registry.list_signals(active_only=active_only),
        "total": len(signal_registry.signals)
    }


@app.get("/signals/{signal_name}")
async def get_signal(signal_name: str):
    """Get details for a specific signal."""
    signal = signal_registry.get_signal(signal_name)
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    return {
        "name": signal.name,
        "metadata": signal.metadata,
        "performance": signal.metadata.performance_stats
    }


@app.post("/signals/register")
async def register_signal(registration: SignalRegistration):
    """Register a new signal dynamically."""
    # This would need actual signal class creation
    # For now, return success
    return {
        "status": "registered",
        "signal": registration.name
    }


# Weight management endpoints
@app.get("/weights")
async def get_weights():
    """Get current signal weights."""
    return {
        "weights": weight_manager.get_current_weights(),
        "method": "current",
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


@app.post("/weights/update")
async def update_weights(request: WeightUpdateRequest):
    """Recalculate and update signal weights."""
    weights = weight_manager.calculate_weights(
        method=request.method,
        lookback_days=request.lookback_days
    )
    
    weight_manager.update_weights(
        weights,
        method=request.method,
        notes=f"API update via {request.method}"
    )
    
    return {
        "weights": weights,
        "method": request.method
    }


# Backtesting endpoints
@app.post("/backtest/run")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run a backtest with specified signals."""
    # Run in background
    background_tasks.add_task(
        execute_backtest,
        request.signal_names,
        request.start_date,
        request.end_date,
        request.initial_capital
    )
    
    return {
        "status": "started",
        "message": "Backtest running in background"
    }


async def execute_backtest(signal_names: List[str], 
                         start_date: datetime,
                         end_date: datetime,
                         initial_capital: float):
    """Execute backtest asynchronously."""
    try:
        backtest = VectorizedBacktest(
            start_date=start_date,
            end_date=end_date,
            signal_names=signal_names,
            initial_capital=initial_capital
        )
        
        backtest.load_data()
        backtest.generate_signals()
        backtest.calculate_positions()
        backtest.calculate_returns()
        
        results = backtest.get_results()
        logger.info(f"Backtest complete: Sharpe={results['sharpe_ratio']:.2f}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")


# Paper trading endpoints
@app.post("/paper/trade")
async def submit_paper_trade(request: PaperTradeRequest):
    """Submit a paper trade."""
    order = PaperOrder(
        order_id=f"API_{int(datetime.now().timestamp() * 1000)}",
        timestamp=datetime.now(timezone.utc),
        source_id=request.source_id,
        market_type=request.market_type,
        bet_name=request.bet_name,
        side=request.side,
        size=request.size,
        limit_price=request.limit_price,
        signal_name=request.signal_name,
        expected_edge=request.expected_edge
    )
    
    fill = await paper_engine.submit_order(order)
    
    if fill:
        return {
            "status": "filled",
            "fill": {
                "price": fill.fill_price,
                "size": fill.fill_size,
                "slippage": fill.slippage,
                "commission": fill.commission
            }
        }
    else:
        return {
            "status": "rejected",
            "reason": "No fill available"
        }


@app.get("/paper/performance")
async def get_paper_performance():
    """Get paper trading performance."""
    metrics = paper_engine.calculate_performance()
    report = paper_engine.generate_report()
    
    return {
        "metrics": metrics,
        "report": report.to_dict(orient="records")[0] if not report.empty else {},
        "portfolio_value": paper_engine.get_portfolio_value()
    }


@app.get("/paper/positions")
async def get_paper_positions():
    """Get current paper trading positions."""
    positions_df = paper_engine.get_open_positions()
    exposure = paper_engine.get_position_exposure()
    
    return {
        "positions": positions_df.to_dict(orient="records") if not positions_df.empty else [],
        "exposure": exposure,
        "cash": paper_engine.current_capital,
        "portfolio_value": paper_engine.get_portfolio_value()
    }


# Session management endpoints
@app.post("/paper/session/start")
async def start_paper_session(capital: Optional[float] = 10000.0, name: Optional[str] = None):
    """Start a new paper trading session."""
    global paper_engine
    
    # Create new engine with session
    paper_engine = PaperTradingEngine(
        initial_capital=capital,
        db_path="paper_trades.db"
    )
    
    session_id = paper_engine.create_session(name)
    
    return {
        "session_id": session_id,
        "status": "started",
        "capital": capital
    }

@app.post("/paper/session/stop")
async def stop_paper_session():
    """Stop the current paper trading session."""
    if not paper_engine.session_active:
        raise HTTPException(status_code=400, detail="No active session")
    
    result = paper_engine.stop_session()
    return result

@app.get("/paper/session")
async def get_current_session():
    """Get current session information."""
    return paper_engine.get_session_info()

@app.get("/paper/sessions")
async def list_paper_sessions(status: Optional[str] = None):
    """List all paper trading sessions."""
    import sqlite3
    
    conn = sqlite3.connect("paper_trades.db")
    cursor = conn.cursor()
    
    query = "SELECT * FROM paper_sessions"
    params = []
    
    if status:
        query += " WHERE status = ?"
        params.append(status)
        
    query += " ORDER BY created_at DESC"
    
    cursor.execute(query, params)
    
    sessions = []
    for row in cursor.fetchall():
        cols = [desc[0] for desc in cursor.description]
        session = dict(zip(cols, row))
        
        # Calculate total return if completed
        if session['status'] == 'completed' and session['final_capital']:
            session['total_return'] = (session['final_capital'] - session['initial_capital']) / session['initial_capital']
        else:
            session['total_return'] = 0
            
        sessions.append(session)
    
    conn.close()
    return sessions


@app.post("/paper/session/{session_id}/archive")
async def archive_session(session_id: str, archive_reason: Optional[str] = None):
    """Archive a completed session."""
    import sqlite3
    
    conn = sqlite3.connect("paper_trades.db")
    cursor = conn.cursor()
    
    # Check session exists and is completed
    cursor.execute("SELECT status FROM paper_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    
    status = row[0]
    if status != 'completed':
        conn.close()
        raise HTTPException(status_code=400, detail=f"Only completed sessions can be archived. Current status: {status}")
    
    # Update session metadata with archive info
    cursor.execute("""
        SELECT metadata FROM paper_sessions WHERE session_id = ?
    """, (session_id,))
    row = cursor.fetchone()
    
    metadata = json.loads(row[0] if row[0] else '{}')
    metadata['archived_at'] = datetime.now(timezone.utc).isoformat()
    metadata['archive_reason'] = archive_reason or "Manual archival"
    
    # Archive the session
    cursor.execute("""
        UPDATE paper_sessions 
        SET status = 'archived', metadata = ?
        WHERE session_id = ?
    """, (json.dumps(metadata), session_id))
    
    conn.commit()
    
    # Get updated session info
    cursor.execute("SELECT * FROM paper_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    cols = [desc[0] for desc in cursor.description]
    session = dict(zip(cols, row))
    
    conn.close()
    
    return {
        "status": "archived",
        "session": session,
        "message": f"Session {session_id} has been archived"
    }


@app.get("/paper/trades/recent")
async def get_recent_paper_trades(limit: int = 20):
    """Get recent paper trading fills."""
    # Get recent fills
    recent_fills = paper_engine.fills[-limit:] if paper_engine.fills else []
    
    fills_data = []
    for fill in recent_fills:
        # Find the corresponding order to get market details
        order = paper_engine.orders.get(fill.order_id)
        
        fill_data = {
            "fill_id": fill.fill_id,
            "order_id": fill.order_id,
            "timestamp": fill.timestamp.isoformat(),
            "fill_price": fill.fill_price,
            "fill_size": fill.fill_size,
            "slippage": fill.slippage,
            "commission": fill.commission,
            "market_impact": fill.market_impact
        }
        
        # Add market details if we have the order
        if order:
            fill_data.update({
                "market_id": order.source_id,
                "market_type": order.market_type,
                "bet_name": order.bet_name,
                "side": order.side,
                "signal_name": order.signal_name,
                "expected_edge": order.expected_edge
            })
            
        fills_data.append(fill_data)
    
    return {
        "fills": fills_data,
        "total_fills": len(paper_engine.fills)
    }


# Portfolio analytics endpoints
@app.get("/paper/analytics/correlation")
async def get_portfolio_correlation():
    """Get correlation analysis of current positions."""
    positions_df = paper_engine.get_open_positions()
    
    if positions_df.empty:
        return {"message": "No open positions"}
    
    # Get correlation data if available
    correlation_data = {
        "positions": positions_df.to_dict(orient="records"),
        "clusters": paper_engine.position_manager.net_cluster_positions() if hasattr(paper_engine.position_manager, 'clusters') else {},
        "correlation_matrix": None  # Would need historical price data
    }
    
    return correlation_data


@app.get("/paper/analytics/exposure")
async def get_portfolio_exposure():
    """Get exposure analysis by various dimensions."""
    positions_df = paper_engine.get_open_positions()
    
    if positions_df.empty:
        return {"message": "No open positions"}
    
    # Calculate exposures
    total_capital = paper_engine.initial_capital
    exposure_data = paper_engine.get_position_exposure()
    
    # Add breakdown by market type
    market_breakdown = {}
    for _, pos in positions_df.iterrows():
        market_id = pos.get('market_id', 'Unknown')
        # Extract sport from market_id if possible
        sport = 'Soccer'  # Default for now
        
        if sport not in market_breakdown:
            market_breakdown[sport] = {
                'count': 0,
                'total_value': 0,
                'exposure_pct': 0
            }
        
        market_breakdown[sport]['count'] += 1
        market_breakdown[sport]['total_value'] += abs(pos.get('current_value', 0))
    
    # Calculate percentages
    for sport in market_breakdown:
        market_breakdown[sport]['exposure_pct'] = market_breakdown[sport]['total_value'] / total_capital
    
    return {
        "total_exposure": exposure_data,
        "by_sport": market_breakdown,
        "risk_utilization": {
            "position_limit": exposure_data.get('largest_position', 0) / 0.1,  # vs 10% limit
            "total_limit": exposure_data.get('total_exposure', 0) / 0.6,  # vs 60% limit
        }
    }


@app.get("/paper/analytics/performance")
async def get_performance_analytics():
    """Get detailed performance analytics."""
    metrics = paper_engine.calculate_performance()
    
    # Add performance attribution by time
    import sqlite3
    conn = sqlite3.connect("paper_trades.db")
    cursor = conn.cursor()
    
    # Get fills by hour of day
    cursor.execute("""
        SELECT 
            strftime('%H', timestamp) as hour,
            COUNT(*) as trades,
            AVG(commission) as avg_commission,
            SUM(slippage) as total_slippage
        FROM paper_fills
        GROUP BY hour
        ORDER BY hour
    """)
    
    hourly_data = []
    for row in cursor.fetchall():
        hourly_data.append({
            "hour": int(row[0]),
            "trades": row[1],
            "avg_commission": row[2],
            "total_slippage": row[3]
        })
    
    # Get performance by signal
    cursor.execute("""
        SELECT 
            o.signal_name,
            COUNT(DISTINCT f.fill_id) as trades,
            AVG(o.expected_edge) as avg_edge
        FROM paper_orders o
        JOIN paper_fills f ON o.order_id = f.order_id
        GROUP BY o.signal_name
    """)
    
    signal_performance = []
    for row in cursor.fetchall():
        signal_performance.append({
            "signal": row[0],
            "trades": row[1],
            "avg_edge": row[2]
        })
    
    conn.close()
    
    return {
        "overall_metrics": metrics,
        "hourly_breakdown": hourly_data,
        "signal_performance": signal_performance
    }


@app.post("/paper/analytics/rebalance")
async def calculate_rebalancing():
    """Calculate portfolio rebalancing trades."""
    # Get current positions
    positions_df = paper_engine.get_open_positions()
    
    if positions_df.empty:
        return {"message": "No positions to rebalance"}
    
    # Import Kelly system function
    from evaluate_open_markets import (
        generate_betting_session_report_and_save,
        SIGNAL_WEIGHTS
    )
    
    # Use Kelly system to calculate target positions
    kelly_results = generate_betting_session_report_and_save(
        kelly_bankroll=1000.0,
        execution_bankroll=paper_engine.current_capital,
        kelly_fraction=0.5,
        cap_per_game=0.1,
        cap_per_bet=0.05,
        cap_per_game_market=0.07,
        min_bet_abs=10.0,
        min_bet_pct=0.001,
        signal_providers=SIGNAL_PROVIDERS,
        signal_weights=SIGNAL_WEIGHTS,
        mode='rebalancing',
        display_md=False,
        save_as_latest=False
    )
    
    if not kelly_results or 'trimmed' not in kelly_results:
        return {"message": "No rebalancing opportunities found"}
    
    # Compare current vs target
    target_positions = kelly_results['trimmed']
    
    # Calculate rebalancing trades
    trades = []
    
    # Create a mapping of current positions by market_id
    current_pos_map = {}
    for _, pos in positions_df.iterrows():
        key = pos['market_id']
        current_pos_map[key] = pos
    
    # Create a mapping of target positions by source_id
    target_pos_map = {}
    for _, target in target_positions.iterrows():
        key = target['source_id']
        target_pos_map[key] = target
    
    # Find positions to close (in current but not in target)
    for market_id, current in current_pos_map.items():
        if market_id not in target_pos_map:
            trades.append({
                'action': 'close',
                'market_id': market_id,
                'bet_name': current.get('bet_name', ''),
                'current_size': current['size'],
                'target_size': 0,
                'size_delta': -current['size'],
                'reason': 'Position not in optimal portfolio'
            })
    
    # Find positions to adjust or open
    for source_id, target in target_pos_map.items():
        current = current_pos_map.get(source_id)
        
        if current is None:
            # New position to open
            if target['stake'] > 0:
                trades.append({
                    'action': 'open',
                    'market_id': source_id,
                    'bet_name': f"{target['market_name']} - {target['normalized_outcome']}",
                    'current_size': 0,
                    'target_size': target['stake'],
                    'size_delta': target['stake'],
                    'target_odds': target['odds'],
                    'expected_edge': target.get('probability', 0) - (1.0 / target['odds']),
                    'reason': 'New position in optimal portfolio'
                })
        else:
            # Existing position - check if adjustment needed
            current_value = abs(current['current_value'])
            target_value = target['stake']
            
            # Only suggest rebalancing if difference is significant (> 5% of target)
            if abs(target_value - current_value) > 0.05 * max(target_value, current_value):
                size_delta = target_value - current_value
                trades.append({
                    'action': 'adjust',
                    'market_id': source_id,
                    'bet_name': current.get('bet_name', ''),
                    'current_size': current_value,
                    'target_size': target_value,
                    'size_delta': size_delta,
                    'target_odds': target['odds'],
                    'expected_edge': target.get('probability', 0) - (1.0 / target['odds']),
                    'reason': 'Rebalance to optimal size'
                })
    
    # Sort trades by absolute size_delta (largest first)
    trades.sort(key=lambda x: abs(x['size_delta']), reverse=True)
    
    return {
        "current_positions": positions_df.to_dict(orient="records"),
        "target_positions": target_positions.to_dict(orient="records") if not target_positions.empty else [],
        "suggested_trades": trades,
        "summary": {
            "positions_to_close": len([t for t in trades if t['action'] == 'close']),
            "positions_to_open": len([t for t in trades if t['action'] == 'open']),
            "positions_to_adjust": len([t for t in trades if t['action'] == 'adjust']),
            "total_rebalancing_trades": len(trades)
        }
    }


@app.post("/paper/analytics/rebalance/execute")
async def execute_rebalancing(max_trades: Optional[int] = None):
    """Execute rebalancing trades."""
    # First calculate what trades are needed
    rebalance_result = await calculate_rebalancing()
    
    if "message" in rebalance_result:
        return rebalance_result
    
    trades = rebalance_result.get("suggested_trades", [])
    
    if not trades:
        return {"message": "No rebalancing trades needed"}
    
    # Limit number of trades if requested
    if max_trades:
        trades = trades[:max_trades]
    
    executed_trades = []
    failed_trades = []
    
    for trade in trades:
        try:
            if trade['action'] == 'close':
                # Create sell order to close position
                order = PaperOrder(
                    order_id=f"RB_CLOSE_{int(time.time() * 1000)}",
                    timestamp=datetime.now(timezone.utc),
                    source_id=trade['market_id'],
                    market_type="",
                    bet_name=trade['bet_name'],
                    side='sell',
                    size=abs(trade['current_size']),
                    limit_price=None,  # Market order
                    signal_name='rebalancing',
                    expected_edge=0
                )
                fill = await paper_engine.submit_order(order)
                
                if fill:
                    executed_trades.append({
                        'trade': trade,
                        'fill': {
                            'price': fill.fill_price,
                            'size': fill.fill_size,
                            'commission': fill.commission
                        }
                    })
                else:
                    failed_trades.append({
                        'trade': trade,
                        'reason': 'Order rejected'
                    })
                    
            elif trade['action'] in ['open', 'adjust']:
                # Create buy order for new or adjusted position
                size = abs(trade['size_delta'])
                side = 'buy' if trade['size_delta'] > 0 else 'sell'
                
                order = PaperOrder(
                    order_id=f"RB_{trade['action'].upper()}_{int(time.time() * 1000)}",
                    timestamp=datetime.now(timezone.utc),
                    source_id=trade['market_id'],
                    market_type="",
                    bet_name=trade['bet_name'],
                    side=side,
                    size=size,
                    limit_price=trade.get('target_odds'),
                    signal_name='rebalancing',
                    expected_edge=trade.get('expected_edge', 0)
                )
                fill = await paper_engine.submit_order(order)
                
                if fill:
                    executed_trades.append({
                        'trade': trade,
                        'fill': {
                            'price': fill.fill_price,
                            'size': fill.fill_size,
                            'commission': fill.commission
                        }
                    })
                else:
                    failed_trades.append({
                        'trade': trade,
                        'reason': 'Order rejected - price or risk limit'
                    })
                    
        except Exception as e:
            logger.error(f"Error executing rebalancing trade: {e}")
            failed_trades.append({
                'trade': trade,
                'reason': str(e)
            })
    
    return {
        "executed_count": len(executed_trades),
        "failed_count": len(failed_trades),
        "executed_trades": executed_trades,
        "failed_trades": failed_trades,
        "portfolio_value_before": paper_engine.get_portfolio_value(),
        "message": f"Executed {len(executed_trades)} of {len(trades)} rebalancing trades"
    }


# Alpha research endpoints
@app.post("/alpha/signals")
async def create_alpha_signal(request: AlphaSignalRequest):
    """Create a new alpha signal for research."""
    signal = AlphaSignal(
        signal_id=f"alpha_{request.name}_{int(datetime.now().timestamp())}",
        name=request.name,
        description=request.description,
        formula=request.formula,
        parameters=request.parameters,
        created_at=datetime.now(timezone.utc),
        current_stage='raw_rd'
    )
    
    alpha_pipeline.register_signal(signal)
    
    return {
        "signal_id": signal.signal_id,
        "status": "registered",
        "stage": signal.current_stage
    }


@app.get("/alpha/signals/{signal_id}")
async def get_alpha_signal(signal_id: str):
    """Get alpha signal details and performance."""
    signal = alpha_pipeline.signals.get(signal_id)
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    report = alpha_pipeline.generate_research_report(signal_id)
    
    return report


@app.post("/alpha/signals/{signal_id}/promote")
async def promote_alpha_signal(signal_id: str, reason: str = "Manual promotion"):
    """Promote signal to next research stage."""
    try:
        alpha_pipeline.promote_signal(signal_id, reason)
        signal = alpha_pipeline.signals[signal_id]
        
        return {
            "status": "promoted",
            "new_stage": signal.current_stage,
            "reason": reason
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Market data endpoints
@app.get("/markets/active")
async def get_active_markets(db: Session = Depends(get_db)):
    """Get currently active markets."""
    # Get markets that haven't started yet
    current_time = datetime.now(timezone.utc)
    
    markets = db.query(Market).filter(
        Market.commence_time > current_time
    ).limit(100).all()
    
    return {
        "markets": [
            {
                "id": m.id,
                "source_id": m.source_id,
                "sport": m.sport_title,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "commence_time": m.commence_time.isoformat()
            }
            for m in markets
        ],
        "count": len(markets)
    }


# GraphQL data endpoints
@app.get("/graphql/markets/{network}")
async def get_graphql_markets(network: str = "optimism"):
    """Get markets from GraphQL."""
    client = OvertimeGraphQLClient(network)
    markets = await client.get_active_markets(limit=50)
    
    return {
        "network": network,
        "markets": [
            {
                "address": m.address,
                "game_id": m.game_id,
                "sport": m.sport,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "maturity": m.maturity_date.isoformat(),
                "liquidity": m.liquidity
            }
            for m in markets
        ],
        "count": len(markets)
    }


# System endpoints
@app.get("/system/config")
async def get_system_config():
    """Get current system configuration."""
    return {
        "environment": deployment.environment,
        "features": deployment.config.feature_flags,
        "network": deployment.config.network,
        "monitoring": deployment.config.monitoring['enabled']
    }


@app.get("/system/metrics")
async def get_system_metrics():
    """Get system performance metrics."""
    # This would integrate with Prometheus
    return {
        "uptime": "Not implemented",
        "trades_today": 0,
        "active_positions": 0,
        "total_pnl": 0
    }


# Background tasks
async def paper_trading_loop():
    """Continuous paper trading based on signals."""
    logger.info("Starting paper trading loop")
    
    # Import Kelly system components
    from evaluate_open_markets import (
        generate_betting_session_report_and_save,
        SIGNAL_WEIGHTS
    )
    
    # Paper trading configuration
    PAPER_TRADING_CONFIG = {
        'kelly_bankroll': 1000.0,  # Kelly calculation bankroll
        'execution_bankroll': paper_engine.current_capital if paper_engine else 10000.0,  # Use actual current capital
        'kelly_fraction': 0.25,  # 25% Kelly - more conservative
        'cap_per_game': 0.05,  # Max 5% per game - reduced for diversification
        'cap_per_bet': 0.03,  # Max 3% per bet - reduced
        'cap_per_game_market': 0.04,  # Max 4% per game-market combo
        'min_bet_abs': 50.0,  # Minimum $50 bet - increased to avoid tiny bets
        'min_bet_pct': 0.005,  # Minimum 0.5% of bankroll
        'min_break_minutes': 60.0,
        'avg_game_duration_minutes': 120.0,
    }
    
    while True:
        try:
            # Use Kelly system to generate optimal bets
            logger.info("[PAPER_TRADE] Running Kelly optimization for current markets")
            
            kelly_results = generate_betting_session_report_and_save(
                kelly_bankroll=PAPER_TRADING_CONFIG['kelly_bankroll'],
                execution_bankroll=paper_engine.current_capital,  # Use actual current capital
                kelly_fraction=PAPER_TRADING_CONFIG['kelly_fraction'],
                cap_per_game=PAPER_TRADING_CONFIG['cap_per_game'],
                cap_per_bet=PAPER_TRADING_CONFIG['cap_per_bet'],
                cap_per_game_market=PAPER_TRADING_CONFIG['cap_per_game_market'],
                min_bet_abs=PAPER_TRADING_CONFIG['min_bet_abs'],
                min_bet_pct=PAPER_TRADING_CONFIG['min_bet_pct'],
                as_of=datetime.now(timezone.utc),
                min_break_minutes=PAPER_TRADING_CONFIG['min_break_minutes'],
                avg_game_duration_minutes=PAPER_TRADING_CONFIG['avg_game_duration_minutes'],
                signal_providers=SIGNAL_PROVIDERS,
                signal_weights=SIGNAL_WEIGHTS,
                mode='paper_trading',
                display_md=False,
                save_as_latest=False
            )
            
            if not kelly_results or 'trimmed' not in kelly_results:
                logger.warning("[PAPER_TRADE] No Kelly recommendations generated")
                await asyncio.sleep(900)  # Wait 15 minutes
                continue
                
            # Get bets with positive stake
            recommended_bets = kelly_results['trimmed']
            bets_to_place = recommended_bets[recommended_bets['stake'] > 0]
            
            if bets_to_place.empty:
                logger.info("[PAPER_TRADE] No bets meet Kelly criteria")
                await asyncio.sleep(900)
                continue
                
            logger.info(f"[PAPER_TRADE] Kelly system recommends {len(bets_to_place)} bets")
            
            # Get current positions for portfolio comparison
            current_positions = paper_engine.get_open_positions()
            current_portfolio = {}
            if not current_positions.empty:
                for _, pos in current_positions.iterrows():
                    current_portfolio[pos['market_id']] = {
                        'size': pos['size'],
                        'value': pos['current_value'],
                        'bet_name': pos.get('bet_name', '')
                    }
                logger.info(f"[PAPER_TRADE] Current portfolio has {len(current_portfolio)} positions")
            
            # Build optimal portfolio from Kelly recommendations
            optimal_portfolio = {}
            for idx, bet_row in bets_to_place.iterrows():
                if bet_row['stake'] > 0:
                    optimal_portfolio[bet_row['source_id']] = {
                        'stake': bet_row['stake'],
                        'odds': bet_row['odds'],
                        'market_name': bet_row['market_name'],
                        'outcome': bet_row['normalized_outcome'],
                        'bet_name': f"{bet_row['market_name']} - {bet_row['normalized_outcome']}",
                        'expected_edge': bet_row.get('probability', 0) - (1.0 / bet_row['odds'])
                    }
            
            logger.info(f"[PAPER_TRADE] Optimal portfolio has {len(optimal_portfolio)} positions")
            
            # Calculate required transactions
            transactions = []
            
            # Transaction thresholds
            MIN_TRADE_ABS = 50.0  # Minimum $50 trade
            MIN_TRADE_PCT = 0.10  # Minimum 10% change to justify transaction
            
            # Check positions that need adjustment or closing
            for market_id, current in current_portfolio.items():
                optimal = optimal_portfolio.get(market_id)
                
                if optimal is None:
                    # Position should be closed
                    if current['size'] > MIN_TRADE_ABS:
                        transactions.append({
                            'action': 'close',
                            'market_id': market_id,
                            'current_size': current['size'],
                            'target_size': 0,
                            'trade_size': -current['size'],
                            'bet_name': current['bet_name']
                        })
                        logger.info(f"[PAPER_TRADE] Close position: {current['bet_name']} (${current['size']:.2f})")
                else:
                    # Position needs adjustment
                    size_diff = optimal['stake'] - current['value']
                    pct_change = abs(size_diff) / current['value'] if current['value'] > 0 else 1.0
                    
                    if abs(size_diff) > MIN_TRADE_ABS and pct_change > MIN_TRADE_PCT:
                        transactions.append({
                            'action': 'adjust',
                            'market_id': market_id,
                            'current_size': current['value'],
                            'target_size': optimal['stake'],
                            'trade_size': size_diff,
                            'bet_name': current['bet_name'],
                            'odds': optimal['odds'],
                            'expected_edge': optimal['expected_edge']
                        })
                        logger.info(f"[PAPER_TRADE] Adjust position: {current['bet_name']} ${current['value']:.2f} -> ${optimal['stake']:.2f}")
            
            # Check new positions to open
            for market_id, optimal in optimal_portfolio.items():
                if market_id not in current_portfolio:
                    if optimal['stake'] > MIN_TRADE_ABS:
                        transactions.append({
                            'action': 'open',
                            'market_id': market_id,
                            'current_size': 0,
                            'target_size': optimal['stake'],
                            'trade_size': optimal['stake'],
                            'bet_name': optimal['bet_name'],
                            'odds': optimal['odds'],
                            'expected_edge': optimal['expected_edge']
                        })
                        logger.info(f"[PAPER_TRADE] Open new position: {optimal['bet_name']} ${optimal['stake']:.2f}")
            
            # Sort transactions by absolute trade size (largest first)
            transactions.sort(key=lambda x: abs(x['trade_size']), reverse=True)
            
            # Execute transactions
            trades_executed = 0
            
            for tx in transactions:
                try:
                    # Determine order parameters based on transaction type
                    if tx['action'] == 'close':
                        # Sell entire position
                        side = 'sell'
                        size = abs(tx['current_size'])
                        limit_price = None  # Market order
                        expected_edge = 0
                    elif tx['action'] == 'adjust':
                        # Buy more or sell some
                        if tx['trade_size'] > 0:
                            side = 'buy'
                            size = tx['trade_size']
                        else:
                            side = 'sell'
                            size = abs(tx['trade_size'])
                        limit_price = tx.get('odds')
                        expected_edge = tx.get('expected_edge', 0)
                    else:  # open
                        side = 'buy'
                        size = tx['trade_size']
                        limit_price = tx.get('odds')
                        expected_edge = tx.get('expected_edge', 0)
                    
                    # Create paper order
                    order = PaperOrder(
                        order_id=f"O{int(time.time() * 1000)}_{trades_executed}",
                        timestamp=datetime.now(timezone.utc),
                        source_id=tx['market_id'],
                        market_type='h2h',  # Default market type
                        bet_name=tx['bet_name'],
                        side=side,
                        size=size,
                        limit_price=limit_price,
                        signal_name='kelly_rebalance',
                        expected_edge=expected_edge
                    )
                    
                    # Submit order
                    fill = await paper_engine.submit_order(order)
                    if fill:
                        trades_executed += 1
                        action_str = {'open': 'Opened', 'close': 'Closed', 'adjust': 'Adjusted'}[tx['action']]
                        logger.info(f"[PAPER_TRADE] {action_str} position: {tx['bet_name']}, "
                                  f"{side} ${size:.2f} @ {fill.fill_price:.3f}, "
                                  f"portfolio impact: ${tx['current_size']:.2f} -> ${tx['target_size']:.2f}")
                    else:
                        logger.warning(f"[PAPER_TRADE] Failed to execute {tx['action']} for {tx['bet_name']}")
                        
                except Exception as e:
                    logger.error(f"Error executing {tx['action']} for {tx.get('bet_name', 'unknown')}: {e}")
                    continue
            
            logger.info(f"[PAPER_TRADE] Rebalancing complete: {trades_executed}/{len(transactions)} transactions executed")
            
            # Log detailed portfolio status
            try:
                portfolio_value = paper_engine.get_portfolio_value()
                positions_df = paper_engine.get_open_positions()
                exposure_info = paper_engine.get_position_exposure()
                
                logger.info("[PAPER_TRADE] Portfolio Status:")
                logger.info(f"  - Total Value: ${portfolio_value:.2f} (Cash: ${paper_engine.current_capital:.2f})")
                logger.info(f"  - Open Positions: {exposure_info['position_count']}")
                logger.info(f"  - Total Exposure: {exposure_info['total_exposure']:.1%}")
                logger.info(f"  - Largest Position: {exposure_info['largest_position']:.1%}")
                
                if not positions_df.empty:
                    logger.info("[PAPER_TRADE] Position Details:")
                    for _, pos in positions_df.head(5).iterrows():  # Show top 5 positions
                        logger.info(f"    {pos['market_id']}: ${pos['current_value']:.2f} "
                                  f"(P&L: {pos['pnl_pct']:.1%})")
                        
                # Check stop losses
                stopped = paper_engine.check_stop_losses()
                if stopped:
                    logger.warning(f"[PAPER_TRADE] Stop losses triggered: {stopped}")
                    
            except Exception as e:
                logger.error(f"[PAPER_TRADE] Error getting portfolio status: {e}")
            
            # Run every 15 minutes
            await asyncio.sleep(900)
            
        except asyncio.CancelledError:
            logger.info("Paper trading loop cancelled")
            break
        except Exception as e:
            logger.error(f"Paper trading error: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes on error


async def metrics_collection_loop():
    """Collect and export metrics."""
    logger.info("Starting metrics collection")
    
    while True:
        try:
            # Collect metrics
            # Export to monitoring system
            await asyncio.sleep(30)  # Every 30 seconds
            
        except asyncio.CancelledError:
            logger.info("Metrics collection cancelled")
            break
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            await asyncio.sleep(60)


# Main entry point
if __name__ == "__main__":
    # Get configuration
    host = "0.0.0.0"
    port = 8000
    
    if deployment.environment == "development":
        # Development mode with auto-reload
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
            log_level="debug"
        )
    else:
        # Production mode
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
