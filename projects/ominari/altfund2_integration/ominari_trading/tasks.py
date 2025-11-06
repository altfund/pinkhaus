"""
Celery tasks for blockchain operations and data synchronization
"""
from celery import shared_task
from celery.utils.log import get_task_logger
from django.utils import timezone
from django.db import transaction
from decimal import Decimal
from typing import List, Dict
import time

from .models import (
    ChainNetwork, TradingSession, Position, BlockchainEvent,
    MarketData, OptimizationRun, Web3Account
)
from .blockchain import OminariBlockchainClient, MockBlockchainClient

logger = get_task_logger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def sync_blockchain_events(self, chain_id: int, from_block: int = None):
    """
    Sync blockchain events for a specific chain
    """
    try:
        chain = ChainNetwork.objects.get(chain_id=chain_id, is_active=True)
        
        # Use mock client for development
        if chain.is_testnet or not chain.trading_engine_address:
            client = MockBlockchainClient(chain)
            logger.info(f"Using mock client for {chain.name}")
            return
        else:
            client = OminariBlockchainClient(chain)
        
        # Determine starting block
        if from_block is None:
            last_event = BlockchainEvent.objects.filter(
                chain=chain
            ).order_by('-block_number').first()
            
            from_block = last_event.block_number + 1 if last_event else 0
        
        # Get current block
        current_block = client.w3.eth.block_number
        
        # Process in chunks to avoid timeouts
        chunk_size = 1000
        to_block = min(from_block + chunk_size, current_block)
        
        logger.info(f"Syncing {chain.name} events from block {from_block} to {to_block}")
        
        # Fetch events
        event_types = ['SessionCreated', 'PositionPlaced', 'PositionSettled', 'SessionClosed', 'PortfolioOptimized']
        
        for event_type in event_types:
            events = client.get_events(event_type, from_block, to_block)
            
            for event in events:
                # Create BlockchainEvent record
                BlockchainEvent.objects.update_or_create(
                    chain=chain,
                    transaction_hash=event['transaction_hash'],
                    log_index=event['log_index'],
                    defaults={
                        'event_type': event_type.lower(),
                        'block_number': event['block_number'],
                        'event_data': event['args'],
                        'block_timestamp': timezone.now(),  # Would get actual timestamp from block
                    }
                )
        
        logger.info(f"Synced {to_block - from_block + 1} blocks for {chain.name}")
        
        # Schedule next sync if not caught up
        if to_block < current_block:
            self.apply_async(args=[chain_id, to_block + 1], countdown=5)
        
    except ChainNetwork.DoesNotExist:
        logger.error(f"Chain with ID {chain_id} not found")
    except Exception as e:
        logger.error(f"Failed to sync blockchain events: {e}")
        raise self.retry(exc=e)


@shared_task(bind=True, max_retries=3)
def process_blockchain_events(self, batch_size: int = 100):
    """
    Process unprocessed blockchain events
    """
    try:
        # Get unprocessed events
        events = BlockchainEvent.objects.filter(
            is_processed=False
        ).order_by('block_number', 'log_index')[:batch_size]
        
        processed_count = 0
        
        for event in events:
            try:
                with transaction.atomic():
                    if event.event_type == 'session_created':
                        process_session_created_event(event)
                    elif event.event_type == 'position_placed':
                        process_position_placed_event(event)
                    elif event.event_type == 'position_settled':
                        process_position_settled_event(event)
                    elif event.event_type == 'session_closed':
                        process_session_closed_event(event)
                    elif event.event_type == 'portfolio_optimized':
                        process_portfolio_optimized_event(event)
                    
                    # Mark as processed
                    event.is_processed = True
                    event.processed_at = timezone.now()
                    event.save()
                    
                    processed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process event {event.id}: {e}")
                event.error = str(e)
                event.save()
        
        logger.info(f"Processed {processed_count} blockchain events")
        
        # Schedule next batch if more events
        if events.count() == batch_size:
            self.apply_async(countdown=1)
        
    except Exception as e:
        logger.error(f"Failed to process blockchain events: {e}")
        raise self.retry(exc=e)


def process_session_created_event(event: BlockchainEvent):
    """Process SessionCreated event"""
    data = event.event_data
    
    # Find or create Web3Account
    web3_account, _ = Web3Account.objects.get_or_create(
        wallet_address=data['trader'].lower(),
        chain=event.chain,
        defaults={'user_id': 1}  # Would need proper user matching in production
    )
    
    # Create TradingSession
    session_id = f"{event.chain.chain_id}-{data['sessionId']}"
    
    TradingSession.objects.update_or_create(
        chain=event.chain,
        session_id=session_id,
        defaults={
            'user': web3_account.user,
            'web3_account': web3_account,
            'transaction_hash': event.transaction_hash,
            'initial_bankroll': Decimal(str(data['bankroll'])) / Decimal('1e18'),
            'current_bankroll': Decimal(str(data['bankroll'])) / Decimal('1e18'),
            'created_at': event.block_timestamp,
        }
    )


def process_position_placed_event(event: BlockchainEvent):
    """Process PositionPlaced event"""
    data = event.event_data
    
    # Find session
    session_id = f"{event.chain.chain_id}-{data['sessionId']}"
    try:
        session = TradingSession.objects.get(
            chain=event.chain,
            session_id=session_id
        )
    except TradingSession.DoesNotExist:
        logger.error(f"Session {session_id} not found for position event")
        return
    
    # Create Position
    Position.objects.update_or_create(
        session=session,
        position_id=str(data['positionId']),
        defaults={
            'market_id': data['marketId'],
            'transaction_hash': event.transaction_hash,
            'stake': Decimal(str(data['stake'])) / Decimal('1e18'),
            'outcome': data['outcome'],
            'placed_at': event.block_timestamp,
            # Other fields would be populated from market data
        }
    )
    
    # Update session stats
    session.total_bets_placed += 1
    session.last_activity = event.block_timestamp
    session.save()


def process_position_settled_event(event: BlockchainEvent):
    """Process PositionSettled event"""
    data = event.event_data
    
    # Find position
    try:
        position = Position.objects.get(
            position_id=str(data['positionId'])
        )
    except Position.DoesNotExist:
        logger.error(f"Position {data['positionId']} not found")
        return
    
    # Update position
    position.is_settled = True
    position.is_won = data['won']
    position.payout = Decimal(str(data['payout'])) / Decimal('1e18')
    position.settled_at = event.block_timestamp
    position.save()
    
    # Update session stats
    session = position.session
    if position.is_won:
        session.total_bets_won += 1
    session.save()


def process_session_closed_event(event: BlockchainEvent):
    """Process SessionClosed event"""
    data = event.event_data
    
    # Find session
    session_id = f"{event.chain.chain_id}-{data['sessionId']}"
    try:
        session = TradingSession.objects.get(
            chain=event.chain,
            session_id=session_id
        )
    except TradingSession.DoesNotExist:
        logger.error(f"Session {session_id} not found")
        return
    
    # Update session
    session.is_active = False
    session.closed_at = event.block_timestamp
    session.current_bankroll = Decimal(str(data['finalBankroll'])) / Decimal('1e18')
    session.total_profit = Decimal(str(data['totalProfit'])) / Decimal('1e18')
    session.save()


def process_portfolio_optimized_event(event: BlockchainEvent):
    """Process PortfolioOptimized event"""
    data = event.event_data
    
    # Find session
    session_id = f"{event.chain.chain_id}-{data['sessionId']}"
    try:
        session = TradingSession.objects.get(
            chain=event.chain,
            session_id=session_id
        )
    except TradingSession.DoesNotExist:
        logger.error(f"Session {session_id} not found")
        return
    
    # Create OptimizationRun
    OptimizationRun.objects.create(
        session=session,
        transaction_hash=event.transaction_hash,
        markets_analyzed=data['marketCount'],
        executed_at=event.block_timestamp,
        # Other fields would be populated from transaction details
        chunk_duration_minutes=120,  # Default
        positions_recommended=0,
        total_stake_allocated=Decimal('0')
    )


@shared_task
def update_session_stats(session_id: int):
    """
    Update session statistics from blockchain
    """
    try:
        session = TradingSession.objects.get(id=session_id)
        
        # Initialize client
        if session.chain.is_testnet:
            client = MockBlockchainClient(session.chain)
        else:
            client = OminariBlockchainClient(session.chain)
        
        # Get on-chain session data
        chain_session_id = int(session.session_id.split('-')[1])
        session_data = client.get_session_data(chain_session_id)
        
        if session_data:
            # Update local data
            session.current_bankroll = Decimal(str(session_data['current_bankroll']))
            session.total_bets_placed = session_data['total_bets_placed']
            session.total_bets_won = session_data['total_bets_won']
            session.total_profit = Decimal(str(session_data['total_profit']))
            session.is_active = session_data['is_active']
            session.save()
            
            logger.info(f"Updated stats for session {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to update session stats: {e}")


@shared_task
def sync_market_data():
    """
    Sync market data from external sources
    """
    # This would integrate with TheGraph or other indexing services
    # to get market data efficiently
    pass


@shared_task
def cleanup_old_events(days: int = 30):
    """
    Clean up old processed blockchain events
    """
    try:
        cutoff_date = timezone.now() - timezone.timedelta(days=days)
        
        count = BlockchainEvent.objects.filter(
            is_processed=True,
            processed_at__lt=cutoff_date
        ).delete()[0]
        
        logger.info(f"Cleaned up {count} old blockchain events")
        
    except Exception as e:
        logger.error(f"Failed to cleanup old events: {e}")