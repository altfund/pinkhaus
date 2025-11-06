"""
Django signals for Ominari trading app
"""
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import BlockchainEvent, TradingSession, Position
from .tasks import process_blockchain_events, update_session_stats
import logging

logger = logging.getLogger(__name__)


@receiver(post_save, sender=BlockchainEvent)
def handle_new_blockchain_event(sender, instance, created, **kwargs):
    """Process new blockchain events"""
    if created and not instance.is_processed:
        # Schedule processing
        process_blockchain_events.apply_async(countdown=1)


@receiver(post_save, sender=TradingSession)
def handle_session_update(sender, instance, created, **kwargs):
    """Update session stats when session changes"""
    if not created and instance.is_active:
        # Schedule stats update
        update_session_stats.apply_async(
            args=[instance.id],
            countdown=5
        )


@receiver(post_save, sender=Position)
def handle_position_update(sender, instance, created, **kwargs):
    """Update session when position changes"""
    if instance.is_settled and not created:
        # Position was just settled
        session = instance.session
        
        # Update profit
        session.total_profit = sum(
            p.profit for p in session.positions.filter(is_settled=True)
        )
        
        # Update current bankroll
        if instance.is_won:
            session.current_bankroll += instance.payout
        
        session.save()