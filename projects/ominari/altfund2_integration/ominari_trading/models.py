from django.db import models
from django.contrib.auth import get_user_model
from django.core.validators import MinValueValidator
from django.utils import timezone
from decimal import Decimal

User = get_user_model()


class ChainNetwork(models.Model):
    """Supported blockchain networks"""
    CHAIN_CHOICES = [
        ('ethereum', 'Ethereum Mainnet'),
        ('polygon', 'Polygon'),
        ('arbitrum', 'Arbitrum One'),
        ('sepolia', 'Sepolia Testnet'),
        ('mumbai', 'Polygon Mumbai'),
    ]
    
    name = models.CharField(max_length=50, choices=CHAIN_CHOICES, unique=True)
    chain_id = models.IntegerField(unique=True)
    rpc_url = models.URLField()
    explorer_url = models.URLField()
    is_testnet = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    
    # Contract addresses
    trading_engine_address = models.CharField(max_length=42, blank=True)
    kelly_optimizer_address = models.CharField(max_length=42, blank=True)
    chunk_manager_address = models.CharField(max_length=42, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['chain_id']
    
    def __str__(self):
        return f"{self.get_name_display()} ({self.chain_id})"


class Web3Account(models.Model):
    """User's Web3 wallet connections"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='web3_accounts')
    wallet_address = models.CharField(max_length=42, db_index=True)
    chain = models.ForeignKey(ChainNetwork, on_delete=models.PROTECT)
    
    # ENS/domain name if available
    ens_name = models.CharField(max_length=255, blank=True)
    
    # Verification status
    is_verified = models.BooleanField(default=False)
    verification_signature = models.TextField(blank=True)
    verified_at = models.DateTimeField(null=True, blank=True)
    
    # Activity tracking
    first_connected = models.DateTimeField(auto_now_add=True)
    last_connected = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['wallet_address', 'chain']
        ordering = ['-last_connected']
    
    def __str__(self):
        return f"{self.wallet_address[:6]}...{self.wallet_address[-4:]} on {self.chain.name}"


class TradingSession(models.Model):
    """On-chain trading session representation"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='trading_sessions')
    web3_account = models.ForeignKey(Web3Account, on_delete=models.CASCADE)
    
    # On-chain data
    session_id = models.CharField(max_length=100)  # Chain ID + contract session ID
    chain = models.ForeignKey(ChainNetwork, on_delete=models.PROTECT)
    transaction_hash = models.CharField(max_length=66)
    
    # Session details (cached from blockchain)
    initial_bankroll = models.DecimalField(max_digits=20, decimal_places=8, validators=[MinValueValidator(0)])
    current_bankroll = models.DecimalField(max_digits=20, decimal_places=8, validators=[MinValueValidator(0)])
    
    # Status
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    last_activity = models.DateTimeField(auto_now=True)
    closed_at = models.DateTimeField(null=True, blank=True)
    
    # Statistics (updated periodically from blockchain)
    total_bets_placed = models.IntegerField(default=0)
    total_bets_won = models.IntegerField(default=0)
    total_profit = models.DecimalField(max_digits=20, decimal_places=8, default=Decimal('0'))
    
    class Meta:
        unique_together = ['chain', 'session_id']
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Session {self.session_id} - {self.user.username}"
    
    @property
    def win_rate(self):
        if self.total_bets_placed == 0:
            return 0
        return (self.total_bets_won / self.total_bets_placed) * 100
    
    @property
    def roi(self):
        if self.initial_bankroll == 0:
            return 0
        return ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll) * 100


class Position(models.Model):
    """Individual betting position"""
    session = models.ForeignKey(TradingSession, on_delete=models.CASCADE, related_name='positions')
    
    # On-chain identifiers
    position_id = models.CharField(max_length=100)
    market_id = models.CharField(max_length=66)
    transaction_hash = models.CharField(max_length=66)
    
    # Position details
    stake = models.DecimalField(max_digits=20, decimal_places=8, validators=[MinValueValidator(0)])
    odds = models.DecimalField(max_digits=10, decimal_places=3, validators=[MinValueValidator(1)])
    outcome = models.IntegerField(choices=[(0, 'Home'), (1, 'Draw'), (2, 'Away')])
    
    # Market info (cached)
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    sport = models.CharField(max_length=50)
    match_date = models.DateTimeField()
    
    # Status
    placed_at = models.DateTimeField(default=timezone.now)
    is_settled = models.BooleanField(default=False)
    is_won = models.BooleanField(null=True)
    payout = models.DecimalField(max_digits=20, decimal_places=8, default=Decimal('0'))
    settled_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        unique_together = ['session', 'position_id']
        ordering = ['-placed_at']
    
    def __str__(self):
        return f"Position {self.position_id} - {self.home_team} vs {self.away_team}"
    
    @property
    def expected_payout(self):
        return self.stake * self.odds
    
    @property
    def profit(self):
        if not self.is_settled:
            return Decimal('0')
        return self.payout - self.stake if self.is_won else -self.stake


class OptimizationRun(models.Model):
    """Kelly optimization execution record"""
    session = models.ForeignKey(TradingSession, on_delete=models.CASCADE, related_name='optimizations')
    
    # Execution details
    transaction_hash = models.CharField(max_length=66, blank=True)
    gas_used = models.IntegerField(null=True)
    
    # Parameters
    chunk_duration_minutes = models.IntegerField()
    markets_analyzed = models.IntegerField()
    positions_recommended = models.IntegerField()
    total_stake_allocated = models.DecimalField(max_digits=20, decimal_places=8)
    
    # Timing
    executed_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-executed_at']
    
    def __str__(self):
        return f"Optimization for session {self.session.session_id} at {self.executed_at}"


class MarketData(models.Model):
    """Cached market data for faster access"""
    market_id = models.CharField(max_length=66, unique=True, db_index=True)
    chain = models.ForeignKey(ChainNetwork, on_delete=models.CASCADE)
    
    # Teams/participants
    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    sport = models.CharField(max_length=50, db_index=True)
    league = models.CharField(max_length=100, blank=True)
    
    # Odds (latest)
    home_odds = models.DecimalField(max_digits=10, decimal_places=3, validators=[MinValueValidator(1)])
    draw_odds = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
    away_odds = models.DecimalField(max_digits=10, decimal_places=3, validators=[MinValueValidator(1)])
    
    # Timing
    match_date = models.DateTimeField(db_index=True)
    is_resolved = models.BooleanField(default=False)
    winning_outcome = models.IntegerField(null=True, choices=[(0, 'Home'), (1, 'Draw'), (2, 'Away')])
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['match_date']
        indexes = [
            models.Index(fields=['sport', 'match_date']),
            models.Index(fields=['is_resolved', 'match_date']),
        ]
    
    def __str__(self):
        return f"{self.home_team} vs {self.away_team} - {self.match_date.date()}"


class BlockchainEvent(models.Model):
    """Raw blockchain events for processing"""
    EVENT_TYPES = [
        ('session_created', 'Session Created'),
        ('position_placed', 'Position Placed'),
        ('position_settled', 'Position Settled'),
        ('session_closed', 'Session Closed'),
        ('portfolio_optimized', 'Portfolio Optimized'),
    ]
    
    chain = models.ForeignKey(ChainNetwork, on_delete=models.CASCADE)
    event_type = models.CharField(max_length=50, choices=EVENT_TYPES)
    
    # Event data
    transaction_hash = models.CharField(max_length=66, db_index=True)
    block_number = models.BigIntegerField(db_index=True)
    log_index = models.IntegerField()
    
    # Raw event data (JSON)
    event_data = models.JSONField()
    
    # Processing status
    is_processed = models.BooleanField(default=False, db_index=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True)
    
    # Timestamps
    block_timestamp = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['chain', 'transaction_hash', 'log_index']
        ordering = ['-block_number', '-log_index']
        indexes = [
            models.Index(fields=['is_processed', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.event_type} - Block {self.block_number}"