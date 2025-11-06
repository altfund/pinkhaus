from rest_framework import serializers
from decimal import Decimal
from ..models import (
    ChainNetwork, Web3Account, TradingSession, Position,
    OptimizationRun, MarketData, BlockchainEvent
)


class ChainNetworkSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChainNetwork
        fields = [
            'id', 'name', 'chain_id', 'is_testnet', 'is_active',
            'explorer_url', 'trading_engine_address'
        ]


class Web3AccountSerializer(serializers.ModelSerializer):
    chain_name = serializers.CharField(source='chain.get_name_display', read_only=True)
    formatted_address = serializers.SerializerMethodField()
    
    class Meta:
        model = Web3Account
        fields = [
            'id', 'wallet_address', 'formatted_address', 'chain', 'chain_name',
            'ens_name', 'is_verified', 'last_connected'
        ]
    
    def get_formatted_address(self, obj):
        return f"{obj.wallet_address[:6]}...{obj.wallet_address[-4:]}"


class CreateSessionSerializer(serializers.Serializer):
    chain_id = serializers.IntegerField()
    initial_bankroll = serializers.DecimalField(max_digits=20, decimal_places=8, min_value=Decimal('0.01'))
    transaction_hash = serializers.CharField(max_length=66, required=False)
    
    def validate_chain_id(self, value):
        try:
            ChainNetwork.objects.get(chain_id=value, is_active=True)
        except ChainNetwork.DoesNotExist:
            raise serializers.ValidationError("Invalid or inactive chain")
        return value


class TradingSessionSerializer(serializers.ModelSerializer):
    chain_name = serializers.CharField(source='chain.get_name_display', read_only=True)
    wallet_address = serializers.CharField(source='web3_account.wallet_address', read_only=True)
    win_rate = serializers.ReadOnlyField()
    roi = serializers.ReadOnlyField()
    duration = serializers.SerializerMethodField()
    
    class Meta:
        model = TradingSession
        fields = [
            'id', 'session_id', 'chain', 'chain_name', 'wallet_address',
            'initial_bankroll', 'current_bankroll', 'is_active',
            'created_at', 'last_activity', 'closed_at',
            'total_bets_placed', 'total_bets_won', 'total_profit',
            'win_rate', 'roi', 'duration', 'transaction_hash'
        ]
    
    def get_duration(self, obj):
        if obj.closed_at:
            delta = obj.closed_at - obj.created_at
        else:
            delta = timezone.now() - obj.created_at
        return int(delta.total_seconds())


class PositionSerializer(serializers.ModelSerializer):
    expected_payout = serializers.ReadOnlyField()
    profit = serializers.ReadOnlyField()
    outcome_display = serializers.CharField(source='get_outcome_display', read_only=True)
    match_display = serializers.SerializerMethodField()
    
    class Meta:
        model = Position
        fields = [
            'id', 'position_id', 'session', 'market_id',
            'stake', 'odds', 'outcome', 'outcome_display',
            'match_display', 'sport', 'match_date',
            'placed_at', 'is_settled', 'is_won', 'payout',
            'settled_at', 'expected_payout', 'profit',
            'transaction_hash'
        ]
    
    def get_match_display(self, obj):
        return f"{obj.home_team} vs {obj.away_team}"


class PlaceBetSerializer(serializers.Serializer):
    session_id = serializers.IntegerField()
    market_id = serializers.CharField(max_length=66)
    stake = serializers.DecimalField(max_digits=20, decimal_places=8, min_value=Decimal('0.001'))
    outcome = serializers.ChoiceField(choices=[(0, 'Home'), (1, 'Draw'), (2, 'Away')])
    transaction_hash = serializers.CharField(max_length=66, required=False)
    
    def validate_session_id(self, value):
        try:
            session = TradingSession.objects.get(id=value, is_active=True)
            if not self.context['request'].user == session.user:
                raise serializers.ValidationError("Not your session")
        except TradingSession.DoesNotExist:
            raise serializers.ValidationError("Invalid or inactive session")
        return value
    
    def validate(self, data):
        # Check if stake doesn't exceed bankroll
        session = TradingSession.objects.get(id=data['session_id'])
        if data['stake'] > session.current_bankroll:
            raise serializers.ValidationError("Stake exceeds available bankroll")
        return data


class OptimizePortfolioSerializer(serializers.Serializer):
    session_id = serializers.IntegerField()
    market_ids = serializers.ListField(
        child=serializers.CharField(max_length=66),
        min_length=1,
        max_length=50
    )
    chunk_duration_minutes = serializers.IntegerField(min_value=60, max_value=480, default=120)
    use_half_kelly = serializers.BooleanField(default=True)
    max_stake_percentage = serializers.IntegerField(min_value=1, max_value=10, default=5)
    
    def validate_session_id(self, value):
        try:
            session = TradingSession.objects.get(id=value, is_active=True)
            if not self.context['request'].user == session.user:
                raise serializers.ValidationError("Not your session")
        except TradingSession.DoesNotExist:
            raise serializers.ValidationError("Invalid or inactive session")
        return value


class OptimizationResultSerializer(serializers.Serializer):
    market_id = serializers.CharField()
    home_team = serializers.CharField()
    away_team = serializers.CharField()
    recommended_outcome = serializers.IntegerField()
    recommended_stake = serializers.DecimalField(max_digits=20, decimal_places=8)
    odds = serializers.DecimalField(max_digits=10, decimal_places=3)
    edge = serializers.DecimalField(max_digits=10, decimal_places=4)
    kelly_fraction = serializers.DecimalField(max_digits=10, decimal_places=4)


class MarketDataSerializer(serializers.ModelSerializer):
    current_time_to_match = serializers.SerializerMethodField()
    
    class Meta:
        model = MarketData
        fields = [
            'id', 'market_id', 'chain', 'home_team', 'away_team',
            'sport', 'league', 'home_odds', 'draw_odds', 'away_odds',
            'match_date', 'is_resolved', 'winning_outcome',
            'current_time_to_match', 'updated_at'
        ]
    
    def get_current_time_to_match(self, obj):
        from django.utils import timezone
        delta = obj.match_date - timezone.now()
        return int(delta.total_seconds()) if delta.total_seconds() > 0 else 0


class BlockchainTransactionSerializer(serializers.Serializer):
    """Serializer for blockchain transaction preparation"""
    function_name = serializers.CharField()
    from_address = serializers.CharField()
    to_address = serializers.CharField()
    value = serializers.CharField()
    gas = serializers.IntegerField()
    gas_price = serializers.CharField()
    nonce = serializers.IntegerField()
    chain_id = serializers.IntegerField()
    data = serializers.CharField()


class WalletVerificationSerializer(serializers.Serializer):
    """Serializer for wallet ownership verification"""
    wallet_address = serializers.CharField()
    signature = serializers.CharField()
    message = serializers.CharField()
    chain_id = serializers.IntegerField()


from django.utils import timezone