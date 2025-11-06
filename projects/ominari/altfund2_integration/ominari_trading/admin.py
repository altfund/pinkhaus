from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils import timezone
from .models import (
    ChainNetwork, Web3Account, TradingSession, Position,
    OptimizationRun, MarketData, BlockchainEvent
)


@admin.register(ChainNetwork)
class ChainNetworkAdmin(admin.ModelAdmin):
    list_display = ['name', 'chain_id', 'is_testnet', 'is_active', 'contract_status']
    list_filter = ['is_testnet', 'is_active']
    search_fields = ['name', 'chain_id']
    
    fieldsets = (
        ('Network Info', {
            'fields': ('name', 'chain_id', 'is_testnet', 'is_active')
        }),
        ('URLs', {
            'fields': ('rpc_url', 'explorer_url')
        }),
        ('Contract Addresses', {
            'fields': (
                'trading_engine_address',
                'kelly_optimizer_address',
                'chunk_manager_address'
            ),
            'description': 'Smart contract addresses on this network'
        })
    )
    
    def contract_status(self, obj):
        if obj.trading_engine_address:
            return format_html(
                '<span style="color: green;">✓ Deployed</span>'
            )
        return format_html(
            '<span style="color: orange;">⚠ Not Deployed</span>'
        )
    contract_status.short_description = 'Contracts'


@admin.register(Web3Account)
class Web3AccountAdmin(admin.ModelAdmin):
    list_display = [
        'formatted_address', 'user', 'chain', 'ens_name',
        'is_verified', 'last_connected'
    ]
    list_filter = ['chain', 'is_verified']
    search_fields = ['wallet_address', 'ens_name', 'user__username']
    readonly_fields = ['verification_signature', 'verified_at']
    
    def formatted_address(self, obj):
        return format_html(
            '<code>{}</code>',
            f"{obj.wallet_address[:6]}...{obj.wallet_address[-4:]}"
        )
    formatted_address.short_description = 'Address'


@admin.register(TradingSession)
class TradingSessionAdmin(admin.ModelAdmin):
    list_display = [
        'session_id', 'user', 'chain', 'status_badge',
        'current_bankroll', 'profit_display', 'win_rate_display',
        'created_at'
    ]
    list_filter = ['chain', 'is_active', 'created_at']
    search_fields = ['session_id', 'user__username', 'transaction_hash']
    readonly_fields = [
        'session_id', 'transaction_hash', 'win_rate', 'roi',
        'created_at', 'last_activity', 'closed_at'
    ]
    
    fieldsets = (
        ('Session Info', {
            'fields': (
                'user', 'web3_account', 'chain', 'session_id',
                'transaction_hash', 'is_active'
            )
        }),
        ('Financial', {
            'fields': (
                'initial_bankroll', 'current_bankroll', 'total_profit',
                'win_rate', 'roi'
            )
        }),
        ('Statistics', {
            'fields': (
                'total_bets_placed', 'total_bets_won',
                'created_at', 'last_activity', 'closed_at'
            )
        })
    )
    
    def status_badge(self, obj):
        if obj.is_active:
            return format_html(
                '<span style="color: green;">● Active</span>'
            )
        return format_html(
            '<span style="color: gray;">● Closed</span>'
        )
    status_badge.short_description = 'Status'
    
    def profit_display(self, obj):
        profit = obj.total_profit
        if profit > 0:
            return format_html(
                '<span style="color: green;">+{:.4f}</span>',
                profit
            )
        elif profit < 0:
            return format_html(
                '<span style="color: red;">{:.4f}</span>',
                profit
            )
        return '0.0000'
    profit_display.short_description = 'Profit'
    
    def win_rate_display(self, obj):
        return f"{obj.win_rate:.1f}%"
    win_rate_display.short_description = 'Win Rate'


@admin.register(Position)
class PositionAdmin(admin.ModelAdmin):
    list_display = [
        'position_id', 'session_link', 'match_display',
        'outcome', 'stake', 'odds', 'status_display',
        'profit_display', 'placed_at'
    ]
    list_filter = ['outcome', 'is_settled', 'is_won', 'sport', 'placed_at']
    search_fields = [
        'position_id', 'market_id', 'transaction_hash',
        'home_team', 'away_team'
    ]
    readonly_fields = [
        'position_id', 'market_id', 'transaction_hash',
        'expected_payout', 'profit', 'placed_at', 'settled_at'
    ]
    
    def session_link(self, obj):
        url = reverse('admin:ominari_trading_tradingsession_change', args=[obj.session.id])
        return format_html('<a href="{}">{}</a>', url, obj.session.session_id)
    session_link.short_description = 'Session'
    
    def match_display(self, obj):
        return f"{obj.home_team} vs {obj.away_team}"
    match_display.short_description = 'Match'
    
    def status_display(self, obj):
        if not obj.is_settled:
            time_to_match = obj.match_date - timezone.now()
            if time_to_match.total_seconds() > 0:
                hours = int(time_to_match.total_seconds() / 3600)
                return format_html(
                    '<span style="color: orange;">⏱ {}h</span>',
                    hours
                )
            return format_html(
                '<span style="color: blue;">▶ Live</span>'
            )
        elif obj.is_won:
            return format_html(
                '<span style="color: green;">✓ Won</span>'
            )
        else:
            return format_html(
                '<span style="color: red;">✗ Lost</span>'
            )
    status_display.short_description = 'Status'
    
    def profit_display(self, obj):
        profit = obj.profit
        if profit > 0:
            return format_html(
                '<span style="color: green;">+{:.4f}</span>',
                profit
            )
        elif profit < 0:
            return format_html(
                '<span style="color: red;">{:.4f}</span>',
                profit
            )
        return '-'
    profit_display.short_description = 'P/L'


@admin.register(MarketData)
class MarketDataAdmin(admin.ModelAdmin):
    list_display = [
        'market_id_short', 'match_display', 'sport', 'league',
        'odds_display', 'match_date', 'status_badge'
    ]
    list_filter = ['sport', 'league', 'is_resolved', 'chain', 'match_date']
    search_fields = ['market_id', 'home_team', 'away_team', 'league']
    readonly_fields = ['market_id', 'created_at', 'updated_at']
    
    def market_id_short(self, obj):
        return format_html(
            '<code title="{}">{}</code>',
            obj.market_id,
            f"{obj.market_id[:8]}..."
        )
    market_id_short.short_description = 'Market ID'
    
    def match_display(self, obj):
        return f"{obj.home_team} vs {obj.away_team}"
    match_display.short_description = 'Match'
    
    def odds_display(self, obj):
        odds_str = f"H:{obj.home_odds}"
        if obj.draw_odds:
            odds_str += f" D:{obj.draw_odds}"
        odds_str += f" A:{obj.away_odds}"
        return odds_str
    odds_display.short_description = 'Odds'
    
    def status_badge(self, obj):
        if obj.is_resolved:
            outcome_map = {0: 'Home', 1: 'Draw', 2: 'Away'}
            outcome = outcome_map.get(obj.winning_outcome, 'Unknown')
            return format_html(
                '<span style="color: gray;">✓ {}</span>',
                outcome
            )
        
        time_to_match = obj.match_date - timezone.now()
        if time_to_match.total_seconds() > 0:
            hours = int(time_to_match.total_seconds() / 3600)
            if hours > 24:
                days = hours // 24
                return format_html(
                    '<span style="color: blue;">{} days</span>',
                    days
                )
            return format_html(
                '<span style="color: orange;">{} hours</span>',
                hours
            )
        return format_html(
            '<span style="color: green;">● Live</span>'
        )
    status_badge.short_description = 'Status'


@admin.register(BlockchainEvent)
class BlockchainEventAdmin(admin.ModelAdmin):
    list_display = [
        'event_type', 'chain', 'block_number', 'tx_hash_short',
        'processing_status', 'block_timestamp'
    ]
    list_filter = ['event_type', 'chain', 'is_processed', 'block_timestamp']
    search_fields = ['transaction_hash', 'event_data']
    readonly_fields = [
        'transaction_hash', 'block_number', 'log_index',
        'event_data', 'error', 'block_timestamp', 'created_at'
    ]
    
    def tx_hash_short(self, obj):
        return format_html(
            '<code title="{}">{}</code>',
            obj.transaction_hash,
            f"{obj.transaction_hash[:10]}..."
        )
    tx_hash_short.short_description = 'TX Hash'
    
    def processing_status(self, obj):
        if obj.is_processed:
            return format_html(
                '<span style="color: green;">✓ Processed</span>'
            )
        elif obj.error:
            return format_html(
                '<span style="color: red;" title="{}">✗ Error</span>',
                obj.error[:100]
            )
        return format_html(
            '<span style="color: orange;">⏳ Pending</span>'
        )
    processing_status.short_description = 'Status'