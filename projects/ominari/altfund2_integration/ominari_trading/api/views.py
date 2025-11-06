from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.db import transaction
from django.utils import timezone
from decimal import Decimal
import logging

from ..models import (
    ChainNetwork, Web3Account, TradingSession, Position,
    OptimizationRun, MarketData
)
from ..blockchain import OminariBlockchainClient, MockBlockchainClient
from ..tasks import sync_blockchain_events, update_session_stats
from .serializers import (
    ChainNetworkSerializer, Web3AccountSerializer, TradingSessionSerializer,
    PositionSerializer, CreateSessionSerializer, PlaceBetSerializer,
    OptimizePortfolioSerializer, OptimizationResultSerializer,
    MarketDataSerializer, BlockchainTransactionSerializer,
    WalletVerificationSerializer
)

logger = logging.getLogger(__name__)


class ChainNetworkViewSet(viewsets.ReadOnlyModelViewSet):
    """View available blockchain networks"""
    queryset = ChainNetwork.objects.filter(is_active=True)
    serializer_class = ChainNetworkSerializer
    permission_classes = [permissions.AllowAny]


class Web3AccountViewSet(viewsets.ModelViewSet):
    """Manage Web3 wallet connections"""
    serializer_class = Web3AccountSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return Web3Account.objects.filter(user=self.request.user)
    
    @action(detail=False, methods=['post'])
    def verify_ownership(self, request):
        """Verify wallet ownership via signature"""
        serializer = WalletVerificationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        chain = get_object_or_404(
            ChainNetwork,
            chain_id=serializer.validated_data['chain_id']
        )
        
        # Initialize blockchain client
        client = OminariBlockchainClient(chain)
        
        # Verify signature
        is_valid = client.verify_wallet_ownership(
            serializer.validated_data['wallet_address'],
            serializer.validated_data['signature'],
            serializer.validated_data['message']
        )
        
        if is_valid:
            # Create or update Web3Account
            account, created = Web3Account.objects.update_or_create(
                wallet_address=serializer.validated_data['wallet_address'].lower(),
                chain=chain,
                defaults={
                    'user': request.user,
                    'is_verified': True,
                    'verification_signature': serializer.validated_data['signature'],
                    'verified_at': timezone.now()
                }
            )
            
            return Response({
                'verified': True,
                'account': Web3AccountSerializer(account).data
            })
        else:
            return Response(
                {'verified': False, 'error': 'Invalid signature'},
                status=status.HTTP_400_BAD_REQUEST
            )


class TradingSessionViewSet(viewsets.ModelViewSet):
    """Manage trading sessions"""
    serializer_class = TradingSessionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return TradingSession.objects.filter(user=self.request.user)
    
    def create(self, request):
        """Create a new trading session"""
        serializer = CreateSessionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Get chain and account
        chain = get_object_or_404(
            ChainNetwork,
            chain_id=serializer.validated_data['chain_id']
        )
        
        # For now, create a mock session
        # In production, this would wait for blockchain confirmation
        with transaction.atomic():
            # Get or create Web3 account
            # In production, would require wallet connection
            web3_account = Web3Account.objects.filter(
                user=request.user,
                chain=chain
            ).first()
            
            if not web3_account:
                return Response(
                    {'error': 'Please connect your wallet first'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Create session
            session = TradingSession.objects.create(
                user=request.user,
                web3_account=web3_account,
                chain=chain,
                session_id=f"{chain.chain_id}-PENDING",
                transaction_hash=serializer.validated_data.get('transaction_hash', '0x0'),
                initial_bankroll=serializer.validated_data['initial_bankroll'],
                current_bankroll=serializer.validated_data['initial_bankroll']
            )
            
            # Schedule blockchain sync
            sync_blockchain_events.delay(chain.chain_id)
            
            return Response(
                TradingSessionSerializer(session).data,
                status=status.HTTP_201_CREATED
            )
    
    @action(detail=True, methods=['post'])
    def refresh(self, request, pk=None):
        """Refresh session data from blockchain"""
        session = self.get_object()
        update_session_stats.delay(session.id)
        return Response({'status': 'refresh scheduled'})
    
    @action(detail=True, methods=['post'])
    def close(self, request, pk=None):
        """Close a trading session"""
        session = self.get_object()
        
        if not session.is_active:
            return Response(
                {'error': 'Session already closed'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check for unsettled positions
        unsettled = session.positions.filter(is_settled=False).count()
        if unsettled > 0:
            return Response(
                {'error': f'{unsettled} unsettled positions exist'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Close session
        session.is_active = False
        session.closed_at = timezone.now()
        session.save()
        
        return Response(TradingSessionSerializer(session).data)


class PositionViewSet(viewsets.ModelViewSet):
    """Manage trading positions"""
    serializer_class = PositionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        queryset = Position.objects.filter(
            session__user=self.request.user
        )
        
        # Filter by session if provided
        session_id = self.request.query_params.get('session')
        if session_id:
            queryset = queryset.filter(session_id=session_id)
        
        # Filter by status
        status_filter = self.request.query_params.get('status')
        if status_filter == 'active':
            queryset = queryset.filter(is_settled=False)
        elif status_filter == 'settled':
            queryset = queryset.filter(is_settled=True)
        
        return queryset.select_related('session')
    
    def create(self, request):
        """Place a new bet"""
        serializer = PlaceBetSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        
        session = get_object_or_404(
            TradingSession,
            id=serializer.validated_data['session_id'],
            user=request.user
        )
        
        # Create position
        position = Position.objects.create(
            session=session,
            position_id=f"{session.session_id}-PENDING",
            market_id=serializer.validated_data['market_id'],
            transaction_hash=serializer.validated_data.get('transaction_hash', '0x0'),
            stake=serializer.validated_data['stake'],
            outcome=serializer.validated_data['outcome'],
            # In production, would get odds and market data from blockchain
            odds=Decimal('2.00'),
            home_team='Team A',
            away_team='Team B',
            sport='Soccer',
            match_date=timezone.now() + timezone.timedelta(hours=2)
        )
        
        # Update session
        session.current_bankroll -= position.stake
        session.total_bets_placed += 1
        session.last_activity = timezone.now()
        session.save()
        
        return Response(
            PositionSerializer(position).data,
            status=status.HTTP_201_CREATED
        )


class MarketDataViewSet(viewsets.ReadOnlyModelViewSet):
    """Browse available markets"""
    queryset = MarketData.objects.filter(
        is_resolved=False,
        match_date__gt=timezone.now()
    )
    serializer_class = MarketDataSerializer
    permission_classes = [permissions.AllowAny]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # Filter by sport
        sport = self.request.query_params.get('sport')
        if sport:
            queryset = queryset.filter(sport__iexact=sport)
        
        # Filter by chain
        chain_id = self.request.query_params.get('chain')
        if chain_id:
            queryset = queryset.filter(chain__chain_id=chain_id)
        
        # Filter by time range
        hours = self.request.query_params.get('hours')
        if hours:
            cutoff = timezone.now() + timezone.timedelta(hours=int(hours))
            queryset = queryset.filter(match_date__lte=cutoff)
        
        return queryset.order_by('match_date')


class OptimizationViewSet(viewsets.ViewSet):
    """Portfolio optimization endpoints"""
    permission_classes = [permissions.IsAuthenticated]
    
    @action(detail=False, methods=['post'])
    def optimize(self, request):
        """Run Kelly optimization for a portfolio"""
        serializer = OptimizePortfolioSerializer(
            data=request.data,
            context={'request': request}
        )
        serializer.is_valid(raise_exception=True)
        
        session = get_object_or_404(
            TradingSession,
            id=serializer.validated_data['session_id'],
            user=request.user
        )
        
        # Get market data
        markets = MarketData.objects.filter(
            market_id__in=serializer.validated_data['market_ids'],
            is_resolved=False
        )
        
        # Run optimization (simplified for demo)
        results = []
        total_stake = Decimal('0')
        
        for market in markets:
            # Simple Kelly calculation
            # In production, would use smart contract
            odds = [market.home_odds, market.draw_odds, market.away_odds]
            probabilities = [1/float(o) for o in odds]
            
            # Find best value
            edges = [(p * float(o) - 1) for p, o in zip(probabilities, odds)]
            best_idx = edges.index(max(edges))
            
            if edges[best_idx] > 0:
                # Positive edge found
                kelly_fraction = min(
                    edges[best_idx] / (float(odds[best_idx]) - 1),
                    serializer.validated_data['max_stake_percentage'] / 100
                )
                
                if serializer.validated_data['use_half_kelly']:
                    kelly_fraction /= 2
                
                stake = session.current_bankroll * Decimal(str(kelly_fraction))
                
                results.append({
                    'market_id': market.market_id,
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'recommended_outcome': best_idx,
                    'recommended_stake': stake,
                    'odds': odds[best_idx],
                    'edge': Decimal(str(edges[best_idx])),
                    'kelly_fraction': Decimal(str(kelly_fraction))
                })
                
                total_stake += stake
        
        # Create optimization record
        OptimizationRun.objects.create(
            session=session,
            chunk_duration_minutes=serializer.validated_data['chunk_duration_minutes'],
            markets_analyzed=len(markets),
            positions_recommended=len(results),
            total_stake_allocated=total_stake
        )
        
        return Response({
            'session_id': session.id,
            'current_bankroll': session.current_bankroll,
            'recommendations': OptimizationResultSerializer(results, many=True).data,
            'total_stake_recommended': total_stake,
            'positions_recommended': len(results)
        })
    
    @action(detail=False, methods=['post'])
    def prepare_transaction(self, request):
        """Prepare a blockchain transaction for signing"""
        # This would build the transaction for the frontend to sign
        # Example implementation
        chain_id = request.data.get('chain_id')
        function_name = request.data.get('function')
        params = request.data.get('params', {})
        
        chain = get_object_or_404(ChainNetwork, chain_id=chain_id)
        client = OminariBlockchainClient(chain)
        
        # Build transaction
        # In production, would properly build based on function
        transaction_data = {
            'function_name': function_name,
            'from_address': request.data.get('from_address'),
            'to_address': chain.trading_engine_address,
            'value': '0',
            'gas': 200000,
            'gas_price': '20000000000',  # 20 gwei
            'nonce': 0,
            'chain_id': chain_id,
            'data': '0x0'
        }
        
        return Response(
            BlockchainTransactionSerializer(transaction_data).data
        )