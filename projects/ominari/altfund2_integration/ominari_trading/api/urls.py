from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'chains', views.ChainNetworkViewSet, basename='chain-network')
router.register(r'wallets', views.Web3AccountViewSet, basename='web3-account')
router.register(r'sessions', views.TradingSessionViewSet, basename='trading-session')
router.register(r'positions', views.PositionViewSet, basename='position')
router.register(r'markets', views.MarketDataViewSet, basename='market-data')
router.register(r'optimize', views.OptimizationViewSet, basename='optimization')

app_name = 'ominari_trading'

urlpatterns = [
    path('', include(router.urls)),
]