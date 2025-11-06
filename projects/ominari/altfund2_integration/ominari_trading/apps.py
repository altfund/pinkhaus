from django.apps import AppConfig


class OminariTradingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ominari_trading'
    verbose_name = 'Ominari Trading DApp'
    
    def ready(self):
        # Import signal handlers
        import ominari_trading.signals