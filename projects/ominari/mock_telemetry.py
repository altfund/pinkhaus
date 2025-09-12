"""Mock telemetry module for testing."""

def get_telemetry():
    """Return a mock telemetry object."""
    class MockTelemetry:
        def record_bet_placed(self, amount, market, strategy):
            pass
    
    return MockTelemetry()

def traced(name):
    """Mock trace decorator."""
    def decorator(func):
        return func
    return decorator

def trace_api_call(func):
    """Mock API call trace decorator."""
    return func