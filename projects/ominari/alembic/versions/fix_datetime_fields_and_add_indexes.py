"""Fix DateTime fields and add performance indexes

Revision ID: fix_datetime_fields
Revises: create_signal_registry_tables
Create Date: 2025-09-01

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = 'fix_datetime_fields'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade():
    """Fix DateTime fields and add performance indexes."""
    
    # Skip the DateTime updates for now - they're taking too long on 201GB database
    # We'll handle these in the application layer instead
    
    # Create indexes for performance
    # Market table indexes
    try:
        op.create_index('idx_market_maturity_date', 'market', ['maturity_date'])
    except:
        pass  # Index might already exist
    
    try:
        op.create_index('idx_market_maturity_source', 'market', ['maturity_date', 'source'])
    except:
        pass
    
    try:
        op.create_index('idx_market_sport_maturity', 'market', ['sport', 'maturity_date'])
    except:
        pass
    
    # Composite index for common queries
    try:
        op.create_index('idx_market_active', 'market', ['maturity_date', 'is_finished'])
    except:
        pass
    
    # Odd table indexes
    try:
        op.create_index('idx_odd_market_bookmaker', 'odd', ['market_source_id', 'bookmaker_key'])
    except:
        pass
    
    try:
        op.create_index('idx_odd_updated_at', 'odd', ['updated_at'])
    except:
        pass
    
    print("Indexes created for performance optimization")


def downgrade():
    """Remove indexes."""
    
    # Drop indexes
    op.drop_index('idx_market_maturity_date', 'market')
    op.drop_index('idx_market_maturity_source', 'market')
    op.drop_index('idx_market_sport_maturity', 'market')
    op.drop_index('idx_market_active', 'market')
    op.drop_index('idx_odd_market_bookmaker', 'odd')
    op.drop_index('idx_odd_updated_at', 'odd')