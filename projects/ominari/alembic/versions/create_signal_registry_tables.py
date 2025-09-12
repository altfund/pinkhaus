"""Create signal registry and alpha research tables

Revision ID: a1b2c3d4e5f6
Revises: 8502c7f8738a
Create Date: 2025-01-31 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '8502c7f8738a'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Signal registry tables
    op.create_table('signal_metadata',
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('version', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('author', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('parameters', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('tags', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('name')
    )
    
    op.create_table('signal_performance',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('signal_name', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('period', sa.Text(), nullable=True),
        sa.Column('n_predictions', sa.Integer(), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('information_ratio', sa.Float(), nullable=True),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('correlations', sa.Text(), nullable=True),
        sa.Column('regime', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['signal_name'], ['signal_metadata.name'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('signal_weights',
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('weights', sa.Text(), nullable=True),
        sa.Column('method', sa.Text(), nullable=True),
        sa.Column('performance_window', sa.Integer(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('timestamp')
    )
    
    # Alpha research tables
    op.create_table('alpha_signals',
        sa.Column('signal_id', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('formula', sa.Text(), nullable=True),
        sa.Column('parameters', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('current_stage', sa.Text(), nullable=True),
        sa.Column('stage_history', sa.Text(), nullable=True),
        sa.Column('performance_metrics', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='1', nullable=True),
        sa.PrimaryKeyConstraint('signal_id')
    )
    
    op.create_table('backtest_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('signal_id', sa.Text(), nullable=False),
        sa.Column('stage', sa.Text(), nullable=False),
        sa.Column('start_date', sa.DateTime(), nullable=True),
        sa.Column('end_date', sa.DateTime(), nullable=True),
        sa.Column('n_bets', sa.Integer(), nullable=True),
        sa.Column('total_return', sa.Float(), nullable=True),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('win_rate', sa.Float(), nullable=True),
        sa.Column('avg_edge', sa.Float(), nullable=True),
        sa.Column('information_ratio', sa.Float(), nullable=True),
        sa.Column('t_statistic', sa.Float(), nullable=True),
        sa.Column('p_value', sa.Float(), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['signal_id'], ['alpha_signals.signal_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('stage_transitions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('signal_id', sa.Text(), nullable=False),
        sa.Column('from_stage', sa.Text(), nullable=True),
        sa.Column('to_stage', sa.Text(), nullable=True),
        sa.Column('transition_date', sa.DateTime(), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('metrics', sa.Text(), nullable=True),
        sa.Column('approved_by', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['signal_id'], ['alpha_signals.signal_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Paper trading tables
    op.create_table('paper_orders',
        sa.Column('order_id', sa.Text(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('source_id', sa.Text(), nullable=False),
        sa.Column('market_type', sa.Text(), nullable=True),
        sa.Column('bet_name', sa.Text(), nullable=True),
        sa.Column('side', sa.Text(), nullable=True),
        sa.Column('size', sa.Float(), nullable=True),
        sa.Column('limit_price', sa.Float(), nullable=True),
        sa.Column('signal_name', sa.Text(), nullable=True),
        sa.Column('expected_edge', sa.Float(), nullable=True),
        sa.Column('status', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('order_id')
    )
    
    op.create_table('paper_fills',
        sa.Column('fill_id', sa.Text(), nullable=False),
        sa.Column('order_id', sa.Text(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('fill_price', sa.Float(), nullable=True),
        sa.Column('fill_size', sa.Float(), nullable=True),
        sa.Column('slippage', sa.Float(), nullable=True),
        sa.Column('commission', sa.Float(), nullable=True),
        sa.Column('market_impact', sa.Float(), nullable=True),
        sa.Column('pnl', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.ForeignKeyConstraint(['order_id'], ['paper_orders.order_id'], ),
        sa.PrimaryKeyConstraint('fill_id')
    )
    
    op.create_table('paper_performance',
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('capital', sa.Float(), nullable=True),
        sa.Column('positions_value', sa.Float(), nullable=True),
        sa.Column('total_value', sa.Float(), nullable=True),
        sa.Column('daily_pnl', sa.Float(), nullable=True),
        sa.Column('total_pnl', sa.Float(), nullable=True),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('win_rate', sa.Float(), nullable=True),
        sa.Column('avg_win', sa.Float(), nullable=True),
        sa.Column('avg_loss', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('timestamp')
    )
    
    # Blockchain data tables
    op.create_table('blockchain_markets',
        sa.Column('market_address', sa.Text(), nullable=False),
        sa.Column('game_id', sa.Text(), nullable=False),
        sa.Column('game_label', sa.Text(), nullable=True),
        sa.Column('maturity_date', sa.Integer(), nullable=True),
        sa.Column('tags', sa.Text(), nullable=True),
        sa.Column('normalized_odds', sa.Text(), nullable=True),
        sa.Column('creation_block', sa.Integer(), nullable=True),
        sa.Column('creation_tx', sa.Text(), nullable=True),
        sa.Column('network', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('market_address')
    )
    
    op.create_table('blockchain_trades',
        sa.Column('tx_hash', sa.Text(), nullable=False),
        sa.Column('block_number', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.Integer(), nullable=True),
        sa.Column('buyer', sa.Text(), nullable=True),
        sa.Column('market_address', sa.Text(), nullable=True),
        sa.Column('position', sa.Integer(), nullable=True),
        sa.Column('amount', sa.Float(), nullable=True),
        sa.Column('susd_paid', sa.Float(), nullable=True),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('gas_used', sa.Integer(), nullable=True),
        sa.Column('gas_price', sa.Float(), nullable=True),
        sa.Column('network', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('tx_hash')
    )
    
    op.create_table('blockchain_odds',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('market_address', sa.Text(), nullable=False),
        sa.Column('timestamp', sa.Integer(), nullable=True),
        sa.Column('block_number', sa.Integer(), nullable=True),
        sa.Column('position', sa.Integer(), nullable=True),
        sa.Column('buy_odds', sa.Float(), nullable=True),
        sa.Column('sell_odds', sa.Float(), nullable=True),
        sa.Column('liquidity', sa.Float(), nullable=True),
        sa.Column('network', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Quote collection table
    op.create_table('quotes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('source_id', sa.Text(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('bid_price', sa.Float(), nullable=True),
        sa.Column('bid_size', sa.Float(), nullable=True),
        sa.Column('ask_price', sa.Float(), nullable=True),
        sa.Column('ask_size', sa.Float(), nullable=True),
        sa.Column('mid_price', sa.Float(), nullable=True),
        sa.Column('spread', sa.Float(), nullable=True),
        sa.Column('liquidity_score', sa.Float(), nullable=True),
        sa.Column('raw_data', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_quotes_source_time', 'quotes', ['source_id', 'timestamp'], unique=False)
    op.create_index('idx_signal_performance_signal_time', 'signal_performance', ['signal_name', 'timestamp'], unique=False)
    op.create_index('idx_backtest_results_signal', 'backtest_results', ['signal_id'], unique=False)
    op.create_index('idx_paper_orders_timestamp', 'paper_orders', ['timestamp'], unique=False)
    op.create_index('idx_blockchain_trades_market', 'blockchain_trades', ['market_address'], unique=False)
    op.create_index('idx_blockchain_odds_market', 'blockchain_odds', ['market_address', 'timestamp'], unique=False)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_blockchain_odds_market', table_name='blockchain_odds')
    op.drop_index('idx_blockchain_trades_market', table_name='blockchain_trades')
    op.drop_index('idx_paper_orders_timestamp', table_name='paper_orders')
    op.drop_index('idx_backtest_results_signal', table_name='backtest_results')
    op.drop_index('idx_signal_performance_signal_time', table_name='signal_performance')
    op.drop_index('idx_quotes_source_time', table_name='quotes')
    
    # Drop tables
    op.drop_table('quotes')
    op.drop_table('blockchain_odds')
    op.drop_table('blockchain_trades')
    op.drop_table('blockchain_markets')
    op.drop_table('paper_performance')
    op.drop_table('paper_fills')
    op.drop_table('paper_orders')
    op.drop_table('stage_transitions')
    op.drop_table('backtest_results')
    op.drop_table('alpha_signals')
    op.drop_table('signal_weights')
    op.drop_table('signal_performance')
    op.drop_table('signal_metadata')