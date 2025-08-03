"""
alembic/versions/f829409364fc_initial_schema.py
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'f829409364fc'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # === 1) If 'market' does not exist, create it with all columns ===
    # This ensures fresh databases get full schema in one step
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if 'market' not in inspector.get_table_names():
        op.create_table(
            'market',
            sa.Column('source_id', sa.String(), primary_key=True, nullable=False),
            sa.Column('sport', sa.String(), nullable=False),
            sa.Column('league_name', sa.String(), nullable=True),
            sa.Column('market_type', sa.String(), nullable=True),
            sa.Column('home_team', sa.String(), nullable=True),
            sa.Column('away_team', sa.String(), nullable=True),
            sa.Column('home_score', sa.Integer(), nullable=True),
            sa.Column('away_score', sa.Integer(), nullable=True),
            sa.Column('home_score_by_period', sa.Text(), nullable=True),
            sa.Column('away_score_by_period', sa.Text(), nullable=True),
            sa.Column('game_status', sa.String(), nullable=True),
            sa.Column('is_finished', sa.Boolean(), nullable=True),
            sa.Column('tournament', sa.String(), nullable=True),
            sa.Column('tournament_round', sa.String(), nullable=True),
            sa.Column('start_time', sa.DateTime(), nullable=True),
            sa.Column('last_update', sa.DateTime(), nullable=True),
            sa.Column('maturity_date', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
        )
    # === 2) Add missing columns for existing DBs ===
    # SQLite supports simple ADD COLUMN for new nullable columns
    existing_cols = inspector.get_columns('market')
    existing_names = {c['name'] for c in existing_cols}

    new_cols = [
        ('home_score', sa.Integer(), True),
        ('away_score', sa.Integer(), True),
        ('home_score_by_period', sa.Text(), True),
        ('away_score_by_period', sa.Text(), True),
        ('game_status', sa.String(), True),
        ('is_finished', sa.Boolean(), True),
        ('tournament', sa.String(), True),
        ('tournament_round', sa.String(), True),
        ('start_time', sa.DateTime(), True),
        ('last_update', sa.DateTime(), True),
    ]
    for name, col_type, nullable in new_cols:
        if name not in existing_names:
            op.add_column('market', sa.Column(name, col_type, nullable=nullable))


def downgrade():
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if 'market' in inspector.get_table_names():
        # drop columns if they exist
        existing = {c['name'] for c in inspector.get_columns('market')}
        for name in [
            'last_update', 'start_time', 'tournament_round', 'tournament',
            'is_finished', 'game_status', 'away_score_by_period',
            'home_score_by_period', 'away_score', 'home_score'
        ]:
            if name in existing:
                op.drop_column('market', name)
    # For fresh DBs, drop the table fully
    if 'market' in inspector.get_table_names():
        op.drop_table('market')

