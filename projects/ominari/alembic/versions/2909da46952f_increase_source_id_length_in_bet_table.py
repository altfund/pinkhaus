"""increase_source_id_length_in_bet_table

Revision ID: 2909da46952f
Revises: 0a9c04660fbb
Create Date: 2025-11-14 21:37:09.415218

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2909da46952f'
down_revision: Union[str, None] = '0a9c04660fbb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Increase source_id column length from VARCHAR(66) to VARCHAR(100)
    op.alter_column('bet', 'source_id',
                    existing_type=sa.String(66),
                    type_=sa.String(100),
                    existing_nullable=False)


def downgrade() -> None:
    # Revert source_id column length back to VARCHAR(66)
    op.alter_column('bet', 'source_id',
                    existing_type=sa.String(100),
                    type_=sa.String(66),
                    existing_nullable=False)
