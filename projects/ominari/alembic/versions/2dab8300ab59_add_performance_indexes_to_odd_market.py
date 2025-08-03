"""add performance indexes to odd & market

Revision ID: 2dab8300ab59
Revises: 7b9d4257785d
Create Date: 2025-07-31 13:15:04.803358

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2dab8300ab59"
down_revision: Union[str, None] = "7b9d4257785d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create only the performance indexes
    op.create_index(
        "idx_market_performance",
        "market",
        ["source", "sport", "market_type", "maturity_date"],
        unique=False,
    )
    op.create_index(
        "idx_odd_performance",
        "odd",
        ["bookmaker", "market_type", "source_id", "updated_at"],
        unique=False,
    )


def downgrade() -> None:
    # Drop them on rollback
    op.drop_index("idx_odd_performance", table_name="odd")
    op.drop_index("idx_market_performance", table_name="market")

    # ### end Alembic commands ###
