"""add position_names to market

Revision ID: 44472f3500d0
Revises: bf44b253f239
Create Date: 2025-07-26 01:06:30.787569

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '44472f3500d0'
down_revision: Union[str, None] = 'bf44b253f239'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Simply tack on the JSON/text column. Existing rows get NULL.
    op.add_column(
        "market",
        sa.Column(
            "position_names",
            sa.Text(),       # or sa.String() if you prefer
            nullable=True
        )
    )


def downgrade() -> None:
    # Drop it on rollback.
    op.drop_column("market", "position_names")

# ### end Alembic commands ###
