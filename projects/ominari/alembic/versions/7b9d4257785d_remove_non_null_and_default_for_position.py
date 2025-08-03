"""remove non-null and default for position

Revision ID: 7b9d4257785d
Revises: 44472f3500d0
Create Date: 2025-07-26 05:18:41.794743

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7b9d4257785d"
down_revision: Union[str, None] = "44472f3500d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Batch alter to drop NOT NULL and DEFAULT
    with op.batch_alter_table("odd", schema=None) as batch_op:
        batch_op.alter_column(
            "position",
            existing_type=sa.Integer(),
            nullable=True,
            server_default=None,
        )


def downgrade():
    # Re-apply the NOT NULL + default if you ever roll back
    with op.batch_alter_table("odd", schema=None) as batch_op:
        batch_op.alter_column(
            "position",
            existing_type=sa.Integer(),
            nullable=False,
            server_default="0",
        )
