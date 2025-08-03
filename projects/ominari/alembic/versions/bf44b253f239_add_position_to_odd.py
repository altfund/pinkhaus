"""add position to odd

Revision ID: bf44b253f239
Revises: f829409364fc
Create Date: 2025-07-26 00:16:14.476889

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "bf44b253f239"
down_revision: Union[str, None] = "f829409364fc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # 1) Quick single‐statement add (fills existing rows with 0)
    op.execute("ALTER TABLE odd ADD COLUMN position INTEGER NOT NULL DEFAULT 0")

    # 2) Enforce uniqueness via an index (SQLite lets you create unique indexes in-place)
    op.create_index(
        "uq_odd_source_position", "odd", ["source_id", "position"], unique=True
    )

    # ### end Alembic commands ###


def downgrade():
    op.drop_index("uq_odd_source_position", table_name="odd")
    # Note: SQLite can’t DROP columns; you’d normally leave `position` there.

    # ### end Alembic commands ###
