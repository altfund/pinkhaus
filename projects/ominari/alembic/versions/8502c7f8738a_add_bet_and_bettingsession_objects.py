"""add bet and bettingsession objects

Revision ID: 8502c7f8738a
Revises: 2dab8300ab59
Create Date: 2025-08-01 00:58:30.362206

"""

from typing import Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine import reflection


# revision identifiers, used by Alembic.
revision: str = "8502c7f8738a"
down_revision: Union[str, None] = "2dab8300ab59"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    insp = reflection.Inspector.from_engine(bind)
    existing = insp.get_table_names()

    # 1) Create the ENUM only if it doesn’t already exist
    session_enum = sa.Enum(
        "backtest", "paper", "live", "simulation", name="session_type_enum"
    )
    session_enum.create(bind, checkfirst=True)

    # 2) Create betting_session iff missing
    if "betting_session" not in existing:
        op.create_table(
            "betting_session",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("as_of", sa.DateTime(), nullable=False, index=True),
            sa.Column(
                "created_at", sa.DateTime(), server_default=sa.func.current_timestamp()
            ),
            sa.Column("session_type", session_enum, nullable=False, index=True),
            sa.Column("strategy_name", sa.String(), nullable=True, index=True),
            # … if you want to add more columns later, do it here or via a new migration …
        )

    # 3) Create bet iff missing
    if "bet" not in existing:
        op.create_table(
            "bet",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "session_id",
                sa.Integer(),
                sa.ForeignKey("betting_session.id"),
                nullable=False,
                index=True,
            ),
            sa.Column(
                "source_id",
                sa.String(),
                sa.ForeignKey("market.source_id"),
                nullable=False,
            ),
            sa.Column("unified_market_type", sa.String(), nullable=False),
            sa.Column("normalized_outcome", sa.String(), nullable=False),
            sa.Column("normalized_line", sa.Float(), nullable=False),
            sa.Column("bet_name", sa.String(), nullable=True),
            sa.Column("probability", sa.Float(), nullable=True),
            sa.Column("odds", sa.Float(), nullable=True),
            sa.Column("stake", sa.Float(), nullable=True),
            sa.Column("execution_stake", sa.Float(), nullable=True),
            sa.Column("fee_amount", sa.Float(), nullable=True),
            sa.Column("fee_pct", sa.Float(), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(), server_default=sa.func.current_timestamp()
            ),
            sa.UniqueConstraint(
                "session_id",
                "source_id",
                "unified_market_type",
                "normalized_outcome",
                "normalized_line",
                name="uq_bet_unique",
            ),
            sa.Index("idx_bet_on_market", "source_id", "unified_market_type"),
        )


def downgrade():
    # Drop tables if they exist
    bind = op.get_bind()
    insp = reflection.Inspector.from_engine(bind)
    existing = insp.get_table_names()

    if "bet" in existing:
        op.drop_table("bet")
    if "betting_session" in existing:
        op.drop_table("betting_session")

    # Drop the enum
    session_enum = sa.Enum(name="session_type_enum")
    session_enum.drop(bind, checkfirst=True)
    # ### end Alembic commands ###
