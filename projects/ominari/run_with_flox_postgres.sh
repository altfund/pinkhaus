#!/bin/bash
# Run Ominari with PostgreSQL via flox

set -e

echo "🚀 Starting Ominari with PostgreSQL via Flox"
echo "============================================"

# Activate flox environment
echo "Activating flox environment..."
eval "$(flox activate)"

# Set up PostgreSQL directories
export PGDATA="$PWD/.flox/postgres/data"
export PGHOST="localhost"  
export PGPORT="5435"
export PGUSER="postgres"

mkdir -p "$PGDATA"
mkdir -p "$PWD/.flox/postgres/log"

# Initialize PostgreSQL if needed
if [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "Initializing PostgreSQL database..."
    initdb -D "$PGDATA" -U postgres --locale=C --encoding=UTF8
    
    # Configure PostgreSQL for port 5435
    echo "port = 5435" >> "$PGDATA/postgresql.conf"
    echo "listen_addresses = 'localhost'" >> "$PGDATA/postgresql.conf"
    echo "shared_buffers = 256MB" >> "$PGDATA/postgresql.conf"
    
    # Set up authentication
    echo "local   all             postgres                                trust" > "$PGDATA/pg_hba.conf"
    echo "host    all             all             127.0.0.1/32            md5" >> "$PGDATA/pg_hba.conf"
fi

# Start PostgreSQL
if ! pg_ctl -D "$PGDATA" status > /dev/null 2>&1; then
    echo "Starting PostgreSQL on port 5435..."
    pg_ctl -D "$PGDATA" -l "$PWD/.flox/postgres/log/postgres.log" start
    sleep 5
    
    # Create Ominari database and user
    echo "Setting up Ominari database..."
    createdb -U postgres ominari_production 2>/dev/null || true
    psql -U postgres -d ominari_production -c "CREATE USER ominari_user WITH PASSWORD 'ominari_2025_secure';" 2>/dev/null || true
    psql -U postgres -d ominari_production -c "GRANT ALL PRIVILEGES ON DATABASE ominari_production TO ominari_user;" 2>/dev/null || true
    psql -U postgres -d ominari_production -c "GRANT ALL ON SCHEMA public TO ominari_user;" 2>/dev/null || true
fi

# Export environment for Python
export PG_HOST=localhost
export PG_PORT=5435
export PG_USER=ominari_user
export PG_PASSWORD=ominari_2025_secure
export PG_DB=ominari_production

echo ""
echo "✅ PostgreSQL is running on port 5435"
echo ""

# Create schema and migrate data
echo "Setting up database schema..."
uv run python -c "
import os
os.environ['PG_PORT'] = '5435'
from sqlalchemy import create_engine
from models import Base

engine = create_engine('postgresql://ominari_user:ominari_2025_secure@localhost:5435/ominari_production')
Base.metadata.create_all(engine)
print('✅ Schema created')
"

# Migrate blockchain data
echo "Migrating blockchain data..."
uv run python -c "
import os
os.environ['PG_PORT'] = '5435'
from sqlalchemy import create_engine, text

pg_engine = create_engine('postgresql://ominari_user:ominari_2025_secure@localhost:5435/ominari_production')
sqlite_engine = create_engine('sqlite:///sport_odds.db')

# Quick migration of some blockchain data
with sqlite_engine.connect() as src:
    markets = src.execute(text('SELECT * FROM market WHERE source LIKE \"blockchain_%\" LIMIT 100')).fetchall()
    print(f'Found {len(markets)} blockchain markets to migrate')
    
    with pg_engine.connect() as dst:
        for m in markets:
            try:
                dst.execute(text('''
                    INSERT INTO market (source_id, source, sport, league_name, 
                                      home_team, away_team, market_type, 
                                      maturity_date, is_finished, updated_at)
                    VALUES (:source_id, :source, :sport, :league_name,
                            :home_team, :away_team, :market_type,
                            :maturity_date, :is_finished, :updated_at)
                    ON CONFLICT (source_id) DO NOTHING
                '''), dict(m._mapping))
            except:
                pass
        dst.commit()
    print('✅ Markets migrated')
"

echo ""
echo "🌐 Starting Web Monitor..."
echo "Dashboard will be available at: http://localhost:8888"
echo "Unified view at: http://localhost:8888/unified"
echo ""

# Start web monitor with PostgreSQL on port 5435
uv run python web_monitor.py