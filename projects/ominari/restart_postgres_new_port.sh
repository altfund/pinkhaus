#!/bin/bash
# Restart PostgreSQL on a different port (5999)

set -e

echo "🐘 Restarting PostgreSQL on port 5999..."

# Activate flox
eval "$(flox activate)"

# Set up new directories and config
export PGDATA="$PWD/.flox/postgres/data"
export PGHOST="$PWD/.flox/postgres/socket"
export PGPORT="5999"

# Stop any existing instance
pg_ctl -D "$PGDATA" stop -m fast 2>/dev/null || true
sleep 2

# Update PostgreSQL config to use port 5999
if [ -f "$PGDATA/postgresql.conf" ]; then
    sed -i "s/port = 5435/port = 5999/g" "$PGDATA/postgresql.conf"
    sed -i "s/port = 5432/port = 5999/g" "$PGDATA/postgresql.conf"
fi

# Start PostgreSQL on new port
echo "Starting PostgreSQL on port 5999..."
pg_ctl -D "$PGDATA" -l "$PWD/.flox/postgres/log/postgres.log" start

# Wait for startup
sleep 5

# Test connection
export PGPASSWORD=ominari_2025_secure
if psql -U ominari_user -h localhost -p 5999 -d ominari_production -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✅ PostgreSQL is running on port 5999"
    
    # Update environment file
    cat > .env.postgres <<EOF
export PG_HOST=localhost
export PG_PORT=5999
export PG_USER=ominari_user
export PG_PASSWORD=ominari_2025_secure
export PG_DB=ominari_production
EOF

    echo "✅ Environment updated for port 5999"
    echo "Source with: source .env.postgres"
    
else
    echo "❌ Failed to connect to PostgreSQL on port 5999"
    exit 1
fi