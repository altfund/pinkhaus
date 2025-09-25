#!/bin/bash
# Setup PostgreSQL in flox environment

set -e

echo "🐘 Setting up PostgreSQL in flox environment..."

# Activate flox
eval "$(flox activate)"

# Set up PostgreSQL directories
export PGDATA="$PWD/.flox/postgres/data"
export PGHOST="$PWD/.flox/postgres/socket"
export PGPORT="5432"

mkdir -p "$PGDATA"
mkdir -p "$PGHOST"
mkdir -p "$PWD/.flox/postgres/log"

# Initialize PostgreSQL if needed
if [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "Initializing PostgreSQL database..."
    initdb -D "$PGDATA" -U postgres --locale=C --encoding=UTF8
    
    # Configure PostgreSQL
    cat >> "$PGDATA/postgresql.conf" <<EOF
port = 5432
listen_addresses = 'localhost'
unix_socket_directories = '$PGHOST'
shared_buffers = 256MB
max_connections = 100
EOF
    
    # Configure authentication
    cat > "$PGDATA/pg_hba.conf" <<EOF
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             postgres                                trust
local   all             ominari_user                           md5
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5
EOF
fi

# Start PostgreSQL
if ! pg_ctl -D "$PGDATA" status > /dev/null 2>&1; then
    echo "Starting PostgreSQL..."
    pg_ctl -D "$PGDATA" -l "$PWD/.flox/postgres/log/postgres.log" start
    sleep 3
fi

# Create Ominari user and database
echo "Setting up Ominari database..."
createuser -U postgres -s ominari_user 2>/dev/null || true
psql -U postgres -c "ALTER USER ominari_user WITH PASSWORD 'ominari_2025_secure';" 2>/dev/null || true
createdb -U postgres -O ominari_user ominari_production 2>/dev/null || true

# Export environment variables
export PG_HOST=localhost
export PG_PORT=5432
export PG_USER=ominari_user
export PG_PASSWORD=ominari_2025_secure
export PG_DB=ominari_production

echo "✅ PostgreSQL is ready!"
echo ""
echo "Connection info:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: ominari_production"
echo "  User: ominari_user"
echo ""

# Save environment for Python
cat > .env.postgres <<EOF
export PGDATA="$PGDATA"
export PGHOST="$PGHOST"
export PGPORT="5432"
export PG_HOST=localhost
export PG_PORT=5432
export PG_USER=ominari_user
export PG_PASSWORD=ominari_2025_secure
export PG_DB=ominari_production
EOF

echo "Environment saved to .env.postgres"