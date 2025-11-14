#!/bin/bash
# Production database migration script

set -e

echo "🗄️  Ominari Database Migration"
echo "============================="

# Load environment
if [ -f ".env.production" ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
else
    echo "❌ Error: .env.production not found"
    exit 1
fi

# Function to create backup
backup_database() {
    echo "💾 Creating database backup..."
    BACKUP_DIR="backups/$(date +%Y%m%d)"
    mkdir -p $BACKUP_DIR
    
    BACKUP_FILE="$BACKUP_DIR/pre_migration_$(date +%H%M%S).sql"
    
    if pg_dump $DATABASE_URL > $BACKUP_FILE; then
        echo "✅ Backup created: $BACKUP_FILE"
        
        # Compress backup
        gzip $BACKUP_FILE
        echo "✅ Backup compressed: ${BACKUP_FILE}.gz"
    else
        echo "❌ Backup failed!"
        exit 1
    fi
}

# Function to run migrations
run_migrations() {
    echo "🚀 Running database migrations..."
    
    # Check current revision
    echo "Current database revision:"
    alembic current
    
    # Show pending migrations
    echo ""
    echo "Pending migrations:"
    alembic history -r current:head
    
    # Confirm before proceeding
    echo ""
    read -p "⚠️  Run migrations? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Migration cancelled."
        exit 0
    fi
    
    # Run migrations
    if alembic upgrade head; then
        echo "✅ Migrations completed successfully"
    else
        echo "❌ Migration failed!"
        exit 1
    fi
}

# Function to verify migration
verify_migration() {
    echo "🔍 Verifying migration..."
    
    # Check current revision
    echo "New database revision:"
    alembic current
    
    # Test database connection
    python -c "
from database_v2 import db_manager
from models import Market
with db_manager.get_db_session() as db:
    count = db.query(Market).count()
    print(f'✅ Database accessible. Market count: {count}')
" || {
        echo "❌ Database verification failed!"
        exit 1
    }
}

# Main execution
case "${1:-migrate}" in
    backup)
        backup_database
        ;;
    migrate)
        backup_database
        run_migrations
        verify_migration
        ;;
    verify)
        verify_migration
        ;;
    rollback)
        echo "🔄 Rolling back last migration..."
        alembic downgrade -1
        verify_migration
        ;;
    *)
        echo "Usage: $0 [backup|migrate|verify|rollback]"
        echo "  backup   - Create database backup only"
        echo "  migrate  - Backup and run migrations (default)"
        echo "  verify   - Verify database connection"
        echo "  rollback - Rollback last migration"
        exit 1
        ;;
esac

echo "✅ Done!"