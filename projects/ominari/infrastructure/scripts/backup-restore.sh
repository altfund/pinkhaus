#!/bin/bash
# Backup and restore script for Ominari

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BACKUP_DIR="$PROJECT_ROOT/backups"

# Load environment
if [ -f "$PROJECT_ROOT/.env.production" ]; then
    export $(cat "$PROJECT_ROOT/.env.production" | grep -v '^#' | xargs)
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Functions
backup_database() {
    echo -e "${GREEN}📦 Creating database backup...${NC}"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/ominari_db_${TIMESTAMP}.sql"
    
    # Create backup
    if pg_dump "$DATABASE_URL" > "$BACKUP_FILE"; then
        # Compress backup
        gzip "$BACKUP_FILE"
        echo -e "${GREEN}✅ Database backup created: ${BACKUP_FILE}.gz${NC}"
        
        # Upload to S3 if configured
        if [ ! -z "$AWS_BACKUP_BUCKET" ]; then
            echo -e "${YELLOW}⬆️  Uploading to S3...${NC}"
            aws s3 cp "${BACKUP_FILE}.gz" "s3://${AWS_BACKUP_BUCKET}/database/${TIMESTAMP}/"
            echo -e "${GREEN}✅ Backup uploaded to S3${NC}"
        fi
    else
        echo -e "${RED}❌ Database backup failed!${NC}"
        exit 1
    fi
}

backup_application() {
    echo -e "${GREEN}📦 Creating application backup...${NC}"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/ominari_app_${TIMESTAMP}.tar.gz"
    
    # Create tarball excluding certain directories
    cd "$PROJECT_ROOT"
    tar -czf "$BACKUP_FILE" \
        --exclude=".git" \
        --exclude="__pycache__" \
        --exclude=".venv" \
        --exclude="backups" \
        --exclude="logs" \
        --exclude="*.log" \
        --exclude=".env*" \
        .
    
    echo -e "${GREEN}✅ Application backup created: $BACKUP_FILE${NC}"
    
    # Upload to S3 if configured
    if [ ! -z "$AWS_BACKUP_BUCKET" ]; then
        echo -e "${YELLOW}⬆️  Uploading to S3...${NC}"
        aws s3 cp "$BACKUP_FILE" "s3://${AWS_BACKUP_BUCKET}/application/${TIMESTAMP}/"
        echo -e "${GREEN}✅ Backup uploaded to S3${NC}"
    fi
}

list_backups() {
    echo -e "${GREEN}📋 Available backups:${NC}"
    echo ""
    
    # Local backups
    echo -e "${YELLOW}Local backups:${NC}"
    if ls -la "$BACKUP_DIR"/*.gz 2>/dev/null; then
        ls -la "$BACKUP_DIR"/*.gz | awk '{print $9, $5}'
    else
        echo "No local backups found"
    fi
    
    echo ""
    
    # S3 backups
    if [ ! -z "$AWS_BACKUP_BUCKET" ]; then
        echo -e "${YELLOW}S3 backups:${NC}"
        aws s3 ls "s3://${AWS_BACKUP_BUCKET}/" --recursive --human-readable
    fi
}

restore_database() {
    BACKUP_FILE="$1"
    
    if [ -z "$BACKUP_FILE" ]; then
        echo -e "${RED}❌ Error: Backup file not specified${NC}"
        echo "Usage: $0 restore-db <backup-file>"
        exit 1
    fi
    
    echo -e "${YELLOW}⚠️  WARNING: This will restore the database from backup${NC}"
    echo -e "${YELLOW}Current data will be overwritten!${NC}"
    read -p "Continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Restore cancelled"
        exit 0
    fi
    
    # Download from S3 if needed
    if [[ "$BACKUP_FILE" == s3://* ]]; then
        echo -e "${YELLOW}⬇️  Downloading from S3...${NC}"
        TEMP_FILE="/tmp/restore_$(date +%s).sql.gz"
        aws s3 cp "$BACKUP_FILE" "$TEMP_FILE"
        BACKUP_FILE="$TEMP_FILE"
    fi
    
    # Decompress if needed
    if [[ "$BACKUP_FILE" == *.gz ]]; then
        echo -e "${YELLOW}📂 Decompressing backup...${NC}"
        gunzip -c "$BACKUP_FILE" > "${BACKUP_FILE%.gz}"
        BACKUP_FILE="${BACKUP_FILE%.gz}"
    fi
    
    # Restore database
    echo -e "${YELLOW}🔄 Restoring database...${NC}"
    if psql "$DATABASE_URL" < "$BACKUP_FILE"; then
        echo -e "${GREEN}✅ Database restored successfully${NC}"
        
        # Clean up temp files
        if [ ! -z "$TEMP_FILE" ]; then
            rm -f "$TEMP_FILE" "${TEMP_FILE%.gz}"
        fi
    else
        echo -e "${RED}❌ Database restore failed!${NC}"
        exit 1
    fi
}

cleanup_old_backups() {
    DAYS_TO_KEEP="${1:-30}"
    
    echo -e "${YELLOW}🧹 Cleaning up backups older than $DAYS_TO_KEEP days...${NC}"
    
    # Clean local backups
    find "$BACKUP_DIR" -name "*.gz" -type f -mtime +$DAYS_TO_KEEP -delete
    echo -e "${GREEN}✅ Local cleanup complete${NC}"
    
    # Clean S3 backups if configured
    if [ ! -z "$AWS_BACKUP_BUCKET" ]; then
        echo -e "${YELLOW}🧹 Cleaning S3 backups...${NC}"
        # This requires lifecycle policy on S3 bucket
        echo "Note: S3 cleanup should be configured via lifecycle policy"
    fi
}

automated_backup() {
    echo -e "${GREEN}🤖 Running automated backup...${NC}"
    echo "Timestamp: $(date)"
    
    # Backup database
    backup_database
    
    # Backup application (optional, once per day)
    HOUR=$(date +%H)
    if [ "$HOUR" -eq "03" ]; then
        backup_application
    fi
    
    # Cleanup old backups
    cleanup_old_backups 30
    
    echo -e "${GREEN}✅ Automated backup complete${NC}"
}

# Main execution
case "${1:-help}" in
    backup-db)
        backup_database
        ;;
    backup-app)
        backup_application
        ;;
    backup-all)
        backup_database
        backup_application
        ;;
    list)
        list_backups
        ;;
    restore-db)
        restore_database "$2"
        ;;
    cleanup)
        cleanup_old_backups "${2:-30}"
        ;;
    auto)
        automated_backup
        ;;
    help|*)
        echo "Ominari Backup & Restore Tool"
        echo ""
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  backup-db      - Backup database only"
        echo "  backup-app     - Backup application files only"
        echo "  backup-all     - Backup both database and application"
        echo "  list          - List available backups"
        echo "  restore-db    - Restore database from backup"
        echo "  cleanup [days] - Remove backups older than N days (default: 30)"
        echo "  auto          - Run automated backup (for cron)"
        echo ""
        echo "Examples:"
        echo "  $0 backup-db"
        echo "  $0 restore-db /path/to/backup.sql.gz"
        echo "  $0 cleanup 7"
        exit 1
        ;;
esac