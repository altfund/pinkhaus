# Complete PostgreSQL Migration Plan

## Executive Summary
Moving from 216GB unresponsive SQLite database to fully normalized PostgreSQL system.

**Current Status:**
- ✅ PostgreSQL schema created with 70% size reduction via normalization
- ✅ 47,049 markets successfully migrated (recent 8 months)
- ✅ 8,796 teams normalized with lookup tables
- ✅ Database size: 33MB PostgreSQL vs 216GB SQLite
- ✅ Performance: 1,700+ records/second migration rate

## Migration Strategy: Complete Historical Data Transfer

### Phase 1: Full Historical Migration (2-4 hours)
**Goal**: Migrate ALL historical data from SQLite to PostgreSQL

#### 1.1 Optimized Bulk Migration
```bash
# Create enhanced migration for ALL data
python enhanced_full_migration.py --target-records 10000000 --batch-size 25000
```

**Expected Results:**
- Full 216GB SQLite → ~70-80GB PostgreSQL (70% reduction)
- All market and odds data preserved
- Normalized schema with lookup tables
- Historical integrity maintained

#### 1.2 Parallel Migration Strategy
- **Market Migration**: 25K records/batch
- **Odds Migration**: 50K records/batch (higher volume)
- **Team Lookup**: Continuous creation during market migration
- **Progress Tracking**: Real-time monitoring and recovery

### Phase 2: Application Updates (2-3 hours)
**Goal**: Update all code to use PostgreSQL exclusively

#### 2.1 Database Access Layer Updates
```python
# Remove hybrid logic, use PostgreSQL-only
class PostgreSQLOnlyAccess:
    def get_markets(self, sport=None, date_range=None):
        # Direct PostgreSQL queries with optimized joins
        # No SQLite fallback
```

#### 2.2 Model and ORM Updates
- Update `models.py` to PostgreSQL-only schemas
- Remove SQLite-specific configurations
- Optimize queries for PostgreSQL performance
- Add proper indexing strategies

#### 2.3 Configuration Updates
```python
# database.py - PostgreSQL only
DATABASE_URL = "postgresql://ominari_user:ominari_2025_secure@localhost:5435/ominari_production"
# Remove SQLite path references
```

### Phase 3: System Validation (1 hour)
**Goal**: Ensure complete system functionality without SQLite

#### 3.1 Comprehensive Testing
- Backtest system with PostgreSQL data
- Signal generation and analysis
- API endpoints and dashboard functionality
- Performance benchmarking

#### 3.2 Data Integrity Verification
- Compare key metrics: market counts, team counts, date ranges
- Validate normalized data relationships
- Check for data consistency and completeness

### Phase 4: Safe SQLite Removal (30 minutes)
**Goal**: Archive and remove 216GB SQLite database

#### 4.1 Create Archive
```bash
# Compress SQLite for long-term storage
gzip sport_odds.db  # Creates sport_odds.db.gz (~50GB)
mv sport_odds.db.gz /archive/historical_backup/
```

#### 4.2 Update Documentation
- Update CLAUDE.md with PostgreSQL-only instructions
- Remove SQLite safety warnings
- Document new PostgreSQL-optimized workflows

## Technical Implementation Details

### Enhanced Migration Script Design
```python
class CompleteMigrator:
    """Migrate ALL data from SQLite to PostgreSQL"""

    def migrate_all_markets(self):
        # Remove date filtering - get ALL historical data
        # Use rowid-based pagination for memory efficiency
        # Batch processing with parallel team creation

    def migrate_all_odds(self):
        # High-volume odds migration
        # Optimized for PostgreSQL partitioned tables
        # Batch size 50K for odds (higher volume)

    def verify_complete_migration(self):
        # Compare SQLite vs PostgreSQL counts
        # Validate data integrity across all tables
```

### Expected Performance Gains
- **Query Speed**: 10-100x faster (normalized schema + PostgreSQL)
- **Storage**: 70% reduction (216GB → 70GB)
- **Maintenance**: No more SQLite timeout/locking issues
- **Scalability**: PostgreSQL handles concurrent access excellently

### Risk Mitigation
1. **Backup Strategy**: Keep compressed SQLite archive
2. **Rollback Plan**: Can restore from PostgreSQL dump if needed
3. **Incremental Testing**: Test each application component before full cutover
4. **Monitoring**: Real-time verification during migration

## Success Metrics
- [ ] All 200M+ markets migrated successfully
- [ ] All applications running on PostgreSQL-only
- [ ] Performance improvements verified (10x+ query speed)
- [ ] Storage reduction achieved (70%+ savings)
- [ ] SQLite database safely archived and removed
- [ ] System stability confirmed over 24-48 hours

## Timeline: Total ~6-8 hours
- **Phase 1**: 2-4 hours (historical migration)
- **Phase 2**: 2-3 hours (application updates)
- **Phase 3**: 1 hour (validation)
- **Phase 4**: 30 minutes (cleanup)

This plan achieves complete PostgreSQL migration while maintaining all historical data and dramatically improving system performance and maintainability.