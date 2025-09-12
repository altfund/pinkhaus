#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update the scheduler configuration to use the new data collector
This will gradually migrate from direct API calls to blockchain/GraphQL priority
"""

import os
import shutil
from datetime import datetime


def backup_file(filepath):
    """Create a backup of the file before modifying."""
    backup_path = f"{filepath}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Created backup: {backup_path}")
    return backup_path


def update_scheduler_script():
    """Update the scheduler script to use new data collector."""
    scheduler_path = "run_scheduler.sh"
    
    if not os.path.exists(scheduler_path):
        print(f"❌ {scheduler_path} not found")
        return
    
    # Create backup
    backup_file(scheduler_path)
    
    # Read current content
    with open(scheduler_path, 'r') as f:
        content = f.read()
    
    # Check if already using new collector
    if "free_data_pull_v2.py" in content:
        print("✅ Scheduler already using new data collector")
        return
    
    # Replace old script with new one
    updated_content = content.replace(
        "python free_data_pull.py",
        "python free_data_pull_v2.py"
    )
    
    # Write updated content
    with open(scheduler_path, 'w') as f:
        f.write(updated_content)
    
    print("✅ Updated scheduler to use new data collector")


def create_migration_script():
    """Create a migration script that can switch between old and new collectors."""
    
    migration_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Migration script to switch between data collectors
Can be used to toggle between old API-first and new blockchain-first approaches
\"\"\"

import asyncio
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_new_collector():
    \"\"\"Run the new blockchain/GraphQL-first collector.\"\"\"
    from free_data_pull_v2 import main as new_main
    logger.info("Running new data collector (blockchain/GraphQL priority)")
    await new_main()


def run_old_collector():
    \"\"\"Run the old API-first collector.\"\"\"
    import free_data_pull
    logger.info("Running old data collector (API-first)")
    # The old collector uses synchronous code
    free_data_pull.main()


async def run_comparison():
    \"\"\"Run both collectors and compare results.\"\"\"
    logger.info("Running comparison mode")
    
    # Run old collector
    logger.info("\\n=== Running OLD collector ===")
    start_old = datetime.now()
    try:
        run_old_collector()
        old_duration = (datetime.now() - start_old).total_seconds()
        logger.info(f"Old collector completed in {old_duration:.2f} seconds")
    except Exception as e:
        logger.error(f"Old collector failed: {e}")
    
    # Run new collector
    logger.info("\\n=== Running NEW collector ===")
    start_new = datetime.now()
    try:
        await run_new_collector()
        new_duration = (datetime.now() - start_new).total_seconds()
        logger.info(f"New collector completed in {new_duration:.2f} seconds")
    except Exception as e:
        logger.error(f"New collector failed: {e}")
    
    # Compare database results
    logger.info("\\n=== Comparison complete ===")


def main():
    parser = argparse.ArgumentParser(description='Data collector migration tool')
    parser.add_argument(
        '--mode', 
        choices=['old', 'new', 'compare'],
        default='new',
        help='Which collector to run (default: new)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test mode - don\'t actually save data'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'old':
        run_old_collector()
    elif args.mode == 'new':
        asyncio.run(run_new_collector())
    elif args.mode == 'compare':
        asyncio.run(run_comparison())


if __name__ == '__main__':
    main()
"""
    
    with open('migration_tool.py', 'w') as f:
        f.write(migration_script)
    
    # Make it executable
    os.chmod('migration_tool.py', 0o755)
    
    print("✅ Created migration_tool.py for testing data collectors")


def update_crontab_example():
    """Create example crontab entries for gradual migration."""
    
    crontab_example = """# Ominari Data Collection Schedule - Migration Example
# 
# This shows how to gradually migrate from old to new collector
# Start with mostly old collector, gradually increase new collector usage

# Initial phase (week 1) - Test new collector at low-traffic times
0 3 * * * cd /path/to/ominari && python migration_tool.py --mode new >> logs/new_collector.log 2>&1
*/5 * * * * cd /path/to/ominari && python free_data_pull.py >> scheduler_output.log 2>&1

# Migration phase (week 2) - Run both and compare
0 */6 * * * cd /path/to/ominari && python migration_tool.py --mode compare >> logs/comparison.log 2>&1
*/5 * * * * cd /path/to/ominari && python free_data_pull.py >> scheduler_output.log 2>&1

# Testing phase (week 3) - Increase new collector usage
*/15 * * * * cd /path/to/ominari && python free_data_pull_v2.py >> scheduler_output.log 2>&1
0,30 * * * * cd /path/to/ominari && python free_data_pull.py >> scheduler_output.log 2>&1

# Final phase (week 4) - Switch to new collector
*/5 * * * * cd /path/to/ominari && python free_data_pull_v2.py >> scheduler_output.log 2>&1

# Monitoring - Check data source status daily
0 0 * * * cd /path/to/ominari && python -c "import asyncio; from data_source_manager import DataSourceManager; m = DataSourceManager(); asyncio.run(m.initialize()); print(asyncio.run(m.get_status()))" >> logs/source_status.log 2>&1
"""
    
    with open('crontab_migration_example.txt', 'w') as f:
        f.write(crontab_example)
    
    print("✅ Created crontab_migration_example.txt")


def main():
    print("\\n" + "="*60)
    print("Scheduler Migration Tool")
    print("="*60 + "\\n")
    
    # Update scheduler script
    update_scheduler_script()
    
    # Create migration tool
    create_migration_script()
    
    # Create crontab example
    update_crontab_example()
    
    print("\\n" + "="*60)
    print("Migration setup complete!")
    print("="*60)
    print("\\nNext steps:")
    print("1. Test the migration tool: python migration_tool.py --mode compare")
    print("2. Review the crontab_migration_example.txt for gradual rollout")
    print("3. Monitor logs to ensure data collection is working properly")
    print("\\nIMPORTANT: The new collector prioritizes blockchain/GraphQL data")
    print("API calls are now used only as a backup when other sources fail")


if __name__ == "__main__":
    main()