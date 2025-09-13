#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migrate codebase to use the enhanced database_v2 module.
"""

import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_imports(file_path: Path):
    """Update imports in a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Update database imports
    patterns = [
        # Direct database imports
        (r'from database import SessionLocal', 'from database_v2 import SessionLocal, db_manager'),
        (r'from database import engine', 'from database_v2 import engine, db_manager'),
        (r'from database import get_db', 'from database_v2 import get_db, db_manager'),
        (r'import database\b', 'import database_v2 as database'),
        
        # Session creation patterns
        (r'db = SessionLocal\(\)', 'with db_manager.get_db_session() as db:'),
        (r'session = SessionLocal\(\)', 'with db_manager.get_db_session() as session:'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # Handle try-finally blocks for sessions
    try_finally_pattern = r'(db|session) = SessionLocal\(\)\s*try:(.*?)finally:\s*\1\.close\(\)'
    
    def replace_try_finally(match):
        var_name = match.group(1)
        body = match.group(2)
        # Dedent the body
        lines = body.split('\n')
        if lines:
            # Find minimum indentation
            min_indent = min(len(line) - len(line.lstrip()) 
                           for line in lines if line.strip())
            # Remove that indentation
            dedented_lines = [line[min_indent:] if len(line) > min_indent else line 
                            for line in lines]
            body = '\n'.join(dedented_lines)
        return f'with db_manager.get_db_session() as {var_name}:\n{body}'
    
    content = re.sub(try_finally_pattern, replace_try_finally, content, flags=re.DOTALL)
    
    if content != original_content:
        logger.info(f"Updated {file_path}")
        return content
    return None


def add_retry_logic(file_path: Path, content: str):
    """Add retry logic to database operations."""
    # This is more complex and would need careful analysis of each file
    # For now, we'll just flag files that might benefit from retry logic
    
    db_operations = [
        'db.query(',
        'session.query(',
        'db.add(',
        'session.add(',
        'db.commit(',
        'session.commit(',
        'db.bulk_',
        'session.bulk_'
    ]
    
    has_db_ops = any(op in content for op in db_operations)
    if has_db_ops and 'OperationalError' not in content:
        logger.warning(f"{file_path} has database operations but no error handling")
    
    return content


def migrate_file(file_path: Path):
    """Migrate a single Python file."""
    if file_path.name == 'database_v2.py':
        return
    
    updated_content = update_imports(file_path)
    if updated_content:
        # Add retry logic analysis
        updated_content = add_retry_logic(file_path, updated_content)
        
        # Write back
        with open(file_path, 'w') as f:
            f.write(updated_content)


def main():
    """Main migration function."""
    logger.info("Starting database migration...")
    
    # Find all Python files
    python_files = list(Path('.').glob('**/*.py'))
    
    # Exclude some directories
    exclude_dirs = {'.venv', '__pycache__', 'venv', 'env', '.git'}
    python_files = [f for f in python_files 
                   if not any(excluded in f.parts for excluded in exclude_dirs)]
    
    logger.info(f"Found {len(python_files)} Python files to check")
    
    for file_path in python_files:
        try:
            migrate_file(file_path)
        except Exception as e:
            logger.error(f"Failed to migrate {file_path}: {e}")
    
    logger.info("Migration complete!")
    logger.info("\nNext steps:")
    logger.info("1. Review the changes")
    logger.info("2. Test the application")
    logger.info("3. Consider adding retry logic to critical database operations")
    logger.info("4. Run: python database_monitor.py report")


if __name__ == "__main__":
    main()