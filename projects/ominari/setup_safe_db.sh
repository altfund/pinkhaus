#!/bin/bash
# Setup safe database access

# Create an alias that overrides sqlite3 for sport_odds.db
echo '# Safe database access for Ominari' >> ~/.bashrc
echo 'sqlite3_safe() {' >> ~/.bashrc
echo '  if [[ "$@" == *"sport_odds.db"* ]]; then' >> ~/.bashrc
echo '    echo "❌ Direct SQLite access blocked for sport_odds.db (216GB)"' >> ~/.bashrc
echo '    echo ""' >> ~/.bashrc
echo '    echo "Use the safe query tool instead:"' >> ~/.bashrc
echo '    echo "  python safe_query.py summary"' >> ~/.bashrc
echo '    echo "  python safe_query.py count Market --filter \"sport LIKE '"'"'%Soccer%'"'"'\""' >> ~/.bashrc
echo '    echo "  python safe_query.py sample Odd --limit 5"' >> ~/.bashrc
echo '    echo ""' >> ~/.bashrc
echo '    echo "Or use Python with ORM:"' >> ~/.bashrc
echo '    echo "  python"' >> ~/.bashrc
echo '    echo "  >>> from database_v2 import db_manager"' >> ~/.bashrc
echo '    echo "  >>> from models import Market"' >> ~/.bashrc
echo '    echo "  >>> with db_manager.get_db_session() as db:"' >> ~/.bashrc
echo '    echo "  >>>     markets = db.query(Market).limit(10).all()"' >> ~/.bashrc
echo '    return 1' >> ~/.bashrc
echo '  else' >> ~/.bashrc
echo '    command sqlite3 "$@"' >> ~/.bashrc
echo '  fi' >> ~/.bashrc
echo '}' >> ~/.bashrc
echo 'alias sqlite3=sqlite3_safe' >> ~/.bashrc

echo "✅ Safe database access configured!"
echo ""
echo "To activate, run: source ~/.bashrc"
echo ""
echo "From now on:"
echo "- sqlite3 sport_odds.db → Will be blocked with helpful message"
echo "- python safe_query.py → Safe queries with automatic limits"
echo "- Use database_v2.py → Automatic retry and safety features"