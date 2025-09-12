# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL: Database Access Safety

The `sport_odds.db` database is **216GB** in size. **NEVER** use direct SQLite queries or raw SQL as they will hang/timeout:

### ❌ NEVER DO THIS:
```bash
sqlite3 sport_odds.db "SELECT ..."  # Will hang!
```

```python
import sqlite3
conn = sqlite3.connect('sport_odds.db')  # Will cause problems!
```

```python
db.execute("SELECT * FROM market")  # Raw SQL - avoid!
```

### ✅ ALWAYS DO THIS:
```python
# Use the safe database manager with ORM
from database_v2 import db_manager
from models import Market, Odd

with db_manager.get_db_session() as db:
    # Always use .limit() for queries
    markets = db.query(Market).filter(
        Market.sport.like('%Soccer%')
    ).limit(100).all()  # Always add limit!
```

### Safe Query Tool:
```bash
# Use the safe query CLI tool
python safe_query.py summary
python safe_query.py count Market --filter "sport LIKE '%Soccer%'"
python safe_query.py sample Odd --limit 10
python safe_query.py recent --hours 24
```

### Key Safety Rules:
1. **Always use ORM** - Never raw SQL
2. **Always add .limit()** - Prevent full table scans
3. **Use database_v2.py** - Has automatic retry and safety features
4. **Filter before counting** - COUNT(*) without WHERE is very slow
5. **Use safe_query.py** for exploration - It has built-in limits

## Common Development Commands

### Dependency Management
- Install dependencies: `uv sync`
- This project uses `uv` as the package manager (not pip or poetry)

### Code Quality Commands
- Run all checks: `just check`
- Format code: `just format` or `uv run ruff format .`
- Check linting: `just lint` or `uv run ruff check .`
- Auto-fix linting issues: `just fix` or `uv run ruff check --fix .`

### Testing
- Run tests: `just test` (requires flox environment)
- No dedicated test directory or unit tests found - testing is done through:
  - Vectorized backtesting with historical data (`vectorized_backtest.py`)
  - Comprehensive backtest experiments (`test_backtests.py`)
  - Main backtest runner (`run_backtest.py`)
  - Dummy gRPC server for signal testing (`dummy_server.py`)
  - Direct script execution for validation

### Database Commands
- Database migrations: `alembic upgrade head`
- Create new migration: `alembic revision --autogenerate -m "description"`

### Protocol Buffers
- Generate Python code from proto files: `just proto-generate`
- Verify proto files match checked-in versions: `just proto-verify`

## Architecture Overview

### Core Components

1. **Database Layer** (`database.py`, `models.py`)
   - SQLite database with WAL mode enabled for better concurrency
   - SQLAlchemy ORM with declarative models
   - Main models: Market, Odd, Bet, BettingSession
   - Database file: `sport_odds.db`

2. **Signal System** (`signals.py`)
   - Base `SignalProvider` class for probability prediction strategies
   - Implementations include:
     - `ImpliedRawSignal`: Uses implied probabilities from odds
     - `ExternalGrpcSignal`: Fetches signals via gRPC from external service
   - Signals return pandas Series of probabilities (0-1 range)

3. **Backtesting Framework** (`vectorized_backtest.py`, `run_backtest.py`)
   - Vectorized implementation for efficient parallel strategy evaluation
   - Evaluates multiple betting strategies against historical data
   - Generates detailed reports and CSV summaries in `backtests/` directory
   - Supports strategy comparison and performance analysis
   - Test suite available in `test_backtests.py` for comprehensive experiments

4. **Live Trading** (`live_trading_overtime.py`)
   - Real-time betting execution
   - Integrates with Overtime betting platform

5. **Data Collection** (`free_data_pull.py`, `get_oracle_odds.py`)
   - Pulls odds data from various sources
   - Stores in SQLite database

6. **Scheduling** (`run_scheduler.sh`, `setup_scheduler.py`)
   - Automated task execution
   - Logs in `scheduler_output.log` and `scheduler_error.log`

### External Dependencies

- **pinkhaus-models**: Local dependency at `../pinkhaus-models`
  - Contains protobuf definitions for gRPC communication
  - Proto files in `pinkhaus_models/proto/`

### Key Design Patterns

1. **Signal Abstraction**: All prediction strategies implement `SignalProvider` interface
2. **Database Sessions**: Uses SQLAlchemy sessionmaker pattern
3. **gRPC Communication**: For external signal providers using protobuf
4. **Alembic Migrations**: Version-controlled database schema changes

### Important Notes

- The project appears to be a sports betting analysis system
- Uses Kelly criterion for bet sizing (`kelly_multimarket.py`)
- Extensive betting session reporting with HTML/Markdown output
- Real-time and historical data analysis capabilities

## Git Workflow & Feature Development

### Branch Structure
- Main branch: `main`
- Feature branches: `feature/<feature-name>` (e.g., `feature/ominari-updates`)
- Current active development on `feature/ominari-updates`

### Creating a New Feature
1. Create feature branch from main:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. Develop your feature:
   - Make changes to relevant files
   - Ensure code formatting: `just format`
   - Check linting: `just check`
   - Commit regularly with descriptive messages

3. Test your changes:
   - For new signals: Create a signal class inheriting from `SignalProvider`
   - Test with dummy server: `python dummy_server.py` (runs gRPC test server)
   - Run backtests: `python backtest.py` with your signal provider
   - Check generated reports in `betting_reports/` directory

### Testing New Features

#### Running Backtests
```bash
# Run standard backtest with multiple strategies
python run_backtest.py

# Run comprehensive experiments
python test_backtests.py

# Run vectorized backtest directly
python vectorized_backtest.py
```

#### Testing a New Signal Provider
```python
# 1. Create your signal in signals.py
class MyNewSignal(SignalProvider):
    name = "my_new_signal"
    
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        # Your probability logic here
        return probabilities

# 2. Add to SIGNAL_PROVIDERS list in signals.py
# 3. Test with dummy server: python dummy_server.py
# 4. Run backtest to evaluate: python run_backtest.py
```

#### Testing Database Changes
1. Create migration: `alembic revision --autogenerate -m "your change"`
2. Review generated migration file in `alembic/versions/`
3. Apply migration: `alembic upgrade head`
4. Test with actual data queries

#### Manual Integration Testing
- Run data collection: `python free_data_pull.py`
- Test signal evaluation: `python evaluate_open_markets.py`
- Run full backtest: `python backtest.py`
- Check outputs in `betting_reports/` for correctness

### Code Submission
1. Push feature branch:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create pull request to main branch
3. Ensure all checks pass (linting, formatting)

### Development Tips
- Always run `just check` before committing
- Test with small date ranges first when backtesting
- Monitor `scheduler_output.log` for automated task issues
- Use `dummy_server.py` for testing gRPC signals without external dependencies