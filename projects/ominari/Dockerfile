# Ominari Blockchain Trading System - Multi-stage Docker build
FROM python:3.11-slim as builder

# Install system dependencies for blockchain trading
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock* .

# Create virtual environment and install dependencies
RUN uv venv
RUN uv sync --no-dev

# Production stage - Blockchain Trading Runtime
FROM python:3.11-slim

# Install runtime dependencies for blockchain trading
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    postgresql-client \
    redis-tools \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 ominari

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=ominari:ominari . .

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Switch to non-root user
USER ominari

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Create required directories and set permissions
RUN mkdir -p /app/logs /app/backtests /app/betting_reports /app/paper_trading_reports \
    && chown -R ominari:ominari /app

# Create startup script
COPY --chown=ominari:ominari docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Default command - can be overridden by docker-compose
CMD ["/app/docker-entrypoint.sh"]