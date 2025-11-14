#!/bin/bash

# GitHub Actions Self-Hosted Runner Setup Script for Ominari
# This script installs and configures a GitHub Actions runner on your local machine

set -e

echo "🚀 GitHub Actions Self-Hosted Runner Setup for Ominari"
echo "====================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}Please do not run this script as root${NC}"
   exit 1
fi

# Configuration
RUNNER_DIR="$HOME/actions-runner-ominari"
REPO_OWNER=$(git remote get-url origin | sed -E 's/.*[:/]([^/]+)\/[^/]+\.git/\1/')
REPO_NAME=$(basename -s .git `git config --get remote.origin.url`)

echo -e "${GREEN}Repository: $REPO_OWNER/$REPO_NAME${NC}"
echo -e "${GREEN}Runner directory: $RUNNER_DIR${NC}"
echo

# Check if runner already exists
if [ -d "$RUNNER_DIR" ]; then
    echo -e "${YELLOW}Runner directory already exists at $RUNNER_DIR${NC}"
    read -p "Do you want to remove and reinstall? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping existing runner service..."
        cd "$RUNNER_DIR"
        sudo ./svc.sh stop || true
        sudo ./svc.sh uninstall || true
        cd -
        rm -rf "$RUNNER_DIR"
    else
        echo "Using existing runner installation..."
        cd "$RUNNER_DIR"
        sudo ./svc.sh start || ./run.sh
        exit 0
    fi
fi

# Create runner directory
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
    linux)
        case "$ARCH" in
            x86_64)
                RUNNER_ARCH="x64"
                ;;
            aarch64|arm64)
                RUNNER_ARCH="arm64"
                ;;
            *)
                echo -e "${RED}Unsupported architecture: $ARCH${NC}"
                exit 1
                ;;
        esac
        ;;
    darwin)
        RUNNER_ARCH="osx-x64"
        if [[ "$ARCH" == "arm64" ]]; then
            RUNNER_ARCH="osx-arm64"
        fi
        ;;
    *)
        echo -e "${RED}Unsupported OS: $OS${NC}"
        exit 1
        ;;
esac

# Get latest runner version
echo "Getting latest runner version..."
LATEST_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')
echo -e "${GREEN}Latest version: v$LATEST_VERSION${NC}"

# Download runner
DOWNLOAD_URL="https://github.com/actions/runner/releases/download/v${LATEST_VERSION}/actions-runner-${OS}-${RUNNER_ARCH}-${LATEST_VERSION}.tar.gz"
echo "Downloading runner from $DOWNLOAD_URL..."
curl -L -o runner.tar.gz "$DOWNLOAD_URL"

# Extract runner
echo "Extracting runner..."
tar xzf runner.tar.gz
rm runner.tar.gz

# Install dependencies on Linux
if [ "$OS" = "linux" ]; then
    echo "Installing dependencies..."
    ./bin/installdependencies.sh || true
fi

echo
echo -e "${YELLOW}⚠️  IMPORTANT: You need a runner registration token from GitHub${NC}"
echo
echo "To get your token:"
echo "1. Go to: https://github.com/$REPO_OWNER/$REPO_NAME/settings/actions/runners/new"
echo "2. Copy the registration token (it starts with 'AAAA...')"
echo "3. Paste it below when prompted"
echo
read -p "Enter your runner registration token: " RUNNER_TOKEN

if [ -z "$RUNNER_TOKEN" ]; then
    echo -e "${RED}Token cannot be empty${NC}"
    exit 1
fi

# Configure runner
echo "Configuring runner..."
./config.sh \
    --url "https://github.com/$REPO_OWNER/$REPO_NAME" \
    --token "$RUNNER_TOKEN" \
    --name "ominari-local-$(hostname)" \
    --labels "self-hosted,Linux,X64,ominari" \
    --work "_work" \
    --unattended \
    --replace

# Install and start as service
echo "Installing runner as service..."
if [ "$OS" = "linux" ]; then
    sudo ./svc.sh install
    sudo ./svc.sh start
    
    # Ensure runner starts on boot
    sudo systemctl enable actions.runner.$REPO_OWNER-$REPO_NAME.ominari-local-$(hostname).service || true
    
    echo
    echo -e "${GREEN}✅ Runner installed and started as systemd service${NC}"
    echo "Service name: actions.runner.$REPO_OWNER-$REPO_NAME.ominari-local-$(hostname)"
else
    # For macOS, create a launch agent
    PLIST_FILE="$HOME/Library/LaunchAgents/com.github.actions.runner.ominari.plist"
    cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.github.actions.runner.ominari</string>
    <key>ProgramArguments</key>
    <array>
        <string>$RUNNER_DIR/run.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>$RUNNER_DIR</string>
    <key>StandardOutPath</key>
    <string>$RUNNER_DIR/runner.log</string>
    <key>StandardErrorPath</key>
    <string>$RUNNER_DIR/runner.err</string>
</dict>
</plist>
EOF
    launchctl load "$PLIST_FILE"
    echo
    echo -e "${GREEN}✅ Runner installed and started as launch agent${NC}"
fi

# Create environment file for Ominari
cat > "$RUNNER_DIR/.env" << EOF
# Ominari environment for GitHub Actions runner
export PG_PORT=5999
export PG_DB=ominari_production
export DATABASE_URL="postgresql://ominari_user:ominari_2025_secure@localhost:5999/ominari_production"
export OMINARI_PROJECT_DIR="$(dirname "$(pwd)")"
export PATH="$OMINARI_PROJECT_DIR/.venv/bin:\$PATH"
EOF

# Create runner startup script
cat > "$RUNNER_DIR/start_ominari.sh" << EOF
#!/bin/bash
# This script is called by GitHub Actions to start Ominari

source "$RUNNER_DIR/.env"
cd "$OMINARI_PROJECT_DIR"

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    uv venv
fi

# Update dependencies
uv sync

# Start Ominari
exec .venv/bin/python main.py
EOF
chmod +x "$RUNNER_DIR/start_ominari.sh"

echo
echo -e "${GREEN}✅ GitHub Actions self-hosted runner setup complete!${NC}"
echo
echo "Runner status:"
if [ "$OS" = "linux" ]; then
    sudo ./svc.sh status
else
    echo "Check runner log at: $RUNNER_DIR/runner.log"
fi

echo
echo "Next steps:"
echo "1. Verify the runner appears at: https://github.com/$REPO_OWNER/$REPO_NAME/settings/actions/runners"
echo "2. Push code to main branch to trigger auto-deployment"
echo "3. Monitor deployment at: https://github.com/$REPO_OWNER/$REPO_NAME/actions"
echo
echo "The runner will:"
echo "- Automatically start on system boot"
echo "- Run the Ominari trading system when you push to main"
echo "- Keep the system running continuously"
echo
echo "To manually control the runner:"
if [ "$OS" = "linux" ]; then
    echo "  Stop:    sudo systemctl stop actions.runner.$REPO_OWNER-$REPO_NAME.ominari-local-$(hostname)"
    echo "  Start:   sudo systemctl start actions.runner.$REPO_OWNER-$REPO_NAME.ominari-local-$(hostname)"
    echo "  Status:  sudo systemctl status actions.runner.$REPO_OWNER-$REPO_NAME.ominari-local-$(hostname)"
    echo "  Logs:    sudo journalctl -u actions.runner.$REPO_OWNER-$REPO_NAME.ominari-local-$(hostname) -f"
else
    echo "  Stop:    launchctl unload $PLIST_FILE"
    echo "  Start:   launchctl load $PLIST_FILE"
    echo "  Logs:    tail -f $RUNNER_DIR/runner.log"
fi