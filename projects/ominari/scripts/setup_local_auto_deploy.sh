#!/bin/bash

# Setup script for local auto-deployment

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMINARI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${GREEN}🚀 Ominari Local Auto-Deploy Setup${NC}"
echo "=================================="
echo
echo "This will set up automatic local deployment that:"
echo "- Monitors your git repository for changes"
echo "- Auto-deploys when you push to main"
echo "- Keeps Ominari running continuously"
echo "- Works entirely on localhost"
echo

# Make scripts executable
chmod +x "$SCRIPT_DIR/local_auto_deploy.sh"

# Create systemd service for Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    SERVICE_FILE="/tmp/ominari-auto-deploy.service"
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Ominari Auto Deploy Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$OMINARI_DIR
ExecStart=$SCRIPT_DIR/local_auto_deploy.sh
Restart=always
RestartSec=10
StandardOutput=append:/tmp/ominari_auto_deploy.log
StandardError=append:/tmp/ominari_auto_deploy.log

[Install]
WantedBy=multi-user.target
EOF

    echo -e "${BLUE}Installing systemd service...${NC}"
    sudo cp "$SERVICE_FILE" /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable ominari-auto-deploy.service
    
    echo -e "${GREEN}✅ Systemd service installed${NC}"
    echo
    echo "To start auto-deployment:"
    echo -e "${YELLOW}  sudo systemctl start ominari-auto-deploy${NC}"
    echo
    echo "To check status:"
    echo -e "${YELLOW}  sudo systemctl status ominari-auto-deploy${NC}"
    echo
    echo "To view logs:"
    echo -e "${YELLOW}  sudo journalctl -u ominari-auto-deploy -f${NC}"
    echo

# Create launch agent for macOS
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLIST_FILE="$HOME/Library/LaunchAgents/com.ominari.autodeploy.plist"
    mkdir -p "$HOME/Library/LaunchAgents"
    
    cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ominari.autodeploy</string>
    <key>ProgramArguments</key>
    <array>
        <string>$SCRIPT_DIR/local_auto_deploy.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>$OMINARI_DIR</string>
    <key>StandardOutPath</key>
    <string>/tmp/ominari_auto_deploy.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/ominari_auto_deploy.log</string>
</dict>
</plist>
EOF

    echo -e "${GREEN}✅ Launch agent created${NC}"
    echo
    echo "To start auto-deployment:"
    echo -e "${YELLOW}  launchctl load $PLIST_FILE${NC}"
    echo
    echo "To check logs:"
    echo -e "${YELLOW}  tail -f /tmp/ominari_auto_deploy.log${NC}"
    echo
fi

# Create convenience scripts
cat > "$SCRIPT_DIR/start_auto_deploy.sh" << 'EOF'
#!/bin/bash
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo systemctl start ominari-auto-deploy
    sudo systemctl status ominari-auto-deploy
elif [[ "$OSTYPE" == "darwin"* ]]; then
    launchctl load "$HOME/Library/LaunchAgents/com.ominari.autodeploy.plist"
    echo "Auto-deploy started. Check logs: tail -f /tmp/ominari_auto_deploy.log"
fi
EOF
chmod +x "$SCRIPT_DIR/start_auto_deploy.sh"

cat > "$SCRIPT_DIR/stop_auto_deploy.sh" << 'EOF'
#!/bin/bash
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo systemctl stop ominari-auto-deploy
elif [[ "$OSTYPE" == "darwin"* ]]; then
    launchctl unload "$HOME/Library/LaunchAgents/com.ominari.autodeploy.plist"
fi
pkill -f "local_auto_deploy.sh" || true
echo "Auto-deploy stopped"
EOF
chmod +x "$SCRIPT_DIR/stop_auto_deploy.sh"

cat > "$SCRIPT_DIR/check_auto_deploy.sh" << 'EOF'
#!/bin/bash
echo "🔍 Checking auto-deploy status..."
echo

if pgrep -f "local_auto_deploy.sh" > /dev/null; then
    echo "✅ Auto-deploy is running"
    echo
    echo "Recent logs:"
    tail -n 20 /tmp/ominari_auto_deploy.log
else
    echo "❌ Auto-deploy is not running"
    echo
    echo "Start it with: ./scripts/start_auto_deploy.sh"
fi

echo
echo "Ominari processes:"
pgrep -af "python.*main.py" || echo "  No Ominari main process found"
EOF
chmod +x "$SCRIPT_DIR/check_auto_deploy.sh"

echo -e "${GREEN}✅ Setup complete!${NC}"
echo
echo "Quick commands:"
echo -e "${BLUE}Start auto-deploy:${NC} ./scripts/start_auto_deploy.sh"
echo -e "${BLUE}Stop auto-deploy:${NC}  ./scripts/stop_auto_deploy.sh"
echo -e "${BLUE}Check status:${NC}     ./scripts/check_auto_deploy.sh"
echo
echo "Once started, the auto-deploy will:"
echo "1. Monitor your git repository every 30 seconds"
echo "2. Auto-pull and deploy when changes are pushed to main"
echo "3. Also deploy if you make local commits"
echo "4. Keep Ominari running at http://localhost:8888"
echo
echo -e "${YELLOW}Would you like to start auto-deploy now? (y/N)${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    "$SCRIPT_DIR/start_auto_deploy.sh"
fi