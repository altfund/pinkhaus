#!/bin/bash
# Setup script for Ominari auto-start

set -e

echo "🚀 Setting up Ominari Trading System for auto-start..."

# Get current user
CURRENT_USER=$(whoami)
INSTALL_DIR=$(pwd)

# Create log directory
echo "Creating log directory..."
sudo mkdir -p /var/log/ominari
sudo chown $CURRENT_USER:$CURRENT_USER /var/log/ominari

# Update service file with correct user and paths
echo "Configuring service file..."
sed -i "s/%USER%/$CURRENT_USER/g" ominari-trading.service
sed -i "s|WorkingDirectory=.*|WorkingDirectory=$INSTALL_DIR|g" ominari-trading.service

# Copy service file to systemd
echo "Installing systemd service..."
sudo cp ominari-trading.service /etc/systemd/system/

# Create startup helper script
cat > start_ominari.sh << 'EOF'
#!/bin/bash
# Ominari startup helper script

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Ensure Graph Node is running
echo "Starting Graph Node infrastructure..."
docker-compose -f docker-compose.graph-node-alt.yml up -d

# Wait for services
echo "Waiting for services to be ready..."
sleep 30

# Start the unified system
echo "Starting Ominari Trading System..."
exec python ominari_unified.py
EOF

chmod +x start_ominari.sh

# Enable service
echo "Enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable ominari-trading.service

echo "✅ Setup complete!"
echo ""
echo "Available commands:"
echo "  Start service:   sudo systemctl start ominari-trading"
echo "  Stop service:    sudo systemctl stop ominari-trading"
echo "  Check status:    sudo systemctl status ominari-trading"
echo "  View logs:       sudo journalctl -u ominari-trading -f"
echo "  View app logs:   tail -f /var/log/ominari/trading.log"
echo ""
echo "The service will start automatically on system boot."
echo ""
echo "To start immediately, run: sudo systemctl start ominari-trading"