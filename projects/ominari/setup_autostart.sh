#!/bin/bash
# Setup auto-start for Ominari Trading System

echo "🔧 Setting up Ominari auto-start"
echo "==============================="

# For development - add to shell profile
if [[ "$1" == "dev" ]]; then
    echo "Setting up development auto-start..."
    
    # Add to bashrc/zshrc
    SHELL_RC="$HOME/.bashrc"
    if [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_RC="$HOME/.zshrc"
    fi
    
    # Check if already added
    if ! grep -q "start_ominari" "$SHELL_RC"; then
        echo "" >> "$SHELL_RC"
        echo "# Auto-start Ominari Trading System" >> "$SHELL_RC"
        echo "alias ominari='cd $(pwd) && ./start_ominari.py'" >> "$SHELL_RC"
        echo "# Uncomment to auto-start on terminal open:" >> "$SHELL_RC"
        echo "# (cd $(pwd) && ./start_ominari.py > /tmp/ominari.log 2>&1 &)" >> "$SHELL_RC"
        
        echo "✅ Added 'ominari' alias to $SHELL_RC"
        echo "   Run 'ominari' to start the system"
    else
        echo "✅ Auto-start already configured"
    fi
    
# For production - install systemd service
elif [[ "$1" == "prod" ]]; then
    if [[ $EUID -ne 0 ]]; then
        echo "❌ Production setup requires root. Run: sudo $0 prod"
        exit 1
    fi
    
    echo "Setting up production auto-start..."
    
    # Create user if doesn't exist
    if ! id "ominari" &>/dev/null; then
        useradd -r -s /bin/bash -d /opt/ominari ominari
        echo "✅ Created ominari user"
    fi
    
    # Create directories
    mkdir -p /opt/ominari /var/log/ominari
    
    # Copy files
    cp -r . /opt/ominari/
    chown -R ominari:ominari /opt/ominari /var/log/ominari
    
    # Install systemd service
    cp infrastructure/systemd/ominari.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable ominari.service
    
    echo "✅ Systemd service installed"
    echo "   Start with: systemctl start ominari"
    echo "   View logs: journalctl -u ominari -f"
    echo "   Auto-starts on boot"
    
else
    echo "Usage: $0 [dev|prod]"
    echo "  dev  - Setup development auto-start (alias)"
    echo "  prod - Setup production auto-start (systemd)"
fi