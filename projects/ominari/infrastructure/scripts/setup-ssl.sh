#!/bin/bash
# SSL setup script using Let's Encrypt

set -e

DOMAIN="ominari.trading"
EMAIL="admin@ominari.trading"  # Update this!

echo "🔐 SSL Certificate Setup for Ominari"
echo "===================================="

# Check if running as root/sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root or with sudo"
    exit 1
fi

# Install certbot if not present
if ! command -v certbot &> /dev/null; then
    echo "📦 Installing Certbot..."
    apt-get update
    apt-get install -y certbot python3-certbot-nginx
fi

# Stop nginx if running (for standalone mode)
echo "🛑 Stopping nginx if running..."
systemctl stop nginx 2>/dev/null || true

# Function to obtain certificate
obtain_certificate() {
    echo "🔑 Obtaining SSL certificate for $DOMAIN..."
    
    certbot certonly \
        --standalone \
        --non-interactive \
        --agree-tos \
        --email $EMAIL \
        --domains $DOMAIN,www.$DOMAIN,api.$DOMAIN \
        --expand
        
    if [ $? -eq 0 ]; then
        echo "✅ SSL certificate obtained successfully!"
    else
        echo "❌ Failed to obtain SSL certificate"
        exit 1
    fi
}

# Function to setup auto-renewal
setup_renewal() {
    echo "⏰ Setting up auto-renewal..."
    
    # Create renewal script
    cat > /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh << 'EOF'
#!/bin/bash
nginx -s reload
EOF
    
    chmod +x /etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh
    
    # Test renewal
    certbot renew --dry-run
    
    if [ $? -eq 0 ]; then
        echo "✅ Auto-renewal configured successfully!"
    else
        echo "⚠️  Auto-renewal test failed, please check configuration"
    fi
}

# Function to configure nginx
configure_nginx() {
    echo "🔧 Configuring nginx..."
    
    # Create directories
    mkdir -p /etc/nginx/sites-available /etc/nginx/sites-enabled
    
    # Copy nginx configuration
    cp infrastructure/nginx/nginx.conf /etc/nginx/sites-available/ominari
    
    # Enable site
    ln -sf /etc/nginx/sites-available/ominari /etc/nginx/sites-enabled/
    
    # Test nginx configuration
    nginx -t
    
    if [ $? -eq 0 ]; then
        echo "✅ Nginx configuration valid!"
        systemctl start nginx
        systemctl enable nginx
    else
        echo "❌ Nginx configuration invalid!"
        exit 1
    fi
}

# Function to generate DH params
generate_dhparam() {
    echo "🔐 Generating DH parameters (this may take a while)..."
    
    if [ ! -f /etc/ssl/certs/dhparam.pem ]; then
        openssl dhparam -out /etc/ssl/certs/dhparam.pem 2048
        echo "✅ DH parameters generated!"
    else
        echo "✅ DH parameters already exist"
    fi
}

# Main execution
case "${1:-all}" in
    cert)
        obtain_certificate
        ;;
    renewal)
        setup_renewal
        ;;
    nginx)
        configure_nginx
        ;;
    dhparam)
        generate_dhparam
        ;;
    all)
        obtain_certificate
        generate_dhparam
        configure_nginx
        setup_renewal
        ;;
    renew)
        certbot renew
        nginx -s reload
        ;;
    test)
        echo "🧪 Testing SSL configuration..."
        curl -I https://$DOMAIN
        echo ""
        echo "SSL Labs test: https://www.ssllabs.com/ssltest/analyze.html?d=$DOMAIN"
        ;;
    *)
        echo "Usage: $0 [cert|renewal|nginx|dhparam|all|renew|test]"
        echo ""
        echo "Commands:"
        echo "  cert     - Obtain SSL certificate only"
        echo "  renewal  - Setup auto-renewal only"
        echo "  nginx    - Configure nginx only"
        echo "  dhparam  - Generate DH parameters only"
        echo "  all      - Complete setup (default)"
        echo "  renew    - Manually renew certificate"
        echo "  test     - Test SSL configuration"
        exit 1
        ;;
esac

echo ""
echo "📋 Next steps:"
echo "1. Update EMAIL in this script to your real email"
echo "2. Ensure DNS A records point to your server:"
echo "   - ominari.trading → your-server-ip"
echo "   - www.ominari.trading → your-server-ip"
echo "   - api.ominari.trading → your-server-ip"
echo "3. Run: sudo $0 all"
echo "4. Test: $0 test"