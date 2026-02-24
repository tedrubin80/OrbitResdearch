#!/bin/bash
# Setup orbitalchaos.online - run with sudo
# Usage: sudo bash /var/www/orbit/scripts/setup_website.sh

set -e

DOMAIN="orbitalchaos.online"
WEBROOT="/var/www/orbit/public"

echo "=== Setting up $DOMAIN ==="

# 1. Create nginx site config
cat > /etc/nginx/sites-available/$DOMAIN << 'NGINX'
# HTTP to HTTPS redirect (added by certbot)
server {
    listen 80;
    listen [::]:80;
    server_name orbitalchaos.online www.orbitalchaos.online;

    root /var/www/orbit/public;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;

    # Gzip
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css application/json application/javascript text/xml;
}
NGINX

echo "[1/4] Nginx config created"

# 2. Enable site
ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/$DOMAIN
echo "[2/4] Site enabled"

# 3. Test and reload nginx
nginx -t
systemctl reload nginx
echo "[3/4] Nginx reloaded"

# 4. Get SSL cert
echo "[4/4] Getting SSL certificate..."
certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --redirect -m admin@$DOMAIN --no-eff-email

echo ""
echo "=== Done! ==="
echo "Visit: https://$DOMAIN"
