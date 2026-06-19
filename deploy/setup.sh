#!/bin/bash
# meeTai サーバセットアップスクリプト（Ubuntu 22.04 / 24.04 対象）
set -e

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_USER="${USER:-ubuntu}"

echo "=== meeTai setup ==="
echo "App directory : $APP_DIR"
echo "Service user  : $SERVICE_USER"
echo ""

# [1] システムパッケージ
echo "[1/5] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y python3 python3-pip python3-venv

# [2] Python venv
echo "[2/5] Creating virtual environment..."
python3 -m venv "$APP_DIR/.venv"

# [3] Python パッケージ
echo "[3/5] Installing Python packages..."
"$APP_DIR/.venv/bin/pip" install --upgrade pip -q
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

# [4] .env
echo "[4/5] Setting up .env..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
    echo "  -> Created .env (edit it to add API keys before starting)"
else
    echo "  -> .env already exists, skipping"
fi

# [5] systemd サービス登録
echo "[5/5] Installing systemd services..."
for svc in meetai-backend meetai-frontend; do
    sudo sed \
        "s|APP_DIR|$APP_DIR|g; s|SERVICE_USER|$SERVICE_USER|g" \
        "$APP_DIR/deploy/$svc.service" \
        | sudo tee "/etc/systemd/system/$svc.service" > /dev/null
done

sudo systemctl daemon-reload
sudo systemctl enable meetai-backend meetai-frontend

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit $APP_DIR/.env and fill in your API keys"
echo "  2. sudo systemctl start meetai-backend meetai-frontend"
echo "  3. Check status: sudo systemctl status meetai-backend meetai-frontend"
echo "  4. Access: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
echo "Firewall (UFW) recommended settings:"
echo "  sudo ufw allow 22    # SSH"
echo "  sudo ufw allow 8501  # Streamlit frontend"
echo "  sudo ufw enable"
echo "  (port 8008 stays closed — frontend calls backend server-side)"
