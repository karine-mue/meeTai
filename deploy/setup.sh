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
echo "[1/7] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y python3 python3-pip python3-venv

# [2] Python venv
echo "[2/7] Creating virtual environment..."
python3 -m venv "$APP_DIR/.venv"

# [3] Python パッケージ
echo "[3/7] Installing Python packages..."
"$APP_DIR/.venv/bin/pip" install --upgrade pip -q
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

# [4] .env
echo "[4/7] Setting up .env..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
    echo "  -> Created .env (edit it to add API keys before starting)"
else
    echo "  -> .env already exists, skipping"
fi

# [5] Streamlit secrets.toml（Google OAuth 設定）
echo "[5/7] Setting up Streamlit secrets..."
mkdir -p "$APP_DIR/.streamlit"
if [ ! -f "$APP_DIR/.streamlit/secrets.toml" ]; then
    cp "$APP_DIR/.streamlit/secrets.toml.example" "$APP_DIR/.streamlit/secrets.toml"
    chmod 600 "$APP_DIR/.streamlit/secrets.toml"
    COOKIE_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i "s|REPLACE_WITH_RANDOM_HEX_STRING|$COOKIE_SECRET|" "$APP_DIR/.streamlit/secrets.toml"
    echo "  -> Created .streamlit/secrets.toml"
    echo "     (YOUR_SERVER_IP → ドメイン名 / CLIENT_ID / CLIENT_SECRET を編集してください)"
else
    echo "  -> .streamlit/secrets.toml already exists, skipping"
fi

# [6] Caddy インストール
echo "[6/7] Installing Caddy..."
if ! command -v caddy &>/dev/null; then
    sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        | sudo tee /etc/apt/sources.list.d/caddy-stable.list > /dev/null
    sudo apt-get update -q
    sudo apt-get install -y caddy
    echo "  -> Caddy installed"
else
    echo "  -> Caddy already installed, skipping"
fi

if [ ! -f /etc/caddy/Caddyfile ] || grep -q "Your site's address" /etc/caddy/Caddyfile; then
    sudo cp "$APP_DIR/deploy/Caddyfile.example" /etc/caddy/Caddyfile
    echo "  -> Copied Caddyfile.example to /etc/caddy/Caddyfile"
    echo "     (meetai.example.com をあなたのドメインに書き換えてください)"
else
    echo "  -> /etc/caddy/Caddyfile already configured, skipping"
fi

# [7] systemd サービス登録
echo "[7/7] Installing systemd services..."
for svc in meetai-backend meetai-frontend; do
    sudo sed \
        "s|APP_DIR|$APP_DIR|g; s|SERVICE_USER|$SERVICE_USER|g" \
        "$APP_DIR/deploy/$svc.service" \
        | sudo tee "/etc/systemd/system/$svc.service" > /dev/null
done

sudo systemctl daemon-reload
sudo systemctl enable meetai-backend meetai-frontend

SERVER_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "=== Setup complete ==="
echo ""
echo "起動前に必要な編集:"
echo ""
echo "  1. API キーを設定:"
echo "       nano $APP_DIR/.env"
echo "       （GOOGLE_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY / ALLOWED_EMAILS）"
echo ""
echo "  2. Google OAuth を設定:"
echo "       nano $APP_DIR/.streamlit/secrets.toml"
echo "       Google Cloud Console: https://console.cloud.google.com/ → APIとサービス → 認証情報"
echo "       callback URL: https://YOUR_DOMAIN/oauth2callback"
echo ""
echo "  3. Caddy のドメインを設定:"
echo "       sudo nano /etc/caddy/Caddyfile"
echo "       meetai.example.com → あなたのドメインに書き換え"
echo ""
echo "起動:"
echo "  sudo systemctl start meetai-backend meetai-frontend"
echo "  sudo systemctl reload caddy"
echo ""
echo "ファイアウォール設定:"
echo "  sudo ufw allow 22/tcp    # SSH"
echo "  sudo ufw allow 80/tcp    # HTTP (Caddy ACME 証明書取得用)"
echo "  sudo ufw allow 443/tcp   # HTTPS"
echo "  sudo ufw deny 8501/tcp   # Streamlit 直接アクセス禁止"
echo "  sudo ufw deny 8008/tcp   # FastAPI 直接アクセス禁止"
echo "  sudo ufw enable"
echo ""
echo "疎通確認の順番:"
echo "  1. curl http://127.0.0.1:8501       # Streamlit が起動しているか"
echo "  2. curl http://127.0.0.1:8008/health # FastAPI が起動しているか"
echo "  3. https://YOUR_DOMAIN でアクセス    # Caddy 経由で到達するか"
