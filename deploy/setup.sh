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
echo "[1/6] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y python3 python3-pip python3-venv

# [2] Python venv
echo "[2/6] Creating virtual environment..."
python3 -m venv "$APP_DIR/.venv"

# [3] Python パッケージ
echo "[3/6] Installing Python packages..."
"$APP_DIR/.venv/bin/pip" install --upgrade pip -q
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

# [4] .env
echo "[4/6] Setting up .env..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
    echo "  -> Created .env (edit it to add API keys before starting)"
else
    echo "  -> .env already exists, skipping"
fi

# [5] Streamlit secrets.toml（GitHub OAuth 設定）
echo "[5/6] Setting up Streamlit secrets..."
mkdir -p "$APP_DIR/.streamlit"
if [ ! -f "$APP_DIR/.streamlit/secrets.toml" ]; then
    cp "$APP_DIR/.streamlit/secrets.toml.example" "$APP_DIR/.streamlit/secrets.toml"
    chmod 600 "$APP_DIR/.streamlit/secrets.toml"
    # cookie_secret をランダム生成して自動埋め込み
    COOKIE_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i "s|REPLACE_WITH_RANDOM_HEX_STRING|$COOKIE_SECRET|" "$APP_DIR/.streamlit/secrets.toml"
    echo "  -> Created .streamlit/secrets.toml"
    echo "     (YOUR_SERVER_IP / CLIENT_ID / CLIENT_SECRET を編集してください)"
else
    echo "  -> .streamlit/secrets.toml already exists, skipping"
fi

# [6] systemd サービス登録
echo "[6/6] Installing systemd services..."
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
echo "次の手順で起動してください:"
echo ""
echo "  1. API キーを設定:"
echo "       nano $APP_DIR/.env"
echo "       （GOOGLE_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY / ALLOWED_EMAILS）"
echo ""
echo "  2. GitHub OAuth を設定:"
echo "       nano $APP_DIR/.streamlit/secrets.toml"
echo "       GitHub OAuth App の登録先: https://github.com/settings/developers"
echo "       callback URL: http://$SERVER_IP:8501/oauth2callback"
echo ""
echo "  3. サービス起動:"
echo "       sudo systemctl start meetai-backend meetai-frontend"
echo ""
echo "  4. アクセス: http://$SERVER_IP:8501"
echo ""
echo "ファイアウォール設定:"
echo "  sudo ufw allow 22    # SSH"
echo "  sudo ufw allow 8501  # Streamlit"
echo "  sudo ufw enable"
echo "  （ポート 8008 は閉じたまま — フロントエンドがサーバ内部で呼ぶため）"
