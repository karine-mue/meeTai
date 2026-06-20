#!/bin/bash
# meeTai シークレットバックアップ＆ローテーション
# 使い方: bash deploy/backup-secrets.sh
#
# バックアップ先: ~/meetai-secrets-backup/meetai/
# 命名規則:      env.YYYYMMDD_HHMMSS / secrets_toml.YYYYMMDD_HHMMSS
# ローテーション: 各種別ごとに直近 KEEP 件を残して古いものを削除

set -e

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="${HOME}/meetai-secrets-backup/meetai"
KEEP=5
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"

backup_file() {
    local src="$1"
    local name="$2"
    if [ -f "$src" ]; then
        local dest="${BACKUP_DIR}/${name}.${TIMESTAMP}"
        cp "$src" "$dest"
        chmod 600 "$dest"
        echo "  backed up: ${name}.${TIMESTAMP}"
    else
        echo "  skip (not found): $src"
    fi
}

rotate() {
    local prefix="$1"
    local files
    mapfile -t files < <(ls -t "${BACKUP_DIR}/${prefix}."* 2>/dev/null)
    local count=${#files[@]}
    if [ "$count" -gt "$KEEP" ]; then
        local i
        for ((i=KEEP; i<count; i++)); do
            rm -f "${files[$i]}"
        done
        echo "  rotated: $((count - KEEP)) old file(s) removed"
    fi
}

echo "=== meeTai secrets backup (${TIMESTAMP}) ==="
backup_file "$APP_DIR/.env"                    "env"
backup_file "$APP_DIR/.streamlit/secrets.toml" "secrets_toml"

echo ""
echo "=== rotation (keep last ${KEEP}) ==="
rotate "env"
rotate "secrets_toml"

echo ""
echo "Backup dir: $BACKUP_DIR"
ls -lt "$BACKUP_DIR"
