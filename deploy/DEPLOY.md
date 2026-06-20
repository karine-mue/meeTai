# VPS デプロイ補足メモ

`deploy/setup.sh` の実行後、追加で確認が必要な事項をまとめる。

---

## 認証：Google OIDC を使う

meeTai の認証は **Google OIDC**（OpenID Connect）を使用している。

### GitHub OAuth App ではなく Google OIDC を選んだ理由

| 項目 | GitHub OAuth App | Google OIDC |
|------|------------------|-------------|
| アカウント普及率 | 開発者向け | 一般ユーザーも使いやすい |
| メール検証 | GitHubアカウントのメール | Googleアカウントのメール（本人確認済み） |
| `ALLOWED_EMAILS` 照合 | GitHubのprimary email | Googleログイン時に返るメール（一致しやすい） |
| Streamlit 対応 | 設定が複雑 | `authlib>=1.3.2` + `[auth.google]` で完結 |

### `.streamlit/secrets.toml` の設定例

```toml
[auth]
redirect_uri   = "https://YOUR_DOMAIN/oauth2callback"
cookie_secret  = "ここに python3 -c \"import secrets; print(secrets.token_hex(32))\" の出力を貼る"

[auth.google]
client_id             = "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com"
client_secret         = "YOUR_GOOGLE_CLIENT_SECRET"
server_metadata_url   = "https://accounts.google.com/.well-known/openid-configuration"
```

`setup.sh` を実行すると `.streamlit/secrets.toml.example` からコピーされ、`cookie_secret` が自動生成される。  
`client_id` / `client_secret` / `redirect_uri` は手動で編集すること。

### `ALLOWED_EMAILS` の注意点

`.env` の `ALLOWED_EMAILS` には、Googleログイン時に返るメールアドレスを指定する。

```env
ALLOWED_EMAILS=you@gmail.com,colleague@gmail.com
```

Googleアカウントで複数のエイリアスを持つ場合、実際にログインで使用するプライマリアドレスを指定すること。

### `authlib>=1.3.2` が必要

Google OIDC 認証には `authlib>=1.3.2` が必要。`requirements.txt` にすでに含まれているため、`setup.sh` 実行時に自動インストールされる。

---

## ファイアウォール：さくらVPS は 2 段構成

さくらVPS は **パケットフィルタ（コントロールパネル）** と **UFW（OS内）** の両方でポートを制御している。  
**どちらか一方だけ開けても通らない。**

### さくらVPS パケットフィルタ（コントロールパネル）

1. さくらVPS コントロールパネル → サーバ → パケットフィルタ設定
2. 以下を許可に追加：

| プロトコル | ポート | 用途 |
|-----------|--------|------|
| TCP | 22 | SSH |
| TCP | 80 | HTTP（Caddy ACME 証明書取得） |
| TCP | 443 | HTTPS |

8501（Streamlit直接）/ 8008（FastAPI直接）は追加しないこと。

### UFW（OS 内ファイアウォール）

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8501/tcp
sudo ufw deny 8008/tcp
sudo ufw enable
```

`setup.sh` の出力にもこのコマンドが表示される。

---

## シークレットバックアップ

`.env` と `.streamlit/secrets.toml` のバックアップには `deploy/backup-secrets.sh` を使う。

```bash
bash deploy/backup-secrets.sh
```

### バックアップ先と命名規則

```
~/meetai-secrets-backup/
  meetai/                        ← リポジトリ名でサブディレクトリを切る
    env.20260620_143012
    secrets_toml.20260620_143012
    env.20260613_091500          ← 直近 5 件を超えると自動削除
```

- ドット始まりなし（`ls` で即見える）
- 種別ごとに直近 5 件を保持してローテーション
- バックアップファイルのパーミッションは `600`、ディレクトリは `700`

### 既存の旧形式ファイルの整理

初回セットアップ時に `~/meetai-secrets-backup/` 直下に旧形式で生成されたファイルは手動で削除する。

```bash
# 確認
ls ~/meetai-secrets-backup/

# 不要なら削除
rm -f ~/meetai-secrets-backup/.env.bak.*
rm -f ~/meetai-secrets-backup/secrets.toml.bak.*
```

---

## 疎通確認の順番

```bash
# 1. Streamlit が起動しているか
curl http://127.0.0.1:8501

# 2. FastAPI が起動しているか
curl http://127.0.0.1:8008/health

# 3. Caddy 経由でアクセスできるか
curl https://YOUR_DOMAIN
```
