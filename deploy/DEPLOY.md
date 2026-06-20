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

Googleアカウントで複数のエイリアスを持つ場合、実際にログインで使用するプライマリアドレスを指定する。

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

8501（Streamlit 直接） / 8008（FastAPI 直接）は追加しないこと。

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

## 一時バックアップファイルの退避

セットアップ作業中に以下のようなバックアップファイルがリポジトリ直下に生成されることがある。

```
.env.bak.YYYYMMDD_HHMMSS
.streamlit/secrets.toml.bak.YYYYMMDD_HHMMSS
```

これらは `.gitignore` に追加済みのため git には追跡されないが、`git status` には untracked として表示される。  
リポジトリ外に退避することを推奨する。

```bash
mkdir -p ~/meetai-secrets-backup
mv .env.bak.* ~/meetai-secrets-backup/ 2>/dev/null || true
mv .streamlit/secrets.toml.bak.* ~/meetai-secrets-backup/ 2>/dev/null || true
chmod 700 ~/meetai-secrets-backup
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
