# ⬡ meeTai

人間とAI複数名で会議するためのローカルシステム。  
FastAPI（バックエンド）+ Streamlit（フロントエンド）+ LangGraph（状態管理）。  
tailscale経由でスマホからもアクセス可能。

---

## 構成

```
multi-ai-meeting/
├── app.py          # バックエンド（FastAPI + LangGraph）
├── frontend.py     # フロントエンド（Streamlit）
├── requirements.txt
└── .env
```

```
[スマホ/PC] → tailscale VPN → 自宅PC
                                ├ uvicorn app.py     :8008  (バックエンド)
                                └ streamlit frontend :8501  (フロントエンド)
```

---

## セットアップ

### 1. 仮想環境

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 環境変数

```bash
cp .env.example .env
```

`.env` を編集：

```env
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=gemini-1.5-pro

ANTHROPIC_API_KEY=your_anthropic_api_key
CLAUDE_MODEL=claude-opus-4-6

# ollama デフォルトポート
QWEN_BASE_URL=http://localhost:11434/v1
QWEN_API_KEY=ollama
QWEN_MODEL=qwen2.5:14b

# GPT は未定義のまま（休眠状態で保持）
# OPENAI_API_KEY=

CHECKPOINT_DB=checkpoints.sqlite
```

### 3. Qwen（ollama）の起動

```bash
ollama serve
ollama run qwen2.5:14b
```

### 4. バックエンド起動

```bash
uvicorn app:app --host 127.0.0.1 --port 8008
```

### 5. フロントエンド起動

```bash
streamlit run frontend.py --server.port 8501
```

### 6. スマホからアクセス

```
http://{tailscale IP}:8501
```

---

## 使い方

### セッション開始

1. ブラウザで `http://localhost:8501` を開く
2. 各エージェントの生死を確認（`●` = 起動中、`○` = 停止中）
3. 参加させるエージェントにチェックを入れる
4. **START SESSION** を押す

### 会議の進め方

| 操作 | 説明 |
|------|------|
| **PHASE ボタン** | FREE / CONTEXT / CRITIQUE / SYNTHESIS を切替 |
| **SEND TO Radio** | 送信先エージェントを1つ選択 |
| **read-only トグル** | ONにすると文脈共有のみ（AIは応答しない） |
| **SEND ▶** | 選択したエージェントへ送信 |

### フェーズの使い分け

| フェーズ | 優先エージェント | 用途 |
|----------|-----------------|------|
| FREE | Gemini → Claude → Qwen | 自由議論 |
| CONTEXT | Gemini → Claude → Qwen | 前提・文脈の整理 |
| CRITIQUE | Claude → Gemini → Qwen | 批評・反証 |
| SYNTHESIS | Qwen → Gemini → Claude | 統合・結論 |

### 複数AIに同じ質問を投げる場合

1. SEND TO で `gemini` を選択 → SEND
2. SEND TO で `claude` に切り替え → SEND（同じ入力）
3. SEND TO で `qwen` に切り替え → SEND（同じ入力）

各AIの応答はタブ（GEMINI / CLAUDE / QWEN）で確認。

### read-only（文脈共有）

前提資料や共有したい情報をAIに「読ませるだけ」にする。

1. read-only トグルを ON
2. 共有したいテキストを入力
3. SEND → 全AIのコンテキストに積まれるが応答なし

### 議事録

**MINUTES** タブに全発言の時系列ログが表示される。  
**↓ EXPORT MINUTES** でテキストファイルとしてダウンロード可能。

---

## エンドポイント（直接叩く場合）

```bash
# ヘルスチェック
curl http://127.0.0.1:8008/health

# セッション開始
curl -X POST http://127.0.0.1:8008/session \
  -H "Content-Type: application/json" \
  -d '{"session_id":"mtg-001","participants":["gemini","claude","qwen"],"phase":"FREE"}'

# メッセージ送信
curl -X POST http://127.0.0.1:8008/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"mtg-001","text":"論点を整理して","phase":"CONTEXT","read_only":false}'

# 参加者の変更
curl -X POST http://127.0.0.1:8008/config \
  -H "Content-Type: application/json" \
  -d '{"session_id":"mtg-001","participants":["claude"]}'
```

---

## 注意事項

- 1ターンに送信できるエージェントは1つ（コンテキスト汚染を防ぐため）
- Qwenが停止中は `need_human` が表示される → 別エージェントへ手動で切り替える
- セッション状態は `checkpoints.sqlite` に永続化される
- セッションを終了する場合は **END SESSION** ボタン

---

## 環境

- Python 3.10+
- ollama（Qwen使用時）
- tailscale（スマホアクセス時）
