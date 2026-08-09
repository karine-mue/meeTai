# ⬡ meeTai

複数のAIと人間が同じコンテキストで会議するためのローカルシステム。  
名前の由来：meeting の meet + me（わたし）と Ai（AI）をかけた造語。

---

## アーキテクチャ

```
[スマホ/PC ブラウザ]
        ↓ tailscale VPN
[自宅PC (WSL/Ubuntu)]
  ├─ uvicorn app.py     :8008  ← FastAPI + LangGraph（バックエンド）
  └─ streamlit frontend :8501  ← Streamlit UI（フロントエンド）
```

### バックエンドの設計原則

**シングルターン・ピンポン構造：**  
1回の `/chat` 呼び出しで1エージェントのみが応答する。複数AIへの同時送信はしない。  
理由：LangGraphの状態DBに同一ユーザー入力を重複記録するコンテキスト汚染を防ぐため。

**コンテキスト共有：**  
`build_context_prompt` が全メッセージ履歴を単一テキストに変換してAPIへ渡す。  
理由：Anthropic APIは `user`/`assistant` の厳密な交互配列を要求するため、マルチエージェントの不規則な発言順序をそのまま渡すと 400 エラーが発生する。

**Reducer によるメッセージ蓄積：**  
`messages: Annotated[List[Message], operator.add]` により差分返却でリストが自動結合される。ノード関数は純粋関数として `dict` を返す設計。

**非同期I/O統一：**  
FastAPI の非同期イベントループをブロックしないよう全クライアントを非同期化。
- `AsyncAnthropic`（anthropic）
- `aio.models.generate_content`（google-genai）
- `ChatOpenAI.ainvoke`（langchain-openai）
- `AsyncSqliteSaver`（langgraph-checkpoint-sqlite）

**状態永続化：**  
`AsyncSqliteSaver` で `checkpoints.sqlite` に全セッション状態を永続化。  
アプリを閉じても履歴はDBに残る（`exporter.py` で復元可能）。

---

## ファイル構成

```
~/python_app/.venv/
├── app.py              # バックエンド（FastAPI + LangGraph）
├── frontend.py         # フロントエンド（Streamlit）
├── openai_gpt.py       # OpenAI model capability profile / request builder / error変換
├── exporter.py         # セッション履歴のCLI抽出ツール
├── tests/              # pytest（実APIへは出ない）
├── requirements.txt
├── .env
├── checkpoints.sqlite  # セッション状態DB（WALモード）
└── exports/            # exporter.py の出力先
```

---

## セットアップ

### 1. 仮想環境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install aiosqlite msgpack  # 追加で必要
```

### 2. .env 設定

```env
# Gemini
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=gemini-2.5-pro       # 現行推奨（stable、reasoning対応）

# Claude
ANTHROPIC_API_KEY=your_anthropic_api_key
CLAUDE_MODEL=claude-opus-4-6

# Qwen (ollama) ※暫定除外中
QWEN_BASE_URL=http://localhost:11434/v1
QWEN_API_KEY=ollama
QWEN_MODEL=qwen2.5:14b

# GPT ※休眠中（APIキー未定義でグラフから自動パージ）
# OPENAI_API_KEY=

# DB
CHECKPOINT_DB=checkpoints.sqlite
```

**Geminiのモデル変遷メモ：**
- `gemini-3.1-pro` → free tier quota厳しくrate limit頻発、非推奨
- `gemini-2.0-flash` → Deprecated（廃止予定）
- `gemini-2.5-flash` → 速度・コスト重視タスク向け
- `gemini-2.5-pro` → 現行推奨。reasoning深度が必要なタスク（コーパス設計等）に適する

### 3. Qwen（ollama）※使う場合

```bash
ollama serve
ollama run qwen2.5:14b
```

コールドスタート（VRAM展開）に2秒以上かかるためヘルスチェックのタイムアウト判定に引っかかりやすい。現在はGemini + Claudeの2エージェントで暫定運用中。

### 4. 起動順

```bash
# 1. バックエンド
uvicorn app:app --host 127.0.0.1 --port 8008

# 2. フロントエンド（別ターミナル）
streamlit run frontend.py --server.port 8501
```

### 5. スマホからアクセス

```
http://{tailscale IP}:8501
```

---

## 使い方

### セッション開始

1. ブラウザで `http://localhost:8501` を開く
2. 各エージェントの状態を確認（`●` 起動中 / `○` 停止中）
3. 参加させるエージェントを選択 → **▶ START SESSION**

### 会議の操作

| 操作 | 説明 |
|------|------|
| **PHASE ボタン** | FREE / CONTEXT / CRITIQUE / SYNTHESIS を切替 |
| **SEND TO Radio** | 送信先を1つ選択（1ターン1エージェント） |
| **read-only トグル** | ON：文脈共有のみ、AIは応答しない |
| **SEND ▶** | 送信（送信中は非活性化→二重送信防止） |
| **✕** | 入力欄をクリア |

送信後は入力欄が自動クリアされる。

### フェーズと優先エージェント

| フェーズ | 優先順 | 用途 |
|----------|--------|------|
| FREE | Gemini → Claude → Qwen | 自由議論 |
| CONTEXT | Gemini → Claude → Qwen | 前提・文脈整理 |
| CRITIQUE | Claude → Gemini → Qwen | 批評・反証 |
| SYNTHESIS | Qwen → Gemini → Claude | 統合・結論 |

### 複数AIに同じ質問を投げる

SEND TO Radio で切り替えながら同じテキストを再送信する。  
（forループによる一括送信はコンテキスト汚染のため廃止済み）

### read-only（文脈共有）

前提資料をAIに「読ませるだけ」にする用途。  
read-only ON → 送信 → DBにメッセージが積まれるがAIは応答しない。

### 議事録

**MINUTES** タブで全発言の時系列ログを確認。  
**↓ EXPORT MINUTES** でテキストファイルをダウンロード。

---

## セッション履歴の復元（exporter.py）

アプリを閉じても `checkpoints.sqlite` に全履歴が残っている。

```bash
pip install msgpack  # 初回のみ
python3 exporter.py
```

```
[Available Sessions: 12]
--------------------------------------------------
  [0] mtg_794e7c78  (31 msgs)  東京から一泊二日で温泉旅行に行くとしたら？
--------------------------------------------------
Enter number to export, 'a' for all, 'q' to quit:
> 0
[Success] Exported: exports/session_mtg_794e7c78_20260228_171456.txt
Export another? (number / 'a' / 'q')
> q
```

**注意：** `checkpoints.sqlite` はWALモードで動作。uvicorn起動中は `.sqlite-shm` / `.sqlite-wal` も同時に存在する。exporter.pyは `PRAGMA wal_checkpoint(PASSIVE)` でWALをフラッシュしてから読む設計になっている。

---

## APIエンドポイント

```bash
# ヘルスチェック
curl http://127.0.0.1:8008/health

# セッション開始
curl -X POST http://127.0.0.1:8008/session \
  -H "Content-Type: application/json" \
  -d '{"session_id":"mtg-001","participants":["gemini","claude"],"phase":"FREE"}'

# メッセージ送信
curl -X POST http://127.0.0.1:8008/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"mtg-001","text":"論点を整理して","phase":"CONTEXT","read_only":false}'

# 参加者の変更（frontend.pyがSEND前に自動で叩く）
curl -X POST http://127.0.0.1:8008/config \
  -H "Content-Type: application/json" \
  -d '{"session_id":"mtg-001","participants":["claude"]}'
```

---

## テスト

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

`pytest.ini` で `testpaths = tests` / `asyncio_mode = auto` を設定済みなので、引数なしの `pytest` で全件走る。実APIへは出ない（OpenAIは `httpx.MockTransport` で差し替え、APIキーは不要）。

| ファイル | 対象 |
|---|---|
| `tests/test_openai_gpt.py` | model capability profile の解決、request body の組み立て、env優先順位、OpenAI error のパースとredaction |
| `tests/test_call_gpt.py` | `call_gpt()` のHTTP error変換、`/chat`・`/ask-all` 両経路の診断保持、error発言がarchive/commitされないこと |

---

## 既知の制約・設計上の注意

**OpenAIの送信パラメータは `openai_gpt.py` に集約：**  
モデルによって送れる optional parameter が変わる（GPT-5系reasoning modelは `temperature` を受け付けず400になる）。判定は `_CAPABILITY_PROFILES` の1箇所だけにあり、`app.py` 側に条件分岐を置かない。モデル追加時に触るのはこの表と対応するテストのみ。

未登録のモデル名は **fail-closed**（必須fieldのみ送信、`reasoning`・`temperature` を付けない）で扱う。「名前が `gpt-5` で始まる」だけの理由でパラメータを送らないため、将来の `gpt-5.x` で同じ400が再発しない。snapshot名（`gpt-5.4-2026-03-05`）は `-` 区切りのprefix matchでbase modelのprofileに解決される。

**reasoning modelの `max_output_tokens`：**  
reasoning tokenも `max_output_tokens` に含まれるため、budgetが小さいとreasoningだけで使い切り、本文が1文字も返らないまま `status=incomplete` / `reason=max_output_tokens` になる。profileの下限（16384）まで自動で引き上げる。

**エラー発言のprefixは `_error_content()` と対：**  
`_error_response()` の `[error] ` と `agent_node()` の `[{agent}] error: ` は `_error_content()` の判定パターンと対になっている。ここを変えるとエラーが正規のAI発言としてarchive/commitされる（サイレントなログ汚染）。`tests/test_call_gpt.py` で両者を縛っている。

**1ターン1エージェント制約：**  
バックエンドのグラフが `supervisor → agent → END` のシングルターン構造のため、1回の `/chat` で応答するエージェントは1つ。

**`as_node="__start__"` について：**  
外部からの状態注入（`aupdate_state`）には `as_node="__start__"` を指定する。  
`as_node="supervisor"` を指定するとsupervisorの実行がバイパスされ `pick_next_agent` がスキップ→AIが無応答になる（サイレントデッドロック）。

**Qwenのコールドスタート問題：**  
ollamaはしばらく使用しないとモデルをVRAMから解放する。ヘルスチェックのタイムアウト（2秒）を超えると停止中判定になる。事前に `ollama run qwen2.5:14b` でロードしておくか、タイムアウト値を延ばすことで回避可能。

---

## 開発経緯メモ（なぜこの設計になったか）

| 問題 | 原因 | 解決策 |
|------|------|--------|
| POST /chat → 500 | `SqliteSaver`（同期）を `ainvoke`（非同期）で使用 | `AsyncSqliteSaver` に変更 |
| POST /session → 500 | `update_state`（同期呼び出し） | `aupdate_state` に変更 |
| Ambiguous update エラー | `aupdate_state` に `as_node` 未指定 | `as_node="__start__"` を追加 |
| AI無応答（サイレントデッドロック） | `as_node="supervisor"` でsupervisorがバイパス | `as_node="__start__"` に修正 |
| コンテキスト汚染 | 複数AIへの `/chat` forループで同一入力が3重記録 | 1ターン1エージェント構造に変更 |
| 議事録の時系列逆転 | AI応答後にhuman発言をappend | humanのappendをAI応答より前に移動 |
| Geminiブロッキング | `genai.generate_content`（同期）使用 | `aio.models.generate_content` に変更 |
| FutureWarning | `google.generativeai`（deprecated） | `google.genai` に移行 |
| exporter.pyでDBが見えない | msgpack形式をjson.loadsで読もうとした | `msgpack.unpackb` でデコード、WAL checkpoint追加 |
| 二重送信 | SEND連打でDBに同一発言が複数記録 | `sending` フラグでボタン非活性化、送信後自動クリア |
| GPTが400（Issue #23） | reasoning付与は `^gpt-5`、temperature除外は `^gpt-5.5` と判定粒度が非対称で、`gpt-5.4` に `temperature` を送っていた | capability profileを `openai_gpt.py` に集約、未登録モデルはfail-closed |
| 400の原因が読めない | `raise_for_status()` がresponse bodyを捨て、httpxの `Client error '400 Bad Request'` だけがUIに出ていた | `parse_openai_error()` で status/message/type/param/code をwhitelist抽出 |

---

## requirements.txt

```
langgraph>=0.3.0
langgraph-checkpoint-sqlite>=1.0.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
anthropic>=0.39.0
google-genai>=0.1.0
fastapi>=0.110.0
uvicorn[standard]>=0.30.0
pydantic>=2.6.0
python-dotenv>=1.0.0
httpx>=0.27.0
streamlit>=1.35.0
aiosqlite
msgpack
```
