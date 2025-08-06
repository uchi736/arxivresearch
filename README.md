# arXiv Research Agent v3.1

研究者が特定の研究トピックに関する最新の学術動向を効率的に把握するための支援ツールです。ユーザーのクエリに基づき、arXivから関連論文を自動的に検索・分析し、構造化された「落合陽一フォーマット」のレポートを生成します。

## 🚀 最新の改善 (v3.1)
- **分析品質の大幅向上**: 論文分析時のテキスト処理量を3,000文字→15,000文字に拡張（5倍）
- **詳細な分析結果保存**: CSV保存時の各フィールド制限を500文字→2,000文字に拡張（4倍）
- **Streamlit UI**: ブラウザベースの使いやすい研究インターフェース

## 主な機能

- 🔍 **多言語対応検索**: 日本語クエリを高精度に英語翻訳し、技術用語辞書を活用した適切なキーワードでarXiv検索
- 📊 **高度な関連性スコアリング**: 独自の重み付けロジックで0〜10点の範囲で論文を評価・選別
- 📝 **Map-Reduce全文分析**: Gemini長文コンテキストを活用したMap-Reduce方式による詳細分析
- 📋 **落合フォーマット出力**: 10項目の構造化された分析結果（これは何か？、先行研究との比較、技術の核心など）
- 🚀 **並列処理**: 複数論文のPDFを並列ダウンロード・処理で高速化
- 🎯 **動的パラメータ調整**: クエリ内容に応じて分析深度と検索期間を自動決定
- 📊 **日本語CSV管理**: 論文データベースの効率的な管理（重複回避、Excel出力対応）
- 🔄 **翻訳機能**: arXiv論文のHTML→日本語翻訳（LaTeXML活用）
- 💾 **バックアップ・リセット**: 安全なデータベース操作機能

## ディレクトリ構成

```
.
├── .env.example                       # 環境変数のサンプルファイル
├── cli_app.py                         # CLIアプリケーションのエントリーポイント
├── research_ui.py                     # Streamlit UIのエントリーポイント
├── README.md                          # このファイル
├── requirements.txt                   # Pythonの依存関係リスト
├── database/                          # 論文データと検索履歴の保存場所
│   ├── analyzed_papers.csv            # 分析済み論文DB（21列日本語フォーマット）
│   └── search_history.csv             # 検索履歴
├── docs/                              # ドキュメント関連
├── outputs/                           # 生成されたレポートや翻訳ファイルの保存場所
├── reports/                           # 分析レポートの保存場所
└── src/                               # ソースコード
    ├── analysis/                      # 論文分析関連モジュール
    ├── core/                          # コアロジック（設定、モデル、ワークフロー等）
    ├── registry/                      # 論文データベース管理モジュール
    ├── search/                        # 論文検索関連モジュール
    ├── translation/                   # 翻訳関連モジュール
    ├── ui/                            # UI関連モジュール
    ├── utils/                         # ユーティリティ関数
    └── workflow/                      # LangGraphのノード定義
        └── nodes/                     # 各処理ステップのノード
```

## クイックスタート

### 1. 環境セットアップ

```bash
# Python 3.11以上が必要
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. Vertex AI の設定

Google Cloud Vertex AI を使用するため、認証設定が必要です：

```bash
# Google Cloud CLIをインストール後
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 3. 実行方法

#### 🌐 Streamlit UI（推奨）

```bash
# ブラウザベースのUIを起動
streamlit run research_ui.py

# ブラウザで http://localhost:8501 にアクセス
```

**UI機能**:
- 検索タブ: 論文検索・分析を視覚的に実行
- データベースタブ: 分析済み論文の閲覧・管理（21項目表示）
- 統計タブ: 研究活動の統計情報表示
- エクスポート機能: Excel形式でのデータ出力

#### 🔍 論文検索・分析（メイン機能）

```bash
# 基本的な使い方
python cli_app.py search "AIエージェントの評価"

# 重複回避付き分析（推奨）
python cli_app.py search "AIエージェントの評価" --skip-analyzed

# オプション指定
python cli_app.py search "AIエージェントの評価" --depth moderate --papers 10

# 全オプション
python cli_app.py search "検索クエリ" \
  --depth {shallow|moderate|deep} \  # 分析の深さ
  --papers 数値 \                    # 分析論文数（デフォルト: 5）
  --output ファイルパス \            # レポート保存先
  --skip-analyzed                    # 分析済み論文をスキップ
```

#### 📊 論文データベース管理

```bash
# 分析済み論文一覧表示
python cli_app.py registry list --limit 10

# 論文検索（データベース内）
python cli_app.py registry search "キーワード" --limit 5

# 統計情報表示
python cli_app.py registry stats --days 30

# Excel出力（日本語項目名）
python cli_app.py registry export --output 研究DB.xlsx

# データベーススキーマ選択
python cli_app.py registry --schema min list     # 最小版（8列）
python cli_app.py registry --schema full export  # 完全版（20列）
```

#### 🔄 論文翻訳

```bash
# arXiv論文の日本語翻訳
python cli_app.py translate "https://arxiv.org/abs/2403.12368"

# 学術モード（高品質翻訳）
python cli_app.py translate "論文URL" --academic

# 出力先指定
python cli_app.py translate "論文URL" --output "翻訳結果.html"
```

#### 💾 バックアップ・リセット

```bash
# データベースバックアップ作成
python cli_app.py registry backup --suffix "before_cleanup"

# データベース完全リセット（要確認）
python cli_app.py registry reset

# 強制リセット（確認なし）
python cli_app.py registry reset --force --no-backup

# バックアップから復元
python cli_app.py registry restore database/backup_20250101_120000
```

#### 使用例

```bash
# 浅い分析（概要把握向け）
python cli_app.py search "強化学習の最新動向" --depth shallow --papers 5 --skip-analyzed

# 標準的な分析
python cli_app.py search "LLMの評価手法" --depth moderate --papers 10 --skip-analyzed

# 深い分析（詳細調査向け）
python cli_app.py search "マルチエージェントシステム" --depth deep --papers 15

# 翻訳付き分析
python cli_app.py search "AIエージェント" --translate 1,3,5
```

## 出力ファイル

実行後、以下のファイルが生成されます：

```
reports/
├── arxiv_advanced_report_YYYYMMDD_HHMMSS.md  # 分析レポート（Markdown）
└── arxiv_research_advanced_YYYYMMDD_HHMMSS.json  # 生データ（JSON）

outputs/
├── translations/
│   └── [arxiv_id]_YYYYMMDD_HHMMSS.html      # 翻訳済み論文（HTML）
└── [arxiv_id]_translated.pdf                # ダウンロードした論文PDF

database/
├── analyzed_papers.csv                       # 分析済み論文DB（21列日本語フォーマット）
├── search_history.csv                       # 検索履歴
└── backup_YYYYMMDD_HHMMSS/                  # バックアップフォルダ
    ├── analyzed_papers.csv
    └── search_history.csv
```

## 📊 CSVスキーマ詳細

### 日本語21列フォーマット（v3.1標準）
| 列名 | 説明 | 例 |
|------|------|-----|
| arxiv_id | arXiv ID（v番号込み） | 2403.12368v1 |
| タイトル | 論文タイトル | Characteristic AI Agents via Large Language Models |
| 著者 | 著者（; 区切り） | Xi Wang; Hongliang Dai |
| 公開日 | arXiv公開日 | 2024-03-19 |
| カテゴリ | arXiv カテゴリ | cs.CL / cs.AI |
| 概要JP | 日本語概要（3-5行） | 大規模言語モデルを用いた特性AI... |
| 手法 | 主な技術要素 | Character100, LLM, Wikipedia |
| 結果 | 定量結果ハイライト | F1 49.11%, BLEU-1 46.18% |

| 列名 | 説明 | 文字数制限 |
|------|------|------------|
| arxiv_id | arXiv ID | - |
| タイトル | 論文タイトル | 2000 |
| 著者 | 著者リスト | 2000 |
| 公開日 | arXiv公開日 | - |
| 取得日 | 分析実施日 | - |
| カテゴリ | arXivカテゴリ | 500 |
| キーワード | 抽出キーワード | 500 |
| 処理状態 | 処理ステータス | - |
| 概要JP | 日本語概要 | 2000 |
| 要点_一言 | キャッチフレーズ | 200 |
| 新規性 | 先行研究との差分 | 2000 |
| 手法 | 技術詳細 | 2000 |
| 実験設定 | 検証方法 | 2000 |
| 結果 | 実験結果 | 2000 |
| 考察 | 議論・解釈 | 2000 |
| 今後の課題 | 限界と展望 | 2000 |
| 応用アイデア | 実用化可能性 | 2000 |
| 重要度 | ★1-5評価 | - |
| リンク_pdf | PDF URL | - |
| 落合フォーマットURL | 詳細分析URL | - |
| 備考 | その他メモ | 1000 |

## レポート形式

生成されるレポートは「落合陽一フォーマット」に準拠し、各論文について以下の10項目で分析されます：

1. **これは何か？** - 研究の核心的な問いと扱っている内容
2. **先行研究との比較** - 既存手法との違いや改善点
3. **技術の核心** - 提案手法の技術的詳細とアルゴリズムの革新的な点
4. **検証方法** - 実験設定、評価指標、データセット
5. **実験結果** - 具体的な数値結果と統計的有意性
6. **議論点** - 結果の解釈、研究の限界、今後の課題
7. **実装詳細** - 実装上の工夫、計算効率、スケーラビリティ
8. **なぜ選ばれたか** - この論文が検索結果に含まれた理由
9. **応用可能性** - 実世界への応用可能性と産業界へのインパクト
10. **次に読むべき論文** - 関連する重要な参考文献

## トラブルシューティング

### タイムアウトエラー

```bash
# タイムアウトが発生する場合は論文数を減らす
python cli_app.py "クエリ" --papers 5
```

### Vertex AI 認証エラー

```bash
# 認証情報を確認
gcloud auth application-default print-access-token

# プロジェクトIDを確認
gcloud config get-value project
```

### メモリ不足

```bash
# 並列処理数を制限（環境変数で設定）
export MAX_PARALLEL_PAPERS=3
python cli_app.py "クエリ"
```

## システムアーキテクチャ

### 技術スタック
- **フレームワーク**: LangGraph (ワークフロー管理)
- **LLM**: Google Vertex AI (Gemini 1.5 Pro/Flash)
- **UI**: Streamlit (Webインターフェース)
- **データ処理**: Pandas, CSV (日本語対応)
- **PDF処理**: PyPDF2, ar5iv (HTML変換)
- **構造化出力**: Pydantic (型安全な分析結果)

### ワークフローノード
本システムはLangGraphフレームワーク上に構築され、以下のノードで構成されています：

1. **Planning Node** - 調査計画の策定、クエリ翻訳、キーワード分類
2. **Query Node** - 検索クエリの生成
3. **Search Node** - arXiv検索と関連性スコアリング
4. **Processing Node** - PDF全文の取得と処理
5. **Analysis Node** - Gemini Map-Reduce分析
6. **Report Node** - 落合フォーマットレポート生成
7. **Save Node** - 結果の保存

## 🔧 高度な使い方

### 効率的な研究ワークフロー

1. **初期調査** - UIで高速スクリーニング
   ```bash
   streamlit run research_ui.py
   # ブラウザで検索実行 → データベースタブで確認
   ```

2. **詳細分析** - 重要論文の深掘り
   ```bash
   python cli_app.py search "詳細トピック" --depth deep --papers 5
   # または UIの詳細分析モードを使用
   ```

3. **継続調査** - 定期的なモニタリング
   ```bash
   # 週次実行（重複回避）
   python cli_app.py "継続トピック" --skip-analyzed
   python cli_app.py registry cleanup --days 90  # 古いデータ削除
   ```

### カスタム設定

`src/core/config.py` で詳細設定が可能：

- モデル選択（gemini-1.5-pro, gemini-1.5-flash など）
- APIのタイムアウト設定
- 並列処理数の調整
- 分析深度別のトークンバジェット設定

### 分析パラメータの自動決定ルール

- **サーベイ系クエリ** ("survey", "review"等) → depth=moderate, time_range=all
- **最新動向系クエリ** ("latest", "recent"等) → depth=shallow, time_range=recent (直近2年)
- **実装・詳細系クエリ** ("implementation", "detailed"等) → depth=deep, time_range=all

### プログラマティックな使用

```python
from src.core.workflow import build_advanced_workflow
from src.core.models import AdvancedAgentState

# ワークフローの作成と実行
workflow = build_advanced_workflow()

initial_state = AdvancedAgentState(
    initial_query="AIエージェントの評価",
    research_plan=None,
    search_queries=[],
    found_papers=[],
    analyzed_papers=[],
    final_report="",
    token_budget=30000,
    analysis_mode="advanced_moderate",
    total_tokens_used=0,
    progress_tracker=None
)

result = workflow.invoke(initial_state)
```

## ❓ よくある質問 (FAQ)

### Q: 分析済み論文の重複を避けたい
A: `--skip-analyzed` フラグを使用してください。一度分析した論文は自動的にスキップされます。

### Q: データベースが大きくなりすぎた
A: 定期的にクリーンアップを実行してください：
```bash
python cli_app.py registry cleanup --days 90  # 90日以前のデータを削除
```

### Q: 間違ってデータベースを削除してしまった
A: バックアップから復元できます：
```bash
python cli_app.py registry restore database/backup_YYYYMMDD_HHMMSS
```

### Q: Excel出力でエラーが出る
A: openpyxlをインストールしてください：
```bash
pip install openpyxl
```

### Q: 翻訳品質を向上させたい
A: 学術モードを使用してください：
```bash
python cli_app.py translate "論文URL" --academic
```

### Q: APIエラーが頻発する
A: 論文数を減らすかタイムアウトを調整してください：
```bash
python cli_app.py "クエリ" --papers 3  # 論文数を減らす
```

### Q: 特定のスキーマでExcel出力したい
A: スキーマを指定してください：
```bash
python cli_app.py registry --schema full export --output 完全版.xlsx
python cli_app.py registry --schema min export --output 最小版.xlsx
```

## 🆕 新機能（v3.1）

### 分析品質の大幅改善
- **5倍の分析深度**: Ochiai分析で15,000文字まで処理（従来3,000文字）
- **4倍の詳細保存**: 各分析フィールド2,000文字まで保存（従来500文字）
- **完全な実験結果**: 数値データ、統計的有意性、詳細な考察を保持

### Streamlit UI
- **ビジュアル検索**: パラメータ設定が直感的
- **リアルタイム進捗**: 分析状況をプログレスバーで表示
- **統合データベース管理**: 21列すべての情報を閲覧・検索可能
- **ワンクリックエクスポート**: Excel形式での即座のデータ出力

## 🆕 新機能（v3.0）

### 論文データベース管理システム
- **重複回避**: 分析済み論文の自動検出とスキップ
- **日本語CSV**: 実用的な日本語項目名での管理
- **スキーマ選択**: 用途に応じた最小版（8列）・完全版（20列）
- **Excel出力**: 日本語ヘッダー対応、チーム共有に最適

### arXiv論文翻訳機能
- **HTML翻訳**: LaTeXML（ar5iv）を活用した高品質翻訳
- **学術モード**: 数式・図表保持、専門用語適切翻訳
- **構造保持**: 論文構造を維持した読みやすい出力

### バックアップ・復元システム
- **自動バックアップ**: 危険操作前の自動保護
- **安全なリセット**: 確認プロンプト付きデータベース初期化
- **簡単復元**: タイムスタンプ付きバックアップからの復旧

## ⚠️ 注意事項

- 大量の論文を処理する場合、Vertex AI の API 制限に注意
- PDF のダウンロードには時間がかかる場合があります
- 一部の論文は著作権の関係でダウンロードできない場合があります
- Windows環境では一部文字化けが発生する場合がありますが、データ自体は正常です
- データベースリセットは不可逆操作です。必要に応じてバックアップを作成してください
- v3.1では分析テキスト量が増加したため、API使用量が増える可能性があります

## ライセンス

MIT License

## 貢献

Issues や Pull Requests は歓迎します。

## 関連情報

- [Vertex AI ドキュメント](https://cloud.google.com/vertex-ai/docs)
- [arXiv API](https://arxiv.org/help/api)
- [LangGraph](https://github.com/langchain-ai/langgraph)
