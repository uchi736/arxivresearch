# arXiv Research Agent v3.0

研究者が特定の研究トピックに関する最新の学術動向を効率的に把握するための支援ツールです。ユーザーのクエリに基づき、arXivから関連論文を自動的に検索・分析し、構造化された「落合陽一フォーマット」のレポートを生成します。

## 主な機能

- 🔍 **多言語対応検索**: 日本語クエリを高精度に英語翻訳し、技術用語辞書を活用した適切なキーワードでarXiv検索
- 📊 **高度な関連性スコアリング**: 独自の重み付けロジックで0〜10点の範囲で論文を評価・選別
- 📝 **Map-Reduce全文分析**: Gemini長文コンテキストを活用したMap-Reduce方式による詳細分析
- 📋 **落合フォーマット出力**: 10項目の構造化された分析結果（これは何か？、先行研究との比較、技術の核心など）
- 🚀 **並列処理**: 複数論文のPDFを並列ダウンロード・処理で高速化
- 🎯 **動的パラメータ調整**: クエリ内容に応じて分析深度と検索期間を自動決定

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

#### CLI版（推奨）

```bash
# 基本的な使い方
python cli_app.py "AIエージェントの評価"

# オプション指定
python cli_app.py "AIエージェントの評価" --depth moderate --papers 10

# 全オプション
python cli_app.py "検索クエリ" \
  --depth {shallow|moderate|deep} \  # 分析の深さ
  --papers 数値 \                    # 分析論文数（デフォルト: 5）
  --output ファイルパス              # レポート保存先
```

#### 使用例

```bash
# 浅い分析（概要把握向け）
python cli_app.py "強化学習の最新動向" --depth shallow --papers 5

# 標準的な分析
python cli_app.py "LLMの評価手法" --depth moderate --papers 10

# 深い分析（詳細調査向け）
python cli_app.py "マルチエージェントシステム" --depth deep --papers 15
```

## 出力ファイル

実行後、以下のファイルが生成されます：

```
reports/
├── arxiv_advanced_report_YYYYMMDD_HHMMSS.md  # 分析レポート（Markdown）
└── arxiv_research_advanced_YYYYMMDD_HHMMSS.json  # 生データ（JSON）

outputs/
└── [arxiv_id]_translated.pdf  # ダウンロードした論文PDF
```

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

本システムはLangGraphフレームワーク上に構築され、以下のノードで構成されています：

1. **Planning Node** - 調査計画の策定、クエリ翻訳、キーワード分類
2. **Query Node** - 検索クエリの生成
3. **Search Node** - arXiv検索と関連性スコアリング
4. **Processing Node** - PDF全文の取得と処理
5. **Analysis Node** - Gemini Map-Reduce分析
6. **Report Node** - 落合フォーマットレポート生成
7. **Save Node** - 結果の保存

## 高度な使い方

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

## 注意事項

- 大量の論文を処理する場合、Vertex AI の API 制限に注意
- PDF のダウンロードには時間がかかる場合があります
- 一部の論文は著作権の関係でダウンロードできない場合があります

## ライセンス

MIT License

## 貢献

Issues や Pull Requests は歓迎します。

## 関連情報

- [Vertex AI ドキュメント](https://cloud.google.com/vertex-ai/docs)
- [arXiv API](https://arxiv.org/help/api)
- [LangGraph](https://github.com/langchain-ai/langgraph)
