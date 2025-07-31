# arXiv Research Agent

論文検索・分析・レポート生成を自動化するAIエージェント

## 特徴

- 🚀 **軽量版**: PDF翻訳機能を除いた高速バージョン
- 📚 **論文検索**: arXivから関連論文を自動検索
- 🔍 **詳細分析**: LLMによる深い論文分析
- 📊 **レポート生成**: 包括的な研究レポートを自動生成
- ⚡ **高速処理**: 翻訳ステップをスキップして大幅高速化

## 必要環境

- Python 3.8+
- Google Cloud Project (Vertex AI有効)
- 環境変数設定

## インストール

1. リポジトリをクローン
```bash
git clone https://github.com/uchi736/arxivresearch.git
cd arxivresearch
```

2. 仮想環境作成・有効化
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux  
source .venv/bin/activate
```

3. 依存関係インストール
```bash
pip install -r requirements.txt
```

4. 環境変数設定
`.env`ファイルを作成し、以下を設定：
```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
```

## 使用方法

### Streamlitアプリ起動
```bash
streamlit run app.py
```

ブラウザで http://localhost:8502 にアクセス

### 基本的な使い方

1. **検索キーワード入力**: 調査したいトピックを入力
2. **分析の深さ選択**: 簡易/標準/詳細から選択
3. **検索・分析開始**: ボタンクリックで自動実行
4. **結果確認**: 論文一覧とレポートを確認

## 機能

### 軽量版（現在）
- ✅ 論文検索
- ✅ 論文分析  
- ✅ レポート生成
- ❌ PDF翻訳（除外）

### 完全版（復元可能）
- ✅ 論文検索
- ✅ 論文分析
- ✅ レポート生成
- ✅ PDF翻訳

## 完全版への復元

PDF翻訳機能付きの完全版に戻すには：

```bash
# ワークフローを復元
cp backup/workflow_with_translation.py src/core/workflow.py

# アプリを復元
cp backup/app_with_translation.py app.py
```

## 設定

主要な設定は `src/core/config.py` で管理：

- `model_name`: 使用するLLMモデル（デフォルト: gemini-2.5-flash）
- `use_vertex_ai`: Vertex AI使用フラグ（デフォルト: True）
- `vertex_ai_location`: リージョン（デフォルト: asia-northeast1）
- `pdf_translation_rate_limit`: API制限（デフォルト: 15回/分）

## ディレクトリ構造

```
arxivresearch/
├── app.py                 # メインStreamlitアプリ
├── requirements.txt       # 依存関係
├── src/                   # ソースコード
│   ├── core/             # コア機能
│   ├── analysis/         # 分析ノード
│   ├── search/           # 検索・RAG
│   ├── ui/               # UI関連
│   └── utils/            # ユーティリティ
├── backup/               # 完全版バックアップ
└── README.md
```

## トラブルシューティング

### よくある問題

1. **Vertex AI初期化エラー**
   - Google Cloud認証を確認
   - プロジェクトIDが正しいか確認

2. **モジュールインポートエラー**  
   - 仮想環境が有効化されているか確認
   - `pip install -r requirements.txt` を再実行

3. **処理が遅い**
   - 軽量版を使用（翻訳機能なし）
   - トークン予算を調整

## 開発者向け

### テスト実行
```bash
python test_quick.py  # 基本機能テスト
```

### API最適化
- バッチ処理によるAPI呼び出し削減
- レート制限による安定化
- エラーハンドリング強化

## ライセンス

MIT License

## 貢献

Issue・PRを歓迎します。

## 更新履歴

- v1.0 - 軽量版リリース（PDF翻訳機能除外）
- v0.9 - PDF翻訳バッチ処理最適化
- v0.8 - Vertex AI対応
- v0.7 - 初期版リリース