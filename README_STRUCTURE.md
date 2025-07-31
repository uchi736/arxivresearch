# arXiv Research Agent

## プロジェクト構造（本番版）

### メインファイル
- `app.py` - メインのStreamlitアプリケーション
- `requirements.txt` - 依存関係
- `.env` - 環境変数設定

### ソースコード (`src/`)
- `core/` - コア機能（設定、モデル、ワークフロー）
- `analysis/` - 分析ノード
- `search/` - 検索・RAGシステム  
- `ui/` - UI関連
- `utils/` - ユーティリティ

## 使用方法
```bash
streamlit run app.py
```

## 環境設定
`.env`ファイルに以下を設定：
```
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
PROJECT_ID=your-project-id
```