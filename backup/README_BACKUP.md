# バックアップファイル

翻訳機能付きの完全版をバックアップしています。

## ファイル構成

- `workflow_with_translation.py` - 翻訳ノード含むワークフロー
- `app_with_translation.py` - 翻訳機能含むStreamlitアプリ

## 復元方法

翻訳機能を戻したい場合：

```bash
# ワークフローを戻す
cp backup/workflow_with_translation.py src/core/workflow.py

# アプリを戻す  
cp backup/app_with_translation.py app.py
```

## 軽量版との違い

完全版：
- PDF翻訳機能あり
- 処理時間：長い（翻訳処理分）
- API呼び出し：多い

軽量版：
- PDF翻訳機能なし
- 処理時間：短い
- API呼び出し：少ない（検索・分析のみ）