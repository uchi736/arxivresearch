# arXiv Research Agent - Claude Code 向けプロジェクト情報

## プロジェクト概要
arXiv論文を検索・分析するAIエージェントシステム。日本語クエリに対応。

## 重要な注意事項
- **Vertex AI (Gemini)** を使用中。プロジェクトID: `gen-lang-client-0613431636`
- **タイムアウト**: LLM呼び出しは40秒以上かかることがある
- **エンコーディング**: Windows環境のため、絵文字は使用禁止

## よく使うコマンド
```bash
# 重複スキップ付き分析（推奨・効率的）
python cli_app.py search "研究クエリ" --skip-analyzed

# 分析済み論文管理
python cli_app.py registry list --limit 10
python cli_app.py registry search "キーワード" --limit 5
python cli_app.py registry stats --days 30

# 簡易デモ実行（LangGraphなし、高速）
python simple_demo.py "AIエージェントの評価"

# メインアプリ（LangGraph使用）
python cli_app.py search "研究クエリ"
```

## 主要ファイル
- `simple_demo.py` - デバッグ用の簡易版（推奨）
- `src/registry/` - CSV論文管理システム（新機能）
- `database/analyzed_papers.csv` - 分析済み論文DB
- `src/analysis/ochiai_structured_analyzer.py` - Pydantic構造化出力
- `src/analysis/gemini_map_reduce_analyzer.py` - Gemini分析
- `src/core/research_planner.py` - 研究計画生成（40秒程度かかる）

## デバッグのヒント
1. **タイムアウト対策**: `simple_demo.py` や限定テストから始める
2. **エラー対策**: 構造化出力（Pydantic）を使用してJSON解析エラーを防ぐ
3. **検索改善**: 英語翻訳されたクエリを優先的に使用

## 既知の問題と対策
- **LangGraphタイムアウト**: `simple_demo.py` を使用
- **Gemini JSON解析エラー**: `OchiaiStructuredAnalyzer` で解決済み
- **関連性スコア超過**: 10.0でキャップ済み
- **文字化け**: Windows CP932エンコーディング問題 → 絵文字を避ける

# Claude Code Instructions

## 🧪 テスト実装の必須要件

### すべての実装には必ず動作テストを含める

**重要**: 機能実装や改善を行った場合、必ず以下の形式で動作テストを作成し、実行結果を報告すること。

### テストスクリプトの構造

```python
#!/usr/bin/env python3
"""
[機能名]の動作テストスクリプト
実装した機能が正しく動作するか確認します
"""
import unittest
import time
import sys
import os

class Test[機能名](unittest.TestCase):
    """[機能名]の動作テスト"""
    
    def setUp(self):
        """テスト前の準備"""
        self.test_start_time = time.time()
        
    def tearDown(self):
        """テスト後の処理とパフォーマンス計測"""
        elapsed = time.time() - self.test_start_time
        print(f"  [TIME] Test completed in {elapsed:.2f}s")
    
    def test_1_基本機能(self):
        """基本的な動作の確認"""
        print("\n[TEST] Testing [具体的な機能]...")
        # 実装
        # アサーション
        print("  [OK] [機能名] is working correctly")
    
    def test_2_エラーハンドリング(self):
        """エラー処理の確認"""
        # エラーケースのテスト
        
    def test_3_パフォーマンス(self):
        """パフォーマンスの測定"""
        # 処理時間の計測と検証

if __name__ == "__main__":
    print("=" * 80)
    print("[機能名] Test Suite")
    print("=" * 80)
    
    # テスト実行
    suite = unittest.TestLoader().loadTestsFromTestCase(Test[機能名])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 結果サマリー
    if result.wasSuccessful():
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[FAILED] {len(result.failures)} failures, {len(result.errors)} errors")
```

### テスト要件

1. **定量的な検証**
   - 処理時間の測定
   - エラー率の確認
   - メモリ使用量（必要に応じて）

2. **Before/After比較**
   - 改善前後のパフォーマンス比較
   - 機能の正確性の確認

3. **エラーケース**
   - 異常入力への対応
   - エラーハンドリングの確認

### テスト結果の報告形式

```
[TEST REPORT]
================================================================================
機能: [実装した機能名]
結果: [SUCCESS/FAILED]
パフォーマンス: [測定結果]
改善率: [改善前後の比較]
================================================================================
```

## 📝 ファイル管理ルール

### 一時ファイルの管理

1. **テストファイルは実行後に削除**
   ```python
   # ❌ 悪い例
   Write("test_something.py")
   Bash("python test_something.py")
   
   # ✅ 良い例
   Write("test_something.py")
   Bash("python test_something.py")
   Bash("rm test_something.py")  # または後でまとめて削除
   ```

2. **正式なテストは`tests/`ディレクトリに配置**
   ```python
   Write("tests/test_new_feature.py")  # 永続的なテスト
   ```

3. **セッション終了時は必ずクリーンアップ**
   ```bash
   make clean  # または手動で一時ファイルを削除
   ```

## 📁 プロジェクト構造の維持

### ディレクトリ構成
```
project/
├── src/           # ソースコード（変更時は慎重に）
├── tests/         # 正式なテストコード
├── outputs/       # 出力ファイル（自動整理対象）
├── reports/       # レポートファイル
└── *.py          # ルートの一時ファイル（削除対象）
```

### ファイル命名規則
- テスト: `test_[機能名]_[タイムスタンプ].py` または `test_[機能名].py`
- デモ: `demo_[機能名].py`
- 一時ファイル: `temp_*.py` または `tmp_*.py`

## ⚠️ Windows環境での注意事項

### エンコーディング
- **絵文字禁止**: CP932エンコーディングエラーを避ける
- 代替表現:
  - ✅ → [OK]
  - ❌ → [FAILED]
  - 🧪 → [TEST]
  - 📊 → [STATS]
  - ⏱️ → [TIME]

### パス表記
- バックスラッシュ(`\`)ではなくスラッシュ(`/`)を使用
- または raw string (`r"path\to\file"`)

## 🔧 実装チェックリスト

### 必須項目
- [ ] 機能の実装
- [ ] ユニットテストの作成
- [ ] 統合テストの実行
- [ ] パフォーマンス測定
- [ ] エラーハンドリング
- [ ] ログ出力（printではなくlogger使用）
- [ ] **動作テストの実装と実行**

### 推奨事項
- [ ] パフォーマンス最適化
- [ ] メモリ効率の考慮
- [ ] 並列処理の活用
- [ ] キャッシング戦略

## 📈 継続的な改善

### レビューサイクル
1. 実装 → テスト → 測定 → 改善
2. 定量的な結果に基づく判断
3. Before/Afterの比較

### パフォーマンス目標
- API呼び出し: < 30秒/リクエスト
- PDF処理: < 10秒/ファイル
- 全体処理: < 3分/クエリ

---

**これらのガイドラインに従って、すべての実装に適切なテストを含めること。**