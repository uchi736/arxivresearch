# arXiv Research Agent - Makefile
# 効率的なコマンド実行のため

.PHONY: help setup demo demo-en test clean lint

# デフォルトターゲット
help:
	@echo "使用可能なコマンド:"
	@echo "  make setup      - 初期セットアップ"
	@echo "  make demo       - 日本語デモ実行（AIエージェントの評価）"
	@echo "  make demo-en    - 英語デモ実行"
	@echo "  make test       - 簡易テスト実行"
	@echo "  make test-quick - 高速テスト（Gemini分析のみ）"
	@echo "  make clean      - 一時ファイル削除"
	@echo "  make run QUERY='クエリ' - 任意のクエリで実行"

# 初期セットアップ
setup:
	@echo "Setting up environment..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Setup complete!"

# 日本語デモ（デフォルト）
demo:
	@echo "Running Japanese demo..."
	python simple_demo.py "AIエージェントの評価"

# 英語デモ
demo-en:
	@echo "Running English demo..."
	python simple_demo.py "evaluation of AI agents"

# 簡易テスト
test:
	@echo "Running limited test..."
	python test_ai_agent_limited.py

# 高速テスト（Gemini分析のみ）
test-quick:
	@echo "Running quick Gemini test..."
	python test_gemini_quick.py

# カスタムクエリ実行
run:
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make run QUERY='your query here'"; \
		exit 1; \
	fi
	python simple_demo.py "$(QUERY)"

# クリーンアップ
clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__ src/__pycache__ src/*/__pycache__
	@rm -f *.pyc src/*.pyc src/*/*.pyc
	@rm -f demo_report_*.md limited_demo_report_*.md
	@echo "Cleanup complete!"

# Lint実行
lint:
	@echo "Running linter..."
	@python -m pylint src/ --disable=C0103,C0114,C0115,C0116,R0903,R0913,W0613 || true
	@echo "Lint complete!"

# 並列実行デモ（高速化）
demo-parallel:
	@echo "Running parallel analysis demo..."
	python -c "import concurrent.futures; print('Parallel execution would go here')"