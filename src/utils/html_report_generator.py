"""
HTML形式のレポート生成ユーティリティ
Markdownレポートを美しいHTMLに変換します
"""

import markdown
from typing import Dict, List
from datetime import datetime


class HTMLReportGenerator:
    """HTML形式のレポート生成クラス"""
    
    def __init__(self):
        """初期化"""
        self.md = markdown.Markdown(extensions=[
            'extra',  # テーブル、フェンスコードブロックなど
            'codehilite',  # コードハイライト
            'toc',  # 目次
            'nl2br',  # 改行をbrタグに変換
            'sane_lists'  # より良いリスト処理
        ])
    
    def generate_html_report(self, markdown_content: str, query: str, timestamp: str = None) -> str:
        """
        Markdownコンテンツから完全なHTMLレポートを生成
        
        Args:
            markdown_content: Markdown形式のレポート内容
            query: 検索クエリ
            timestamp: タイムスタンプ（省略時は現在時刻）
            
        Returns:
            完全なHTML文書
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        
        # MarkdownをHTMLに変換
        html_content = self.md.convert(markdown_content)
        
        # HTMLテンプレート
        html_template = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>arXiv論文調査レポート - {query}</title>
    <style>
        body {{
            font-family: 'Segoe UI', 'Yu Gothic', 'Meiryo', sans-serif;
            line-height: 1.8;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-left: 10px;
            border-left: 4px solid #3498db;
        }}
        
        h3 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        
        h4 {{
            color: #7f8c8d;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        
        ul, ol {{
            margin-bottom: 20px;
            padding-left: 30px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        
        a {{
            color: #3498db;
            text-decoration: none;
            transition: color 0.3s;
        }}
        
        a:hover {{
            color: #2980b9;
            text-decoration: underline;
        }}
        
        code {{
            background-color: #f7f7f7;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
        }}
        
        pre {{
            background-color: #f7f7f7;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin-bottom: 20px;
        }}
        
        blockquote {{
            border-left: 4px solid #bdc3c7;
            padding-left: 20px;
            margin: 20px 0;
            color: #7f8c8d;
            font-style: italic;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 40px 0;
        }}
        
        .header-info {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
            font-size: 0.9em;
        }}
        
        .paper-section {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border: 1px solid #e9ecef;
        }}
        
        .ochiai-section {{
            margin-bottom: 20px;
            padding-left: 20px;
        }}
        
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        
        .toc h2 {{
            margin-top: 0;
            font-size: 1.2em;
            color: #34495e;
        }}
        
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        
        .toc li {{
            margin-bottom: 5px;
        }}
        
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        
        @media print {{
            body {{
                background-color: white;
            }}
            
            .container {{
                box-shadow: none;
                padding: 0;
            }}
            
            .header-info {{
                background-color: #f8f9fa;
            }}
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            
            .container {{
                padding: 20px;
            }}
            
            h1 {{
                font-size: 1.5em;
            }}
            
            h2 {{
                font-size: 1.3em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-info">
            <strong>生成日時:</strong> {timestamp}<br>
            <strong>検索クエリ:</strong> {query}
        </div>
        
        {html_content}
        
        <div class="footer">
            <p>このレポートはarXiv Research Agentによって自動生成されました。</p>
        </div>
    </div>
</body>
</html>"""
        
        return html_template
    
    def save_html_report(self, markdown_content: str, query: str, filename: str, timestamp: str = None):
        """
        HTMLレポートをファイルに保存
        
        Args:
            markdown_content: Markdown形式のレポート内容
            query: 検索クエリ
            filename: 保存するファイル名
            timestamp: タイムスタンプ（省略時は現在時刻）
        """
        html_content = self.generate_html_report(markdown_content, query, timestamp)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)