#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional ArXiv Translator - 高品質な学術論文翻訳システム
完全なUTF-8対応と高度な文書構造解析を実装
"""
import os
import re
import json
import time
import codecs
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag, Comment
import requests

from src.core.config import create_llm_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentSection:
    """文書セクションの構造"""
    level: int
    title: str
    number: str
    content: List[Any] = field(default_factory=list)
    subsections: List['DocumentSection'] = field(default_factory=list)
    figures: List[Dict] = field(default_factory=list)
    tables: List[Dict] = field(default_factory=list)
    equations: List[Dict] = field(default_factory=list)
    

@dataclass
class DocumentStructure:
    """文書全体の構造"""
    title: str
    authors: List[str]
    abstract: str
    sections: List[DocumentSection]
    references: List[Dict]
    terminology: Dict[str, str]
    

class ProfessionalArxivTranslator:
    """
    プロフェッショナルグレードのarXiv論文翻訳器
    """
    
    def __init__(self):
        self.model = create_llm_model()
        self.encoding = 'utf-8'
        
        # 基本的な学術用語辞書（UTF-8で正しくエンコード）
        self.base_terminology = {
            "agent": "エージェント",
            "evaluation": "評価",
            "benchmark": "ベンチマーク",
            "framework": "フレームワーク",
            "methodology": "手法",
            "performance": "性能",
            "metrics": "指標",
            "baseline": "ベースライン",
            "dataset": "データセット",
            "algorithm": "アルゴリズム",
            "architecture": "アーキテクチャ",
            "model": "モデル",
            "training": "学習",
            "validation": "検証",
            "testing": "テスト",
            "accuracy": "精度",
            "efficiency": "効率",
            "scalability": "スケーラビリティ",
            "robustness": "頑健性",
            "optimization": "最適化"
        }
        
        # 保護すべきHTML要素
        self.protected_tags = {
            'math', 'svg', 'img', 'table', 'code', 'pre',
            'script', 'style', 'meta', 'link'
        }
        
        # 保護すべきクラス
        self.protected_classes = {
            'ltx_equation', 'ltx_math', 'ltx_cite', 'ltx_ref',
            'ltx_bibitem', 'ltx_listing', 'ltx_verbatim', 'ltx_code',
            'ltx_figure', 'ltx_table', 'ltx_graphics'
        }
        
        # セクション番号のパターン
        self.section_pattern = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')
        
        logger.info("ProfessionalArxivTranslator initialized with UTF-8 encoding")
        
    def ensure_utf8(self, text: str) -> str:
        """テキストをUTF-8として確実に処理"""
        if isinstance(text, bytes):
            return text.decode('utf-8', errors='replace')
        return str(text)
        
    def normalize_unicode(self, text: str) -> str:
        """Unicode正規化"""
        return unicodedata.normalize('NFC', text)
        
    def arxiv_to_ar5iv_url(self, arxiv_url: str) -> str:
        """Convert arXiv URL to ar5iv URL"""
        arxiv_id_match = re.search(r'(\d{4}\.\d{4,5})', arxiv_url)
        if not arxiv_id_match:
            raise ValueError(f"Invalid arXiv URL: {arxiv_url}")
        arxiv_id = arxiv_id_match.group(1)
        return f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        
    def check_ar5iv_availability(self, arxiv_url: str) -> bool:
        """Check if paper is available on ar5iv"""
        try:
            ar5iv_url = self.arxiv_to_ar5iv_url(arxiv_url)
            response = requests.head(ar5iv_url, timeout=10)
            return response.status_code == 200
        except:
            return False
            
    def fetch_arxiv_html(self, arxiv_url: str) -> str:
        """ar5ivからHTMLを取得（UTF-8保証）"""
        ar5iv_url = self.arxiv_to_ar5iv_url(arxiv_url)
        
        logger.info(f"Fetching from {ar5iv_url}")
        
        response = requests.get(ar5iv_url, timeout=30)
        response.raise_for_status()
        
        # Force UTF-8 encoding
        response.encoding = 'utf-8'
        content = response.text
        
        # Ensure it's properly decoded
        return self.ensure_utf8(content)
        
    def analyze_document_structure(self, soup: BeautifulSoup) -> DocumentStructure:
        """文書構造を詳細に解析"""
        logger.info("Analyzing document structure...")
        
        # Extract title
        title_elem = soup.find('h1', class_='ltx_title')
        title = title_elem.get_text(strip=True) if title_elem else "Untitled"
        
        # Extract authors
        authors = []
        for author in soup.find_all('span', class_='ltx_personname'):
            authors.append(author.get_text(strip=True))
            
        # Extract abstract
        abstract = ""
        abstract_div = soup.find('div', class_='ltx_abstract')
        if abstract_div:
            abstract_p = abstract_div.find('p')
            if abstract_p:
                abstract = abstract_p.get_text(strip=True)
                
        # Build section hierarchy
        sections = self._build_section_hierarchy(soup)
        
        # Extract references
        references = self._extract_references(soup)
        
        # Build initial terminology
        terminology = self._extract_terminology(soup)
        
        return DocumentStructure(
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            references=references,
            terminology=terminology
        )
        
    def _build_section_hierarchy(self, soup: BeautifulSoup) -> List[DocumentSection]:
        """セクション階層を構築"""
        sections = []
        current_section = None
        section_stack = []
        
        # Find all headers
        headers = soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6'])
        
        for header in headers:
            level = int(header.name[1])  # h2 -> 2
            text = header.get_text(strip=True)
            
            # Extract section number if present
            match = self.section_pattern.match(text)
            if match:
                number = match.group(1)
                title = match.group(2)
            else:
                number = ""
                title = text
                
            section = DocumentSection(
                level=level,
                title=title,
                number=number
            )
            
            # Find appropriate parent
            while section_stack and section_stack[-1].level >= level:
                section_stack.pop()
                
            if section_stack:
                section_stack[-1].subsections.append(section)
            else:
                sections.append(section)
                
            section_stack.append(section)
            
            # Collect content until next header
            self._collect_section_content(header, section)
            
        return sections
        
    def _collect_section_content(self, header: Tag, section: DocumentSection):
        """セクションのコンテンツを収集"""
        current = header.next_sibling
        
        while current and not (isinstance(current, Tag) and current.name in ['h2', 'h3', 'h4', 'h5', 'h6']):
            if isinstance(current, Tag):
                # Check for figures
                if 'ltx_figure' in current.get('class', []):
                    figure_info = self._extract_figure_info(current)
                    if figure_info:
                        section.figures.append(figure_info)
                        
                # Check for tables
                elif 'ltx_table' in current.get('class', []):
                    table_info = self._extract_table_info(current)
                    if table_info:
                        section.tables.append(table_info)
                        
                # Check for equations
                elif 'ltx_equation' in current.get('class', []):
                    eq_info = self._extract_equation_info(current)
                    if eq_info:
                        section.equations.append(eq_info)
                        
                # Regular content
                else:
                    section.content.append(current)
                    
            current = current.next_sibling if hasattr(current, 'next_sibling') else None
            
    def _extract_figure_info(self, figure_elem: Tag) -> Optional[Dict]:
        """図の情報を抽出"""
        info = {'type': 'figure'}
        
        # Extract figure number
        caption = figure_elem.find('figcaption')
        if caption:
            info['caption'] = caption.get_text(strip=True)
            
        # Extract image info
        img = figure_elem.find('img')
        if img:
            info['src'] = img.get('src', '')
            info['alt'] = img.get('alt', '')
            
        return info if 'caption' in info else None
        
    def _extract_table_info(self, table_elem: Tag) -> Optional[Dict]:
        """表の情報を抽出"""
        info = {'type': 'table'}
        
        # Extract caption
        caption = table_elem.find('caption')
        if caption:
            info['caption'] = caption.get_text(strip=True)
            
        # Keep table structure
        info['element'] = table_elem
        
        return info
        
    def _extract_equation_info(self, eq_elem: Tag) -> Optional[Dict]:
        """数式の情報を抽出"""
        info = {'type': 'equation'}
        
        # Extract equation number
        eq_num = eq_elem.find(class_='ltx_tag')
        if eq_num:
            info['number'] = eq_num.get_text(strip=True)
            
        # Keep equation element
        info['element'] = eq_elem
        
        return info
        
    def _extract_references(self, soup: BeautifulSoup) -> List[Dict]:
        """参考文献を抽出"""
        references = []
        
        for bib_item in soup.find_all(class_='ltx_bibitem'):
            ref = {
                'id': bib_item.get('id', ''),
                'text': bib_item.get_text(strip=True)
            }
            references.append(ref)
            
        return references
        
    def _extract_terminology(self, soup: BeautifulSoup) -> Dict[str, str]:
        """専門用語を抽出して辞書を構築"""
        terminology = self.base_terminology.copy()
        
        # Extract from abstract and introduction
        # This is a simplified version - could be enhanced with NLP
        
        return terminology
        
    def translate_text_with_placeholders(self, text: str, placeholders: Dict[str, str]) -> str:
        """
        プレースホルダーをLLMから保護し、テキスト部分のみを翻訳する（区切り翻訳）。
        翻訳の前後でプレースホルダーの数が一致することを検証する。
        """
        if not placeholders:
            return self.translate_with_context(text, {})

        # プレースホルダーを長い順にソートして、誤分割を防止
        sorted_keys = sorted(placeholders.keys(), key=len, reverse=True)
        
        # 正規表現パターンを作成
        pattern = re.compile("(" + "|".join(map(re.escape, sorted_keys)) + ")")
        
        # テキストをプレースホルダーで分割
        parts = pattern.split(text)
        
        translated_parts = []
        for part in parts:
            if part in placeholders:
                # プレースホルダーはそのまま追加
                translated_parts.append(part)
            elif part.strip():
                # テキスト部分のみを翻訳
                translated_parts.append(self.translate_with_context(part, {}))
            else:
                # 空白なども保持
                translated_parts.append(part)
        
        result = "".join(translated_parts)
        
        # --- 整合性検証 ---
        # 翻訳後のプレースホルダーの数を数える
        post_translation_count = sum(result.count(key) for key in sorted_keys)
        
        if len(sorted_keys) != post_translation_count:
            logger.warning(
                f"Placeholder count mismatch after translation. "
                f"Before: {len(sorted_keys)}, After: {post_translation_count}. "
                "Falling back to original text to prevent corruption."
            )
            # 整合性が取れない場合は、破壊を防ぐために元テキストを返す
            return text
            
        return result
    
    def translate_with_context(self, text: str, context: Dict) -> str:
        """コンテキストを考慮した翻訳"""
        # Build context prompt
        context_info = []
        
        if 'section_title' in context:
            context_info.append(f"セクション: {context['section_title']}")
            
        if 'terminology' in context:
            context_info.append("専門用語の統一:")
            for eng, jpn in list(context['terminology'].items())[:10]:
                context_info.append(f"  {eng} → {jpn}")
                
        if 'previous_paragraph' in context:
            context_info.append(f"前の段落: {context['previous_paragraph'][:100]}...")
            
        prompt = f"""以下の学術論文のテキストを日本語に翻訳してください。

【コンテキスト】
{chr(10).join(context_info)}

【翻訳ルール】
1. 学術的で正確な翻訳
2. 専門用語は一貫性を保つ
3. 文体は「である調」で統一
4. 数式記号、引用番号、図表番号は変更しない
5. 文章の論理構造を保持

【原文】
{text}

【翻訳】"""

        try:
            response = self.model.invoke(prompt)
            return self.ensure_utf8(response.content)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
            
    def _restore_placeholders_in_string(self, text: str, placeholders: Dict[str, str]) -> str:
        """翻訳済み文字列内のプレースホルダーを元のHTMLに戻す"""
        for key, val in placeholders.items():
            text = text.replace(key, val)
        return text
        
    def protect_special_nodes(self, elem: Tag) -> Dict[str, str]:
        """特殊ノードを保護してプレースホルダーに置換"""
        # 重複処理防止チェック
        if hasattr(elem, '_placeholders_protected'):
            return getattr(elem, '_placeholders_dict', {})
        
        placeholders = {}
        
        # 数式、参照、引用などを保護
        special_selectors = [
            'math', 'svg', '[class*="ltx_equation"]', '[class*="ltx_math"]',
            '[class*="ltx_cite"]', '[class*="ltx_ref"]', 'img'
        ]
        
        placeholder_counter = 0
        for selector in special_selectors:
            for special_elem in elem.select(selector):
                placeholder = f"@@@SPECIAL_{placeholder_counter}@@@"
                placeholders[placeholder] = str(special_elem)
                
                # プレースホルダーで置換
                placeholder_span = elem.new_tag('span')
                placeholder_span.string = placeholder
                special_elem.replace_with(placeholder_span)
                
                placeholder_counter += 1
        
        # フラグを設定
        elem._placeholders_protected = True
        elem._placeholders_dict = placeholders
        
        return placeholders if placeholders else {}
    
    def handle_forest_structures(self, soup: BeautifulSoup):
        """
        Forest構造を安全に処理する。
        - wrap() を使って元の位置を保持する。
        - 既に処理済みの場合は二重に処理しない（冪等性）。
        """
        import re
        
        # `find_all`はリストを返すので、イテレート中にDOMを変更しても安全
        for text_node in soup.find_all(text=re.compile(r'\{forest\}|for tree=')):
            container = text_node.parent
            
            # 親がない、または既に処理済みの場合はスキップ
            if not container or container.find_parent('details', class_='forest-wrapper'):
                continue

            # 元の要素を<details>でラップして位置を保持
            details = soup.new_tag('details', attrs={'class': 'forest-wrapper'})
            summary = soup.new_tag('summary')
            summary.string = "[図: 論文構造のツリー図（クリックで展開）]"
            
            # 元のコンテナをラップし、その後にsummaryを追加
            container.wrap(details)
            details.insert(0, summary)
            
            # 元のコンテンツをさらにdivでラップして、スタイル適用や識別を容易にする
            original_content_div = soup.new_tag('div', attrs={'class': 'forest-original'})
            container.wrap(original_content_div)
    
    def _looks_japanese(self, s: str) -> bool:
        """文字列に日本語（ひらがな、カタカナ、漢字）が含まれるかチェック"""
        return bool(re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', s))

    def apply_translations(self, soup: BeautifulSoup, doc_structure: DocumentStructure):
        """翻訳結果をHTMLに適用 - DOM構造を保持"""
        logger.info("--- Applying all translations to DOM ---")
        
        # Update title
        title_elem = soup.find('h1', class_='ltx_title')
        if title_elem:
            logger.debug(f"Applying title translation: '{doc_structure.title}'")
            self._replace_text_nodes_only(title_elem, doc_structure.title)
            
        # Update abstract
        abstract_div = soup.find('div', class_='ltx_abstract')
        if abstract_div and doc_structure.abstract:
            abstract_p = abstract_div.find('p')
            if abstract_p:
                logger.debug(f"Applying abstract translation: '{doc_structure.abstract[:100]}...'")
                self._replace_text_nodes_only(abstract_p, doc_structure.abstract)
                
        # Update sections
        all_paragraphs = soup.find_all('p')
        logger.info(f"Found {len(all_paragraphs)} total paragraphs to process.")

        for i, p_element in enumerate(all_paragraphs):
            original_text = p_element.get_text(strip=True)
            if not original_text or len(original_text) < 30:
                continue

            logger.debug(f"Processing paragraph {i+1}/{len(all_paragraphs)}: '{original_text[:70]}...'")
            
            # 1. 保護
            placeholders = self.protect_special_nodes(p_element)
            text_to_translate = p_element.get_text(strip=True)

            # 2. 翻訳
            translated_text = self.translate_text_with_placeholders(text_to_translate, placeholders)

            # 3. 復元と適用
            # プレースホルダーを翻訳済みテキストに戻す
            restored_text = self._restore_placeholders_in_string(translated_text, placeholders)
            
            # 新しいsoupオブジェクトにパースして、テキスト以外のノードを復元
            temp_soup = BeautifulSoup(restored_text, 'html.parser')
            
            # 元のp要素の子を全て削除
            p_element.clear()
            
            # 新しい内容を追加
            for content_node in temp_soup.contents:
                p_element.append(content_node.extract())

            # 日本語化検証
            if not self._looks_japanese(p_element.get_text(" ", strip=True)):
                logger.warning(f"Paragraph {i+1} may not be translated correctly.")

        logger.info("--- Finished applying translations ---")
    
    # インライン要素とスキップ要素の定義
    INLINE_TAGS = {'a', 'em', 'i', 'b', 'strong', 'span', 'code', 'kbd', 'samp', 'sub', 'sup'}
    SKIP_TAGS = {'math', 'code', 'pre', 'script', 'style'}
    SKIP_CLASSES = {'ltx_Math', 'ltx_ref', 'ltx_cite', 'ltx_equation', 'ltx_eref'}

    def _extract_text_runs(self, elem: Tag) -> List:
        """要素から翻訳対象のテキストランを抽出"""
        runs = []
        for t in elem.find_all(string=True, recursive=True):
            p = t.parent
            if not p:
                continue
            if p.name in self.SKIP_TAGS or (set(p.get('class', [])) & self.SKIP_CLASSES):
                continue
            if t.strip():
                runs.append(t)
        return runs

    def _choose_host_node(self, elem: Tag, runs: List):
        """テキスト配置のホストノードを選択（堅牢版）"""
        BLOCK_TAGS = {'p', 'div', 'section', 'article', 'li', 'td', 'th', 'body', 'main'}
        
        # 1. elem直下のテキストノードを最優先
        for t in runs:
            if t.parent == elem:
                return t
                
        # 2. ブロック要素の直接の子であるテキストノードを優先
        for t in runs:
            if t.parent and (t.parent.name in BLOCK_TAGS):
                return t

        # 3. インライン要素の子ではないテキストノードを優先
        for t in runs:
            if t.parent.name not in self.INLINE_TAGS:
                return t
                
        # 4. フォールバックとして最初のランを返す
        return runs[0] if runs else None

    def _replace_text_nodes_only(self, elem: Tag, new_text: str):
        """
        子要素の構造を保持しつつ、可視テキストだけを差し替える堅牢な実装。
        StopIterationを回避し、インライン要素の破壊を防ぐ。
        """
        if elem is None:
            return

        # 1. 子ノードが全くない場合は、新しいテキストを追加して終了 (StopIteration回避)
        if not elem.contents:
            elem.append(NavigableString(new_text))
            return

        # 2. 翻訳対象となるテキストランを抽出
        runs = self._extract_text_runs(elem)
        
        # 3. テキストランが見つからない場合、テキストを追記して終了
        #    (例: <p><br/></p> のようなケースで、構造を破壊しない)
        if not runs:
            elem.append(NavigableString(new_text))
            return

        # 4. 翻訳テキストを配置するのに最適な「ホスト」ノードを選択
        host = self._choose_host_node(elem, runs)
        
        # 5. ホストノードを翻訳テキストで置換
        host.replace_with(NavigableString(new_text))
        
        # 6. 他のテキストランは空文字列に置換して削除
        for r in runs:
            # host自体や、既に親がいないノードはスキップ
            if r is not host and r.parent:
                try:
                    r.replace_with('')
                except Exception:
                    # ノードが既にデタッチされている場合などは無視
                    pass
    
    def _restore_placeholders(self, elem: Tag, placeholders: Dict[str, str]):
        """プレースホルダーを元の要素に復元"""
        if not placeholders:
            return
            
        for placeholder, original_html in placeholders.items():
            # プレースホルダーを含むspanを検索
            placeholder_spans = elem.find_all('span', string=placeholder)
            for span in placeholder_spans:
                # 元のHTMLを解析して復元
                temp_soup = BeautifulSoup(original_html, 'html.parser')
                if temp_soup.contents:
                    restored_elem = temp_soup.contents[0]
                    span.replace_with(restored_elem)
        
    def add_metadata_and_styles(self, soup: BeautifulSoup):
        """メタデータとスタイルを追加"""
        if not soup.head:
            soup.head = soup.new_tag('head')
            
        # Ensure UTF-8
        # Remove existing charset tags
        for meta in soup.head.find_all('meta', attrs={'charset': True}):
            meta.decompose()
        for meta in soup.head.find_all('meta', attrs={'http-equiv': 'content-type'}):
            meta.decompose()
            
        # Add UTF-8 meta tag
        charset_meta = soup.new_tag('meta', charset='utf-8')
        soup.head.insert(0, charset_meta)
        
        # Add translation metadata
        trans_meta = soup.new_tag('meta', attrs={
            'name': 'translator',
            'content': 'ProfessionalArxivTranslator v2.0'
        })
        soup.head.append(trans_meta)
        
        # Add custom styles
        style = soup.new_tag('style')
        style.string = """
        /* Professional Academic Translation Styles */
        body {
            font-family: 'Noto Serif JP', 'Yu Mincho', serif;
            line-height: 1.8;
            color: #333;
        }
        
        .ltx_p {
            text-align: justify;
            margin-bottom: 1em;
        }
        
        .translation-note {
            background: #f0f8ff;
            border-left: 4px solid #4169e1;
            padding: 10px;
            margin: 20px 0;
            font-style: italic;
        }
        
        .terminology-unified {
            border-bottom: 1px dotted #666;
            cursor: help;
        }
        
        /* Ensure proper encoding display */
        * {
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        """
        soup.head.append(style)
        
    def _default_output_path_from_url(self, arxiv_url: str) -> str:
        """URLからデフォルトの出力ファイル名を生成する"""
        try:
            arxiv_id = to_arxiv_id(arxiv_url)
            return f"{arxiv_id}.html"
        except ValueError:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"translation_{timestamp}.html"

    def _resolve_output_path(self, output_path: Optional[str], arxiv_url: str) -> Path:
        """
        出力パスを絶対Pathに正規化。
        - output_path が None/空 の場合は URL から既定ファイル名を生成
        - ディレクトリが指定された場合（既存ディレクトリ or 拡張子なし等）は既定ファイル名を付与
        """
        if not output_path:
            name = self._default_output_path_from_url(arxiv_url)
            return Path(name).resolve()

        p = Path(output_path).resolve()
        if p.is_dir():
            name = self._default_output_path_from_url(arxiv_url)
            p = p / name
        return p

    def _assert_invariants(self, before_soup: BeautifulSoup, after_soup: BeautifulSoup):
        """翻訳前後の健全性チェック（要素数の一致など）"""
        logger.info("Verifying structural invariants after translation...")
        
        def count_elements(soup, selector):
            return len(soup.select(selector))

        invariants = {
            "Math elements": 'math, span.ltx_Math, [class*="ltx_equation"]',
            "References": 'a.ltx_ref, span.ltx_ref, cite, a[href^="#bib"]'
        }

        all_ok = True
        for name, selector in invariants.items():
            before_count = count_elements(before_soup, selector)
            after_count = count_elements(after_soup, selector)
            if before_count != after_count:
                logger.error(
                    f"Invariant check failed for '{name}'. "
                    f"Count before: {before_count}, Count after: {after_count}"
                )
                all_ok = False
            else:
                logger.info(f"Invariant check passed for '{name}': {after_count} elements.")
        
        if not all_ok:
            raise RuntimeError("Structural integrity check failed. Translation aborted.")

    def translate_arxiv_paper(self, arxiv_url: str, output_path: str) -> bool:
        """メイン翻訳関数"""
        logger.info(f"Starting professional translation of {arxiv_url}")
        start_time = time.time()

        if not output_path:
            try:
                arxiv_id = to_arxiv_id(arxiv_url)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"outputs/translations/{arxiv_id}_{timestamp}.html"
                logger.warning(f"Output path was not provided. Defaulting to: {output_path}")
            except ValueError:
                logger.error("Could not generate default output path due to invalid arXiv URL.")
                return False
        
        try:
            # Fetch and parse HTML
            html_content = self.fetch_arxiv_html(arxiv_url)
            soup = BeautifulSoup(html_content, 'lxml')
            
            # 翻訳前の状態を保存
            original_soup = BeautifulSoup(str(soup), 'lxml')

            # Handle special structures first
            self.handle_forest_structures(soup)
            
            # Analyze structure
            doc_structure = self.analyze_document_structure(soup)
            logger.info(f"Document structure: {len(doc_structure.sections)} sections found")
            
            # Translate title and abstract
            doc_structure.title = self.translate_with_context(
                doc_structure.title,
                {'terminology': doc_structure.terminology}
            )
            
            if doc_structure.abstract:
                doc_structure.abstract = self.translate_with_context(
                    doc_structure.abstract,
                    {'terminology': doc_structure.terminology}
                )
            
            # Apply translations to HTML with DOM preservation
            self.apply_translations(soup, doc_structure)
            
            # Add metadata and styles
            self.add_metadata_and_styles(soup)
            
            # Post-process: natural space handling
            self._improve_text_spacing(soup)
            
            # 健全性チェック
            self._assert_invariants(original_soup, soup)

            # pathlibを使用して安全にパスを解決し、保存
            out_path = self._resolve_output_path(output_path, arxiv_url)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with out_path.open('w', encoding='utf-8', newline='\n') as f:
                f.write(str(soup))
                
            elapsed = time.time() - start_time
            logger.info(f"Professional translation completed in {elapsed:.1f}s")
            logger.info(f"Output saved to {out_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _improve_text_spacing(self, soup: BeautifulSoup):
        """テキストの自然な空白処理を改善"""
        import re
        
        # 段落内のテキストで不自然な空白を修正
        for para in soup.find_all('p'):
            for text_node in para.find_all(text=True):
                if isinstance(text_node, NavigableString):
                    # 複数の空白を1つに
                    normalized = re.sub(r'\s+', ' ', str(text_node))
                    # 句読点前の空白を削除
                    normalized = re.sub(r'\s+([。、，．])', r'\1', normalized)
                    # 改行文字をスペースに
                    normalized = re.sub(r'\n+', ' ', normalized)
                    
                    if normalized != str(text_node):
                        text_node.replace_with(NavigableString(normalized.strip()))
    
    def _verify_element_preservation(self, soup: BeautifulSoup):
        """要素保持の検証"""
        # 数式要素数をカウント
        math_count = len(soup.select('math, span.ltx_Math, [class*="ltx_equation"]'))
        ref_count = len(soup.select('a.ltx_ref, span.ltx_ref, cite, a[href^="#bib"]'))
        
        logger.info(f"Element preservation check - Math: {math_count}, References: {ref_count}")
        
        if math_count == 0:
            logger.warning("No math elements found - may indicate processing issue")
        if ref_count == 0:
            logger.warning("No reference elements found - may indicate processing issue")
            
    def verify_output_encoding(self, output_path: str) -> bool:
        """出力ファイルのエンコーディングを検証"""
        try:
            with codecs.open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Check for common encoding issues
                if '' in content or '?' * 3 in content:
                    logger.warning("Potential encoding issues detected in output")
                    return False
            return True
        except UnicodeDecodeError:
            logger.error("Output file has encoding errors")
            return False

</file_content>

Now that you have the latest state of the file, try the operation again with fewer, more precise SEARCH blocks. For large files especially, it may be prudent to try to limit yourself to <5 SEARCH/REPLACE blocks at a time, then wait for the user to respond with the result of the operation before following up with another replace_in_file call to make additional edits.
(If you run into this error 3 times in a row, you may use the write_to_file tool as a fallback.)
</error><environment_details>
# VSCode Visible Files
src/translation/professional_arxiv_translator.py

# VSCode Open Tabs
outputs/translations/2503.16416_20250805_183500.html
cli_app.py
fixed_translation.html
src/translation/professional_arxiv_translator.py

# Current Time
2025/8/5 午後9:54:52 (Asia/Tokyo, UTC+9:00)

# Context Window Usage
794,095 / 1,048.576K tokens used (76%)

# Current Mode
ACT MODE
</environment_details>

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
src/translation/professional_arxiv_translator.py
- [Pylance Error] Line 863: 式が必要です
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: 式が必要です
- [Pylance Error] Line 866: "(" が閉じられませんでした
- [Pylance Error] Line 867: 式が必要です
- [Pylance Error] Line 878: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 878: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 878: 式が必要です
- [Pylance Error] Line 881: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 881: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 881: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 881: 式が必要です
- [Pylance Error] Line 881: "(" が閉じられませんでした
- [Pylance Error] Line 884: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 885: 式が必要です
- [Pylance Error] Line 887: 式が必要です
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: 式が必要です
- [Pylance Error] Line 893: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 893: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 893: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 893: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 893: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 893: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 893: 式が必要です
- [Pylance Error] Line 895: "[" は閉じられていません
- [Pylance Error] Line 896: "[" は閉じられていません
- [Pylance Error] Line 897: "[" は閉じられていません
- [Pylance Error] Line 898: "[" は閉じられていません
- [Pylance Error] Line 899: "[" は閉じられていません
- [Pylance Error] Line 900: "[" は閉じられていません
- [Pylance Error] Line 901: "[" は閉じられていません
- [Pylance Error] Line 902: "[" は閉じられていません
- [Pylance Error] Line 903: "[" は閉じられていません
- [Pylance Error] Line 904: "[" は閉じられていません
- [Pylance Error] Line 905: "[" は閉じられていません
- [Pylance Error] Line 906: "[" は閉じられていません
- [Pylance Error] Line 907: "[" は閉じられていません
- [Pylance Error] Line 908: "[" は閉じられていません
- [Pylance Error] Line 909: "[" は閉じられていません
- [Pylance Error] Line 910: "[" は閉じられていません
- [Pylance Error] Line 911: "[" は閉じられていません
- [Pylance Error] Line 912: "[" は閉じられていません
- [Pylance Error] Line 913: "[" は閉じられていません
- [Pylance Error] Line 914: "[" は閉じられていません
- [Pylance Error] Line 915: "[" は閉じられていません
- [Pylance Error] Line 916: "[" は閉じられていません
- [Pylance Error] Line 917: "[" は閉じられていません
- [Pylance Error] Line 928: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 928: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 928: 式が必要です
- [Pylance Error] Line 931: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 931: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 931: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 931: 式が必要です
- [Pylance Error] Line 931: "(" が閉じられませんでした
- [Pylance Error] Line 934: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 935: 式が必要です<environment_details>
# VSCode Visible Files
src/translation/professional_arxiv_translator.py

# VSCode Open Tabs
outputs/translations/2503.16416_20250805_183500.html
cli_app.py
fixed_translation.html
src/translation/professional_arxiv_translator.py

# Current Time
2025/8/5 午後9:58:59 (Asia/Tokyo, UTC+9:00)

# Context Window Usage
794,095 / 1,048.576K tokens used (76%)

# Current Mode
ACT MODE
</environment_details>

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.



New problems detected after saving the file:
src/translation/professional_arxiv_translator.py
- [Pylance Error] Line 863: 式が必要です
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 865: 式が必要です
- [Pylance Error] Line 866: "(" が閉じられませんでした
- [Pylance Error] Line 867: 式が必要です
- [Pylance Error] Line 878: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 878: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 878: 式が必要です
- [Pylance Error] Line 881: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 881: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 881: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 881: 式が必要です
- [Pylance Error] Line 881: "(" が閉じられませんでした
- [Pylance Error] Line 884: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 885: 式が必要です
- [Pylance Error] Line 887: 式が必要です
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 889: 式が必要です
- [Pylance Error] Line 893: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 893: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 893: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 893: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 893: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 893: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 893: 式が必要です
- [Pylance Error] Line 895: "[" は閉じられていません
- [Pylance Error] Line 896: "[" は閉じられていません
- [Pylance Error] Line 897: "[" は閉じられていません
- [Pylance Error] Line 898: "[" は閉じられていません
- [Pylance Error] Line 899: "[" は閉じられていません
- [Pylance Error] Line 900: "[" は閉じられていません
- [Pylance Error] Line 901: "[" は閉じられていません
- [Pylance Error] Line 902: "[" は閉じられていません
- [Pylance Error] Line 903: "[" は閉じられていません
- [Pylance Error] Line 904: "[" は閉じられていません
- [Pylance Error] Line 905: "[" は閉じられていません
- [Pylance Error] Line 906: "[" は閉じられていません
- [Pylance Error] Line 907: "[" は閉じられていません
- [Pylance Error] Line 908: "[" は閉じられていません
- [Pylance Error] Line 909: "[" は閉じられていません
- [Pylance Error] Line 910: "[" は閉じられていません
- [Pylance Error] Line 911: "[" は閉じられていません
- [Pylance Error] Line 912: "[" は閉じられていません
- [Pylance Error] Line 913: "[" は閉じられていません
- [Pylance Error] Line 914: "[" は閉じられていません
- [Pylance Error] Line 915: "[" は閉じられていません
- [Pylance Error] Line 916: "[" は閉じられていません
- [Pylance Error] Line 917: "[" は閉じられていません
- [Pylance Error] Line 928: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 928: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 928: 式が必要です
- [Pylance Error] Line 931: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 931: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 931: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 931: 式が必要です
- [Pylance Error] Line 931: "(" が閉じられませんでした
- [Pylance Error] Line 934: ステートメントは改行またはセミコロンで区切る必要があります
- [Pylance Error] Line 935: 式が必要です<environment_details>
# VSCode Visible Files
src/translation/professional_arxiv_translator.py

# VSCode Open Tabs
outputs/translations/2503.16416_20250805_183500.html
cli_app.py
fixed_translation.html
src/translation/professional_arxiv_translator.py

# Current Time
2025/8/5 午後10:00:00 (Asia/Tokyo, UTC+9:00)

# Context Window Usage
794,095 / 1,048.576K tokens used (76%)

# Current Mode
ACT MODE
</environment_details>
