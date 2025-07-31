"""
LangGraph nodes for paper analysis workflow

This module contains all the node functions for the analysis workflow.
"""

import json
import os
import requests
import tempfile
from datetime import datetime
from typing import Dict, List

from langchain.prompts import ChatPromptTemplate
from src.core.models import (
    AdvancedAgentState, PaperMemory, ResearchPlan, SearchQuery,
    OchiaiFormatAdvanced, OCHIAI_SECTION_MAPPING
)
from src.core.config import get_model_config, get_analysis_config, create_llm_model
from src.search.rag_system import PaperRAGSystem
from src.analysis.paper_processor import extract_fulltext_advanced, update_chunk_page_numbers
from src.search.arxiv_search import search_arxiv_papers
from src.utils.pdf_translator import PDFTranslatorWithReportLab
from src.core.progress_tracker import ProgressTracker, StepStatus

# Initialize models - will be done lazily in functions
model = None
creative_model = None

def get_model():
    """Get or create the main model"""
    global model
    if model is None:
        model = create_llm_model()
    return model

def get_creative_model():
    """Get or create the creative model"""
    global creative_model
    if creative_model is None:
        creative_model = create_llm_model(temperature=0.7)
    return creative_model

# Initialize RAG system - will be done lazily
rag_system = None
paper_memories: Dict[str, PaperMemory] = {}

def get_rag_system():
    """Get or create the RAG system"""
    global rag_system
    if rag_system is None:
        rag_system = PaperRAGSystem()
    return rag_system

# Analysis prompt template
analyze_with_memory_prompt = ChatPromptTemplate.from_template(
    """あなたは論文分析の専門家です。
    
与えられた論文の部分的な情報から、落合フォーマットの特定の質問に答えてください。

質問: {question}
必要なセクション: {required_sections}

関連するテキスト:
{relevant_chunks}

既に抽出された主張:
{existing_claims}

以下の形式で回答してください:
1. 質問への直接的な回答
2. 根拠となる証拠（ページ番号付き）
3. 不足している情報があれば指摘

注意: 与えられた情報のみに基づいて回答し、推測は避けてください。"""
)


# Helper functions for progress tracking
def get_progress_tracker(state: AdvancedAgentState) -> ProgressTracker:
    """Get or create progress tracker from state"""
    if state.get("progress_tracker") is None:
        tracker = ProgressTracker()
    else:
        # Reconstruct from dict
        tracker = ProgressTracker()
        tracker_data = state["progress_tracker"]
        tracker.start_time = tracker_data.get("start_time", tracker.start_time)
        tracker.is_complete = tracker_data.get("is_complete", False)
        tracker.error_occurred = tracker_data.get("error_occurred", False)
        # Restore step states
        for step_name, step_data in tracker_data.get("steps", {}).items():
            if step_name in tracker.steps:
                step = tracker.steps[step_name]
                step.status = StepStatus(step_data.get("status", "pending"))
                step.start_time = step_data.get("start_time")
                step.end_time = step_data.get("end_time")
                step.current_item = step_data.get("current_item")
                step.total_items = step_data.get("total_items", 0)
                step.completed_items = step_data.get("completed_items", 0)
                step.error_message = step_data.get("error_message")
                step.details = step_data.get("details", {})
    return tracker


def save_progress_tracker(tracker: ProgressTracker) -> Dict:
    """Convert progress tracker to dict for state storage"""
    return {
        "start_time": tracker.start_time,
        "is_complete": tracker.is_complete,
        "error_occurred": tracker.error_occurred,
        "steps": {
            name: {
                "status": step.status.value,
                "start_time": step.start_time,
                "end_time": step.end_time,
                "current_item": step.current_item,
                "total_items": step.total_items,
                "completed_items": step.completed_items,
                "error_message": step.error_message,
                "details": step.details
            } for name, step in tracker.steps.items()
        }
    }


def plan_research_advanced_node(state: AdvancedAgentState):
    """Advanced research planning node"""
    print("--- 高度な調査計画を策定中 ---")
    
    # Get current model config
    from src.core.config import get_model_config
    model_config = get_model_config()
    print(f"[DEBUG] Current model config - use_vertex_ai: {model_config.use_vertex_ai}")
    print(f"[DEBUG] Model name: {model_config.model_name}")
    print(f"[DEBUG] Location: {model_config.vertex_ai_location}")
    print(f"[DEBUG] Project: {model_config.vertex_ai_project}")
    
    # Ensure model uses current config
    global model
    model = None  # Reset to force recreation with current config
    print("[DEBUG] Model will be created with current config")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    tracker.start_step("研究計画策定")
    
    try:
        print("[DEBUG] Creating plan prompt...")
        plan_prompt = ChatPromptTemplate.from_template(
            """ユーザーのarXiv論文調査リクエストを分析し、詳細な調査計画を立ててください。
            
リクエスト: {query}

以下をJSON形式で回答してください:
{{
  "main_topic": "メイントピック",
  "sub_topics": ["サブトピック1", "サブトピック2", "サブトピック3"],
  "num_papers": 3,
  "focus_areas": ["注目分野1", "注目分野2"],
  "full_text_analysis": true,
  "analysis_depth": "moderate"
}}

注意:
- sub_topicsは3-5個
- num_papersは1以上の整数
- analysis_depthは "shallow", "moderate", "deep" のいずれか
- shallow: 要旨のみで十分
- moderate: 重要セクションの抽出
- deep: 全セクション詳細分析"""
        )
        
        # Generate plan using regular model call
        print(f"[DEBUG] Invoking model with query: {state['initial_query'][:50]}...")
        import time
        invoke_start = time.time()
        try:
            formatted_prompt = plan_prompt.format(query=state["initial_query"])
            print("[DEBUG] Prompt formatted successfully")
            
            # Create model with timeout
            llm = get_model()
            print(f"[DEBUG] Model created: {type(llm).__name__}")
            
            response = llm.invoke(formatted_prompt)
            invoke_time = time.time() - invoke_start
            print(f"[DEBUG] Model invocation completed in {invoke_time:.1f}s")
        except Exception as e:
            invoke_time = time.time() - invoke_start
            print(f"[ERROR] Model invocation failed after {invoke_time:.1f}s: {e}")
            raise
        
        # Parse the JSON response
        import json
        try:
            # Extract JSON from response content
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            plan_data = json.loads(content.strip())
            
            # Create ResearchPlan object
            plan = ResearchPlan.parse_obj(plan_data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing plan JSON: {e}")
            # Create default plan
            plan = ResearchPlan(
                main_topic=state["initial_query"],
                sub_topics=["General research", "Related work", "Applications"],
                num_papers=10,
                focus_areas=["Key concepts", "Methods"],
                full_text_analysis=True,
                analysis_depth="moderate"
            )
        
        # Set token budget based on analysis depth
        analysis_config = get_analysis_config()
        if plan.analysis_depth == "shallow":
            token_budget = analysis_config.token_budget_shallow
        elif plan.analysis_depth == "moderate":
            token_budget = analysis_config.token_budget_moderate
        else:  # deep
            token_budget = analysis_config.token_budget_deep
        
        # Complete the step
        tracker.complete_step("研究計画策定", {
            "main_topic": plan.main_topic,
            "subtopics": len(plan.sub_topics),
            "analysis_depth": plan.analysis_depth
        })
        
        return {
            "research_plan": plan.dict(),
            "token_budget": token_budget,
            "progress_tracker": save_progress_tracker(tracker)
        }
    
    except Exception as e:
        import traceback
        print("\n--- EXCEPTION in plan_research_advanced_node ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Initial query: {state.get('initial_query', 'N/A')}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("--- END EXCEPTION ---\n")
        
        tracker.error_step("研究計画策定", str(e))
        
        # Create a default plan to continue the workflow
        default_plan = ResearchPlan(
            main_topic=state.get("initial_query", "Research topic"),
            sub_topics=["General overview", "Related work", "Applications"],
            num_papers=10,
            focus_areas=["Key concepts", "Methods"],
            full_text_analysis=True,
            analysis_depth="moderate"
        )
        
        return {
            "research_plan": default_plan.dict(),
            "token_budget": state.get("token_budget", 30000),
            "progress_tracker": save_progress_tracker(tracker),
            "analysis_mode": "advanced_moderate"
        }


def generate_queries_node(state: AdvancedAgentState):
    """Generate search queries node"""
    print("--- 検索クエリを生成中 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    tracker.start_step("検索クエリ生成")
    
    try:
        if state.get("research_plan") is None:
            raise ValueError("Research plan is None - likely an error in plan_research_advanced_node")
        plan = ResearchPlan.parse_obj(state["research_plan"])
        
        generate_queries_prompt = ChatPromptTemplate.from_template(
            """以下の調査計画に基づいて、arXiv検索用のクエリをJSON形式で生成してください。

調査計画:
- メイントピック: {main_topic}
- サブトピック: {sub_topics}
- 注目分野: {focus_areas}

各トピックに対して効果的な検索クエリを作成してください。
重要: 
- キーワードは2-3個に限定し、シンプルにしてください
- 全クエリの合計で{num_papers}件以内の論文を取得するよう、max_resultsを調整してください
- あまり具体的すぎないキーワードを使用してください

出力は以下のJSON形式に従ってください:
{{
  "queries": [
    {{
      "keywords": ["keyword1", "keyword2"],
      "max_results": 1,
      "sort_by": "relevance",
      "category": "cs.AI"
    }}
  ]
}}
"""
        )
        
        response = get_model().invoke(
            generate_queries_prompt.format(
                main_topic=plan.main_topic,
                sub_topics=", ".join(plan.sub_topics),
                focus_areas=", ".join(plan.focus_areas),
                num_papers=plan.num_papers
            )
        )
        
        # Extract JSON from response
        json_str = response.content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:-3].strip()
        
        queries_data = json.loads(json_str)
        queries = [SearchQuery.parse_obj(q) for q in queries_data["queries"]]
        
        # Complete the step
        tracker.complete_step("検索クエリ生成", {
            "total_queries": len(queries),
            "topics": [q.keywords[0] if q.keywords else "" for q in queries]
        })
        
        return {
            "search_queries": [q.dict() for q in queries],
            "progress_tracker": save_progress_tracker(tracker)
        }
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  クエリ生成のJSONパースに失敗: {e}")
        # Fallback to simple query
        queries = [{
            "keywords": [plan.main_topic], 
            "max_results": 10, 
            "sort_by": "relevance", 
            "category": None
        }]
        
        tracker.complete_step("検索クエリ生成", {
            "total_queries": 1,
            "fallback": True,
            "error": str(e)
        })
        
        return {
            "search_queries": queries,
            "progress_tracker": save_progress_tracker(tracker)
        }
    
    except Exception as e:
        import traceback
        print(f"\n--- EXCEPTION in generate_queries_node ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Research plan: {state.get('research_plan', 'N/A')}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("--- END EXCEPTION ---\n")
        
        # Create minimal fallback query
        fallback_query = [{
            "keywords": [state.get("initial_query", "research")], 
            "max_results": 10, 
            "sort_by": "relevance", 
            "category": None
        }]
        
        tracker.error_step("検索クエリ生成", str(e))
        
        return {
            "search_queries": fallback_query,
            "progress_tracker": save_progress_tracker(tracker)
        }


def search_papers_node(state: AdvancedAgentState):
    """Search papers node"""
    print("--- arXivで論文を検索中 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    tracker.start_step("論文検索", {
        "total_items": len(state["search_queries"])
    })
    
    all_papers = []
    
    for i, query_dict in enumerate(state["search_queries"]):
        try:
            query = SearchQuery.parse_obj(query_dict)
            query_str = ' '.join(query.keywords)
            print(f"検索中: {query_str}")
            
            # Update progress
            tracker.update_step("論文検索", 
                              f"検索中: {query_str}",
                              {"completed_items": i, "found": len(all_papers)})
            
            papers = search_arxiv_papers(query)
            all_papers.extend(papers)
            print(f"  → {len(papers)}件の論文を取得")
        except Exception as e:
            print(f"  → エラー: {e}")
            continue
    
    # Remove duplicates
    unique_papers = []
    seen_ids = set()
    for paper in all_papers:
        if paper.arxiv_id not in seen_ids:
            seen_ids.add(paper.arxiv_id)
            unique_papers.append(paper)
    
    # Limit to research plan's num_papers if we got too many
    if state.get("research_plan"):
        plan = ResearchPlan.parse_obj(state["research_plan"]) 
        if len(unique_papers) > plan.num_papers:
            print(f"取得論文数を{plan.num_papers}件に制限します")
            unique_papers = unique_papers[:plan.num_papers]
    
    # Complete the step
    tracker.complete_step("論文検索", {
        "total_papers": len(unique_papers),
        "queries_processed": len(state["search_queries"])
    })
    
    print(f"--- {len(unique_papers)}件の論文を発見 ---")
    return {
        "found_papers": [p.dict() for p in unique_papers],
        "progress_tracker": save_progress_tracker(tracker)
    }


def translate_pdfs_node(state: AdvancedAgentState):
    """Translate PDFs to Japanese"""
    print("--- PDF翻訳を開始 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    
    translator = PDFTranslatorWithReportLab()
    
    # Limit to 5 papers max
    papers_to_translate = state["found_papers"][:5]
    
    # Start the step
    tracker.start_step("論文翻訳", {
        "total_items": len(papers_to_translate)
    })
    
    for i, paper in enumerate(papers_to_translate):
        print(f"  翻訳中 ({i+1}/{len(papers_to_translate)}): {paper['title']}")
        
        # Update progress
        tracker.update_step("論文翻訳",
                          f"翻訳中: {paper['title'][:50]}...",
                          {"completed_items": i})
        
        try:
            # Download PDF to temp file
            response = requests.get(paper["pdf_url"], timeout=30)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name

            # Execute translation
            output_filename = f"{paper['arxiv_id'].replace('.', '_')}_translated.pdf"
            translator.translate_pdf(tmp_file_path, output_filename)
            
            os.unlink(tmp_file_path)
            print(f"  → 翻訳済みPDFを 'outputs/{output_filename}' に保存しました")

        except Exception as e:
            print(f"  → 翻訳エラー: {e}")
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            continue
    
    # Complete the step
    tracker.complete_step("論文翻訳", {
        "translated_papers": len(papers_to_translate)
    })
    
    return {"progress_tracker": save_progress_tracker(tracker)}


def advanced_fulltext_processing_node(state: AdvancedAgentState):
    """Advanced full text processing node"""
    print("--- 高度な全文処理を開始 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    
    paper_memories.clear()
    
    # Adjust number of papers based on analysis depth
    analysis_config = get_analysis_config()
    depth = state["research_plan"].get("analysis_depth", "moderate")
    # Get max papers based on depth
    if depth == "shallow":
        max_papers = analysis_config.max_papers_shallow
    elif depth == "moderate":
        max_papers = analysis_config.max_papers_moderate
    else:
        max_papers = analysis_config.max_papers_deep
    
    papers_to_process = state["found_papers"][:max_papers]
    
    # Start the step
    tracker.start_step("フルテキスト処理", {
        "total_items": len(papers_to_process)
    })
    
    for i, paper in enumerate(papers_to_process):
        print(f"\n処理中 ({i+1}/{len(papers_to_process)}): {paper['title']}")
        
        # Update progress
        tracker.update_step("フルテキスト処理",
                          f"処理中: {paper['title'][:50]}...",
                          {"completed_items": i})
        
        try:
            # Download and extract PDF
            pdf_data = extract_fulltext_advanced(paper["pdf_url"])
            if not pdf_data:
                print("  → PDF処理失敗、スキップ")
                continue
            
            # Extract sections
            sections = get_rag_system().extract_sections(pdf_data["full_text"])
            print(f"  → {len(sections)}個のセクションを検出")
            
            # Chunk by sections
            chunks = get_rag_system().chunk_by_section(sections)
            print(f"  → {len(chunks)}個のチャンクを生成")
            
            # Compute embeddings
            get_rag_system().compute_embeddings(chunks)
            
            # Update page numbers
            update_chunk_page_numbers(chunks, pdf_data["page_texts"])
            
            # Save to memory
            paper_memory = PaperMemory(
                paper_id=paper["arxiv_id"],
                sections={},
                claims=[],
                chunks=chunks,
                coverage_map={section: 0.0 for section in sections.keys()},
                token_budget_used=0
            )
            
            paper_memories[paper["arxiv_id"]] = paper_memory
            
        except Exception as e:
            print(f"  → 処理エラー: {e}")
            continue
    
    # Complete the step
    tracker.complete_step("フルテキスト処理", {
        "processed_papers": len(paper_memories)
    })
    
    return {"progress_tracker": save_progress_tracker(tracker)}


def ochiai_focused_analysis_node(state: AdvancedAgentState):
    """Ochiai format focused analysis node"""
    print("--- 落合フォーマット特化分析を開始 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    tracker.start_step("論文分析", {
        "total_items": len(paper_memories)
    })
    
    analyzed_papers = []
    
    for idx, (paper_id, memory) in enumerate(paper_memories.items()):
        print(f"\n分析中: {paper_id}")
        
        # Update progress
        tracker.update_step("論文分析",
                          f"分析中: {paper_id}",
                          {"completed_items": idx})
        
        # Get corresponding paper metadata
        paper_meta = next((p for p in state["found_papers"] if p["arxiv_id"] == paper_id), None)
        if not paper_meta:
            continue
        
        ochiai_results = {}
        evidence_map = {}
        total_tokens_used = 0
        
        # Process each Ochiai format question
        for ochiai_key, mapping in OCHIAI_SECTION_MAPPING.items():
            print(f"  → {ochiai_key}を分析中...")
            
            # Get relevant chunks from required sections
            relevant_chunks = []
            for section in mapping["sections"]:
                section_chunks = [c for c in memory.chunks if section in c.section.lower()]
                relevant_chunks.extend(section_chunks)
            
            if not relevant_chunks:
                # Fall back to all chunks
                relevant_chunks = memory.chunks
            
            # Determine number of chunks based on token budget
            token_budget_for_question = int(state["token_budget"] * mapping["weight"])
            k_chunks = min(len(relevant_chunks), max(3, token_budget_for_question // 1000))
            
            # Select diverse chunks using MMR
            query = f"{ochiai_key} {paper_meta['title']}"
            selected_chunks = get_rag_system().mmr_select(query, relevant_chunks, k=k_chunks)
            
            # Extract claims from selected chunks
            claims = []
            for chunk in selected_chunks:
                chunk_claims = get_rag_system().extract_claims(chunk, get_model())
                claims.extend(chunk_claims)
                memory.claims.extend(chunk_claims)
            
            # Answer the question
            relevant_text = "\n\n".join([f"[Page {c.page_num}, {c.section}]\n{c.text[:500]}" 
                                        for c in selected_chunks])
            existing_claims_text = "\n".join([f"- {claim.claim} (p.{claim.page_num})" 
                                            for claim in claims])
            
            response = get_model().invoke(
                analyze_with_memory_prompt.format(
                    question=ochiai_key,
                    required_sections=", ".join(mapping["sections"]),
                    relevant_chunks=relevant_text,
                    existing_claims=existing_claims_text
                )
            )
            
            ochiai_results[ochiai_key] = response.content
            evidence_map[ochiai_key] = [(c.page_num, c.chunk_id) for c in selected_chunks]
            
            # Update coverage
            for chunk in selected_chunks:
                memory.coverage_map[chunk.section] = min(1.0, 
                    memory.coverage_map.get(chunk.section, 0) + 0.2)
        
        # Add remaining fields
        ochiai_results["next_papers"] = ["References not extracted - see PDF"]
        ochiai_results["why_selected"] = f"Query: {state['initial_query']}"
        ochiai_results["applicability"] = "Based on the analysis above"
        
        # Create Ochiai format result
        ochiai_format = OchiaiFormatAdvanced(
            what_is_it=ochiai_results.get("what_is_it", ""),
            comparison_with_prior_work=ochiai_results.get("comparison_with_prior_work", ""),
            key_technique=ochiai_results.get("key_technique", ""),
            validation_method=ochiai_results.get("validation_method", ""),
            experimental_results=ochiai_results.get("experimental_results", ""),
            discussion_points=ochiai_results.get("discussion_points", ""),
            next_papers=ochiai_results.get("next_papers", []),
            implementation_details=ochiai_results.get("implementation_details", ""),
            why_selected=ochiai_results.get("why_selected", ""),
            applicability=ochiai_results.get("applicability", ""),
            evidence_map=evidence_map
        )
        
        analyzed_paper = {
            "metadata": paper_meta,
            "analysis": ochiai_format.dict(),
            "analysis_type": "advanced_rag",
            "coverage": memory.coverage_map,
            "tokens_used": total_tokens_used,
            "num_claims": len(memory.claims)
        }
        
        analyzed_papers.append(analyzed_paper)
        
        print(f"  → 完了: {len(memory.claims)}個の主張を抽出, "
              f"カバレッジ: {sum(memory.coverage_map.values())/len(memory.coverage_map)*100:.1f}%")
    
    # Complete the step
    tracker.complete_step("論文分析", {
        "analyzed_papers": len(analyzed_papers),
        "total_claims": sum(len(p.get("claims", [])) for p in analyzed_papers)
    })
    
    return {
        "analyzed_papers": analyzed_papers,
        "progress_tracker": save_progress_tracker(tracker)
    }


def generate_advanced_report_node(state: AdvancedAgentState):
    """Generate advanced report node"""
    print("--- 高度なレポートを生成中 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    tracker.start_step("レポート生成")
    
    # Map-Reduce approach for integration
    section_summaries = {
        "overview": [],
        "methods": [],
        "results": [],
        "insights": []
    }
    
    for paper in state["analyzed_papers"]:
        metadata = paper["metadata"]
        analysis = paper["analysis"]
        
        # Distribute information to sections
        section_summaries["overview"].append(
            f"**{metadata['title']}**\n{analysis['what_is_it']}"
        )
        section_summaries["methods"].append(
            f"**{metadata['title']}**\n{analysis['key_technique']}\n{analysis['implementation_details']}"
        )
        section_summaries["results"].append(
            f"**{metadata['title']}**\n{analysis['experimental_results']}"
        )
        section_summaries["insights"].append(
            f"**{metadata['title']}**\n{analysis['discussion_points']}"
        )
    
    # Summarize each section
    report_sections = {}
    for section_name, contents in section_summaries.items():
        if contents:
            section_prompt = f"""
            以下の情報を統合して、{section_name}セクションの要約を作成してください:
            
            {chr(10).join(contents)}
            
            簡潔で洞察に富んだ要約を作成してください。
            """
            
            summary = get_creative_model().invoke(section_prompt)
            report_sections[section_name] = summary.content
    
    # Build final report
    final_report = f"""# arXiv論文調査レポート（高度解析版）

## 調査概要
- **クエリ**: {state['initial_query']}
- **分析モード**: {state['analysis_mode']}
- **分析論文数**: {len(state['analyzed_papers'])}
- **総トークン使用量**: {sum(p['tokens_used'] for p in state['analyzed_papers'])}

## 研究概要
{report_sections.get('overview', '')}

## 手法・技術詳細
{report_sections.get('methods', '')}

## 実験結果・評価
{report_sections.get('results', '')}

## 洞察・今後の展望
{report_sections.get('insights', '')}

## 分析詳細
"""
    
    # Add detailed information for each paper
    for paper in state["analyzed_papers"]:
        metadata = paper["metadata"]
        coverage = paper.get("coverage", {})
        
        final_report += f"""
### {metadata['title']}
- **著者**: {', '.join(metadata['authors'])}
- **arXiv**: [{metadata['arxiv_id']}]({metadata['pdf_url']})
- **分析カバレッジ**: {sum(coverage.values())/max(len(coverage), 1)*100:.1f}%
- **抽出された主張数**: {paper.get('num_claims', 0)}
"""
    
    # Complete the step
    tracker.complete_step("レポート生成", {
        "report_length": len(final_report)
    })
    
    return {
        "final_report": final_report,
        "progress_tracker": save_progress_tracker(tracker)
    }


def save_results_node(state: AdvancedAgentState):
    """Save results node"""
    print("--- 結果を保存中 ---")
    
    # Get progress tracker
    tracker = get_progress_tracker(state)
    tracker.start_step("結果保存")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/arxiv_research_advanced_{timestamp}.json"
    
    os.makedirs("reports", exist_ok=True)
    
    total_tokens = sum(p.get('tokens_used', 0) for p in state["analyzed_papers"])
    
    # Prepare save data
    save_data = {
        "query": state["initial_query"],
        "research_plan": state["research_plan"],
        "analysis_mode": state["analysis_mode"],
        "analyzed_papers": state["analyzed_papers"],
        "report": state["final_report"],
        "timestamp": timestamp,
        "total_tokens_used": total_tokens
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    # Also save the report as markdown
    report_filename = f"reports/arxiv_advanced_report_{timestamp}.md"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(state["final_report"])
    
    print(f"--- 結果を {filename} に保存しました ---")
    print(f"--- レポートを {report_filename} に保存しました ---")
    
    # Complete the step
    tracker.complete_step("結果保存", {
        "json_file": filename,
        "report_file": report_filename
    })
    
    # Mark workflow as complete
    tracker.is_complete = True
    
    return {
        "total_tokens_used": total_tokens,
        "progress_tracker": save_progress_tracker(tracker)
    }
