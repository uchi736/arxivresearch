"""
Unified research planning module with multilingual support

This module provides a common interface for creating research plans
that works across both standard and optimized workflows.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from src.core.models import ImprovedResearchPlan, ResearchPlan
from src.core.config import create_llm_model

# Technical term dictionary for Japanese to English translation
TECH_TERMS_DICT = {
    # AI/ML terms
    "機械学習": ["machine learning", "ML"],
    "深層学習": ["deep learning", "DL", "neural networks"],
    "強化学習": ["reinforcement learning", "RL"],
    "自然言語処理": ["natural language processing", "NLP"],
    "画像認識": ["image recognition", "computer vision", "CV"],
    "生成AI": ["generative AI", "generative models"],
    "大規模言語モデル": ["large language models", "LLM"],
    "トランスフォーマー": ["transformer", "transformers"],
    "敵対的生成ネットワーク": ["generative adversarial networks", "GAN"],
    "畳み込みニューラルネットワーク": ["convolutional neural networks", "CNN"],
    "再帰的ニューラルネットワーク": ["recurrent neural networks", "RNN"],
    "注意機構": ["attention mechanism", "attention"],
    "事前学習": ["pre-training", "pretraining"],
    "ファインチューニング": ["fine-tuning", "finetuning"],
    "転移学習": ["transfer learning"],
    
    # AI Agent terms (NEW)
    "AIエージェント": ["AI agent", "artificial intelligence agent", "intelligent agent"],
    "エージェント": ["agent", "AI agent", "software agent"],
    "評価": ["evaluation", "assessment", "benchmarking"],
    "ベンチマーク": ["benchmark", "evaluation benchmark", "standardized test"],
    "性能評価": ["performance evaluation", "agent assessment", "evaluation metric"],
    "会話エージェント": ["conversational agent", "dialogue agent", "chatbot"],
    "自律エージェント": ["autonomous agent", "self-directed agent"],
    "マルチエージェント": ["multi-agent", "multi-agent system", "MAS"],
    "エージェント評価": ["agent evaluation", "AI agent assessment", "agent benchmark"],
    
    # Robotics/Control
    "ロボット工学": ["robotics", "robot engineering"],
    "制御理論": ["control theory", "control systems"],
    "自動運転": ["autonomous driving", "self-driving"],
    "ドローン": ["drone", "UAV", "unmanned aerial vehicle"],
    
    # Data Science
    "データサイエンス": ["data science"],
    "ビッグデータ": ["big data"],
    "データマイニング": ["data mining"],
    "予測分析": ["predictive analytics"],
    "統計学": ["statistics", "statistical analysis"],
    
    # Quantum Computing
    "量子コンピュータ": ["quantum computing", "quantum computer"],
    "量子アルゴリズム": ["quantum algorithms"],
    "量子もつれ": ["quantum entanglement"],
    
    # Other CS terms
    "ブロックチェーン": ["blockchain"],
    "暗号技術": ["cryptography", "encryption"],
    "分散システム": ["distributed systems"],
    "エッジコンピューティング": ["edge computing"],
    "クラウドコンピューティング": ["cloud computing"],
}


class ResearchPlanner:
    """Unified research planner with enhanced capabilities"""
    
    def __init__(self, model=None):
        self.model = model or create_llm_model()
    
    def detect_language(self, query: str) -> str:
        """Detect if query contains Japanese characters"""
        # Check for Japanese characters (Hiragana, Katakana, Kanji)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', query):
            return "ja"
        return "en"
    
    def extract_tech_terms(self, query: str) -> Dict[str, List[str]]:
        """Extract technical terms and their translations from query"""
        found_terms = {}
        query_lower = query.lower()
        
        for ja_term, en_terms in TECH_TERMS_DICT.items():
            if ja_term in query_lower:
                found_terms[ja_term] = en_terms
        
        return found_terms
    
    def determine_num_papers(self, query: str, analysis_depth: str) -> int:
        """Determine optimal number of papers based on query complexity"""
        # Check for survey/review indicators
        survey_keywords = ["サーベイ", "survey", "review", "概要", "overview", "比較", "comparison"]
        is_survey = any(keyword in query.lower() for keyword in survey_keywords)
        
        # Base numbers by depth
        base_numbers = {
            "shallow": 5,
            "moderate": 10,
            "deep": 15
        }
        
        num_papers = base_numbers.get(analysis_depth, 10)
        
        # Adjust for survey requests
        if is_survey:
            num_papers = int(num_papers * 1.5)
        
        # Cap at reasonable limits
        return min(max(num_papers, 3), 30)
    
    def determine_analysis_depth(self, query: str) -> str:
        """Automatically determine analysis depth from query"""
        query_lower = query.lower()
        
        # Shallow indicators
        if any(term in query_lower for term in ["基礎", "入門", "概要", "basics", "introduction", "overview"]):
            return "shallow"
        
        # Deep indicators
        if any(term in query_lower for term in ["詳細", "実装", "詳しく", "深く", "implementation", "detailed", "in-depth"]):
            return "deep"
        
        # Default to moderate
        return "moderate"
    
    def determine_time_range(self, query: str) -> str:
        """Determine time range preference from query"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["最新", "最近", "recent", "latest", "new"]):
            return "recent"
        elif any(term in query_lower for term in ["基礎", "古典", "foundational", "classical", "seminal"]):
            return "foundational"
        
        return "all"
    
    def categorize_keywords(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Categorize keywords into different search strategies"""
        categories = {
            "core": [],
            "technical": [],
            "application": [],
            "method": []
        }
        
        # Simple categorization based on keyword patterns
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if any(term in keyword_lower for term in ["algorithm", "method", "approach", "technique"]):
                categories["method"].append(keyword)
            elif any(term in keyword_lower for term in ["application", "use case", "applied"]):
                categories["application"].append(keyword)
            elif len(keyword.split()) == 1:  # Single words are often core terms
                categories["core"].append(keyword)
            else:
                categories["technical"].append(keyword)
        
        # Ensure core has at least one keyword
        if not categories["core"] and keywords:
            categories["core"].append(keywords[0])
        
        return categories
    
    async def create_research_plan(self, query: str, analysis_mode: str = "moderate") -> ImprovedResearchPlan:
        """Create an improved research plan from user query"""
        
        # Detect language
        query_language = self.detect_language(query)
        
        # Extract technical terms if Japanese
        tech_terms = {}
        if query_language == "ja":
            tech_terms = self.extract_tech_terms(query)
        
        # Determine analysis parameters
        analysis_depth = self.determine_analysis_depth(query) if "moderate" in analysis_mode else analysis_mode.split("_")[-1]
        num_papers = self.determine_num_papers(query, analysis_depth)
        time_range = self.determine_time_range(query)
        
        # Create prompt for LLM
        prompt = self._create_planning_prompt(query, query_language, tech_terms, analysis_depth)
        
        # Get LLM response
        response = await self._get_llm_response(prompt)
        
        # Parse response and create plan
        plan_data = self._parse_llm_response(response, query, query_language, tech_terms)
        
        # Add determined parameters
        plan_data.update({
            "num_papers": num_papers,
            "analysis_depth": analysis_depth,
            "time_range": time_range
        })
        
        return ImprovedResearchPlan(**plan_data)
    
    def create_research_plan_sync(self, query: str, analysis_mode: str = "moderate") -> ImprovedResearchPlan:
        """Synchronous version of create_research_plan"""
        
        # Detect language
        query_language = self.detect_language(query)
        
        # Extract technical terms if Japanese
        tech_terms = {}
        if query_language == "ja":
            tech_terms = self.extract_tech_terms(query)
        
        # Determine analysis parameters
        analysis_depth = self.determine_analysis_depth(query) if "moderate" in analysis_mode else analysis_mode.split("_")[-1]
        num_papers = self.determine_num_papers(query, analysis_depth)
        time_range = self.determine_time_range(query)
        
        # Create prompt for LLM
        prompt = self._create_planning_prompt(query, query_language, tech_terms, analysis_depth)
        
        # Get LLM response (synchronous)
        response = self.model.invoke(prompt)
        
        # Parse response and create plan
        plan_data = self._parse_llm_response(response, query, query_language, tech_terms)
        
        # Add determined parameters
        plan_data.update({
            "num_papers": num_papers,
            "analysis_depth": analysis_depth,
            "time_range": time_range
        })
        
        return ImprovedResearchPlan(**plan_data)
    
    def _create_planning_prompt(self, query: str, language: str, tech_terms: Dict, analysis_depth: str) -> str:
        """Create prompt for research planning"""
        
        tech_terms_str = ""
        if tech_terms:
            tech_terms_str = "\n既に識別された技術用語:\n"
            for ja_term, en_terms in tech_terms.items():
                tech_terms_str += f"- {ja_term}: {', '.join(en_terms)}\n"
        
        # Check if this is an AI agent evaluation query
        is_agent_eval = any(term in query.lower() for term in ["エージェント", "agent", "評価", "evaluation", "assessment"])
        
        domain_guidance = ""
        if is_agent_eval:
            domain_guidance = """

SPECIAL GUIDANCE FOR AI AGENT EVALUATION:
- Focus on AI/ML specific evaluation terms, NOT general "performance metrics"
- Use domain-specific terms like "agent benchmark", "AI evaluation", "LLM assessment"
- Avoid generic terms that could match non-AI domains (e.g., "performance metrics" matches battery papers)
- Include specific agent types: "conversational agent", "LLM agent", "autonomous agent"
- Prioritize arXiv categories: cs.AI, cs.LG, cs.CL, cs.MA
"""
        
        prompt = f"""You are an expert research planner for academic paper searches on arXiv.
Create a comprehensive research plan for the following query.

Query: {query}
Query Language: {language}
Analysis Depth: {analysis_depth}
{tech_terms_str}{domain_guidance}

Please provide a research plan in the following JSON format:
{{
    "translated_query": "English translation of the query (if Japanese) or the original query",
    "main_topic": "Main research topic in English",
    "sub_topics": ["3-5 subtopics in English"],
    "search_keywords": {{
        "core": ["main keywords - be specific to the domain"],
        "technical": ["technical terms - domain-specific"],
        "application": ["application areas"],
        "method": ["methods/algorithms"]
    }},
    "synonyms": {{
        "keyword1": ["synonym1", "synonym2"],
        "keyword2": ["synonym1", "synonym2"]
    }},
    "arxiv_categories": ["cs.AI", "cs.LG", etc.],
    "focus_areas": ["Key areas to focus on in analysis"],
    "plan_reasoning": "Brief explanation of the research strategy"
}}

CRITICAL GUIDELINES:
1. If the query is in Japanese, provide accurate technical translations
2. BE DOMAIN-SPECIFIC - avoid generic terms that match unrelated fields
3. For AI/agent queries: use "agent evaluation", "AI benchmark", NOT "performance metrics"
4. Include both general and specific search terms within the target domain
5. Select appropriate arXiv categories based on the topic
6. Explain your reasoning for the research strategy"""

        return prompt
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM (async)"""
        response = await self.model.ainvoke(prompt)
        return response.content
    
    def _parse_llm_response(self, response, query: str, language: str, tech_terms: Dict) -> Dict:
        """Parse LLM response and create plan data"""
        try:
            # Extract JSON from response
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Find JSON block
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group()
                plan_data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
            
            # Ensure all required fields
            result = {
                "original_query": query,
                "translated_query": plan_data.get("translated_query", query),
                "query_language": language,
                "main_topic": plan_data.get("main_topic", query),
                "sub_topics": plan_data.get("sub_topics", []),
                "search_keywords": plan_data.get("search_keywords", {"core": [query]}),
                "synonyms": plan_data.get("synonyms", {}),
                "arxiv_categories": plan_data.get("arxiv_categories", []),
                "focus_areas": plan_data.get("focus_areas", ["General analysis"]),
                "plan_reasoning": plan_data.get("plan_reasoning", ""),
                "full_text_analysis": True,
                "confidence_score": 0.9
            }
            
            return result
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # Fallback plan
            return {
                "original_query": query,
                "translated_query": query,
                "query_language": language,
                "main_topic": query,
                "sub_topics": [query + " applications", query + " methods"],
                "search_keywords": {"core": [query]},
                "synonyms": {},
                "arxiv_categories": [],
                "focus_areas": ["General analysis"],
                "plan_reasoning": "Fallback plan due to parsing error",
                "full_text_analysis": True,
                "confidence_score": 0.5
            }
    
    def convert_to_legacy_plan(self, improved_plan: ImprovedResearchPlan) -> ResearchPlan:
        """Convert ImprovedResearchPlan to legacy ResearchPlan for compatibility"""
        return ResearchPlan(
            main_topic=improved_plan.main_topic,
            sub_topics=improved_plan.sub_topics,
            num_papers=improved_plan.num_papers,
            focus_areas=improved_plan.focus_areas,
            full_text_analysis=improved_plan.full_text_analysis,
            analysis_depth=improved_plan.analysis_depth
        )