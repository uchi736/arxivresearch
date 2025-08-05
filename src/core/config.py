"""
Configuration management for arXiv Research Agent

Central configuration file to manage API keys, model settings, and other configurations.
"""

import os
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.utils.logger import get_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)


class ModelConfig(BaseModel):
    """Configuration for language models"""
    model_name: str = Field(default="gemini-1.5-flash")
    creative_model_name: str = Field(default="gemini-1.5-pro") # More powerful model for reports
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    creative_temperature: float = Field(default=0.7, ge=0.0, le=2.0) # Higher temp for creative tasks
    google_api_key: Optional[str] = Field(default=None)
    # Vertex AI configuration
    use_vertex_ai: bool = Field(default=True)  # Use Vertex AI by default
    vertex_ai_project: Optional[str] = Field(default=None)
    vertex_ai_location: str = Field(default="asia-northeast1")  # 東京リージョンで動作確認済み
    # 高速分析の設定
    use_fast_analysis: bool = Field(default=True)  # 高速分析モードを有効化
    fast_analysis_model: str = Field(default="gemini-2.5-flash")  # 高速モデル
    
    # PDF翻訳の設定
    pdf_translation_rate_limit: int = Field(default=15)  # API calls per minute
    pdf_max_text_length: int = Field(default=4000)  # Max characters per API call
    pdf_image_max_dimension: int = Field(default=1200)  # Max image dimension in pixels
    pdf_image_quality: int = Field(default=85)  # JPEG quality (1-100)
    pdf_retry_attempts: int = Field(default=3)  # Number of retry attempts
    
    class Config:
        protected_namespaces = ()
    
    def __init__(self, **data):
        """Initialize model with validation"""
        super().__init__(**data)
        
        if self.use_vertex_ai:
            # For Vertex AI, check project configuration
            if not self.vertex_ai_project:
                self.vertex_ai_project = os.getenv("GOOGLE_CLOUD_PROJECT")
                if not self.vertex_ai_project:
                    raise ValueError("GOOGLE_CLOUD_PROJECT not found in environment variables for Vertex AI")
        else:
            # For regular Google AI, check API key
            if not self.google_api_key:
                self.google_api_key = os.getenv("GOOGLE_API_KEY")
                if not self.google_api_key:
                    raise ValueError("Google API Key not found in environment variables")


class AnalysisConfig(BaseModel):
    """Configuration for analysis settings"""
    batch_concurrency: int = Field(default=2, gt=0)  # API concurrency for batch jobs
    analysis_depth: Literal["shallow", "moderate", "deep"] = Field(default="moderate")
    max_papers_shallow: int = Field(default=10, gt=0)
    max_papers_moderate: int = Field(default=5, gt=0)
    max_papers_deep: int = Field(default=3, gt=0)
    token_budget_shallow: int = Field(default=10000, gt=0)
    token_budget_moderate: int = Field(default=30000, gt=0)
    token_budget_deep: int = Field(default=50000, gt=0)


class AppConfig(BaseModel):
    """Main application configuration"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    data_dir: str = Field(default="./arxiv_data")
    outputs_dir: str = Field(default="./outputs")
    reports_dir: str = Field(default="./reports")

    def get_token_budget(self, depth: str) -> int:
        """Get token budget based on analysis depth"""
        if depth == "shallow":
            return self.analysis.token_budget_shallow
        elif depth == "moderate":
            return self.analysis.token_budget_moderate
        else:
            return self.analysis.token_budget_deep
    
    def get_max_papers(self, depth: str) -> int:
        """Get maximum papers to analyze based on depth"""
        if depth == "shallow":
            return self.analysis.max_papers_shallow
        elif depth == "moderate":
            return self.analysis.max_papers_moderate
        else:
            return self.analysis.max_papers_deep


# Global configuration instance
config = AppConfig()

# Convenience functions
def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return config.model

def get_analysis_config() -> AnalysisConfig:
    """Get analysis configuration"""
    return config.analysis

def create_llm_model(creative: bool = False):
    """Create an LLM model based on configuration"""
    import time
    start_time = time.time()
    model_config = get_model_config()
    
    # Use creative model settings if requested
    if creative:
        model_name = model_config.creative_model_name
        temperature = model_config.creative_temperature
    else:
        model_name = model_config.model_name
        temperature = model_config.temperature
    
    logger.debug(f"Creating model: {model_name}, use_vertex_ai={model_config.use_vertex_ai}")
    
    if model_config.use_vertex_ai:
        try:
            from langchain_google_vertexai import ChatVertexAI
            logger.debug(f"Initializing Vertex AI with project={model_config.vertex_ai_project}, location={model_config.vertex_ai_location}")
            llm = ChatVertexAI(
                project=model_config.vertex_ai_project,
                location=model_config.vertex_ai_location,
                model=model_name,  # Note: parameter name is 'model' not 'model_name'
                temperature=temperature,
                max_output_tokens=2048,
                model_kwargs={
                    "timeout": 60,  # 60 second timeout
                    "candidate_count": 1
                }
            )
            elapsed = time.time() - start_time
            logger.info(f"Vertex AI initialized successfully in {elapsed:.1f}s")
            return llm
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"Failed to initialize Vertex AI after {elapsed:.1f}s: {e}")
            logger.info("Falling back to Google AI API...")
            model_config.use_vertex_ai = False
    
    # Use Google AI API as fallback or default
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=model_config.google_api_key
    )
