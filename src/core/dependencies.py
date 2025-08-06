"""
Dependency Injection Container

This module provides a simple container for managing shared resources
and dependencies across the application, such as language models,
RAG systems, and configuration objects. This helps to decouple components
and improve testability.
"""

from functools import lru_cache
from typing import Optional

from langchain_core.language_models.base import BaseLanguageModel
from src.core.config import AppConfig, config as app_config
from src.search.relevance_scorer import RelevanceScorer
from src.analysis.simple_pdf_processor import SimplePDFProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AppContainer:
    """
    A simple dependency injection container for managing application-wide resources.
    """
    def __init__(self, config: Optional[AppConfig] = None):
        logger.info("Initializing AppContainer...")
        self._config = config if config else app_config
        self._llm_model: Optional[BaseLanguageModel] = None
        self._creative_llm_model: Optional[BaseLanguageModel] = None
        self._relevance_scorer: Optional[RelevanceScorer] = None
        self._pdf_processor: Optional[SimplePDFProcessor] = None

    @property
    def config(self) -> AppConfig:
        """Returns the application configuration."""
        return self._config

    @property
    def llm_model(self) -> BaseLanguageModel:
        """Provides a singleton instance of the LLMModel."""
        if self._llm_model is None:
            logger.info("Creating LLMModel instance...")
            # This will use the model configuration from the AppConfig
            from src.core.config import create_llm_model
            self._llm_model = create_llm_model()
        return self._llm_model

    @property
    def creative_llm_model(self) -> BaseLanguageModel:
        """Provides a singleton instance of the creative LLMModel."""
        if self._creative_llm_model is None:
            logger.info("Creating creative LLMModel instance...")
            from src.core.config import create_llm_model
            self._creative_llm_model = create_llm_model(creative=True)
        return self._creative_llm_model

    @property
    def relevance_scorer(self) -> RelevanceScorer:
        """Provides a singleton instance of the RelevanceScorer."""
        if self._relevance_scorer is None:
            logger.info("Creating RelevanceScorer instance...")
            self._relevance_scorer = RelevanceScorer(llm_model=self.llm_model)
        return self._relevance_scorer

    @property
    def pdf_processor(self) -> SimplePDFProcessor:
        """Provides a singleton instance of the SimplePDFProcessor."""
        if self._pdf_processor is None:
            logger.info("Creating SimplePDFProcessor instance...")
            self._pdf_processor = SimplePDFProcessor()
        return self._pdf_processor

@lru_cache(maxsize=1)
def get_container() -> AppContainer:
    """
    Provides a global, cached instance of the AppContainer.
    """
    logger.info("Accessing global AppContainer instance.")
    return AppContainer()

def clear_container_cache():
    """
    Clear the cached AppContainer instance.
    Useful for resetting state between runs.
    """
    logger.info("Clearing AppContainer cache.")
    get_container.cache_clear()
