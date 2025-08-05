"""
Unified Error Handling System for arXiv Research Agent

This module provides a centralized error handling system with:
- Custom exception hierarchy
- Consistent error messages
- Error tracking and reporting
- Recovery strategies
"""

import sys
import traceback
from typing import Optional, Dict, Any, Callable, Type, Union
from enum import Enum
from functools import wraps
import logging
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"  # Can continue, minor issue
    MEDIUM = "medium"  # Should log, may affect results
    HIGH = "high"  # Critical, but recoverable
    CRITICAL = "critical"  # Must stop execution


class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"  # Network/API related
    PARSING = "parsing"  # Data parsing errors
    VALIDATION = "validation"  # Data validation errors
    RESOURCE = "resource"  # Resource not found/available
    CONFIGURATION = "configuration"  # Config/setup issues
    PROCESSING = "processing"  # General processing errors
    TIMEOUT = "timeout"  # Operation timeouts
    UNKNOWN = "unknown"  # Uncategorized errors


# Base exception classes
class ArxivAgentError(Exception):
    """Base exception for all arXiv agent errors"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.severity = severity
        self.category = category
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now()
        
        # Log the error
        self._log_error()
    
    def _log_error(self):
        """Log error based on severity"""
        log_message = f"[{self.category.value.upper()}] {str(self)}"
        
        if self.severity == ErrorSeverity.LOW:
            logger.debug(log_message)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        else:  # CRITICAL
            logger.critical(log_message)
        
        if self.cause:
            logger.debug(f"Caused by: {type(self.cause).__name__}: {str(self.cause)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization"""
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "severity": self.severity.value,
            "category": self.category.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None
        }


# Specific exception classes
class NetworkError(ArxivAgentError):
    """Network-related errors"""
    def __init__(self, message: str, url: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK,
            **kwargs
        )
        if url:
            self.details["url"] = url


class PDFProcessingError(ArxivAgentError):
    """PDF processing errors"""
    def __init__(self, message: str, pdf_url: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            **kwargs
        )
        if pdf_url:
            self.details["pdf_url"] = pdf_url


class ParsingError(ArxivAgentError):
    """Data parsing errors"""
    def __init__(self, message: str, data_type: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PARSING,
            **kwargs
        )
        if data_type:
            self.details["data_type"] = data_type


class ValidationError(ArxivAgentError):
    """Data validation errors"""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            **kwargs
        )
        if field:
            self.details["field"] = field


class ConfigurationError(ArxivAgentError):
    """Configuration errors"""
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )
        if config_key:
            self.details["config_key"] = config_key


class TimeoutError(ArxivAgentError):
    """Operation timeout errors"""
    def __init__(self, message: str, operation: Optional[str] = None, timeout: Optional[float] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TIMEOUT,
            **kwargs
        )
        if operation:
            self.details["operation"] = operation
        if timeout:
            self.details["timeout_seconds"] = timeout


class ResourceNotFoundError(ArxivAgentError):
    """Resource not found errors"""
    def __init__(self, message: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.RESOURCE,
            **kwargs
        )
        if resource_type:
            self.details["resource_type"] = resource_type
        if resource_id:
            self.details["resource_id"] = resource_id


# Error handler class
class ErrorHandler:
    """Centralized error handler with recovery strategies"""
    
    def __init__(self):
        self.error_history: list[ArxivAgentError] = []
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.max_history_size = 100
    
    def register_recovery(self, error_type: Type[Exception], strategy: Callable):
        """Register a recovery strategy for an error type"""
        self.recovery_strategies[error_type] = strategy
        logger.debug(f"Registered recovery strategy for {error_type.__name__}")
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        raise_after: bool = True
    ) -> Optional[Any]:
        """
        Handle an error with optional recovery
        
        Args:
            error: The exception to handle
            context: Additional context for error handling
            raise_after: Whether to re-raise after handling
            
        Returns:
            Recovery result if available, None otherwise
        """
        # Convert to ArxivAgentError if needed
        if not isinstance(error, ArxivAgentError):
            agent_error = self._convert_to_agent_error(error, context)
        else:
            agent_error = error
            if context:
                agent_error.details.update(context)
        
        # Add to history
        self._add_to_history(agent_error)
        
        # Try recovery
        recovery_result = self._attempt_recovery(agent_error)
        
        if raise_after and recovery_result is None:
            raise agent_error
        
        return recovery_result
    
    def _convert_to_agent_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ArxivAgentError:
        """Convert standard exception to ArxivAgentError"""
        error_map = {
            ConnectionError: NetworkError,
            TimeoutError: TimeoutError,
            ValueError: ValidationError,
            KeyError: ValidationError,
            FileNotFoundError: ResourceNotFoundError,
            json.JSONDecodeError: ParsingError,
        }
        
        for base_type, agent_type in error_map.items():
            if isinstance(error, base_type):
                return agent_type(
                    str(error),
                    details=context or {},
                    cause=error
                )
        
        # Default conversion
        return ArxivAgentError(
            str(error),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.UNKNOWN,
            details=context or {},
            cause=error
        )
    
    def _add_to_history(self, error: ArxivAgentError):
        """Add error to history with size limit"""
        self.error_history.append(error)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
    
    def _attempt_recovery(self, error: ArxivAgentError) -> Optional[Any]:
        """Attempt to recover from error"""
        error_type = type(error)
        
        # Check for exact match
        if error_type in self.recovery_strategies:
            try:
                logger.info(f"Attempting recovery for {error_type.__name__}")
                return self.recovery_strategies[error_type](error)
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
                return None
        
        # Check for subclass match
        for registered_type, strategy in self.recovery_strategies.items():
            if isinstance(error, registered_type):
                try:
                    logger.info(f"Attempting recovery for {error_type.__name__} using {registered_type.__name__} strategy")
                    return strategy(error)
                except Exception as e:
                    logger.error(f"Recovery failed: {e}")
                    return None
        
        return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        summary = {
            "total_errors": len(self.error_history),
            "by_category": {},
            "by_severity": {},
            "recent_errors": []
        }
        
        for error in self.error_history:
            # Count by category
            category = error.category.value
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
            
            # Count by severity
            severity = error.severity.value
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        
        # Add recent errors
        summary["recent_errors"] = [
            e.to_dict() for e in self.error_history[-5:]
        ]
        
        return summary


# Global error handler instance
_error_handler = ErrorHandler()


# Decorator for error handling
def handle_errors(
    default_return: Any = None,
    error_types: Optional[tuple[Type[Exception], ...]] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    log_traceback: bool = True
):
    """
    Decorator to handle errors in functions
    
    Args:
        default_return: Value to return on error
        error_types: Specific error types to catch (None = catch all)
        severity: Error severity level
        log_traceback: Whether to log full traceback
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if we should handle this error type
                if error_types and not isinstance(e, error_types):
                    raise
                
                # Create context
                context = {
                    "function": func.__name__,
                    "args": str(args)[:100],  # Truncate for safety
                    "kwargs": str(kwargs)[:100]
                }
                
                # Log traceback if requested
                if log_traceback and not isinstance(e, ArxivAgentError):
                    logger.debug(f"Traceback for {func.__name__}:\n{traceback.format_exc()}")
                
                # Handle the error
                recovery_result = _error_handler.handle_error(
                    e,
                    context=context,
                    raise_after=False
                )
                
                # Return recovery result or default
                return recovery_result if recovery_result is not None else default_return
        
        return wrapper
    return decorator


# Recovery strategy helpers
def retry_with_backoff(max_attempts: int = 3, backoff_factor: float = 2.0):
    """
    Decorator to retry function with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for wait time between attempts
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            last_error = None
            wait_time = 1.0
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        logger.info(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        wait_time *= backoff_factor
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            # If all attempts failed, raise the last error
            if last_error:
                raise last_error
        
        return wrapper
    return decorator


# Common recovery strategies
def network_retry_strategy(error: NetworkError) -> Optional[Any]:
    """Recovery strategy for network errors - returns None to trigger retry"""
    logger.info(f"Network error recovery: {error.details.get('url', 'unknown URL')}")
    return None


def parsing_fallback_strategy(error: ParsingError) -> Optional[Dict[str, Any]]:
    """Recovery strategy for parsing errors - returns empty structure"""
    logger.info(f"Parsing error recovery: returning empty {error.details.get('data_type', 'data')}")
    return {}


# Register default recovery strategies
_error_handler.register_recovery(NetworkError, network_retry_strategy)
_error_handler.register_recovery(ParsingError, parsing_fallback_strategy)


# Convenience functions
def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    return _error_handler


def report_error(
    message: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    **details
) -> ArxivAgentError:
    """
    Report an error without raising it
    
    Args:
        message: Error message
        severity: Error severity
        category: Error category
        **details: Additional error details
        
    Returns:
        The created error instance
    """
    error = ArxivAgentError(
        message=message,
        severity=severity,
        category=category,
        details=details
    )
    _error_handler._add_to_history(error)
    return error


# Import guard for json
import json