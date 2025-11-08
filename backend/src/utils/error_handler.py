# src/utils/error_handler.py
"""
Comprehensive error handling
"""

import logging
import traceback
from functools import wraps
from typing import Callable, Any
import sys

# Configure logging
def setup_logging(log_file='outputs/logs/system.log'):
    """Setup comprehensive logging"""
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simple)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

class MedicalAssistantError(Exception):
    """Base exception for medical assistant"""
    pass

class DatabaseConnectionError(MedicalAssistantError):
    """Neo4j connection error"""
    pass

class QueryParsingError(MedicalAssistantError):
    """Error parsing user query"""
    pass

class ReasoningError(MedicalAssistantError):
    """Error during reasoning process"""
    pass

def handle_errors(func: Callable) -> Callable:
    """Decorator for error handling"""
    
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = logging.getLogger(func.__module__)
        
        try:
            return func(*args, **kwargs)
        
        except DatabaseConnectionError as e:
            logger.error(f"Database connection failed: {e}")
            return {
                'error': 'database_connection',
                'message': 'Could not connect to knowledge graph. Please check Neo4j is running.',
                'details': str(e)
            }
        
        except QueryParsingError as e:
            logger.error(f"Query parsing failed: {e}")
            return {
                'error': 'query_parsing',
                'message': 'Could not understand your question. Please rephrase.',
                'details': str(e)
            }
        
        except ReasoningError as e:
            logger.error(f"Reasoning failed: {e}")
            return {
                'error': 'reasoning',
                'message': 'Could not find answer. Try a simpler question.',
                'details': str(e)
            }
        
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            return {
                'error': 'unexpected',
                'message': 'An unexpected error occurred. Please try again.',
                'details': str(e)
            }
    
    return wrapper

def safe_neo4j_operation(operation: Callable, fallback_value=None):
    """Safe wrapper for Neo4j operations"""
    logger = logging.getLogger(__name__)
    
    try:
        return operation()
    except Exception as e:
        logger.error(f"Neo4j operation failed: {e}")
        if fallback_value is not None:
            logger.info(f"Using fallback value: {fallback_value}")
            return fallback_value
        raise DatabaseConnectionError(f"Neo4j operation failed: {e}")

# Apply to existing reasoner
from src.reasoning.optimized_search import OptimizedReasoner as BaseReasoner

class RobustReasoner(BaseReasoner):
    """Reasoner with comprehensive error handling"""
    
    @handle_errors
    def answer_question(self, query: str):
        """Answer question with error handling"""
        
        if not query or len(query.strip()) == 0:
            raise QueryParsingError("Empty query provided")
        
        if len(query) > 500:
            raise QueryParsingError("Query too long (max 500 characters)")
        
        # Check Neo4j connection
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            raise DatabaseConnectionError(f"Neo4j not available: {e}")
        
        # Run reasoning
        try:
            result = super().answer_question(query)
            
            if not result or 'paths' not in result:
                raise ReasoningError("No reasoning paths found")
            
            return result
        
        except Exception as e:
            raise ReasoningError(f"Reasoning failed: {e}")