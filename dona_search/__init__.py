from .interfaces import SearchResult
from .output import format_result, format_results
from .pipeline import SearchPipeline

__version__ = '0.1.0'
__all__ = ['SearchPipeline', 'SearchResult', 'format_result', 'format_results']
