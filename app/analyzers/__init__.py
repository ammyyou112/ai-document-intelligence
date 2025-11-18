"""
Analyzers package for document analysis
"""
try:
    from app.analyzers.document_complexity_analyzer import DocumentComplexityAnalyzer, ComplexityResult
    __all__ = ['DocumentComplexityAnalyzer', 'ComplexityResult']
except ImportError:
    try:
        from document_complexity_analyzer import DocumentComplexityAnalyzer, ComplexityResult
        __all__ = ['DocumentComplexityAnalyzer', 'ComplexityResult']
    except ImportError:
        __all__ = []

