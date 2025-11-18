"""
Processors package for OCR engines and document intelligence
"""
try:
    from app.processors.simple_ocr_engine import SimpleOCREngine
    from app.processors.hybrid_ocr_router import HybridOCRRouter
    from app.processors.document_classifier import DocumentClassifier, ClassificationResult
    from app.processors.metadata_extractor import MetadataExtractor, MetadataResult
    from app.processors.research_paper_structurer import ResearchPaperStructurer, StructureResult, Section
    from app.processors.enhanced_ocr_pipeline import EnhancedOCRPipeline
    __all__ = [
        'SimpleOCREngine', 
        'HybridOCRRouter',
        'DocumentClassifier',
        'ClassificationResult',
        'MetadataExtractor',
        'MetadataResult',
        'ResearchPaperStructurer',
        'StructureResult',
        'Section',
        'EnhancedOCRPipeline'
    ]
except ImportError:
    try:
        from simple_ocr_engine import SimpleOCREngine
        from hybrid_ocr_router import HybridOCRRouter
        from document_classifier import DocumentClassifier, ClassificationResult
        from metadata_extractor import MetadataExtractor, MetadataResult
        from research_paper_structurer import ResearchPaperStructurer, StructureResult, Section
        from enhanced_ocr_pipeline import EnhancedOCRPipeline
        __all__ = [
            'SimpleOCREngine', 
            'HybridOCRRouter',
            'DocumentClassifier',
            'ClassificationResult',
            'MetadataExtractor',
            'MetadataResult',
            'ResearchPaperStructurer',
            'StructureResult',
            'Section',
            'EnhancedOCRPipeline'
        ]
    except ImportError:
        __all__ = []

