# Enhanced OCR Pipeline

Complete document processing pipeline integrating all document intelligence features.

## Overview

The Enhanced OCR Pipeline (`enhanced_ocr_pipeline.py`) provides a complete end-to-end solution for document processing with:

- **Hybrid OCR Routing** - Automatically selects optimal OCR engine
- **Document Classification** - Identifies document type
- **Metadata Extraction** - Extracts title, authors, dates, etc.
- **Research Paper Structuring** - Structures academic papers into sections
- **Quality Scoring** - Calculates overall document quality

## Features

### 1. Intelligent OCR Routing
- Analyzes document complexity
- Routes simple documents → EasyOCR (fast)
- Routes complex documents → DeepSeek-OCR (accurate)
- Tracks engine usage statistics

### 2. Document Classification
- Classifies into 5 types:
  - `research_paper`
  - `invoice`
  - `financial_report`
  - `technical_manual`
  - `general_document`

### 3. Metadata Extraction
- **Title** - Document title
- **Authors** - Author names
- **Date** - Publication/creation date
- **Organization** - Institution/company
- **Document Number** - Invoice numbers, DOIs, etc.

### 4. Research Paper Structuring
- Detects sections: Abstract, Introduction, Methodology, Results, Discussion, Conclusion, References
- Extracts section text and word counts
- Only applied to research papers

### 5. Quality Scoring
- Calculates quality score (0.0-1.0) based on:
  - Text extraction confidence
  - Classification confidence
  - Metadata completeness
  - Document structure
  - Content richness

## Output Format

The pipeline returns a comprehensive JSON structure:

```json
{
  "document_metadata": {
    "document_id": "uuid",
    "filename": "document.pdf",
    "type": "research_paper",
    "engines_used": ["simple", "deepseek"],
    "processed_date": "2024-01-01T12:00:00",
    "processing_time_seconds": 45.2,
    "total_pages": 10
  },
  "content_metadata": {
    "title": "Document Title",
    "authors": ["Author 1", "Author 2"],
    "date": "2024-01-01",
    "organization": "University Name",
    "document_number": "DOI:10.1234/example",
    "pages": 10,
    "words": 5000,
    "classification_confidence": 0.85
  },
  "document_structure": {
    "abstract": {"text": "...", "word_count": 150},
    "introduction": {"text": "...", "word_count": 300},
    ...
  },
  "pages": [
    {
      "page_number": 1,
      "blocks": [...],
      "text": "...",
      "word_count": 500,
      "engine_used": "deepseek",
      "processing_time": 2.3,
      "complexity": {
        "complexity": "complex",
        "confidence": 0.8,
        "reasons": [...]
      }
    },
    ...
  ],
  "training_annotations": {
    "quality_score": 0.85,
    "has_tables": true,
    "engines_distribution": {
      "simple": 2,
      "deepseek": 8,
      "simple_percentage": 20.0,
      "deepseek_percentage": 80.0
    },
    "average_processing_time": 4.5,
    "total_blocks": 150,
    "document_type": "research_paper"
  },
  "classification_details": {
    "type": "research_paper",
    "confidence": 0.85,
    "keywords_found": [...],
    "patterns_matched": 5
  },
  "router_statistics": {
    "total_calls": 10,
    "simple_calls": 2,
    "deepseek_calls": 8,
    ...
  }
}
```

## Usage

### In Flask App

The pipeline is automatically initialized and used in `app.py`:

```python
# Pipeline is initialized at startup
enhanced_pipeline = EnhancedOCRPipeline(
    deepseek_processor=deepseek_processor,
    complexity_threshold=0.7,
    pdf_dpi=300
)

# Used in OCR endpoint
result = enhanced_pipeline.process_document(file_path, filename)
```

### Standalone Usage

```python
from app.processors.enhanced_ocr_pipeline import EnhancedOCRPipeline
from deepseek_ocr_wrapper import DeepSeekOCR

# Initialize
deepseek_ocr = DeepSeekOCR()
pipeline = EnhancedOCRPipeline(
    deepseek_processor=deepseek_ocr.process,
    complexity_threshold=0.7,
    pdf_dpi=300
)

# Process document
result = pipeline.process_document("document.pdf", "document.pdf")

# Access results
print(f"Type: {result['document_metadata']['type']}")
print(f"Title: {result['content_metadata']['title']}")
print(f"Quality: {result['training_annotations']['quality_score']}")
```

## Configuration

### Parameters

- `deepseek_processor`: DeepSeekOCR instance or callable
- `complexity_threshold`: Routing threshold (0.0-1.0), default 0.7
- `pdf_dpi`: PDF conversion DPI, default 300

### PDF Processing

- Converts PDFs to images at specified DPI (default 300)
- Processes each page through hybrid router
- Automatically cleans up temporary images

## Logging

Comprehensive logging throughout:

- Document processing start/end
- Page-by-page processing
- Classification and metadata extraction
- Engine routing decisions
- Error handling

Logs are written to:
- Console (stdout)
- File: `app_output.log`

## Error Handling

- Graceful fallbacks if components unavailable
- Automatic cleanup of temporary files
- Detailed error messages
- Fallback to basic OCR if enhanced pipeline fails

## Performance

- **Simple documents**: Fast processing with EasyOCR
- **Complex documents**: Accurate processing with DeepSeek-OCR
- **Automatic optimization**: Routes to optimal engine per page
- **Statistics tracking**: Monitors performance and usage

## Integration

The pipeline integrates seamlessly with:

- Flask application (`app.py`)
- Hybrid OCR Router
- Document Classifier
- Metadata Extractor
- Research Paper Structurer

All components work together to provide comprehensive document intelligence.

