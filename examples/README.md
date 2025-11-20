# Examples Directory

This directory contains example scripts demonstrating how to use various components of the DeepSeek-OCR system.

## Example Scripts

### `example_basic_ocr.py`
**Purpose**: Basic OCR processing example  
**Shows**: 
- How to initialize DeepSeekOCR model
- How to process a single image
- How to extract and display text results

**Usage**:
```bash
python examples/example_basic_ocr.py
```

**Note**: Update the `image_path` variable in the script to point to your own image file.

### `example_hybrid_usage.py`
**Purpose**: Hybrid OCR routing example  
**Shows**:
- How to use the HybridOCRRouter
- Automatic engine selection (EasyOCR vs DeepSeek-OCR)
- Complexity analysis and routing decisions

**Usage**:
```bash
python examples/example_hybrid_usage.py
```

### `example_document_intelligence.py`
**Purpose**: Document intelligence features example  
**Shows**:
- Document classification
- Metadata extraction
- Research paper structuring

**Usage**:
```bash
python examples/example_document_intelligence.py
```

### `example_full_integration.py`
**Purpose**: Complete pipeline integration example  
**Shows**:
- Full document processing pipeline
- Integration of all components
- Complete JSON output structure

**Usage**:
```bash
python examples/example_full_integration.py
```

## Running Examples

All examples should be run from the project root directory:

```bash
# From project root
python examples/example_basic_ocr.py
python examples/example_hybrid_usage.py
python examples/example_document_intelligence.py
python examples/example_full_integration.py
```

## Prerequisites

Before running examples:
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure you have internet connection (for model download on first run)
3. Have sufficient disk space (~several GB for model weights)

## Customization

Most examples have configurable variables at the top:
- Image paths
- Model parameters
- Processing options

Modify these to test with your own files and settings.

