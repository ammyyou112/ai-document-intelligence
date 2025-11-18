# Hybrid OCR Routing System

This hybrid routing system intelligently routes documents to the appropriate OCR engine based on document complexity analysis.

## Architecture

### Components

1. **Document Complexity Analyzer** (`app/analyzers/document_complexity_analyzer.py`)
   - Analyzes image quality (DPI, sharpness, noise)
   - Detects layout complexity (columns, regions)
   - Detects content type (tables, formulas, diagrams)
   - Returns complexity score and recommendation

2. **Simple OCR Engine** (`app/processors/simple_ocr_engine.py`)
   - Uses EasyOCR for fast processing
   - Extracts text with bounding boxes [x1, y1, x2, y2]
   - Optimized for simple documents

3. **Hybrid OCR Router** (`app/processors/hybrid_ocr_router.py`)
   - Analyzes document complexity
   - Routes simple docs → EasyOCR
   - Routes complex docs → DeepSeek-OCR
   - Tracks statistics and performance

## Installation

Install required dependencies:

```bash
pip install easyocr opencv-python
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from PIL import Image
from app.processors.hybrid_ocr_router import HybridOCRRouter
from deepseek_ocr_wrapper import DeepSeekOCR

# Initialize router
router = HybridOCRRouter(complexity_threshold=0.7)

# Initialize DeepSeek OCR
deepseek_ocr = DeepSeekOCR()

# Load image
image = Image.open("document.png")

# Process
result = router.process(
    image=image,
    deepseek_processor=deepseek_ocr.process
)

# Access results
print(f"Engine used: {result['engine_used']}")
print(f"Blocks: {result['blocks']}")
print(f"Processing time: {result['processing_time']}")
```

### Result Format

```python
{
    'blocks': [
        {
            'text': 'Extracted text',
            'bbox': [x1, y1, x2, y2],
            'confidence': 0.95
        }
    ],
    'engine_used': 'simple' | 'deepseek',
    'complexity_analysis': ComplexityResult,
    'processing_time': 1.23,
    'stats': {...}
}
```

### Complexity Analysis

The analyzer evaluates:

- **Image Quality**: Sharpness, DPI, noise, contrast
- **Layout**: Multi-column detection, region density
- **Content**: Tables, formulas, diagrams

### Configuration

```python
# Custom thresholds
router = HybridOCRRouter(
    complexity_threshold=0.7,  # Confidence threshold (0.0-1.0)
    force_engine=None  # 'simple', 'deepseek', or None for auto
)

# Force specific engine
router = HybridOCRRouter(force_engine='simple')
```

### Statistics

Track performance and routing decisions:

```python
stats = router.get_stats()
print(f"Simple OCR: {stats['simple_calls']} calls")
print(f"DeepSeek OCR: {stats['deepseek_calls']} calls")
print(f"Average time: {stats['avg_processing_time']:.2f}s")
```

## Integration with Flask App

To integrate with your existing Flask app:

```python
from app.processors.hybrid_ocr_router import HybridOCRRouter

# Initialize router (once at startup)
router = HybridOCRRouter()

# In your OCR endpoint
@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    # ... get image ...
    
    # Process with hybrid router
    result = router.process(
        image=image,
        deepseek_processor=deepseek_ocr_model.process
    )
    
    # Use result['blocks'] for structured data
    # Use result['engine_used'] for logging
```

## Features

- ✅ Automatic complexity analysis
- ✅ Intelligent routing based on document type
- ✅ Performance statistics tracking
- ✅ Error handling and logging
- ✅ Type hints throughout
- ✅ PIL Image support
- ✅ Bounding box extraction

## Performance

- **Simple OCR (EasyOCR)**: Fast processing for simple documents
- **DeepSeek-OCR**: Accurate processing for complex documents
- **Automatic selection**: Optimizes speed vs accuracy

## Error Handling

All components include comprehensive error handling:

- Missing dependencies (EasyOCR, OpenCV)
- Image conversion errors
- Processing failures
- Graceful fallbacks

## Logging

All components use Python's logging module:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Type Hints

All functions include type hints for better IDE support and documentation.

