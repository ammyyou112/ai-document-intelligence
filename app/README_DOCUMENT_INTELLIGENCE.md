# Document Intelligence System

Document intelligence features for the hybrid OCR system, including classification, metadata extraction, and research paper structuring.

## Components

### 1. Document Classifier (`document_classifier.py`)

Classifies documents into types:
- `research_paper` - Academic papers, journal articles
- `invoice` - Invoices, bills, payment documents
- `financial_report` - Financial statements, annual reports
- `technical_manual` - User guides, technical documentation
- `general_document` - Fallback category

**Features:**
- Keyword-based classification
- Regex pattern matching
- Confidence scoring
- Returns matched keywords and patterns

**Usage:**
```python
from app.processors.document_classifier import DocumentClassifier

classifier = DocumentClassifier()
result = classifier.classify(text, blocks)

print(f"Type: {result.type}")
print(f"Confidence: {result.confidence}")
print(f"Keywords: {result.keywords_found}")
```

### 2. Metadata Extractor (`metadata_extractor.py`)

Extracts metadata from document first page:
- **Title** - Document title
- **Authors** - Author names
- **Date** - Publication/creation date
- **Organization** - Institution/company
- **Document Number** - Invoice numbers, DOIs, etc.

**Features:**
- Regex-based extraction
- Spatial analysis using bounding boxes
- Multiple pattern matching
- Handles various date formats

**Usage:**
```python
from app.processors.metadata_extractor import MetadataExtractor

extractor = MetadataExtractor()
metadata = extractor.extract(text, blocks)

print(f"Title: {metadata.title}")
print(f"Authors: {metadata.authors}")
print(f"Date: {metadata.date}")
print(f"Organization: {metadata.organization}")
print(f"Doc Number: {metadata.document_number}")
```

### 3. Research Paper Structurer (`research_paper_structurer.py`)

Structures research papers into sections:
- Abstract
- Introduction
- Methodology
- Results
- Discussion
- Conclusion
- References
- Acknowledgment
- Appendix

**Features:**
- Case-insensitive section detection
- Multiple pattern matching per section
- Word count per section
- Section ordering

**Usage:**
```python
from app.processors.research_paper_structurer import ResearchPaperStructurer

structurer = ResearchPaperStructurer()
structure = structurer.structure(text)

# Access sections
for section_name in structure.section_order:
    section = structure.sections[section_name]
    print(f"{section_name}: {section.word_count} words")
    print(f"Text: {section.text[:100]}...")

# Get specific section
abstract = structurer.get_section_text(structure, 'abstract')

# Convert to dictionary
structure_dict = structurer.to_dict(structure)
```

## Integration with Hybrid OCR

Complete pipeline example:

```python
from PIL import Image
from app.processors.hybrid_ocr_router import HybridOCRRouter
from app.processors.document_classifier import DocumentClassifier
from app.processors.metadata_extractor import MetadataExtractor
from app.processors.research_paper_structurer import ResearchPaperStructurer
from deepseek_ocr_wrapper import DeepSeekOCR

# Initialize
router = HybridOCRRouter()
classifier = DocumentClassifier()
extractor = MetadataExtractor()
structurer = ResearchPaperStructurer()
deepseek_ocr = DeepSeekOCR()

# Process image
image = Image.open("document.png")
ocr_result = router.process(image, deepseek_processor=deepseek_ocr.process)

# Extract text
full_text = '\n'.join([b['text'] for b in ocr_result['blocks']])

# Classify
classification = classifier.classify(full_text, ocr_result['blocks'])

# Extract metadata
metadata = extractor.extract(full_text[:2000], ocr_result['blocks'][:20])

# Structure (if research paper)
if classification.type == 'research_paper':
    structure = structurer.structure(full_text)
```

## Result Formats

### Classification Result
```python
ClassificationResult(
    type='research_paper',
    confidence=0.85,
    keywords_found=['abstract', 'introduction', 'references'],
    patterns_matched=[r'\babstract\b', r'\bintroduction\b']
)
```

### Metadata Result
```python
MetadataResult(
    title='Deep Learning for Document Understanding',
    authors=['John Smith', 'Jane Doe'],
    date='2023-12-15',
    organization='Stanford University',
    document_number='DOI:10.1234/example',
    raw_metadata={...}
)
```

### Structure Result
```python
StructureResult(
    sections={
        'abstract': Section(name='abstract', text='...', word_count=150, ...),
        'introduction': Section(name='introduction', text='...', word_count=300, ...),
        ...
    },
    section_order=['abstract', 'introduction', 'methodology', ...],
    total_sections=7,
    total_words=5000
)
```

## Features

- ✅ **Modular Design** - Each component works independently
- ✅ **Type Hints** - Full type annotations
- ✅ **Error Handling** - Graceful fallbacks
- ✅ **Logging** - Comprehensive logging support
- ✅ **Flexible** - Works with text or OCR blocks
- ✅ **Extensible** - Easy to add new patterns/types

## Pattern Customization

All components use configurable patterns. You can extend:

1. **Document Classifier**: Add keywords/patterns in `__init__()`
2. **Metadata Extractor**: Add regex patterns for new fields
3. **Research Paper Structurer**: Add section patterns

## Performance

- **Classification**: Fast (regex matching)
- **Metadata Extraction**: Fast (pattern matching)
- **Structuring**: Fast (line-by-line scanning)

All operations are O(n) where n is text length.

## Error Handling

All components handle:
- Empty text input
- Missing patterns
- Invalid data formats
- Graceful fallbacks

## Examples

See:
- `example_document_intelligence.py` - Individual component usage
- `example_full_integration.py` - Complete pipeline integration

