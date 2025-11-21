# Test Suite

Comprehensive test suite for the hybrid OCR system.

## Structure

- `test_system.py` - Main test suite with all tests consolidated

## Running Tests

### Run All Tests
```bash
python tests/test_system.py
```

### Run Specific Test
```bash
python tests/test_system.py --test bbox          # Test bounding box extraction
python tests/test_system.py --complexity        # Test complexity analysis
python tests/test_system.py --routing           # Test hybrid routing
python tests/test_system.py --pipeline          # Test full pipeline
python tests/test_system.py --validation        # Test output validation
```

### Test with Specific PDF
```bash
python tests/test_system.py --pdf path/to/document.pdf
```

## Test Coverage

1. **Bounding Box Extraction** - Verifies DeepSeek-OCR extracts valid bounding boxes
2. **Complexity Analysis** - Tests document complexity analyzer on multiple pages
3. **Hybrid Routing** - Verifies pages are routed to correct OCR engine
4. **Full Pipeline** - End-to-end test of complete processing pipeline
5. **Output Validation** - Validates JSON output structure and content

## Test Output

Tests generate:
- `tests/test_page_1.png` - Temporary test image (auto-generated)
- `tests/test_output.json` - Full pipeline output for validation

These files are automatically ignored by git (see `.gitignore`).

## Expected Results

All tests should pass for a properly functioning system:
- ✅ Bbox extraction: All bounding boxes should be valid (not [0,0,0,0])
- ✅ Complexity analysis: Should classify pages as simple or complex
- ✅ Hybrid routing: Should route pages to appropriate engines
- ✅ Full pipeline: Should produce complete JSON output
- ✅ Output validation: JSON should have all required keys and valid data

