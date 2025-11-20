"""
Full Integration Example

Shows how to use Hybrid OCR Router with Document Intelligence features:
1. OCR processing (hybrid routing)
2. Document classification
3. Metadata extraction
4. Research paper structuring (if applicable)
"""
import sys
import os

# Add parent directory to path (so we can import from root)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from PIL import Image
from app.processors.hybrid_ocr_router import HybridOCRRouter
from app.processors.document_classifier import DocumentClassifier
from app.processors.metadata_extractor import MetadataExtractor
from app.processors.research_paper_structurer import ResearchPaperStructurer
from deepseek_ocr_wrapper import DeepSeekOCR
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

def process_document_with_intelligence(image_path: str):
    """
    Complete document processing pipeline with intelligence
    
    Args:
        image_path: Path to document image
    """
    print("=" * 60)
    print("Full Document Intelligence Pipeline")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing components...")
    router = HybridOCRRouter(complexity_threshold=0.7)
    classifier = DocumentClassifier()
    extractor = MetadataExtractor()
    structurer = ResearchPaperStructurer()
    
    try:
        deepseek_ocr = DeepSeekOCR()
        print("   ✓ DeepSeek OCR initialized")
    except Exception as e:
        print(f"   ⚠ DeepSeek OCR not available: {e}")
        deepseek_ocr = None
    
    # Load image
    print(f"\n2. Loading image: {image_path}")
    try:
        image = Image.open(image_path)
        print(f"   ✓ Image loaded: {image.size}")
    except FileNotFoundError:
        print(f"   ✗ Image not found: {image_path}")
        return
    except Exception as e:
        print(f"   ✗ Error loading image: {e}")
        return
    
    # Step 1: OCR Processing
    print("\n3. OCR Processing (Hybrid Routing)...")
    try:
        ocr_result = router.process(
            image=image,
            deepseek_processor=deepseek_ocr.process if deepseek_ocr else None
        )
        print(f"   ✓ Engine used: {ocr_result['engine_used']}")
        print(f"   ✓ Processing time: {ocr_result['processing_time']:.2f}s")
        print(f"   ✓ Blocks extracted: {len(ocr_result['blocks'])}")
        
        # Combine text from blocks
        full_text = '\n'.join([block.get('text', '') for block in ocr_result['blocks']])
        print(f"   ✓ Total text length: {len(full_text)} characters")
        
    except Exception as e:
        print(f"   ✗ OCR processing failed: {e}")
        return
    
    # Step 2: Document Classification
    print("\n4. Document Classification...")
    try:
        classification = classifier.classify(full_text, ocr_result['blocks'])
        print(f"   ✓ Document Type: {classification.type}")
        print(f"   ✓ Confidence: {classification.confidence:.2f}")
        print(f"   ✓ Keywords: {', '.join(classification.keywords_found[:5])}")
    except Exception as e:
        print(f"   ✗ Classification failed: {e}")
        classification = None
    
    # Step 3: Metadata Extraction
    print("\n5. Metadata Extraction...")
    try:
        # Extract from first page (first 2000 chars or first 20 blocks)
        first_page_text = full_text[:2000] if len(full_text) > 2000 else full_text
        first_page_blocks = ocr_result['blocks'][:20] if len(ocr_result['blocks']) > 20 else ocr_result['blocks']
        
        metadata = extractor.extract(first_page_text, first_page_blocks)
        print(f"   ✓ Title: {metadata.title[:60] if metadata.title else 'Not found'}...")
        print(f"   ✓ Authors: {', '.join(metadata.authors[:3]) if metadata.authors else 'Not found'}")
        print(f"   ✓ Date: {metadata.date or 'Not found'}")
        print(f"   ✓ Organization: {metadata.organization or 'Not found'}")
        print(f"   ✓ Doc Number: {metadata.document_number or 'Not found'}")
    except Exception as e:
        print(f"   ✗ Metadata extraction failed: {e}")
        metadata = None
    
    # Step 4: Research Paper Structuring (if research paper)
    structure = None
    if classification and classification.type == 'research_paper':
        print("\n6. Research Paper Structuring...")
        try:
            structure = structurer.structure(full_text)
            print(f"   ✓ Sections found: {structure.total_sections}")
            print(f"   ✓ Total words: {structure.total_words}")
            print(f"   ✓ Section order: {' → '.join(structure.section_order)}")
            
            # Show section word counts
            for section_name in structure.section_order[:5]:
                section = structure.sections[section_name]
                print(f"      - {section_name}: {section.word_count} words")
        except Exception as e:
            print(f"   ✗ Structuring failed: {e}")
    else:
        print("\n6. Research Paper Structuring...")
        print(f"   ⏭ Skipped (document type: {classification.type if classification else 'unknown'})")
    
    # Compile results
    print("\n7. Compiling Results...")
    results = {
        'ocr': {
            'engine_used': ocr_result['engine_used'],
            'processing_time': ocr_result['processing_time'],
            'total_blocks': len(ocr_result['blocks']),
            'text_length': len(full_text)
        },
        'classification': {
            'type': classification.type if classification else None,
            'confidence': classification.confidence if classification else None,
            'keywords': classification.keywords_found if classification else []
        },
        'metadata': {
            'title': metadata.title if metadata else None,
            'authors': metadata.authors if metadata else [],
            'date': metadata.date if metadata else None,
            'organization': metadata.organization if metadata else None,
            'document_number': metadata.document_number if metadata else None
        },
        'structure': structurer.to_dict(structure) if structure else None,
        'statistics': router.get_stats()
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))
    
    return results

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/document.png"  # Replace with actual path
    
    print("Full Integration Example")
    print("=" * 60)
    print(f"Processing: {image_path}")
    print("\nNote: Update image_path with a valid image file to test")
    print("=" * 60)
    
    # Uncomment to run:
    # results = process_document_with_intelligence(image_path)

