#!/usr/bin/env python3
"""
Comprehensive test for the full hybrid OCR pipeline

Tests the complete EnhancedOCRPipeline including:
- Document complexity analysis
- Hybrid routing (simple → fast, complex → DeepSeek)
- Document classification
- Metadata extraction
- Research paper structuring
"""

import os
import sys
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print a header with formatting"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

def print_step(step_num: int, text: str):
    """Print a step indicator"""
    emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟']
    emoji = emojis[step_num - 1] if step_num <= len(emojis) else '▶️'
    print(f"{Colors.BOLD}{Colors.OKCYAN}{emoji} {text}{Colors.ENDC}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKBLUE}ℹ️  {text}{Colors.ENDC}")

def find_test_pdf() -> Optional[str]:
    """Find a test PDF in common directories"""
    print_step(1, "Finding test PDF...")
    
    # Check common directories
    search_dirs = [
        'uploads',
        'examples',
        '.',
        'test_documents',
        'test'
    ]
    
    found_pdfs = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for path in Path(search_dir).glob("*.pdf"):
                found_pdfs.append(str(path))
    
    if found_pdfs:
        # Prefer PDFs in uploads/ or examples/
        preferred = [pdf for pdf in found_pdfs if 'upload' in pdf.lower() or 'example' in pdf.lower()]
        if preferred:
            selected = preferred[0]
        else:
            selected = found_pdfs[0]
        
        print_success(f"Found test PDF: {selected}")
        return selected
    else:
        print_warning("No test PDFs found in common directories")
        print_info("Please provide a path to a test PDF")
        user_path = input(f"{Colors.BOLD}Enter PDF path: {Colors.ENDC}").strip().strip('"').strip("'")
        
        if os.path.exists(user_path):
            print_success(f"Using provided PDF: {user_path}")
            return user_path
        else:
            print_error(f"PDF not found: {user_path}")
            return None

def initialize_pipeline() -> Tuple[Optional[Any], Optional[Callable]]:
    """Initialize the Enhanced OCR Pipeline with DeepSeek processor"""
    print_step(2, "Initializing pipeline...")
    
    try:
        # Import EnhancedOCRPipeline
        try:
            from app.processors.enhanced_ocr_pipeline import EnhancedOCRPipeline
            print_info("Imported EnhancedOCRPipeline")
        except ImportError:
            print_error("Failed to import EnhancedOCRPipeline")
            print_info("Make sure app/processors/enhanced_ocr_pipeline.py exists")
            return None, None
        
        # Initialize DeepSeek processor
        deepseek_processor = None
        try:
            from deepseek_ocr_wrapper import DeepSeekOCR
            print_info("Initializing DeepSeek-OCR (this may take a while on first run)...")
            
            # Create a lazy loader function
            _deepseek_model = None
            def deepseek_processor_func(image_path: str):
                """Wrapper for DeepSeek OCR processing"""
                nonlocal _deepseek_model
                if _deepseek_model is None:
                    print_info("Loading DeepSeek-OCR model (first use)...")
                    _deepseek_model = DeepSeekOCR()
                return _deepseek_model.process(image_path)
            
            deepseek_processor = deepseek_processor_func
            print_success("DeepSeek processor ready (will load on first use)")
        except ImportError as e:
            print_warning(f"DeepSeek-OCR not available: {e}")
            print_info("Pipeline will use Simple OCR only (no complex document processing)")
            deepseek_processor = None
        except Exception as e:
            print_warning(f"DeepSeek-OCR initialization error: {e}")
            print_info("Pipeline will use Simple OCR only")
            deepseek_processor = None
        
        # Initialize pipeline
        try:
            pipeline = EnhancedOCRPipeline(
                deepseek_processor=deepseek_processor,
                complexity_threshold=0.7,
                pdf_dpi=300
            )
            print_success("EnhancedOCRPipeline loaded")
            return pipeline, deepseek_processor
        except Exception as e:
            print_error(f"Failed to initialize pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None, None
            
    except Exception as e:
        print_error(f"Unexpected error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def validate_bboxes(blocks: List[Dict]) -> Dict:
    """Validate bounding boxes in blocks"""
    total = len(blocks)
    valid = 0
    invalid = 0
    
    for block in blocks:
        bbox = block.get('bbox', [0, 0, 0, 0])
        is_valid = (
            bbox != [0, 0, 0, 0] and
            len(bbox) == 4 and
            isinstance(bbox[0], (int, float)) and
            bbox[2] > bbox[0] and  # x2 > x1
            bbox[3] > bbox[1]      # y2 > y1
        )
        if is_valid:
            valid += 1
        else:
            invalid += 1
    
    return {
        'total': total,
        'valid': valid,
        'invalid': invalid,
        'valid_percentage': (valid / total * 100) if total > 0 else 0.0,
        'invalid_percentage': (invalid / total * 100) if total > 0 else 0.0
    }

def analyze_pages(pages: List[Dict]) -> Dict:
    """Analyze page-level statistics"""
    total_pages = len(pages)
    simple_count = 0
    deepseek_count = 0
    total_blocks = 0
    total_words = 0
    pages_with_valid_bboxes = 0
    
    for page in pages:
        engine = page.get('engine_used', 'unknown')
        if engine == 'simple':
            simple_count += 1
        elif engine == 'deepseek':
            deepseek_count += 1
        
        blocks = page.get('blocks', [])
        total_blocks += len(blocks)
        total_words += page.get('word_count', 0)
        
        # Check if page has valid bboxes
        bbox_stats = validate_bboxes(blocks)
        if bbox_stats['valid'] > 0:
            pages_with_valid_bboxes += 1
    
    return {
        'total_pages': total_pages,
        'simple_count': simple_count,
        'deepseek_count': deepseek_count,
        'simple_percentage': (simple_count / total_pages * 100) if total_pages > 0 else 0.0,
        'deepseek_percentage': (deepseek_count / total_pages * 100) if total_pages > 0 else 0.0,
        'total_blocks': total_blocks,
        'total_words': total_words,
        'pages_with_valid_bboxes': pages_with_valid_bboxes,
        'pages_with_valid_bboxes_percentage': (pages_with_valid_bboxes / total_pages * 100) if total_pages > 0 else 0.0
    }

def display_results(result: Dict, pdf_path: str):
    """Display comprehensive results"""
    print_header("Full Hybrid OCR Pipeline Test Results")
    
    # Extract data
    doc_meta = result.get('document_metadata', {})
    content_meta = result.get('content_metadata', {})
    pages = result.get('pages', [])
    doc_structure = result.get('document_structure')
    training_ann = result.get('training_annotations', {})
    classification = result.get('classification_details', {})
    router_stats = result.get('router_statistics', {})
    
    # Analyze pages
    page_stats = analyze_pages(pages)
    
    # 3️⃣ Document Analysis
    print_step(3, "Document Analysis")
    print(f"\n{Colors.BOLD}Type:{Colors.ENDC} {Colors.OKCYAN}{classification.get('type', 'unknown')}{Colors.ENDC} ({classification.get('confidence', 0.0)*100:.1f}% confidence)")
    print(f"{Colors.BOLD}Pages:{Colors.ENDC} {Colors.OKBLUE}{doc_meta.get('total_pages', 0)}{Colors.ENDC}")
    print(f"{Colors.BOLD}Total words:{Colors.ENDC} {Colors.OKBLUE}{content_meta.get('words', 0):,}{Colors.ENDC}")
    print(f"{Colors.BOLD}Processing time:{Colors.ENDC} {Colors.OKBLUE}{doc_meta.get('processing_time_seconds', 0):.2f}s{Colors.ENDC}")
    
    # 4️⃣ Engine Usage
    print_step(4, "Engine Usage")
    engines_dist = training_ann.get('engines_distribution', {})
    simple_pct = engines_dist.get('simple_percentage', 0.0)
    deepseek_pct = engines_dist.get('deepseek_percentage', 0.0)
    
    print(f"\n{Colors.OKGREEN}Simple OCR:{Colors.ENDC}  {page_stats['simple_count']:3d} pages ({simple_pct:.1f}%)")
    print(f"{Colors.OKCYAN}DeepSeek:{Colors.ENDC}     {page_stats['deepseek_count']:3d} pages ({deepseek_pct:.1f}%)")
    
    # Show page-by-page routing
    print(f"\n{Colors.BOLD}Page routing:{Colors.ENDC}")
    for page in pages[:10]:  # Show first 10 pages
        page_num = page.get('page_number', 0)
        engine = page.get('engine_used', 'unknown')
        complexity = page.get('complexity', {})
        complexity_type = complexity.get('complexity', 'unknown') if complexity else 'unknown'
        
        if engine == 'simple':
            emoji = '⚡'
            color = Colors.OKGREEN
        elif engine == 'deepseek':
            emoji = '🧠'
            color = Colors.OKCYAN
        else:
            emoji = '❓'
            color = Colors.WARNING
        
        print(f"   {color}{emoji} Page {page_num:3d}: {complexity_type.capitalize():7s} → {engine.upper():8s}{Colors.ENDC}")
    
    if len(pages) > 10:
        print(f"   {Colors.OKBLUE}... and {len(pages) - 10} more pages{Colors.ENDC}")
    
    # 5️⃣ Bounding Boxes
    print_step(5, "Bounding Boxes")
    valid_pages = page_stats['pages_with_valid_bboxes']
    total_pages = page_stats['total_pages']
    valid_pct = page_stats['pages_with_valid_bboxes_percentage']
    
    print(f"\n{Colors.OKGREEN}Valid:{Colors.ENDC}   {valid_pages:3d}/{total_pages} pages ({valid_pct:.1f}%)")
    print(f"{Colors.WARNING}Invalid:{Colors.ENDC} {total_pages - valid_pages:3d}/{total_pages} pages ({100 - valid_pct:.1f}%)")
    
    # 6️⃣ Metadata Extracted
    print_step(6, "Metadata Extracted")
    title = content_meta.get('title')
    authors = content_meta.get('authors', [])
    date = content_meta.get('date')
    org = content_meta.get('organization')
    doc_num = content_meta.get('document_number')
    
    if title:
        print(f"\n{Colors.BOLD}Title:{Colors.ENDC} {Colors.OKCYAN}\"{title}\"{Colors.ENDC}")
    else:
        print(f"\n{Colors.BOLD}Title:{Colors.ENDC} {Colors.WARNING}Not found{Colors.ENDC}")
    
    if authors:
        authors_str = ', '.join(authors[:5])
        if len(authors) > 5:
            authors_str += f" (+{len(authors) - 5} more)"
        print(f"{Colors.BOLD}Authors:{Colors.ENDC} {Colors.OKCYAN}[{authors_str}]{Colors.ENDC}")
    else:
        print(f"{Colors.BOLD}Authors:{Colors.ENDC} {Colors.WARNING}Not found{Colors.ENDC}")
    
    if date:
        print(f"{Colors.BOLD}Date:{Colors.ENDC} {Colors.OKCYAN}{date}{Colors.ENDC}")
    else:
        print(f"{Colors.BOLD}Date:{Colors.ENDC} {Colors.WARNING}Not found{Colors.ENDC}")
    
    if org:
        print(f"{Colors.BOLD}Organization:{Colors.ENDC} {Colors.OKCYAN}{org}{Colors.ENDC}")
    else:
        print(f"{Colors.BOLD}Organization:{Colors.ENDC} {Colors.WARNING}Not found{Colors.ENDC}")
    
    if doc_num:
        print(f"{Colors.BOLD}Document Number:{Colors.ENDC} {Colors.OKCYAN}{doc_num}{Colors.ENDC}")
    
    # 7️⃣ Document Structure (if research paper)
    if doc_structure and classification.get('type') == 'research_paper':
        print_step(7, "Document Structure")
        sections = doc_structure.get('sections', {})
        section_names = ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion', 'references']
        
        print()
        for section_name in section_names:
            section_data = sections.get(section_name)
            if section_data:
                word_count = section_data.get('word_count', 0)
                print(f"   {Colors.OKGREEN}✓{Colors.ENDC} {section_name.capitalize():15s} found ({word_count:,} words)")
            else:
                print(f"   {Colors.WARNING}✗{Colors.ENDC} {section_name.capitalize():15s} not found")
        
        total_sections = doc_structure.get('total_sections', 0)
        total_words = doc_structure.get('total_words', 0)
        print(f"\n{Colors.BOLD}Total sections:{Colors.ENDC} {Colors.OKBLUE}{total_sections}{Colors.ENDC}")
        print(f"{Colors.BOLD}Total words in structure:{Colors.ENDC} {Colors.OKBLUE}{total_words:,}{Colors.ENDC}")
    else:
        print_step(7, "Document Structure")
        print(f"\n{Colors.WARNING}Not a research paper - structure analysis skipped{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Document type: {classification.get('type', 'unknown')}{Colors.ENDC}")
    
    # 8️⃣ Performance Stats
    print_step(8, "Performance Statistics")
    avg_time = router_stats.get('avg_processing_time', 0.0)
    simple_time = router_stats.get('simple_avg_time', 0.0)
    deepseek_time = router_stats.get('deepseek_avg_time', 0.0)
    quality_score = training_ann.get('quality_score', 0.0)
    
    print(f"\n{Colors.BOLD}Average processing time per page:{Colors.ENDC} {Colors.OKBLUE}{avg_time:.2f}s{Colors.ENDC}")
    if simple_time > 0:
        print(f"{Colors.BOLD}Simple OCR avg time:{Colors.ENDC} {Colors.OKGREEN}{simple_time:.2f}s{Colors.ENDC}")
    if deepseek_time > 0:
        print(f"{Colors.BOLD}DeepSeek avg time:{Colors.ENDC} {Colors.OKCYAN}{deepseek_time:.2f}s{Colors.ENDC}")
    print(f"{Colors.BOLD}Quality score:{Colors.ENDC} {Colors.OKBLUE}{quality_score:.2f}/1.00{Colors.ENDC}")
    print(f"{Colors.BOLD}Total blocks extracted:{Colors.ENDC} {Colors.OKBLUE}{page_stats['total_blocks']:,}{Colors.ENDC}")
    print(f"{Colors.BOLD}Has tables:{Colors.ENDC} {Colors.OKBLUE}{'Yes' if training_ann.get('has_tables') else 'No'}{Colors.ENDC}")

def save_json_output(result: Dict, pdf_path: str) -> str:
    """Save full JSON output to file"""
    output_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}_{output_id}.json"
    
    output_data = {
        'test_info': {
            'timestamp': datetime.now().isoformat(),
            'pdf_path': pdf_path,
            'test_type': 'full_pipeline_test'
        },
        'result': result
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return filename

def main():
    """Main test function"""
    print_header("Full Hybrid OCR Pipeline Test")
    
    try:
        # Step 1: Find test PDF
        pdf_path = find_test_pdf()
        if not pdf_path:
            print_error("No PDF file provided. Exiting.")
            sys.exit(1)
        
        # Step 2: Initialize pipeline
        pipeline, deepseek_processor = initialize_pipeline()
        if not pipeline:
            print_error("Failed to initialize pipeline. Exiting.")
            sys.exit(1)
        
        # Step 3: Process document
        print_step(3, f"Processing document: {os.path.basename(pdf_path)}")
        print()
        print_info("Analyzing complexity...")
        print_info("Routing pages through hybrid system...")
        print()
        
        start_time = datetime.now()
        try:
            result = pipeline.process_document(
                file_path=pdf_path,
                filename=os.path.basename(pdf_path)
            )
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            print_success(f"Document processed successfully in {processing_time:.2f}s")
        except Exception as e:
            print_error(f"Failed to process document: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Step 4: Display results
        display_results(result, pdf_path)
        
        # Step 5: Save JSON output
        print_step(9, "Saving Results")
        output_file = save_json_output(result, pdf_path)
        print_success(f"Full JSON saved to: {Colors.BOLD}{output_file}{Colors.ENDC}")
        
        # Final summary
        print_header("Test Complete")
        print(f"{Colors.BOLD}Summary:{Colors.ENDC}")
        print(f"  PDF: {os.path.basename(pdf_path)}")
        print(f"  Pages: {result.get('document_metadata', {}).get('total_pages', 0)}")
        print(f"  Type: {result.get('classification_details', {}).get('type', 'unknown')}")
        print(f"  Processing time: {result.get('document_metadata', {}).get('processing_time_seconds', 0):.2f}s")
        print(f"  Quality score: {result.get('training_annotations', {}).get('quality_score', 0.0):.2f}/1.00")
        print(f"\n{Colors.OKBLUE}Full results saved to: {output_file}{Colors.ENDC}\n")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Test interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

