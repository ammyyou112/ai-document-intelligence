#!/usr/bin/env python3
"""
Full Hybrid Pipeline Test - PDF Focused

Tests the complete Enhanced OCR Pipeline with PDF documents.
Shows page-by-page progress and comprehensive results.
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

def find_pdf() -> Optional[str]:
    """Find a PDF file or ask user for path"""
    print_step(1, "Finding PDF file...")
    
    # Check common directories
    search_dirs = [
        'uploads',
        'examples',
        '.',
        'test_documents',
        'test',
        'content'
    ]
    
    found_pdfs = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for path in Path(search_dir).glob("*.pdf"):
                found_pdfs.append(str(path))
    
    if found_pdfs:
        # Prefer PDFs in uploads/ or examples/
        preferred = [pdf for pdf in found_pdfs if 'upload' in pdf.lower() or 'example' in pdf.lower() or 'content' in pdf.lower()]
        if preferred:
            selected = preferred[0]
        else:
            selected = found_pdfs[0]
        
        print_success(f"Using: {selected}")
        return selected
    else:
        print_warning("No PDFs found in common directories")
        print_info("Please provide a path to a PDF file")
        user_path = input(f"{Colors.BOLD}Enter PDF path: {Colors.ENDC}").strip().strip('"').strip("'")
        
        if os.path.exists(user_path) and user_path.lower().endswith('.pdf'):
            print_success(f"Using: {user_path}")
            return user_path
        elif os.path.exists(user_path):
            print_error(f"File is not a PDF: {user_path}")
            return None
        else:
            print_error(f"PDF not found: {user_path}")
            return None

def initialize_pipeline() -> Tuple[Optional[Any], Optional[Callable]]:
    """Initialize the Enhanced OCR Pipeline with DeepSeek processor"""
    print_step(2, "Initializing Enhanced OCR Pipeline...")
    
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
            print_info("Setting up DeepSeek-OCR processor (lazy loading)...")
            
            # Create a lazy loader function
            _deepseek_model = None
            def deepseek_processor_func(image_path: str):
                """Wrapper for DeepSeek OCR processing"""
                nonlocal _deepseek_model
                if _deepseek_model is None:
                    print_info("   Loading DeepSeek-OCR model (first use)...")
                    _deepseek_model = DeepSeekOCR()
                return _deepseek_model.process(image_path)
            
            deepseek_processor = deepseek_processor_func
            print_success("DeepSeek processor ready")
        except ImportError as e:
            print_warning(f"DeepSeek-OCR not available: {e}")
            print_info("Pipeline will use Simple OCR only")
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
            print_success("Pipeline ready (all components loaded)")
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
        'valid_percentage': (valid / total * 100) if total > 0 else 0.0
    }

def display_page_progress(pages: List[Dict], total_pages: int):
    """Display progress for each page"""
    print()
    for page in pages:
        page_num = page.get('page_number', 0)
        engine = page.get('engine_used', 'unknown')
        complexity = page.get('complexity', {})
        complexity_type = complexity.get('complexity', 'unknown') if complexity else 'unknown'
        blocks = page.get('blocks', [])
        
        # Validate bboxes for this page
        bbox_stats = validate_bboxes(blocks)
        
        # Determine colors and emojis
        if engine == 'simple':
            engine_display = 'Fast OCR'
            engine_color = Colors.OKGREEN
            engine_emoji = '⚡'
        elif engine == 'deepseek':
            engine_display = 'DeepSeek-OCR'
            engine_color = Colors.OKCYAN
            engine_emoji = '🧠'
        else:
            engine_display = engine.upper()
            engine_color = Colors.WARNING
            engine_emoji = '❓'
        
        complexity_display = complexity_type.capitalize()
        if complexity_type == 'simple':
            complexity_color = Colors.OKGREEN
        elif complexity_type == 'complex':
            complexity_color = Colors.OKCYAN
        else:
            complexity_color = Colors.WARNING
        
        # Bbox status
        if bbox_stats['valid'] == bbox_stats['total'] and bbox_stats['total'] > 0:
            bbox_status = f"{Colors.OKGREEN}✓{Colors.ENDC}"
            bbox_text = f"{bbox_stats['valid']}/{bbox_stats['total']} valid"
        elif bbox_stats['valid'] > 0:
            bbox_status = f"{Colors.WARNING}⚠{Colors.ENDC}"
            bbox_text = f"{bbox_stats['valid']}/{bbox_stats['total']} valid ({bbox_stats['invalid']} invalid)"
        else:
            bbox_status = f"{Colors.FAIL}✗{Colors.ENDC}"
            bbox_text = f"0/{bbox_stats['total']} valid"
        
        # Print page info
        print(f"Page {page_num:3d}/{total_pages}: {complexity_color}{complexity_display:7s}{Colors.ENDC} → {engine_color}{engine_emoji} {engine_display:12s}{Colors.ENDC}")
        print(f"   Bboxes: {bbox_text} {bbox_status}")
        print()

def analyze_results(result: Dict) -> Dict:
    """Analyze results and extract statistics"""
    pages = result.get('pages', [])
    doc_meta = result.get('document_metadata', {})
    content_meta = result.get('content_metadata', {})
    training_ann = result.get('training_annotations', {})
    engines_dist = training_ann.get('engines_distribution', {})
    
    # Page statistics
    total_pages = len(pages)
    simple_count = engines_dist.get('simple', 0)
    deepseek_count = engines_dist.get('deepseek', 0)
    simple_percentage = engines_dist.get('simple_percentage', 0.0)
    deepseek_percentage = engines_dist.get('deepseek_percentage', 0.0)
    
    # Bbox validation
    pages_with_valid_bboxes = 0
    total_blocks = 0
    total_valid_bboxes = 0
    
    for page in pages:
        blocks = page.get('blocks', [])
        total_blocks += len(blocks)
        bbox_stats = validate_bboxes(blocks)
        total_valid_bboxes += bbox_stats['valid']
        if bbox_stats['valid'] > 0:
            pages_with_valid_bboxes += 1
    
    bbox_valid_percentage = (pages_with_valid_bboxes / total_pages * 100) if total_pages > 0 else 0.0
    
    return {
        'total_pages': total_pages,
        'simple_count': simple_count,
        'deepseek_count': deepseek_count,
        'simple_percentage': simple_percentage,
        'deepseek_percentage': deepseek_percentage,
        'total_words': content_meta.get('words', 0),
        'pages_with_valid_bboxes': pages_with_valid_bboxes,
        'bbox_valid_percentage': bbox_valid_percentage,
        'total_blocks': total_blocks,
        'total_valid_bboxes': total_valid_bboxes
    }

def display_results(result: Dict, pdf_path: str):
    """Display comprehensive results"""
    print_step(4, "Results Summary")
    print()
    
    # Extract data
    classification = result.get('classification_details', {})
    content_meta = result.get('content_metadata', {})
    doc_structure = result.get('document_structure')
    stats = analyze_results(result)
    
    # Document Classification
    print(f"{Colors.BOLD}📄 Document Classification:{Colors.ENDC}")
    doc_type = classification.get('type', 'unknown')
    confidence = classification.get('confidence', 0.0) * 100
    print(f"   Type: {Colors.OKCYAN}{doc_type}{Colors.ENDC}")
    print(f"   Confidence: {Colors.OKBLUE}{confidence:.1f}%{Colors.ENDC}")
    print()
    
    # Processing Stats
    print(f"{Colors.BOLD}📊 Processing Stats:{Colors.ENDC}")
    print(f"   Total Pages: {Colors.OKBLUE}{stats['total_pages']}{Colors.ENDC}")
    print(f"   Simple Pages: {Colors.OKGREEN}{stats['simple_count']}{Colors.ENDC} ({stats['simple_percentage']:.1f}%) → Fast OCR")
    print(f"   Complex Pages: {Colors.OKCYAN}{stats['deepseek_count']}{Colors.ENDC} ({stats['deepseek_percentage']:.1f}%) → DeepSeek")
    print(f"   Total Words: {Colors.OKBLUE}{stats['total_words']:,}{Colors.ENDC}")
    print()
    
    # Bounding Box Validation
    print(f"{Colors.BOLD}📐 Bounding Box Validation:{Colors.ENDC}")
    valid_pages = stats['pages_with_valid_bboxes']
    total_pages = stats['total_pages']
    invalid_pages = total_pages - valid_pages
    print(f"   Valid: {Colors.OKGREEN}{valid_pages}/{total_pages} pages{Colors.ENDC} ({stats['bbox_valid_percentage']:.1f}%)")
    print(f"   Invalid: {Colors.WARNING}{invalid_pages}/{total_pages} pages{Colors.ENDC} ({100 - stats['bbox_valid_percentage']:.1f}%)")
    print()
    
    # Metadata Extracted
    print(f"{Colors.BOLD}📝 Metadata Extracted:{Colors.ENDC}")
    title = content_meta.get('title')
    authors = content_meta.get('authors', [])
    date = content_meta.get('date')
    org = content_meta.get('organization')
    doc_num = content_meta.get('document_number')
    
    if title:
        print(f"   Title: {Colors.OKCYAN}\"{title}\"{Colors.ENDC}")
    else:
        print(f"   Title: {Colors.WARNING}Not found{Colors.ENDC}")
    
    if authors:
        authors_str = ', '.join([f'"{a}"' for a in authors[:3]])
        if len(authors) > 3:
            authors_str += f" (+{len(authors) - 3} more)"
        print(f"   Authors: [{Colors.OKCYAN}{authors_str}{Colors.ENDC}]")
    else:
        print(f"   Authors: {Colors.WARNING}Not found{Colors.ENDC}")
    
    if date:
        print(f"   Date: {Colors.OKCYAN}{date}{Colors.ENDC}")
    else:
        print(f"   Date: {Colors.WARNING}Not found{Colors.ENDC}")
    
    if org:
        print(f"   Organization: {Colors.OKCYAN}{org}{Colors.ENDC}")
    else:
        print(f"   Organization: {Colors.WARNING}Not found{Colors.ENDC}")
    
    if doc_num:
        print(f"   Doc Number: {Colors.OKCYAN}{doc_num}{Colors.ENDC}")
    print()
    
    # Document Structure (if research paper)
    if doc_structure and classification.get('type') == 'research_paper':
        print(f"{Colors.BOLD}📚 Document Structure (Research Paper):{Colors.ENDC}")
        sections = doc_structure.get('sections', {})
        section_names = ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion', 'references']
        
        for section_name in section_names:
            section_data = sections.get(section_name)
            if section_data:
                word_count = section_data.get('word_count', 0)
                if section_name == 'references':
                    # References might have entry count instead
                    ref_text = section_data.get('text', '')
                    if ref_text:
                        # Count reference entries (rough estimate)
                        ref_count = len([line for line in ref_text.split('\n') if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('['))])
                        if ref_count > 0:
                            print(f"   {Colors.OKGREEN}✓{Colors.ENDC} {section_name.capitalize():15s}: {Colors.OKBLUE}{ref_count} entries{Colors.ENDC}")
                        else:
                            print(f"   {Colors.OKGREEN}✓{Colors.ENDC} {section_name.capitalize():15s}: {Colors.OKBLUE}{word_count} words{Colors.ENDC}")
                    else:
                        print(f"   {Colors.OKGREEN}✓{Colors.ENDC} {section_name.capitalize():15s}: {Colors.OKBLUE}{word_count} words{Colors.ENDC}")
                else:
                    print(f"   {Colors.OKGREEN}✓{Colors.ENDC} {section_name.capitalize():15s}: {Colors.OKBLUE}{word_count:,} words{Colors.ENDC}")
            else:
                print(f"   {Colors.WARNING}✗{Colors.ENDC} {section_name.capitalize():15s}: {Colors.WARNING}Not found{Colors.ENDC}")
        print()
    else:
        print(f"{Colors.BOLD}📚 Document Structure:{Colors.ENDC}")
        print(f"   {Colors.WARNING}Not a research paper - structure analysis skipped{Colors.ENDC}")
        print(f"   Document type: {Colors.OKBLUE}{classification.get('type', 'unknown')}{Colors.ENDC}")
        print()

def save_json_output(result: Dict, pdf_path: str) -> str:
    """Save full JSON output to file"""
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{pdf_basename}_{timestamp}.json"
    
    output_data = {
        'test_info': {
            'timestamp': datetime.now().isoformat(),
            'pdf_path': pdf_path,
            'test_type': 'full_pipeline_pdf_test'
        },
        'result': result
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return filename

def main():
    """Main test function"""
    print_header("Full Hybrid Pipeline - PDF Test")
    
    try:
        # Step 1: Find PDF
        pdf_path = find_pdf()
        if not pdf_path:
            print_error("No PDF file provided. Exiting.")
            sys.exit(1)
        
        # Step 2: Initialize pipeline
        pipeline, deepseek_processor = initialize_pipeline()
        if not pipeline:
            print_error("Failed to initialize pipeline. Exiting.")
            sys.exit(1)
        
        # Step 3: Process PDF
        print_step(3, f"Processing PDF ({os.path.basename(pdf_path)})...")
        
        # We need to process and show progress page by page
        # The pipeline processes all pages, but we can show progress
        print()
        print_info("Processing all pages through hybrid OCR system...")
        print()
        
        start_time = datetime.now()
        try:
            result = pipeline.process_document(
                file_path=pdf_path,
                filename=os.path.basename(pdf_path)
            )
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Get pages for progress display
            pages = result.get('pages', [])
            total_pages = len(pages)
            
            # Display page-by-page progress
            display_page_progress(pages, total_pages)
            
            print_success(f"PDF processed successfully in {processing_time:.2f}s")
            print()
            
        except Exception as e:
            print_error(f"Failed to process PDF: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Step 4: Display results
        display_results(result, pdf_path)
        
        # Step 5: Save JSON output
        output_file = save_json_output(result, pdf_path)
        print_success(f"Full JSON saved to: {Colors.BOLD}{output_file}{Colors.ENDC}")
        print()
        
        # Final summary
        print_header("Test Complete")
        print(f"{Colors.BOLD}Summary:{Colors.ENDC}")
        print(f"  PDF: {os.path.basename(pdf_path)}")
        print(f"  Pages: {result.get('document_metadata', {}).get('total_pages', 0)}")
        print(f"  Type: {result.get('classification_details', {}).get('type', 'unknown')}")
        print(f"  Processing time: {processing_time:.2f}s")
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

