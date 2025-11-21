#!/usr/bin/env python3
"""
Test script to verify complexity analyzer fix

Tests that simple pages route to Fast OCR and complex pages route to DeepSeek.
"""

import os
import sys
import logging
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path

# Setup logging to see detailed analysis
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Color codes
class Colors:
    OKGREEN = '\033[92m'
    OKCYAN = '\033[96m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(text: str):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_info(text: str):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def find_pdf():
    """Find a PDF file"""
    search_dirs = ['uploads', 'examples', '.', 'content']
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for path in Path(search_dir).glob("*.pdf"):
                return str(path)
    
    user_path = input(f"{Colors.BOLD}Enter PDF path: {Colors.ENDC}").strip().strip('"').strip("'")
    if os.path.exists(user_path):
        return user_path
    return None

def test_complexity_analyzer():
    """Test the complexity analyzer"""
    print_header("Testing Complexity Analyzer Fix")
    
    # Find PDF
    pdf_path = find_pdf()
    if not pdf_path:
        print(f"{Colors.FAIL}❌ No PDF found{Colors.ENDC}")
        return
    
    print_info(f"Testing with: {os.path.basename(pdf_path)}")
    
    # Import analyzer
    try:
        from app.analyzers.document_complexity_analyzer import DocumentComplexityAnalyzer
        analyzer = DocumentComplexityAnalyzer()
        print_success("Complexity analyzer loaded")
    except ImportError as e:
        print(f"{Colors.FAIL}❌ Failed to import analyzer: {e}{Colors.ENDC}")
        return
    
    # Convert PDF to images
    try:
        print_info("Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=300)
        print_success(f"Converted to {len(images)} page(s)")
    except Exception as e:
        print(f"{Colors.FAIL}❌ Failed to convert PDF: {e}{Colors.ENDC}")
        return
    
    # Analyze each page
    print()
    print_header("Page-by-Page Complexity Analysis")
    
    simple_count = 0
    complex_count = 0
    
    for i, image in enumerate(images, 1):
        print(f"\n{Colors.BOLD}Page {i}/{len(images)}:{Colors.ENDC}")
        result = analyzer.analyze(image)
        
        if result.complexity == 'simple':
            simple_count += 1
            engine = f"{Colors.OKGREEN}Fast OCR{Colors.ENDC}"
            emoji = "⚡"
        else:
            complex_count += 1
            engine = f"{Colors.OKCYAN}DeepSeek-OCR{Colors.ENDC}"
            emoji = "🧠"
        
        print(f"   {emoji} Complexity: {Colors.BOLD}{result.complexity.upper()}{Colors.ENDC} (confidence: {result.confidence:.2f})")
        print(f"   → Engine: {engine}")
        if result.reasons:
            print(f"   → Reasons: {', '.join(result.reasons)}")
    
    # Summary
    print()
    print_header("Summary")
    print(f"Total pages: {len(images)}")
    print(f"{Colors.OKGREEN}Simple pages: {simple_count} → Fast OCR{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Complex pages: {complex_count} → DeepSeek-OCR{Colors.ENDC}")
    print(f"Simple percentage: {simple_count/len(images)*100:.1f}%")
    print(f"Complex percentage: {complex_count/len(images)*100:.1f}%")
    
    # Expected result for NASA PDF
    if len(images) == 5:
        print()
        print_info("Expected for 5-page NASA PDF:")
        print("   Pages 1-4: SIMPLE → Fast OCR")
        print("   Page 5: COMPLEX → DeepSeek")
        if simple_count == 4 and complex_count == 1:
            print_success("✅ Test PASSED! Routing matches expected behavior")
        else:
            print(f"{Colors.WARNING}⚠️  Test needs review: Expected 4 simple + 1 complex, got {simple_count} simple + {complex_count} complex{Colors.ENDC}")

if __name__ == '__main__':
    try:
        test_complexity_analyzer()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Test interrupted{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()

