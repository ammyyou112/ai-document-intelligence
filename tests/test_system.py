#!/usr/bin/env python3
"""
Comprehensive test suite for hybrid OCR system.

Tests: bbox extraction, complexity analysis, hybrid routing, full pipeline, output validation.

Usage:
    python tests/test_system.py                    # Run all tests
    python tests/test_system.py --test bbox        # Run specific test
    python tests/test_system.py --pdf sample.pdf   # Test on specific PDF
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.processors.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.analyzers.document_complexity_analyzer import DocumentComplexityAnalyzer
from deepseek_ocr_wrapper import DeepSeekOCR
from pdf2image import convert_from_path
import json


class SystemTester:
    """Comprehensive system testing"""
    
    def __init__(self):
        self.results = {}
    
    def test_bbox_extraction(self, image_path):
        """Test 1: Verify bbox extraction works"""
        print("\n" + "="*70)
        print("TEST 1: Bounding Box Extraction")
        print("="*70)
        
        try:
            ocr = DeepSeekOCR()
            result = ocr.process(image_path)
            
            structured_data = result.get('structured_data', [])
            total = len(structured_data)
            valid = sum(1 for item in structured_data if item.get('bbox', [0,0,0,0]) != [0,0,0,0])
            
            print(f"📊 Total blocks: {total}")
            if total > 0:
                print(f"📐 Valid bboxes: {valid}/{total} ({valid/total*100:.1f}%)")
            else:
                print(f"📐 Valid bboxes: {valid}/{total} (0.0%)")
            
            success = valid == total and total > 0
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}\n")
            
            self.results['bbox_extraction'] = success
            return success
        except Exception as e:
            print(f"❌ FAIL: {str(e)}\n")
            self.results['bbox_extraction'] = False
            return False
    
    def test_complexity_analysis(self, pdf_path):
        """Test 2: Verify complexity analysis"""
        print("\n" + "="*70)
        print("TEST 2: Complexity Analysis")
        print("="*70)
        
        try:
            analyzer = DocumentComplexityAnalyzer()
            images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=3)
            
            results = []
            for i, image in enumerate(images, 1):
                analysis = analyzer.analyze(image)
                results.append(analysis)
                print(f"Page {i}: {analysis.complexity.upper()} "
                      f"(conf: {analysis.confidence:.2f}) - {analysis.reasons[0] if analysis.reasons else 'N/A'}")
            
            success = len(results) > 0
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}\n")
            
            self.results['complexity_analysis'] = success
            return success
        except Exception as e:
            print(f"❌ FAIL: {str(e)}\n")
            self.results['complexity_analysis'] = False
            return False
    
    def test_hybrid_routing(self, pdf_path):
        """Test 3: Verify hybrid routing"""
        print("\n" + "="*70)
        print("TEST 3: Hybrid Routing")
        print("="*70)
        
        try:
            pipeline = EnhancedOCRPipeline()
            result = pipeline.process_pdf(pdf_path)
            
            engines = result.get('training_annotations', {}).get('engines_distribution', {})
            simple_pages = engines.get('simple', 0)
            complex_pages = engines.get('deepseek', 0)
            
            print(f"🔀 Engine distribution: {engines}")
            print(f"⚡ Simple pages (Fast OCR): {simple_pages}")
            print(f"🧠 Complex pages (DeepSeek): {complex_pages}")
            
            success = len(engines) > 0
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}\n")
            
            self.results['hybrid_routing'] = success
            return success
        except Exception as e:
            print(f"❌ FAIL: {str(e)}\n")
            self.results['hybrid_routing'] = False
            return False
    
    def test_full_pipeline(self, pdf_path):
        """Test 4: End-to-end pipeline"""
        print("\n" + "="*70)
        print("TEST 4: Full Pipeline End-to-End")
        print("="*70)
        
        pipeline = EnhancedOCRPipeline()
        output_path = 'tests/test_output.json'
        
        try:
            result = pipeline.process_pdf(pdf_path, output_path)
            
            required_keys = ['document_metadata', 'content_metadata', 'pages', 'training_annotations']
            has_all_keys = all(key in result for key in required_keys)
            has_output_file = os.path.exists(output_path)
            
            print(f"📄 Pages processed: {result.get('content_metadata', {}).get('pages', 'N/A')}")
            print(f"📋 Document type: {result.get('document_metadata', {}).get('type', 'N/A')}")
            print(f"✓ Required keys: {has_all_keys}")
            print(f"✓ Output file: {has_output_file}")
            
            success = has_all_keys and has_output_file
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}\n")
            
            self.results['full_pipeline'] = success
            return success
            
        except Exception as e:
            print(f"❌ FAIL: {str(e)}\n")
            self.results['full_pipeline'] = False
            return False
    
    def test_output_validation(self):
        """Test 5: Validate output JSON"""
        print("\n" + "="*70)
        print("TEST 5: Output Validation")
        print("="*70)
        
        output_path = 'tests/test_output.json'
        
        if not os.path.exists(output_path):
            print("❌ FAIL: Output file not found\n")
            self.results['output_validation'] = False
            return False
        
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            checks = {
                'Has pages': len(data.get('pages', [])) > 0,
                'Has metadata': 'document_metadata' in data,
                'Has bboxes': any(
                    any(b.get('bbox', [0,0,0,0]) != [0,0,0,0] for b in page.get('blocks', []))
                    for page in data.get('pages', [])
                ),
                'Has engine stats': 'engines_distribution' in data.get('training_annotations', {})
            }
            
            for check, passed in checks.items():
                print(f"{'✓' if passed else '✗'} {check}")
            
            success = all(checks.values())
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}\n")
            
            self.results['output_validation'] = success
            return success
            
        except Exception as e:
            print(f"❌ FAIL: {str(e)}\n")
            self.results['output_validation'] = False
            return False
    
    def run_all_tests(self, pdf_path):
        """Run all tests"""
        print("\n" + "="*70)
        print("🧪 HYBRID OCR SYSTEM - COMPREHENSIVE TEST SUITE")
        print("="*70)
        
        # Convert first page for bbox test
        try:
            images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
            test_image_path = 'tests/test_page_1.png'
            os.makedirs('tests', exist_ok=True)
            images[0].save(test_image_path)
        except Exception as e:
            print(f"⚠️  Could not create test image: {e}")
            test_image_path = None
        
        # Run all tests
        if test_image_path:
            self.test_bbox_extraction(test_image_path)
        self.test_complexity_analysis(pdf_path)
        self.test_hybrid_routing(pdf_path)
        self.test_full_pipeline(pdf_path)
        self.test_output_validation()
        
        # Summary
        print("="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        for test_name, passed in self.results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        total = len(self.results)
        passed = sum(self.results.values())
        
        if total > 0:
            print(f"\n🎯 Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
        
        return all(self.results.values())


def main():
    parser = argparse.ArgumentParser(description='Test hybrid OCR system')
    parser.add_argument('--pdf', help='PDF file to test with')
    parser.add_argument('--test', choices=['bbox', 'complexity', 'routing', 'pipeline', 'validation', 'all'],
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    # Find PDF
    if not args.pdf:
        pdf_candidates = [
            'uploads/test.pdf',
            '19840015553-1-5.pdf',
            'test.pdf',
            'sample.pdf'
        ]
        for pdf in pdf_candidates:
            if os.path.exists(pdf):
                args.pdf = pdf
                break
        
        if not args.pdf:
            print("❌ No PDF found. Provide --pdf argument")
            sys.exit(1)
    
    print(f"📄 Testing with: {args.pdf}")
    
    # Create tests directory
    os.makedirs('tests', exist_ok=True)
    
    tester = SystemTester()
    
    if args.test == 'all':
        success = tester.run_all_tests(args.pdf)
    else:
        if args.test == 'bbox':
            images = convert_from_path(args.pdf, dpi=300, first_page=1, last_page=1)
            test_image = 'tests/test_page_1.png'
            os.makedirs('tests', exist_ok=True)
            images[0].save(test_image)
            success = tester.test_bbox_extraction(test_image)
        elif args.test == 'complexity':
            success = tester.test_complexity_analysis(args.pdf)
        elif args.test == 'routing':
            success = tester.test_hybrid_routing(args.pdf)
        elif args.test == 'pipeline':
            success = tester.test_full_pipeline(args.pdf)
        elif args.test == 'validation':
            success = tester.test_output_validation()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

