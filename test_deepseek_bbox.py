#!/usr/bin/env python3
"""
Test script to verify DeepSeek-OCR bounding box extraction

Tests whether the DeepSeek-OCR wrapper correctly extracts real bounding boxes
instead of dummy [0, 0, 0, 0] values.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
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
    emoji = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟'][step_num - 1]
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

def find_test_image() -> str:
    """Find a test image in common directories"""
    print_step(1, "Finding test image...")
    
    # Check common directories
    search_dirs = [
        'uploads',
        'examples',
        '.',
        'test_images',
        'test'
    ]
    
    # Supported image extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    
    found_images = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for ext in image_extensions:
                pattern = f"*{ext}"
                for path in Path(search_dir).glob(pattern):
                    found_images.append(str(path))
    
    if found_images:
        # Prefer images in uploads/ or examples/
        preferred = [img for img in found_images if 'upload' in img.lower() or 'example' in img.lower()]
        if preferred:
            selected = preferred[0]
        else:
            selected = found_images[0]
        
        print_success(f"Found test image: {selected}")
        return selected
    else:
        print_warning("No test images found in common directories")
        print_info("Please provide a path to a test image")
        user_path = input(f"{Colors.BOLD}Enter image path: {Colors.ENDC}").strip()
        
        if os.path.exists(user_path):
            print_success(f"Using provided image: {user_path}")
            return user_path
        else:
            print_error(f"Image not found: {user_path}")
            sys.exit(1)

def analyze_bboxes(structured_data: List[Dict]) -> Dict:
    """Analyze bounding boxes in structured data"""
    total = len(structured_data)
    valid = 0
    invalid = 0
    valid_bboxes = []
    invalid_bboxes = []
    
    for item in structured_data:
        bbox = item.get('bbox', [0, 0, 0, 0])
        text = item.get('text', '')
        confidence = item.get('confidence', 0.0)
        
        # Check if bbox is valid (not [0, 0, 0, 0] and has positive dimensions)
        is_valid = (
            bbox != [0, 0, 0, 0] and
            len(bbox) == 4 and
            isinstance(bbox[0], (int, float)) and
            bbox[2] > bbox[0] and  # x2 > x1
            bbox[3] > bbox[1]      # y2 > y1
        )
        
        bbox_info = {
            'text': text[:50] + '...' if len(text) > 50 else text,
            'bbox': bbox,
            'confidence': confidence,
            'width': bbox[2] - bbox[0] if len(bbox) == 4 else 0,
            'height': bbox[3] - bbox[1] if len(bbox) == 4 else 0
        }
        
        if is_valid:
            valid += 1
            valid_bboxes.append(bbox_info)
        else:
            invalid += 1
            invalid_bboxes.append(bbox_info)
    
    return {
        'total': total,
        'valid': valid,
        'invalid': invalid,
        'valid_percentage': (valid / total * 100) if total > 0 else 0.0,
        'invalid_percentage': (invalid / total * 100) if total > 0 else 0.0,
        'valid_bboxes': valid_bboxes[:10],  # First 10 valid
        'invalid_bboxes': invalid_bboxes[:5],  # First 5 invalid
        'all_bboxes': [item.get('bbox', [0, 0, 0, 0]) for item in structured_data]
    }

def print_bbox_statistics(stats: Dict):
    """Print detailed bbox statistics"""
    print_step(6, "Bounding Box Statistics")
    print(f"\n{Colors.BOLD}Total Blocks: {stats['total']}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}Valid Bboxes: {stats['valid']} ({stats['valid_percentage']:.1f}%){Colors.ENDC}")
    print(f"{Colors.FAIL}Invalid Bboxes: {stats['invalid']} ({stats['invalid_percentage']:.1f}%){Colors.ENDC}")
    
    if stats['valid'] > 0:
        print(f"\n{Colors.BOLD}Sample Valid Bboxes:{Colors.ENDC}")
        for i, bbox_info in enumerate(stats['valid_bboxes'][:3], 1):
            print(f"  {Colors.OKGREEN}{i}.{Colors.ENDC} Text: {Colors.BOLD}'{bbox_info['text']}'{Colors.ENDC}")
            print(f"     Bbox: {Colors.OKCYAN}{bbox_info['bbox']}{Colors.ENDC}")
            print(f"     Size: {bbox_info['width']}x{bbox_info['height']} px")
            print(f"     Confidence: {bbox_info['confidence']:.2f}")
            print()
    
    if stats['invalid'] > 0:
        print(f"\n{Colors.BOLD}Sample Invalid Bboxes:{Colors.ENDC}")
        for i, bbox_info in enumerate(stats['invalid_bboxes'][:3], 1):
            print(f"  {Colors.FAIL}{i}.{Colors.ENDC} Text: {Colors.BOLD}'{bbox_info['text']}'{Colors.ENDC}")
            print(f"     Bbox: {Colors.WARNING}{bbox_info['bbox']}{Colors.ENDC} (invalid)")
            print()

def get_verdict(stats: Dict) -> Tuple[str, str]:
    """Determine test verdict based on statistics"""
    valid_pct = stats['valid_percentage']
    
    if valid_pct >= 80:
        return "SUCCESS", f"{Colors.OKGREEN}✅ SUCCESS: {valid_pct:.1f}% valid bboxes - Bbox extraction is working!{Colors.ENDC}"
    elif valid_pct > 0:
        return "PARTIAL", f"{Colors.WARNING}⚠️  PARTIAL: {valid_pct:.1f}% valid bboxes - Some bboxes extracted, but needs improvement{Colors.ENDC}"
    else:
        return "FAILED", f"{Colors.FAIL}❌ FAILED: 0% valid bboxes - All bboxes are [0,0,0,0] - Fix needed!{Colors.ENDC}"

def main():
    """Main test function"""
    print_header("DeepSeek-OCR Bounding Box Test")
    
    try:
        # Step 1: Find test image
        image_path = find_test_image()
        
        # Step 2: Import and initialize DeepSeekOCR
        print_step(2, "Initializing DeepSeek-OCR...")
        try:
            from deepseek_ocr_wrapper import DeepSeekOCR
            print_info("Importing DeepSeekOCR wrapper...")
        except ImportError as e:
            print_error(f"Failed to import DeepSeekOCR: {e}")
            print_info("Make sure deepseek_ocr_wrapper.py is in the project root")
            sys.exit(1)
        
        try:
            print_info("Initializing model (this may take a while on first run)...")
            ocr = DeepSeekOCR()
            print_success("DeepSeek-OCR initialized successfully")
        except Exception as e:
            print_error(f"Failed to initialize DeepSeek-OCR: {e}")
            print_info("Make sure the model is properly installed and dependencies are available")
            sys.exit(1)
        
        # Step 3: Process image
        print_step(3, f"Processing image: {os.path.basename(image_path)}")
        try:
            print_info("Running OCR inference...")
            result = ocr.process(
                image_path,
                extract_structure=True,
                base_size=640,  # Smaller size for faster testing
                image_size=640
            )
            print_success("Image processed successfully")
        except Exception as e:
            print_error(f"Failed to process image: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Step 4: Extract structured data
        print_step(4, "Extracting structured data...")
        structured_data = result.get('structured_data', [])
        if not structured_data:
            print_warning("No structured data found in result")
            print_info("Result keys: " + str(list(result.keys())))
            structured_data = []
        else:
            print_success(f"Found {len(structured_data)} text blocks")
        
        # Step 5: Analyze bboxes
        print_step(5, "Analyzing bounding boxes...")
        stats = analyze_bboxes(structured_data)
        
        # Step 6: Print statistics
        print_bbox_statistics(stats)
        
        # Step 7: Determine verdict
        print_step(7, "Test Verdict")
        verdict, verdict_msg = get_verdict(stats)
        print(f"\n{Colors.BOLD}{verdict_msg}{Colors.ENDC}\n")
        
        # Step 8: Save results
        print_step(8, "Saving results to test_bbox_results.json...")
        results = {
            'test_timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'verdict': verdict,
            'statistics': stats,
            'full_result': {
                'total_blocks': len(structured_data),
                'full_text_length': len(result.get('full_text', '')),
                'has_markdown': bool(result.get('markdown_output')),
                'metadata': result.get('metadata', {})
            },
            'sample_structured_data': structured_data[:20]  # First 20 blocks
        }
        
        output_file = 'test_bbox_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print_success(f"Results saved to {output_file}")
        
        # Final summary
        print_header("Test Complete")
        print(f"{Colors.BOLD}Summary:{Colors.ENDC}")
        print(f"  Image: {os.path.basename(image_path)}")
        print(f"  Total Blocks: {stats['total']}")
        print(f"  Valid Bboxes: {stats['valid']} ({stats['valid_percentage']:.1f}%)")
        print(f"  Invalid Bboxes: {stats['invalid']} ({stats['invalid_percentage']:.1f}%)")
        print(f"  Verdict: {verdict}")
        print(f"\n{Colors.OKBLUE}Full results saved to: {output_file}{Colors.ENDC}\n")
        
        # Exit code based on verdict
        if verdict == "SUCCESS":
            sys.exit(0)
        elif verdict == "PARTIAL":
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # Failed
            
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

