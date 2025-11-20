"""
Basic OCR Example

This example demonstrates how to use the DeepSeekOCR wrapper for basic OCR processing.

PURPOSE:
    Shows the simplest way to:
    - Initialize the DeepSeekOCR model
    - Process a single image
    - Extract and display text results

USAGE:
    python examples/example_basic_ocr.py
    
    Or modify the image_path variable to point to your own image.

WHAT IT DOES:
    1. Initializes the DeepSeekOCR model
    2. Processes a test image
    3. Displays extracted text and metadata
    4. Shows error handling

NOTE:
    This is an EXAMPLE script, not a unit test. For actual testing,
    use pytest or unittest framework in a tests/ directory.
"""
import os
import sys

# Add parent directory to path (so we can import from root)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from deepseek_ocr_wrapper import DeepSeekOCR

# Example: Process an image
# Change this path to your own image file
image_path = 'uploads/e8105941-3b57-4fb3-855f-47d581d5fb2f_19840015553_page_1.png'

if not os.path.exists(image_path):
    print(f"Image not found: {image_path}")
    print("\nPlease update the 'image_path' variable in this script to point to a valid image file.")
    print("Example: image_path = 'path/to/your/image.png'")
    sys.exit(1)

print("=" * 60)
print("Basic OCR Example - DeepSeek OCR Processing")
print("=" * 60)
print(f"Image: {image_path}")
print(f"Image exists: {os.path.exists(image_path)}")
print()

try:
    print("Step 1: Initializing DeepSeekOCR model...")
    print("   (This may take a while on first run - model will be downloaded)")
    model = DeepSeekOCR()
    print(f"   ✓ Model initialized on device: {model.device}")
    print()
    
    print("Step 2: Processing image...")
    result = model.process(image_path, base_size=640, image_size=640, crop_mode=False)
    
    print("   ✓ Processing completed")
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Text extracted: {len(result.get('full_text', ''))} characters")
    print(f"Lines: {result.get('metadata', {}).get('total_lines', 0)}")
    print(f"Words: {result.get('metadata', {}).get('total_words', 0)}")
    print(f"Average confidence: {result.get('metadata', {}).get('average_confidence', 0.0):.2f}")
    
    if result.get('full_text'):
        print(f"\nFirst 200 characters of extracted text:")
        print("-" * 60)
        print(result.get('full_text', '')[:200])
        print("-" * 60)
    
    print("\n✓ Example completed successfully!")
    
except Exception as e:
    print()
    print("=" * 60)
    print("Error Occurred")
    print("=" * 60)
    print(f"Error: {str(e)}")
    print("\nTroubleshooting:")
    print("  1. Ensure the image file exists and is readable")
    print("  2. Check that DeepSeekOCR dependencies are installed")
    print("  3. Verify you have internet connection (for model download)")
    print("  4. Check that you have sufficient disk space (~several GB)")
    print()
    import traceback
    traceback.print_exc()
    sys.exit(1)

